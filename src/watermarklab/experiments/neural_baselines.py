"""Optional *pretrained* neural watermark comparisons on the project's host images.

TrustMark:    pip install trustmark
ChunkySeal:   install facebookresearch/videoseal and a compatible torch build.

These are native-capacity benchmarks (TrustMark BCH_5: typically 61 information
bits; ChunkySeal: 1024 information bits), NOT 4096-bit matched comparisons.
No neural model is retrained on the test images.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from watermarklab.common.attack import apply_attack, default_attack_suite
from watermarklab.common.metrics import psnr, ssim
from watermarklab.experiments.review_audit import load_rgb

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METHODS = ("trustmark", "chunkyseal")
COLUMNS = [
    "method_id", "method_name", "image", "attack", "phase",
    "psnr", "ssim", "nc", "ncc", "ber", "embed_time_sec",
    "extract_time_sec", "native_capacity_bits", "payload_kind",
    "decoded", "detected", "ber_if_missing_one", "model_device",
    "attack_applied", "error",
]


class TrustMarkAdapter:
    """Official TrustMark binary encoder; weights downloaded by upstream on first use."""
    name = "TrustMark"

    def __init__(self, model_type: str = "Q", device: str = "auto"):
        try:
            from trustmark import TrustMark
        except ImportError as exc:
            raise RuntimeError("TrustMark dependency unavailable. Install: pip install trustmark") from exc
        self.model = TrustMark(verbose=False, model_type=model_type, encoding_type=TrustMark.Encoding.BCH_5)
        self.capacity = int(self.model.schemaCapacity())
        self.device = "upstream-selected"
        if self.capacity <= 0:
            raise ValueError(f"Invalid TrustMark native payload capacity: {self.capacity}")

    def embed(self, rgb: np.ndarray, bits: np.ndarray) -> np.ndarray:
        encoded = "".join(str(int(bit)) for bit in bits)
        if len(encoded) != self.capacity:
            raise ValueError(f"TrustMark requires {self.capacity} information bits")
        image = self.model.encode(Image.fromarray(rgb), encoded, MODE="binary")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray | None, bool]:
        decoded, present, _schema = self.model.decode(Image.fromarray(rgb), MODE="binary")
        if not present or not isinstance(decoded, str) or len(decoded) != self.capacity:
            return None, False
        if set(decoded) - {"0", "1"}:
            return None, False
        return np.fromiter((int(char) for char in decoded), dtype=np.uint8), True


class ChunkySealAdapter:
    """Official VideoSeal model zoo's 1024-bit pretrained ChunkySeal model."""
    name = "ChunkySeal"

    def __init__(self, model_type: str = "Q", device: str = "auto"):
        try:
            import torch
            import videoseal
        except ImportError as exc:
            raise RuntimeError(
                "ChunkySeal requires torch and facebookresearch/videoseal; see docs/NEURAL_BASELINES.md"
            ) from exc
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for ChunkySeal but CUDA is unavailable")
        # Upstream model cards point to configs/attenuation.yaml. Some pip builds
        # omit this *data* file despite downloading the model checkpoint.
        # Recover ONLY from an official source checkout, never invent a YAML.
        package_dir = Path(videoseal.__file__).resolve().parent
        required = package_dir / "configs" / "attenuation.yaml"
        if not required.is_file():
            candidates = [
                package_dir.parent / "configs" / "attenuation.yaml",
                Path.cwd() / "configs" / "attenuation.yaml",
                Path.cwd() / "videoseal-src" / "configs" / "attenuation.yaml",
                Path('/kaggle/working/videoseal-src/configs/attenuation.yaml'),
            ]
            available = next((p for p in candidates if p.is_file()), None)
            if available is None:
                raise RuntimeError(
                    "VideoSeal's official configs/attenuation.yaml is missing. "
                    "Clone facebookresearch/videoseal into /kaggle/working/videoseal-src "
                    "and install it in editable mode before loading ChunkySeal."
                )
            required.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(available, required)
        self.model = videoseal.load("chunkyseal").eval().to(self.device)
        self.capacity = 1024

    def _tensor(self, rgb: np.ndarray):
        return (self.torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0)

    def embed(self, rgb: np.ndarray, bits: np.ndarray) -> np.ndarray:
        if bits.size != self.capacity:
            raise ValueError(f"ChunkySeal requires {self.capacity} bits")
        with self.torch.no_grad():
            msg = self.torch.from_numpy(bits.reshape(1, -1).astype(np.float32)).to(self.device)
            result = self.model.embed(self._tensor(rgb), msgs=msg, is_video=False)
        marked = result["imgs_w"]
        if marked.ndim != 4 or marked.shape[0] != 1:
            raise RuntimeError(f"Unexpected ChunkySeal image output shape: {tuple(marked.shape)}")
        marked = marked[0].detach().cpu().clamp(0, 1)
        return (marked.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, None]:
        with self.torch.no_grad():
            result = self.model.detect(self._tensor(rgb), is_video=False)
        predictions = result["preds"].detach().cpu().float()
        # Expected [batch, capacity + 1] or [batch, capacity + 1, H, W].
        if predictions.shape[0] != 1 or predictions.shape[1] != self.capacity + 1:
            raise RuntimeError(f"Unexpected ChunkySeal prediction shape: {tuple(predictions.shape)}")
        if predictions.ndim > 2:
            predictions = predictions.mean(dim=tuple(range(2, predictions.ndim)))
        bits = (predictions[0, 1:] > 0).numpy().astype(np.uint8)
        # Do NOT invent a calibrated presence threshold from the first logit.
        return bits, None


def _nc_ncc(reference: np.ndarray, estimated: np.ndarray) -> tuple[float, float]:
    a = reference.astype(np.float64)
    b = estimated.astype(np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    nc = float(np.dot(a, b) / den) if den else float(np.array_equal(reference, estimated))
    x, y = a - a.mean(), b - b.mean()
    centered_den = np.linalg.norm(x) * np.linalg.norm(y)
    ncc = float(np.dot(x, y) / centered_den) if centered_den else float(np.array_equal(reference, estimated))
    return float(np.clip(nc, -1, 1)), float(np.clip(ncc, -1, 1))


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _create_adapter(name: str, model_type: str = "Q", device: str = "auto"):
    if name == "trustmark":
        return TrustMarkAdapter(model_type=model_type, device=device)
    if name == "chunkyseal":
        return ChunkySealAdapter(device=device)
    raise ValueError(f"Unknown neural model: {name}")


def _select_methods(methods: str) -> list[str]:
    selected = [v.strip().lower() for v in methods.split(",") if v.strip()]
    if selected in (["all"], ["neural"], ["neural-all"]):
        return list(METHODS)
    if not selected or any(v not in METHODS for v in selected):
        raise ValueError("Neural phase --methods must be: trustmark,chunkyseal,all")
    return list(dict.fromkeys(selected))


def _hosts(host_dir: Path, max_images: int | None) -> list[Path]:
    if host_dir.is_file() and host_dir.suffix.lower() in IMAGE_EXTENSIONS:
        candidates = [host_dir]
    else:
        if not host_dir.is_dir():
            raise FileNotFoundError(f"Host directory does not exist: {host_dir}")
        candidates = sorted(p for p in host_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images must be >= 1")
        candidates = candidates[:max_images]
    if not candidates:
        raise ValueError(f"No supported images in {host_dir}")
    return candidates


def _measurement(adapter, source: np.ndarray, watermarked: np.ndarray, image: np.ndarray,
                 payload: np.ndarray, image_name: str, attack: str, phase: str,
                 embed_seconds: float, attack_applied: bool) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        estimated, present = adapter.detect(image)
        extract_seconds = time.perf_counter() - start
        if estimated is not None:
            estimated = np.asarray(estimated, dtype=np.uint8).ravel()
            if estimated.size != payload.size or not np.isin(estimated, [0, 1]).all():
                raise ValueError(f"Invalid decoded bit vector: expected {payload.size} binary bits, got {estimated.size}")
        valid_payload = estimated is not None and present is not False and phase != "control"
        if valid_payload:
            raw_ber = float(np.mean(payload != estimated))
            corr, centered_corr = _nc_ncc(payload, estimated)
        else:
            raw_ber = corr = centered_corr = None
        return dict(method_id=adapter.name.lower(), method_name=adapter.name, image=image_name,
                    attack=attack, phase=phase, psnr=psnr(source, image), ssim=ssim(source, image),
                    nc=corr, ncc=centered_corr, ber=raw_ber, embed_time_sec=embed_seconds,
                    extract_time_sec=extract_seconds, native_capacity_bits=adapter.capacity,
                    payload_kind="native_random_information_bits", decoded=bool(valid_payload),
                    detected=present, ber_if_missing_one=(raw_ber if raw_ber is not None else (1.0 if phase != "control" else None)),
                    model_device=adapter.device, attack_applied=attack_applied, error="")
    except Exception as exc:
        return dict(method_id=adapter.name.lower(), method_name=adapter.name, image=image_name,
                    attack=attack, phase=phase, psnr=psnr(source, image), ssim=ssim(source, image),
                    nc=None, ncc=None, ber=None, embed_time_sec=embed_seconds,
                    extract_time_sec=time.perf_counter() - start, native_capacity_bits=adapter.capacity,
                    payload_kind="native_random_information_bits", decoded=False,
                    detected=None, ber_if_missing_one=(1.0 if phase != "control" else None),
                    model_device=adapter.device, attack_applied=attack_applied, error=repr(exc))


def _write_results(output: Path, rows: list[dict], failures: list[dict], manifest: dict) -> None:
    import pandas as pd
    _csv(output / "per_image_attack_results.csv", rows, COLUMNS)
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if rows:
        df = pd.DataFrame(rows)
        # Negative-control rows must never contaminate marked/attacked averages.
        measured = df[df.phase.isin(["before_attack", "after_attack"])].copy()
        summaries = []
        for (method_id, method_name, phase), frame in measured.groupby(["method_id", "method_name", "phase"], sort=True):
            summaries.append({
                "method_id": method_id, "method_name": method_name, "phase": phase,
                "n": len(frame), "decoded_n": int(frame.decoded.sum()),
                "decode_success_rate": float(frame.decoded.mean()),
                "native_capacity_bits": int(frame.native_capacity_bits.iloc[0]),
                **{f"{col}_mean": float(frame[col].mean()) for col in
                   ["psnr", "ssim", "nc", "ncc", "ber", "ber_if_missing_one", "embed_time_sec", "extract_time_sec"]},
            })
        summary = pd.DataFrame(summaries)
        summary.to_csv(output / "summary_by_method_phase.csv", index=False)
        if len(summary):
            wide = summary.pivot(index=["method_id", "method_name", "native_capacity_bits"],
                                 columns="phase", values=["psnr_mean", "ssim_mean", "nc_mean", "ncc_mean", "ber_mean", "ber_if_missing_one_mean", "decode_success_rate"])
            wide.columns = [f"{metric}_{phase}" for metric, phase in wide.columns]
            wide.reset_index().to_csv(output / "compare_psnr_nc_ber_ncc_before_after_attack.csv", index=False)
    manifest["successful_rows"] = len(rows)
    manifest["failures"] = len(failures)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_dataset(*, methods: str, host_dir: str | Path, output: str | Path,
                attack_preset: str = "lite", max_images: int | None = None,
                max_attacks: int = 0, seed: int = 3101, device: str = "auto",
                model_type: str = "Q", save_images: bool = False,
                adapter_factory=None) -> list[dict]:
    """Run all hosts, separate ML/DL outputs, checkpoint CSV after each host.

    `adapter_factory` exists only to allow deterministic offline integration tests.
    """
    if max_attacks < 0:
        raise ValueError("--neural-max-attacks must be >= 0 (0 means all)")
    selected = _select_methods(methods)
    hosts = _hosts(Path(host_dir), max_images)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = default_attack_suite(include_none=False, preset=attack_preset)
    if max_attacks:
        cases = cases[:max_attacks]
    rows: list[dict] = []
    failures: list[dict] = []
    manifest = {"phase": "neural", "methods_requested": selected, "host_count": len(hosts),
                "hosts": [p.name for p in hosts], "attack_preset": attack_preset,
                "attack_names": [c.name for c in cases], "seed": seed,
                "payload_kind": "native_random_information_bits",
                "capacity_matched_4096_bits": False,
                "comparability": "NOT payload-capacity matched to the 4096-bit classical benchmark",
                "device_requested": device, "model_variant": model_type,
                "upstream_checkpoint_identifiers": "record externally before publication",
                "status": "running"}
    success_methods = set()
    for method_idx, name in enumerate(selected):
        try:
            adapter = (adapter_factory(name, model_type, device) if adapter_factory
                       else _create_adapter(name, model_type, device))
        except Exception as exc:
            failures.append({"method": name, "stage": "model_load", "error": repr(exc),
                             "traceback": traceback.format_exc()})
            print(f"[NEURAL ERROR] model {name}: {exc}", flush=True)
            _write_results(output, rows, failures, manifest)
            continue
        successful_hosts = 0
        for idx, path in enumerate(hosts):
            try:
                original = load_rgb(path, None)
                # Deterministic by host index and method index; never tune to test hosts.
                rng = np.random.default_rng(np.random.SeedSequence([seed, method_idx, idx]))
                payload = rng.integers(0, 2, adapter.capacity, dtype=np.uint8)
                tic = time.perf_counter()
                watermarked = adapter.embed(original, payload)
                embed_sec = time.perf_counter() - tic
                if watermarked.shape != original.shape or watermarked.dtype != np.uint8:
                    raise ValueError("Neural adapter must return same-size uint8 RGB image")
                if save_images:
                    dest = output / "marked_images" / name
                    dest.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(watermarked).save(dest / (path.stem + ".png"))
                clean_row = _measurement(adapter, original, watermarked, watermarked, payload,
                                         path.stem, "no_attack", "before_attack", embed_sec, False)
                rows.append(clean_row)
                if clean_row["error"]:
                    failures.append({"method": name, "host": path.name, "stage": "clean_decode",
                                     "error": clean_row["error"]})
                rows.append(_measurement(adapter, original, watermarked, original, payload,
                                         path.stem, "original_unmarked", "control", embed_sec, False))
                for config in cases:
                    try:
                        attacked = apply_attack(watermarked, config)
                        row = _measurement(adapter, original, watermarked, attacked, payload,
                                           path.stem, config.name, "after_attack", embed_sec, True)
                        rows.append(row)
                        if row["error"]:
                            failures.append({"method": name, "host": path.name, "attack": config.name,
                                             "stage": "decode", "error": row["error"]})
                    except Exception as exc:
                        failures.append({"method": name, "host": path.name,
                                         "attack": config.name, "stage": "apply_attack", "error": repr(exc)})
                successful_hosts += 1
                print(f"[NEURAL] {name} {idx+1}/{len(hosts)}: {path.name} complete", flush=True)
            except Exception as exc:
                failures.append({"method": name, "host": path.name, "stage": "host", "error": repr(exc),
                                 "traceback": traceback.format_exc()})
                print(f"[NEURAL ERROR] {name} {path.name}: {exc}", flush=True)
            _write_results(output, rows, failures, manifest)  # supports Kaggle partial collection
        if successful_hosts:
            success_methods.add(name)
    manifest["status"] = "complete" if len(success_methods) == len(selected) and not failures else "partial_or_failed"
    manifest["methods_completed"] = sorted(success_methods)
    _write_results(output, rows, failures, manifest)
    if len(success_methods) != len(selected) or failures:
        raise RuntimeError(f"Neural benchmark incomplete: {len(success_methods)}/{len(selected)} methods ran; {len(failures)} errors. Inspect {output/'failures.json'}")
    return rows


def run(method_name: str, host: Path, out: Path, *, attack_preset: str = "lite",
        max_attacks: int = 5, seed: int = 3101):
    """Backward-compatible one-image interface (now uses the shared report schema)."""
    return run_dataset(methods=method_name, host_dir=host, output=out, attack_preset=attack_preset,
                       max_attacks=max_attacks, seed=seed, save_images=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attack-preset", default="lite")
    parser.add_argument("--max-attacks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args(argv)
    run(args.method, args.host, args.output, attack_preset=args.attack_preset,
        max_attacks=args.max_attacks, seed=args.seed)
    print(f"Saved neural results: {args.output/'compare_psnr_nc_ber_ncc_before_after_attack.csv'}")


if __name__ == "__main__":
    cli()
