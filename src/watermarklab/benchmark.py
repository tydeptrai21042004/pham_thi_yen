from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary, save_image, list_image_files
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.common.metrics import psnr, ssim, nc, ncc, ber
from watermarklab.methods import build_methods, BASELINE_METHOD_IDS
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams, ProposalQHDWTHess, optimization_param_snapshot
from watermarklab.paper_reported import write_paper_reported


OPT_PARAM_FORMAT = "watermarklab_proposal_optimization_params_v1"
DEFAULT_PROPOSAL_PARAM_FILE = "results/proposal_optimized_params.json"


def _safe_num(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return str(x)
    return x


def _parse_repeat(value: str | int | None):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s in {"auto", "none", "full", "faithful", "script"}:
        return None
    return int(s)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return _safe_num(obj)


def _extract_params_dict(record: Any) -> dict[str, Any] | None:
    """Extract a parameter dictionary from several supported file formats."""
    if record is None:
        return None
    if isinstance(record, dict):
        # Our optimization snapshot: {q4_tau, ..., source_script_names: {Q4_TAU, ...}}
        if any(k in record for k in ["q4_tau", "Q4_TAU", "h01_q", "H01_Q"]):
            return dict(record.get("source_script_names", record)) if "source_script_names" in record and not any(k in record for k in ["q4_tau", "h01_q"]) else dict(record)
        for key in ["best_params", "params", "parameter_snapshot"]:
            if key in record:
                found = _extract_params_dict(record[key])
                if found:
                    return found
    return None


def load_proposal_param_file(path: str | Path) -> dict[str, Any]:
    """Load optimized proposal parameters if present.

    Supported inputs:
      1. This package's ``proposal_optimized_params.json``.
      2. A standalone script ``best_params.json`` containing uppercase keys.
      3. A standalone/script CSV such as ``all_images_optimal_params.csv``.
    """
    path = Path(path)
    if not path.exists():
        return {"loaded": False, "path": str(path), "reason": "file_not_found", "per_image": {}, "global_params": None}

    per_image: dict[str, dict[str, Any]] = {}
    global_params: dict[str, Any] | None = None
    raw: Any = None

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rec = row.to_dict()
            params = {k.replace("param_", ""): v for k, v in rec.items() if str(k).startswith("param_") and pd.notna(v)}
            params = _extract_params_dict(params) or params
            host = str(rec.get("host_image", rec.get("image", ""))).strip()
            if host and params:
                per_image[host] = params
                per_image[Path(host).stem] = params
        if per_image:
            first_key = next(iter(per_image))
            global_params = per_image[first_key]
        raw = {"csv_rows": len(df)}
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            # Standalone all_images_best_results.json style.
            for rec in raw:
                params = _extract_params_dict(rec)
                host = str(rec.get("host_image", rec.get("image", ""))).strip() if isinstance(rec, dict) else ""
                if host and params:
                    per_image[host] = params
                    per_image[Path(host).stem] = params
            if raw:
                global_params = _extract_params_dict(raw[0])
        elif isinstance(raw, dict):
            if raw.get("format") == OPT_PARAM_FORMAT:
                for key, rec in dict(raw.get("per_image", {})).items():
                    params = _extract_params_dict(rec)
                    if params:
                        per_image[str(key)] = params
                        per_image[Path(str(key)).stem] = params
                        if isinstance(rec, dict) and rec.get("filename"):
                            per_image[str(rec["filename"])] = params
                global_params = _extract_params_dict(raw.get("global_best"))
            else:
                # Single best_params.json or best_result.json.
                global_params = _extract_params_dict(raw)

    return {
        "loaded": bool(global_params or per_image),
        "path": str(path),
        "per_image": per_image,
        "global_params": global_params,
        "raw_format": raw.get("format") if isinstance(raw, dict) else type(raw).__name__,
    }


def _select_proposal_params_for_image(host_path: Path, base_params: ProposalParams, loaded_payload: dict[str, Any] | None):
    if not loaded_payload or not loaded_payload.get("loaded"):
        return ProposalParams.from_dict(base_params.to_dict()), "default"
    per_image = dict(loaded_payload.get("per_image", {}))
    for key in [host_path.name, host_path.stem, str(host_path)]:
        if key in per_image:
            merged = base_params.to_dict()
            merged.update(per_image[key])
            return ProposalParams.from_dict(merged), f"optimized:{loaded_payload.get('path')}:{key}"
    if loaded_payload.get("global_params"):
        merged = base_params.to_dict()
        merged.update(loaded_payload["global_params"])
        return ProposalParams.from_dict(merged), f"optimized:{loaded_payload.get('path')}:global"
    return ProposalParams.from_dict(base_params.to_dict()), "default"


def run_benchmark(
    host_dir: str | Path,
    watermark_path: str | Path,
    output_dir: str | Path,
    selected_methods: list[str] | None = None,
    max_images: int | None = None,
    save_outputs: bool = True,
    invert_watermark: bool = False,
    attack_preset: str = "lite",
    proposal_options: dict[str, Any] | None = None,
    baseline_modes: dict[str, str] | None = None,
):
    host_dir = Path(host_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_modes = dict(baseline_modes or {})
    proposal_options = dict(proposal_options or {})
    optimized_payload = proposal_options.pop("optimized_payload", None)

    if selected_methods == ["baselines"]:
        selected_methods = list(BASELINE_METHOD_IDS)
    elif selected_methods == ["all"]:
        selected_methods = None

    original_report_methods = [m for m, mode in baseline_modes.items() if mode == "original" and (selected_methods is None or m in selected_methods)]
    if original_report_methods:
        write_paper_reported(output_dir / "original_reported", original_report_methods)
    runtime_selected = None if selected_methods is None else [m for m in selected_methods if m not in original_report_methods]
    if runtime_selected == []:
        return {"results": pd.DataFrame(), "summary": pd.DataFrame(), "comparison": pd.DataFrame(), "failures": [], "reported_methods": original_report_methods}

    runtime_baseline_modes = {m: mode for m, mode in baseline_modes.items() if mode != "original"}
    methods = build_methods(runtime_selected, proposal_options=proposal_options, baseline_modes=runtime_baseline_modes)
    host_paths = list_image_files(host_dir)
    if max_images is not None:
        host_paths = host_paths[: int(max_images)]
    if not host_paths:
        raise ValueError(f"No host images found in {host_dir}")

    watermark = load_watermark_binary(watermark_path, invert=invert_watermark)
    attacks = default_attack_suite(include_none=True, preset=attack_preset)

    rows: list[dict] = []
    failures: list[dict] = []

    for method_id, method in methods.items():
        for host_path in host_paths:
            image_name = host_path.stem
            host = load_host_rgb(host_path)
            param_source = "not_proposal"
            try:
                if method_id == "proposal" and isinstance(method, ProposalQHDWTHess):
                    selected_params, param_source = _select_proposal_params_for_image(host_path, method.params, optimized_payload)
                    method.params = selected_params
                    # In two-phase normal mode, do not run per-image optimizer again.
                    if optimized_payload and optimized_payload.get("loaded"):
                        method.use_optimizer = False

                t0 = time.perf_counter()
                watermarked, key = method.embed(host, watermark)
                embed_time = time.perf_counter() - t0
                t1 = time.perf_counter()
                extracted_clean = method.extract(watermarked, key, host_rgb=host)
                extract_clean_time = time.perf_counter() - t1

                key_info = {"proposal_param_source": param_source} if method_id == "proposal" else {}
                if hasattr(key, "repeat_factor"):
                    key_info["proposal_repeat_factor"] = getattr(key, "repeat_factor")
                    key_info["proposal_usable_blocks"] = getattr(key, "usable_blocks")
                    key_info["proposal_total_blocks"] = getattr(key, "total_blocks")
                    flags = getattr(key, "flags", [])
                    key_info["proposal_q4_used"] = int(sum(1 for x in flags if int(x) == 0))
                    key_info["proposal_h_used"] = int(sum(1 for x in flags if int(x) == 1))
                    key_info["proposal_skip_used"] = int(sum(1 for x in flags if int(x) == 2))
                    if hasattr(key, "params"):
                        key_info["proposal_q4_tau"] = float(key.params.q4_tau)
                        key_info["proposal_q4_margin"] = float(key.params.q4_margin)
                        key_info["proposal_h01_q"] = float(key.params.h01_q)
                        key_info["proposal_h01_margin"] = float(key.params.h01_margin)
                        key_info["proposal_dwt_mode"] = str(key.params.dwt_mode)

                if save_outputs:
                    base = output_dir / "images" / method_id / image_name
                    save_image(base / "watermarked.png", watermarked)
                    save_image(base / "extracted_no_attack.png", extracted_clean)

                rows.append({
                    "method_id": method_id,
                    "method_name": method.name,
                    "image": image_name,
                    "attack": "no_attack",
                    "phase": "before_attack",
                    "psnr": psnr(host, watermarked),
                    "ssim": ssim(host, watermarked),
                    "nc": nc(watermark, extracted_clean),
                    "ncc": ncc(watermark, extracted_clean),
                    "ber": ber(watermark, extracted_clean),
                    "embed_time_sec": embed_time,
                    "extract_time_sec": extract_clean_time,
                    **key_info,
                })

                for attack in attacks:
                    if attack.name == "no_attack":
                        continue
                    try:
                        attacked = apply_attack(watermarked, attack)
                        t2 = time.perf_counter()
                        extracted = method.extract(attacked, key, host_rgb=host)
                        extract_time = time.perf_counter() - t2
                        if save_outputs:
                            base = output_dir / "images" / method_id / image_name
                            if attack.name in {"jpeg_q70", "jpeg_q90", "gaussian_noise_sigma5", "rotation_2deg", "script_rotation_45deg"}:
                                save_image(base / f"attacked_{attack.name}.png", attacked)
                                save_image(base / f"extracted_{attack.name}.png", extracted)
                        rows.append({
                            "method_id": method_id,
                            "method_name": method.name,
                            "image": image_name,
                            "attack": attack.name,
                            "phase": "after_attack",
                            "psnr": psnr(host, attacked),
                            "ssim": ssim(host, attacked),
                            "nc": nc(watermark, extracted),
                            "ncc": ncc(watermark, extracted),
                            "ber": ber(watermark, extracted),
                            "embed_time_sec": embed_time,
                            "extract_time_sec": extract_time,
                            **key_info,
                        })
                    except Exception as e:
                        failures.append({"method_id": method_id, "image": image_name, "attack": attack.name, "error": repr(e)})
            except Exception as e:
                failures.append({"method_id": method_id, "image": image_name, "attack": "embedding_or_clean_extraction", "error": repr(e)})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_image_attack_results.csv", index=False)

    if not df.empty:
        summary = (
            df.groupby(["method_id", "method_name", "phase"], dropna=False)
            .agg(
                psnr_mean=("psnr", "mean"),
                ssim_mean=("ssim", "mean"),
                nc_mean=("nc", "mean"),
                ncc_mean=("ncc", "mean"),
                ber_mean=("ber", "mean"),
                psnr_std=("psnr", "std"),
                nc_min=("nc", "min"),
                ber_max=("ber", "max"),
                images=("image", "nunique"),
                rows=("image", "count"),
                embed_time_mean_sec=("embed_time_sec", "mean"),
                extract_time_mean_sec=("extract_time_sec", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "summary_by_method_phase.csv", index=False)
        pivot = summary.pivot_table(
            index=["method_id", "method_name"],
            columns="phase",
            values=["psnr_mean", "nc_mean", "ncc_mean", "ber_mean"],
        )
        pivot.columns = [f"{metric}_{phase}" for metric, phase in pivot.columns]
        pivot = pivot.reset_index()
        pivot.to_csv(output_dir / "compare_psnr_nc_ber_ncc_before_after_attack.csv", index=False)
    else:
        summary = pd.DataFrame()
        pivot = pd.DataFrame()

    with open(output_dir / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False, default=_json_default)

    return {"results": df, "summary": summary, "comparison": pivot, "failures": failures, "reported_methods": original_report_methods}


def run_proposal_optimization_phase(
    host_dir: str | Path,
    watermark_path: str | Path,
    output_file: str | Path,
    *,
    max_images: int | None = None,
    invert_watermark: bool = False,
    repeat: int | None = None,
    attack_preset: str = "lite",
    n_fireflies: int = 4,
    n_generations: int = 2,
    alpha: float = 0.18,
    beta0: float = 1.0,
    gamma: float = 1.0,
    alpha_decay: float = 0.80,
    seed: int = 123,
) -> dict[str, Any]:
    host_dir = Path(host_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    host_paths = list_image_files(host_dir)
    if max_images is not None:
        host_paths = host_paths[: int(max_images)]
    if not host_paths:
        raise ValueError(f"No host images found in {host_dir}")

    watermark = load_watermark_binary(watermark_path, invert=invert_watermark)
    attack_suite = default_attack_suite(include_none=False, preset=attack_preset)
    base_params = ProposalParams.from_dict({"repeat": repeat})

    per_image: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    global_best_obj = -1e18

    for idx, host_path in enumerate(host_paths):
        host = load_host_rgb(host_path)
        method = ProposalQHDWTHess(params=base_params, use_optimizer=False, optimizer_trials=n_fireflies, optimizer_seed=seed + idx * 100000)
        best_params, record = method.optimize_params(
            host,
            watermark,
            n_fireflies=n_fireflies,
            n_generations=n_generations,
            alpha=alpha,
            beta0=beta0,
            gamma=gamma,
            alpha_decay=alpha_decay,
            seed=seed + idx * 100000,
            attack_suite=attack_suite,
        )
        best_result = dict(record.get("best_result", {}))
        objective = float(best_result.get("objective", -1e9))
        entry = {
            "image": host_path.stem,
            "filename": host_path.name,
            "params": best_params.to_dict(),
            "optimized_params": optimization_param_snapshot(best_params),
            "objective": objective,
            "best_result": best_result,
            "optimizer": record.get("optimizer", {}),
            "history": record.get("history", []),
        }
        per_image[host_path.stem] = entry
        per_image[host_path.name] = entry
        if objective > global_best_obj:
            global_best_obj = objective
            global_best = {"image": host_path.stem, "filename": host_path.name, "params": best_params.to_dict(), "optimized_params": optimization_param_snapshot(best_params), "objective": objective}
        row = {
            "image": host_path.stem,
            "filename": host_path.name,
            "objective": objective,
            "clean_psnr": best_result.get("clean_psnr"),
            "clean_nc": best_result.get("clean_nc"),
            "clean_ber": best_result.get("clean_ber"),
            "mean_attack_nc": best_result.get("mean_attack_nc"),
            "min_attack_nc": best_result.get("min_attack_nc"),
            "q4_used": best_result.get("q4_used"),
            "hpos_used": best_result.get("hpos_used"),
            "skip_used": best_result.get("skip_used"),
            "repeat_factor": best_result.get("repeat_factor"),
            "error": best_result.get("error"),
            "q4_tau": float(best_params.q4_tau),
            "q4_margin": float(best_params.q4_margin),
            "h01_q": float(best_params.h01_q),
            "h01_margin": float(best_params.h01_margin),
        }
        rows.append(row)
        print(f"[OPT] {host_path.name}: objective={objective:.6f}, q4_tau={best_params.q4_tau:.6f}, q4_margin={best_params.q4_margin:.6f}, h01_q={best_params.h01_q:.6f}, h01_margin={best_params.h01_margin:.6f}")

    payload = {
        "format": OPT_PARAM_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_dir": str(host_dir),
        "watermark_path": str(watermark_path),
        "attack_preset": str(attack_preset),
        "repeat": "full" if repeat is None else int(repeat),
        "param_names": ["q4_tau", "q4_margin", "h01_q", "h01_margin"],
        "global_best": global_best,
        "per_image": per_image,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    csv_path = output_file.with_suffix(".csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved optimized proposal parameters to: {output_file}")
    print(f"Saved optimized proposal parameter CSV to: {csv_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run the cleaned watermarking benchmark on 512x512 RGB host images and a 64x64 binary watermark.")
    parser.add_argument("--phase", default="normal", choices=["normal", "optimize", "optimization"], help="normal: run benchmark; optimize: search proposal parameters and export them.")
    parser.add_argument("--host-dir", default="data/host")
    parser.add_argument("--watermark", default="data/watermark/wm.png")
    parser.add_argument("--output", default="results/common_benchmark")
    parser.add_argument("--methods", default="all", help="Comma-separated: all,baselines,kumar2021,guo2017_dwt_qr_fa,gaata2022_dwt_hess_fwa,dwt_hd_svd_2025,proposal")
    parser.add_argument("--max-images", type=int, default=None, help="Optional quick-run limit.")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--invert-watermark", action="store_true")
    parser.add_argument("--attack-preset", default="lite", choices=["none", "lite", "full", "stress", "script", "grid"], help="Attack preset for normal benchmark.")
    parser.add_argument("--baseline-mode", default="adapt", choices=["adapt", "original", "original-rerun"], help="Global baseline mode: adapt runs local adapted benchmark; original writes paper-reported tables; original-rerun attempts a stricter local rerun.")
    parser.add_argument("--kumar-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--guo-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--gaata-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--dwt-hd-svd-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])

    parser.add_argument("--proposal-param-file", default=DEFAULT_PROPOSAL_PARAM_FILE, help="JSON/CSV file written by optimization phase. Normal phase loads it automatically when it exists.")
    parser.add_argument("--proposal-param-mode", default="auto", choices=["auto", "ignore", "require"], help="auto: use param file if present; ignore: always defaults; require: fail if missing.")
    parser.add_argument("--proposal-use-optimizer", action="store_true", help="Legacy one-pass adaptive mode. For reproducible two-phase use --phase optimize first, then --phase normal.")
    parser.add_argument("--proposal-optimizer-trials", type=int, default=4, help="Legacy alias for --proposal-optimizer-fireflies.")
    parser.add_argument("--proposal-optimizer-fireflies", type=int, default=None)
    parser.add_argument("--proposal-optimizer-generations", type=int, default=2)
    parser.add_argument("--proposal-optimizer-attack-preset", default="lite", choices=["none", "lite", "full", "stress", "script", "grid"], help="Attack preset used while optimizing parameters.")
    parser.add_argument("--proposal-optimizer-alpha", type=float, default=0.18)
    parser.add_argument("--proposal-optimizer-beta0", type=float, default=1.0)
    parser.add_argument("--proposal-optimizer-gamma", type=float, default=1.0)
    parser.add_argument("--proposal-optimizer-alpha-decay", type=float, default=0.80)
    parser.add_argument("--proposal-optimizer-seed", type=int, default=123)
    parser.add_argument("--proposal-repeat", default="full", help="Default full/faithful uses all source-script structured repetition; use an integer such as 3 only for quick practical runs.")

    args = parser.parse_args()
    fireflies = int(args.proposal_optimizer_fireflies if args.proposal_optimizer_fireflies is not None else args.proposal_optimizer_trials)
    repeat_value = _parse_repeat(args.proposal_repeat)

    if args.phase in {"optimize", "optimization"}:
        run_proposal_optimization_phase(
            host_dir=args.host_dir,
            watermark_path=args.watermark,
            output_file=args.proposal_param_file,
            max_images=args.max_images,
            invert_watermark=args.invert_watermark,
            repeat=repeat_value,
            attack_preset=args.proposal_optimizer_attack_preset,
            n_fireflies=fireflies,
            n_generations=int(args.proposal_optimizer_generations),
            alpha=float(args.proposal_optimizer_alpha),
            beta0=float(args.proposal_optimizer_beta0),
            gamma=float(args.proposal_optimizer_gamma),
            alpha_decay=float(args.proposal_optimizer_alpha_decay),
            seed=int(args.proposal_optimizer_seed),
        )
        return

    selected = [s.strip() for s in args.methods.split(",") if s.strip()]
    if selected == ["all"]:
        selected = None

    def _mode(value: str) -> str:
        return args.baseline_mode if value == "inherit" else value

    baseline_modes = {
        "kumar2021": _mode(args.kumar_mode),
        "guo2017_dwt_qr_fa": _mode(args.guo_mode),
        "gaata2022_dwt_hess_fwa": _mode(args.gaata_mode),
        "dwt_hd_svd_2025": _mode(args.dwt_hd_svd_mode),
    }

    optimized_payload = None
    if args.proposal_param_mode != "ignore":
        optimized_payload = load_proposal_param_file(args.proposal_param_file)
        if args.proposal_param_mode == "require" and not optimized_payload.get("loaded"):
            raise FileNotFoundError(f"Required proposal parameter file was not loaded: {args.proposal_param_file}")
        if optimized_payload.get("loaded"):
            print(f"[NORMAL] Loaded optimized proposal parameters from: {args.proposal_param_file}")
        else:
            print(f"[NORMAL] No optimized proposal parameter file found; using default proposal parameters. Checked: {args.proposal_param_file}")

    proposal_options = {
        "use_optimizer": bool(args.proposal_use_optimizer),
        "optimizer_trials": fireflies,
        "params": {"repeat": repeat_value, "dwt_mode": "pywt"},
        "optimized_payload": optimized_payload,
    }

    result = run_benchmark(
        host_dir=args.host_dir,
        watermark_path=args.watermark,
        output_dir=args.output,
        selected_methods=selected,
        max_images=args.max_images,
        save_outputs=not args.no_save_images,
        invert_watermark=args.invert_watermark,
        attack_preset=args.attack_preset,
        proposal_options=proposal_options,
        baseline_modes=baseline_modes,
    )
    if not result["results"].empty:
        print(f"Saved per-image results to: {Path(args.output) / 'per_image_attack_results.csv'}")
        print(f"Saved comparison table to: {Path(args.output) / 'compare_psnr_nc_ber_ncc_before_after_attack.csv'}")
        print(result["comparison"].to_string(index=False) if not result["comparison"].empty else "No successful rows")
    if result.get("reported_methods"):
        print(f"Saved original paper-reported tables to: {Path(args.output) / 'original_reported' / 'paper_reported_results.csv'}")
    if result["failures"]:
        print(f"Failures: {len(result['failures'])}. See failures.json")


if __name__ == "__main__":
    main()
