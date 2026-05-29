from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary, save_image, list_image_files
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.common.metrics import psnr, ssim, nc, ncc, ber
from watermarklab.methods import build_methods, BASELINE_METHOD_IDS
from watermarklab.paper_reported import write_paper_reported


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
    if s in {"auto", "none", "full", "faithful"}:
        return None
    return int(s)


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
            try:
                t0 = time.perf_counter()
                watermarked, key = method.embed(host, watermark)
                embed_time = time.perf_counter() - t0
                t1 = time.perf_counter()
                extracted_clean = method.extract(watermarked, key, host_rgb=host)
                extract_clean_time = time.perf_counter() - t1
                key_info = {}
                if hasattr(key, "repeat_factor"):
                    key_info["proposal_repeat_factor"] = getattr(key, "repeat_factor")
                    key_info["proposal_usable_blocks"] = getattr(key, "usable_blocks")
                    key_info["proposal_total_blocks"] = getattr(key, "total_blocks")
                    flags = getattr(key, "flags", [])
                    key_info["proposal_q4_used"] = int(sum(1 for x in flags if int(x) == 0))
                    key_info["proposal_h_used"] = int(sum(1 for x in flags if int(x) == 1))
                    key_info["proposal_skip_used"] = int(sum(1 for x in flags if int(x) == 2))
                if save_outputs:
                    base = output_dir / "images" / method_id / image_name
                    save_image(base / "watermarked.png", watermarked)
                    save_image(base / "extracted_no_attack.png", extracted_clean)
                rows.append(
                    {
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
                    }
                )

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
                            if attack.name in {"jpeg_q70", "gaussian_noise_sigma5", "rotation_2deg"}:
                                save_image(base / f"attacked_{attack.name}.png", attacked)
                                save_image(base / f"extracted_{attack.name}.png", extracted)
                        rows.append(
                            {
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
                            }
                        )
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
        json.dump(failures, f, indent=2, ensure_ascii=False, default=_safe_num)

    return {"results": df, "summary": summary, "comparison": pivot, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description="Run the cleaned watermarking benchmark on 512x512 RGB host images and a 64x64 binary watermark.")
    parser.add_argument("--host-dir", default="data/host")
    parser.add_argument("--watermark", default="data/watermark/wm.png")
    parser.add_argument("--output", default="results/common_benchmark")
    parser.add_argument("--methods", default="all", help="Comma-separated: all,baselines,kumar2021,guo2017_dwt_qr_fa,gaata2022_dwt_hess_fwa,dwt_hd_svd_2025,proposal")
    parser.add_argument("--max-images", type=int, default=None, help="Optional quick-run limit.")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--invert-watermark", action="store_true")
    parser.add_argument("--attack-preset", default="lite", choices=["none", "lite", "full", "stress"], help="none runs clean extraction only; lite is quick; full includes many attacks; stress is compact but harsher.")
    parser.add_argument("--baseline-mode", default="adapt", choices=["adapt", "original", "original-rerun"], help="Global baseline mode: adapt runs local adapted benchmark; original writes paper-reported tables; original-rerun attempts a stricter local rerun.")
    parser.add_argument("--kumar-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--guo-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--gaata-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--dwt-hd-svd-mode", default="inherit", choices=["inherit", "adapt", "original", "original-rerun"])
    parser.add_argument("--proposal-use-optimizer", action="store_true", help="Adaptive proposal mode. OFF by default for fair comparison.")
    parser.add_argument("--proposal-optimizer-trials", type=int, default=4)
    parser.add_argument("--proposal-repeat", default="full", help="Default full/faithful uses all source-script structured repetition; use an integer such as 3 only for quick practical runs.")
    args = parser.parse_args()
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

    proposal_options = {
        "use_optimizer": bool(args.proposal_use_optimizer),
        "optimizer_trials": int(args.proposal_optimizer_trials),
        "params": {"repeat": _parse_repeat(args.proposal_repeat)},
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
