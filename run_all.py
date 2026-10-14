"""Run the common benchmark on the frozen 12-host held-out test set.

For publication-quality results, first tune parameters exclusively on the nine
new training images using ``scripts/kaggle_review_run.py``. Running this file
without that parameter file is a functional check, not a tuned benchmark.
"""
from pathlib import Path
import sys

# Allow running from the unpacked project without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from watermarklab.benchmark import load_proposal_param_file, run_benchmark
from watermarklab.experiments.splits import build_split

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    split = build_split(root / "data/host", root / "results/common_benchmark/split.json",
                        create_dirs=True)
    params_path = root / "results/reviewer_9train_12test/proposal_optimized_params.json"
    optimized = load_proposal_param_file(params_path)
    if not optimized.get("loaded"):
        print("WARNING: no parameters tuned on the nine new training images. "
              "Output is an untuned functional check, not a final held-out comparison.")
    run_benchmark(
        host_dir=split["test_dir"],
        watermark_path=root / "data/watermark/wm.png",
        output_dir=root / "results/common_benchmark",
        selected_methods=["all"],
        max_images=None,
        save_outputs=True,
        attack_preset="lite",
        proposal_options={
            "params": {"repeat": None, "dwt_mode": "pywt"},
            "use_optimizer": False,
            "optimized_payload": optimized if optimized.get("loaded") else None,
        },
    )
    print("Done. See results/common_benchmark/compare_psnr_nc_ber_ncc_before_after_attack.csv")
