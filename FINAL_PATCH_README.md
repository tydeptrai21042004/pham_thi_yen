# FINAL merged patch — current repository: 2026-10-21 12:09

Apply onto `pham_thi_yen-main(20261021-120927).zip` using the paths within this ZIP. It contains ONLY new/modified source, scripts, tests and docs, not dataset files, checkpoints, figures, models or manuscript results.

What changed:
1. Adds versioned `manuscript43` (original, 43 attacks + clean) and `reviewer45` (45 attacks + clean), identical between W1 and W2.
2. `reviewer45` retains 42 of 43 conditions, removes duplicated `lowpass_3x3` (same underlying operation as average_3x3), and adds `resize_2p0`, `crop_resize_50`, `shear_x_0p10`.
3. Restores stage-based 64/128 source-watermark comparisons, using the SAME selected attack preset for both sizes, shared 512x512 host, and frozen TRAIN-only global proposal parameters.
4. Supports corresponding 43/45 attack families for ablation and BSS/channel experiments, and preserves baseline corrections already in the latest repository.
5. Adds `scripts/run_reviewer45_experiments.py` for W1/W2 and `scripts/plot_reviewer45_hparams.py` for the EXACT LaTeX figure name and optional visual comparison to the user-supplied reference PDF.
6. Strictly validates host splits, full attack grids, finite data, paired size-grid report, and incomplete hyperparameter sweeps.

Quick check:
  pip install -r requirements.txt
  PYTHONPATH=src pytest -q tests/test_reviewer45_protocol.py tests/test_manuscript43_and_128_protocol.py
  python scripts/run_manuscript43_and_128.py --attack-preset reviewer45 --stages preflight --output results/reviewer45_preflight

Full 64-vs-128 both-watermark run:
  python scripts/run_manuscript43_and_128.py --attack-preset reviewer45 --hosts /kaggle/working/test12 --proposal-params /kaggle/working/proposal_optimized_params.json --output /kaggle/working/reviewer45_size --stages resize128
  python scripts/report_manuscript43_resize.py --attack-preset reviewer45 --input /kaggle/working/reviewer45_size/resized64_vs_128

Full W1/W2 45-attack proposal comparison and hparam:
  python scripts/run_reviewer45_experiments.py --mode evaluate --test-hosts /kaggle/working/test12 --proposal-param-file /kaggle/working/proposal_optimized_params.json --proposal-param-mode require --repeat full --output /kaggle/working/reviewer45
  python scripts/run_reviewer45_experiments.py --mode sweep --train-hosts /kaggle/working/train9 --repeat full --output /kaggle/working/reviewer45
  python scripts/plot_reviewer45_hparams.py --w1 /kaggle/working/reviewer45/hyperparams_W1/proposal_hparam_sweep_summary_by_combo.csv --w2 /kaggle/working/reviewer45/hyperparams_W2/proposal_hparam_sweep_summary_by_combo.csv --reference-pdf /kaggle/working/hparam_tradeoff_tauQ_qH_consistent.pdf --output /kaggle/working/reviewer45/figures

Pilot validation: 72 selected existing/new tests passed. One-host W1 and W2 each yielded complete 45-attack rows; one-host 128x128 W1 yielded complete 45-attack rows; hparam PDF and side-by-side reference plot were generated from controlled test CSV records. This is NOT full paper evidence. In the pilot environment PyWavelets was absent and a Haar fallback was used. Real paper evaluation must install PyWavelets and use strict DWT enforcement; never insert pilot numbers into the manuscript.

See docs/REVIEWER45_PROTOCOL.md for full instructions and limitations.
