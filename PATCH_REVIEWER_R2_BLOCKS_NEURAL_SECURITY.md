# Reviewer experimental patch: preserved 4x4 proposal, additional size extension

## What changed
- `proposal_qh_dwt_hess.py`: **only** extend the fixed Givens rotation from `I_4` to `I_b` for b=8/16, leaving the 4x4 path, H indices, BSS, quantizer, Arnold, FA and decoding rules the same.
- `benchmark.py`: optimized parameter snapshots no longer overwrite ablation-specific flags. Record effective settings.
- `experiments/block_size_comparison.py`: true 4/8/16 embedding, extraction, attack measures, per-category CSV and failure manifests. This study is the only change to the proposed algorithm.
- `experiments/review_audit.py`: run global (training-only) optimized parameters, wrong-key/metadata controls, flag uncalibrated threshold and suspicious false positives explicitly.
- `experiments/bss_validation.py`: support 9 test hosts and attack-group BSS vs negative MSE rank correlation; no numerical performance guaranteed.
- `experiments/neural_baselines.py`: retire ChunkySeal; run TrustMark + pretrained WAM (32 bits), preserving conditional decode metrics and negative controls.
- `scripts/kaggle_review_run.py` and `scripts/kaggle_neural_one_cell.py`: update Kaggle orchestration and checkpoint downloads.

## Representative commands (run from repo root)
```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m watermarklab.experiments.block_size_comparison \
 --host-dir /path/to/heldout9 --watermark data/watermark/wm.png \
 --param-file /path/to/training_params.json --block-sizes 4,8,16 \
 --attack-preset reviewer_stress --max-attacks 0 --output results/reviewer_block_sizes
PYTHONPATH=src python -m watermarklab.experiments.review_audit \
 --host-dir /path/to/heldout9 --watermark data/watermark/wm.png \
 --param-file /path/to/training_params.json --max-images 9 --host-size 512 \
 --watermark-size 64 --output results/reviewer_security
PYTHONPATH=src python -m watermarklab.experiments.bss_validation \
 --host-dir /path/to/heldout9 --watermark data/watermark/wm.png \
 --param-file /path/to/training_params.json --max-hosts 9 --host-size 512 \
 --watermark-size 64 --output results/reviewer_bss
```

Only results actually obtained from these commands can be claimed. Need to install PyWavelets for strict reproducibility. Official WAM checkpoints, external dependencies, GPU and full Kaggle data have NOT been run in the offline unit environment.

## Data and evaluation cautions
- 4x4 originally has 16 repetitions at 512x512 / 64x64 W LL payload, 8x8 four, 16x16 one. Failed votes reduce actual coverage, so report the measured factor and erasures.
- WAM (32 bits) and TrustMark (61 bits) have different native capacities, detection semantics and post-processing. Conditional NC alone is not a fair comparison.
- A very high full-watermark NC on *unmarked* images is a security concern due to the saved LH/HL/HH watermark detail subbands. The patch does not silently change the method or pretend the problem is solved.
- The full-model original 4x4 behavior should be checked against historical outputs when PyWavelets is installed. Smoke tests on a small sample cannot establish accuracy on the final 9-host suite.
