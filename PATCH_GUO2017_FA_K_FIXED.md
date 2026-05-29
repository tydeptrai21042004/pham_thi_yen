# Patch: Guo 2017 DWT-QR-FA Firefly lambda + paper-style K

This patch changes only the two requested Guo 2017 reproduction issues:

1. `src/watermarklab/methods/guo2017_dwt_qr_fa.py`
   - Adds a real Firefly optimization routine for the embedding strength `lambda`.
   - Uses the paper objective:
     `f(lambda) = [1 - SSIM(X, Xw)] + 30 * mean(BER(w, w_i'))`.
   - Uses paper FA defaults: `alpha_fa=0.01`, `beta0=1`, `gamma=1`, `n_fireflies=10`, `n_iterations=10`.
   - Changes K generation to paper-style random integral vector from `{-1, 0, 1}` with rejection of zero-variance vectors.

2. `src/watermarklab/benchmark.py`
   - Adds Guo optimization phase export to JSON/CSV.
   - Adds normal-phase loading of optimized Guo lambda.
   - Adds CLI flags:
     - `--guo-param-file`
     - `--guo-param-mode`
     - `--guo-optimizer-fireflies`
     - `--guo-optimizer-generations`
     - `--guo-optimizer-attack-preset`
     - `--guo-optimizer-alpha`
     - `--guo-optimizer-beta0`
     - `--guo-optimizer-gamma`
     - `--guo-optimizer-weight`
     - `--guo-lambda-min`
     - `--guo-lambda-max`
     - `--guo-optimizer-seed`

3. `src/watermarklab/methods/__init__.py`
   - Passes Guo options into the Guo method builder.

## Run optimization

```bash
PYTHONPATH=src python main.py \
  --phase optimize \
  --methods guo2017_dwt_qr_fa \
  --guo-mode original-rerun \
  --guo-param-file results/guo2017_lambda.json \
  --guo-optimizer-fireflies 10 \
  --guo-optimizer-generations 10 \
  --guo-optimizer-alpha 0.01 \
  --guo-optimizer-beta0 1 \
  --guo-optimizer-gamma 1 \
  --guo-optimizer-weight 30
```

## Run normal phase using optimized lambda

```bash
PYTHONPATH=src python main.py \
  --phase normal \
  --methods guo2017_dwt_qr_fa \
  --guo-mode original-rerun \
  --guo-param-file results/guo2017_lambda.json \
  --guo-param-mode require \
  --output results/guo2017_after_optimization
```
