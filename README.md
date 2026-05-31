# WatermarkLab cleaned baseline project

This is the cleaned version of your watermarking project. It keeps only:

1. **Paper 1 — Kumar 2021**: DWT + maximum entropy + alpha blending.
2. **Paper 2 — Guo 2017**: DWT + QR + Firefly-Algorithm baseline family.
3. **Paper 3 — Gaata 2022**: DWT + Hessenberg + Firework Algorithm.
4. **Paper 4 — DWT-HD-SVD 2024/2025**: DWT + HD/Hessenberg + SVD + logistic chaos.
5. **Paper 5 — Hess-Nha 2023**: blind Hessenberg H(2,2) quantization, adapted to a 64x64 binary watermark.
6. **Paper 6 — Roy 2018**: YCbCr-Y blockwise DWT-SVD side-information baseline.
7. **Your proposal method**.

Older unrelated baselines have been removed.

---

## Project layout

```text
watermarklab_baselines_cleaned/
├── data/
│   ├── host/                    # 512x512 RGB host images
│   └── watermark/               # 64x64 binary watermark
├── docs/
│   ├── BASELINE_METHODS.md
│   └── baseline_paper_reported_results.csv
├── src/watermarklab/
│   ├── benchmark.py             # unified runner
│   ├── common/                  # DWT, metrics, attacks, color, entropy, helpers
│   ├── methods/                 # the baseline methods + proposal
│   └── vendor/dwt_hess_fwa/     # Gaata 2022 support code only
├── tests/                       # lightweight smoke tests
├── main.py
├── run_all.py
├── requirements.txt
└── pyproject.toml
```

---

## Install

```bash
pip install -r requirements.txt
```

For reproducibility, run with:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src
```

On Windows PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:PYTHONPATH="src"
```

---

## Run tests

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
```

---

## Run only the baselines

Clean/no-attack quick check:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods baselines \
  --max-images 1 \
  --no-save-images \
  --attack-preset none \
  --output results/baseline_clean_smoke
```

Lite attack benchmark:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods baselines \
  --attack-preset lite \
  --output results/baseline_lite
```

Full attack benchmark:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods baselines \
  --attack-preset full \
  --output results/baseline_full
```

---

## Run baselines + proposal

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods all \
  --attack-preset lite \
  --proposal-repeat 3 \
  --output results/common_benchmark
```

---

## Run one method only

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods guo2017_dwt_qr_fa \
  --max-images 1 \
  --attack-preset lite \
  --output results/guo2017_lite_smoke
```

Valid method IDs:

```text
kumar2021
guo2017_dwt_qr_fa
gaata2022_dwt_hess_fwa
dwt_hd_svd_2025
hess_nha2023
roy2018_dwt_svd
proposal
baselines
all
```

---


## Hess-Nha2023 64x64 adapter

The new baseline ID is:

```text
hess_nha2023
```

It implements the Nha et al. blind Hessenberg idea with:

```text
64x64 binary watermark -> Arnold scrambling -> 4x4 host blocks -> blue-channel H(2,2) quantization -> blind extraction with 2x2 majority voting
```

By default, `--hess-nha-mode adapt` uses the notebook's 64x64 setting `T=15` and `alpha=3.2`. Use `--hess-nha-mode paper` or `--hess-nha-mode original-rerun` to use the paper quantization step `T=65` while still keeping the 64x64 project watermark adapter.

The requested attack preset also includes:

```text
occlusion_25pct
occlusion_50pct
```

## Roy2018 DWT-SVD baseline

The new baseline ID is:

```text
roy2018_dwt_svd
```

It follows Roy and Pal 2018: RGB host images are converted to YCbCr, the Y channel is split into `32x32` blocks, each block is transformed by three-level Haar DWT, and each `4x4` watermark block is embedded in the singular-value matrix with `alpha=0.02`. Extraction uses side information from the embedding phase (`Sp`, `UWp`, `VWp`), so this is not a fully blind method.

Run only Roy2018:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods roy2018_dwt_svd \
  --attack-preset none \
  --max-images 1 \
  --output results/roy2018_clean_smoke
```

## Output files

Each run writes:

```text
per_image_attack_results.csv
summary_by_method_phase.csv
compare_psnr_nc_ber_ncc_before_after_attack.csv
failures.json
```

The most useful file for manuscript tables is usually:

```text
compare_psnr_nc_ber_ncc_before_after_attack.csv
```

---

## Important note about paper numbers

The local baseline outputs may not exactly match the original paper tables. This project is now organized to avoid confusing those categories:

- `docs/baseline_paper_reported_results.csv` = original paper-reported scale.
- `results/...` = local unified benchmark results.

Use separate table labels in your manuscript: **paper-reported**, **paper-guided reproduction**, and **unified benchmark**.

## Original vs adapted baseline modes

```bash
# Exact paper-reported tables copied from the four papers
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods baselines \
  --baseline-mode original \
  --output results/original_reported

# Local unified benchmark on your input data
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods baselines,proposal \
  --baseline-mode adapt \
  --attack-preset lite \
  --output results/adapt_lite
```

`original` writes paper-reported tables. `adapt` is the actual runnable comparison on your dataset. Keep them separate in the manuscript. See `docs/ORIGINAL_VS_ADAPT_MODES.md`.

---

## Proposal two-phase workflow

The proposal method now supports the requested two phases.

### Phase 1 — optimization phase

This phase runs the Firefly parameter search and exports a parameter file.
Only the four source-script parameters are optimized:

```text
Q4_TAU, Q4_MARGIN, H01_Q, H01_MARGIN
```

Example quick optimization:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase optimize \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --max-images 1 \
  --proposal-repeat full \
  --proposal-optimizer-fireflies 4 \
  --proposal-optimizer-generations 2 \
  --proposal-optimizer-attack-preset script \
  --proposal-param-file results/proposal_optimized_params.json
```

The optimization phase writes:

```text
results/proposal_optimized_params.json
results/proposal_optimized_params.csv
```

The JSON contains both per-image optimized parameters and a global best fallback.

### Phase 2 — normal phase

In normal phase, the benchmark checks `--proposal-param-file` automatically.
If the file exists, it uses the optimized parameters. If the file does not exist,
it uses the default source-script parameters.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase normal \
  --methods proposal \
  --attack-preset script \
  --proposal-repeat full \
  --proposal-param-file results/proposal_optimized_params.json \
  --output results/proposal_normal_with_optimized_params
```

Useful parameter-file modes:

```text
--proposal-param-mode auto      # default: use optimized file if it exists, else defaults
--proposal-param-mode ignore    # force default parameters
--proposal-param-mode require   # fail if optimized file is missing
```

### Attack presets

Available presets are now:

```text
none    clean extraction only
lite    quick mixed attack suite
script  attacks aligned with the standalone Python script
full    broad attack suite
stress  compact harsh attack suite
grid    large parameter sweep with many attack levels
```

Use `--attack-preset grid` when you want many attack variables for sensitivity testing.
