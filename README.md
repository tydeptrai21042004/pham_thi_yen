# WatermarkLab cleaned baseline project

This is the cleaned version of your watermarking project. It keeps only:

1. **Paper 1 — Kumar 2021**: DWT + maximum entropy + alpha blending.
2. **Paper 2 — Guo 2017**: DWT + QR + Firefly-Algorithm baseline family.
3. **Paper 3 — Gaata 2022**: DWT + Hessenberg + Firework Algorithm.
4. **Paper 4 — DWT-HD-SVD 2024/2025**: DWT + HD/Hessenberg + SVD + logistic chaos.
5. **Your proposal method**.

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
│   ├── methods/                 # only four baselines + proposal
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

## Run only the four baselines

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
proposal
baselines
all
```

---

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
