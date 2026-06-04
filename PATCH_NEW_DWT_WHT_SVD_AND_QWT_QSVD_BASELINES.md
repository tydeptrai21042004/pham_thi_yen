# Patch: add DWT-WHT-SVD2024 and Zhang2022 QWT-QSVD baselines

This patch adds two baselines that satisfy the unified benchmark requirement:

- host image: 512x512 RGB/color image
- watermark: 64x64 binary watermark = 4096 payload bits/symbols
- extraction mode preserved from the paper

## Added method IDs

```text
dwt_wht_svd_2024
qwt_qsvd_zhang2022_blind
qwt_qsvd_zhang2022_semiblind
```

## DWT-WHT-SVD2024

Paper setting preserved:

- RGB -> YCbCr
- embed in Y component
- 3-level DWT through HH branch: Y -> HH1 -> HH2 -> HH3
- for a 512x512 host, HH3 is 64x64, so it matches the 64x64 watermark
- WHT + SVD
- Arnold scrambling
- semi-blind extraction with side information: H_PC, U_W, V_W, alpha

## Zhang2022 QWT-QSVD

Paper setting preserved at benchmark level:

- RGB -> YCbCr
- embed in Y component
- one-level low-frequency QWT/Q1 domain
- split low-frequency domain into 4x4 blocks
- 256x256 Q1 domain gives 64x64 = 4096 blocks
- one 64x64 binary payload bit per block
- QIM extraction
- supports both blind and semi-blind variants

Implementation note: QSVD is implemented through the complex adjoint representation of quaternion matrices. The QWT low-frequency Q1 branch is implemented in a deterministic no-extra-dependency form to keep the benchmark reproducible in this repository.

## Quick commands

Clean smoke test:

```bash
PYTHONPATH=src python main.py \
  --methods dwt_wht_svd_2024,qwt_qsvd_zhang2022_blind,qwt_qsvd_zhang2022_semiblind \
  --max-images 1 \
  --attack-preset none \
  --no-save-images \
  --guo-param-mode ignore \
  --gaata-param-mode ignore \
  --proposal-param-mode ignore \
  --output results/new_baselines_clean
```

Lite attack test:

```bash
PYTHONPATH=src python main.py \
  --methods dwt_wht_svd_2024,qwt_qsvd_zhang2022_blind \
  --max-images 1 \
  --attack-preset lite \
  --no-save-images \
  --guo-param-mode ignore \
  --gaata-param-mode ignore \
  --proposal-param-mode ignore \
  --output results/new_baselines_lite
```

Targeted tests:

```bash
PYTHONPATH=src pytest -q \
  tests/test_new_baselines_2024_2022.py \
  tests/test_clean_registry.py \
  tests/test_baseline_smoke.py
```
