# Patch: Gaata 2022 uint8-safe adapted mode

## Problem fixed

The previous adapted implementation used the paper-style selected-decimal-digit parity rule with:

```python
D = 3
Delta = 1 / 10**D = 0.001
```

That change was visible in the floating-point reconstructed image but disappeared after the benchmark converted the watermarked image to `uint8`. As a result, Gaata failed even under `no_attack` with BER close to 0.5 and `PSNR = inf`.

## Correction

The paper-style decimal rule is still available for `original-rerun`, but the default `adapt` mode now uses a quantization-aware Hessenberg parity rule:

```python
embedding_rule = "qim"
qim_step = 16.0
```

The corrected adapted mode still follows the same main path:

```text
RGB -> DWT detail bands -> keyed embedding matrix -> 4x4 Hessenberg blocks -> one bit per selected H coefficient -> inverse path
```

The difference is only the bit decision rule for the selected Hessenberg coefficient:

- old adapted rule: parity of selected decimal digit;
- new adapted rule: parity of the nearest quantization index, `round(H[pos] / qim_step) % 2`.

This makes the embedded signal survive the common benchmark's RGB reconstruction and `uint8` rounding.

## Changed files

- `src/watermarklab/vendor/dwt_hess_fwa/watermark.py`
- `src/watermarklab/methods/gaata2022_dwt_hess_fwa.py`
- `src/watermarklab/benchmark.py`
- `tests/test_baseline_smoke.py`

## Validation

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
```

Result:

```text
6 passed
```

Clean adapted benchmark on one image:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods all \
  --max-images 1 \
  --no-save-images \
  --attack-preset none \
  --gaata-param-mode ignore \
  --guo-param-mode ignore \
  --proposal-param-mode ignore \
  --output results/check_all_none
```

Gaata no-attack result after patch:

```text
BER ≈ 0.006836
NC  ≈ 0.993177
PSNR ≈ 50.33 dB
```

Before the patch, Gaata had BER near 0.5 after `uint8` conversion.

## Usage

Default adapted mode now uses the fixed uint8-safe rule:

```bash
python main.py --methods gaata2022_dwt_hess_fwa --attack-preset none --gaata-param-mode ignore
```

To force the old paper-style decimal rule:

```bash
python main.py \
  --methods gaata2022_dwt_hess_fwa \
  --attack-preset none \
  --gaata-param-mode ignore \
  --gaata-embedding-rule decimal
```

To tune the adapted quantization strength:

```bash
python main.py \
  --methods gaata2022_dwt_hess_fwa \
  --attack-preset none \
  --gaata-param-mode ignore \
  --gaata-embedding-rule qim \
  --gaata-qim-step 16
```
