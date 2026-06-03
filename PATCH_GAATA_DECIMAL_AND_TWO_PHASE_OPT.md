# Patch: Gaata2022 decimal digit + two-phase optimization

## Fixed

- `Gaata2022DWTHessFWA` adapt mode now uses `decimal_position=3` by default.
- This follows the paper-style rule: use a digit after the floating point for parity embedding/extraction.
- The old adapt default `decimal_position=-1` was a pragmatic quantization shortcut and should not be described as the paper decimal-digit rule.

## Added

Gaata2022 now supports a two-phase workflow similar to Guo2017 and the proposal method.

### Phase 1: optimize chaotic key parameters

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase optimize \
  --methods gaata2022_dwt_hess_fwa \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --gaata-param-file results/gaata2022_key_params.json \
  --gaata-optimizer-population 8 \
  --gaata-optimizer-iterations 3 \
  --gaata-optimizer-sparks 3
```

### Phase 2: normal benchmark loading optimized parameters

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase normal \
  --methods gaata2022_dwt_hess_fwa \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --attack-preset requested \
  --gaata-param-mode require \
  --gaata-param-file results/gaata2022_key_params.json \
  --output results/gaata2022_requested_optimized
```

## Important numerical note

Using the paper-style decimal digit after the floating point is highly sensitive to uint8 rounding. Therefore, old results generated with `decimal_position=-1` should not be compared directly with new paper-style Gaata results.
