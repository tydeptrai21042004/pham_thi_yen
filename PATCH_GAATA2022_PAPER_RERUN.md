# Gaata 2022 paper-rerun correction

This patch corrects the Gaata et al. 2022 DWT-Hessenberg-FWA baseline in the `original-rerun` mode.

## Corrected points

1. **Decimal/parity rule**
   - The previous quick `adapt` mode used a quantization-aware shortcut (`decimal_position=-1`).
   - `original-rerun` now uses the paper-style decimal digit rule (`decimal_position=3`) on the selected Hessenberg `H(4,4)` coefficient.

2. **Chaotic-key amplitude**
   - The previous common benchmark setting used `key_strength=0.020`.
   - `original-rerun` now uses a stronger `key_strength=1.0` so the generated chaotic keys have a meaningful effect on the embedding matrix before Hessenberg embedding.

3. **Firework Algorithm**
   - The previous package default used no FWA, and the old `original-rerun` used only a tiny search.
   - `original-rerun` now enables FWA and uses at least:
     - population size `N = 100`,
     - sparks per firework `= 5`,
     - iterations `>= 10`.

## How to run

```bash
PYTHONPATH=src python main.py \
  --methods gaata2022_dwt_hess_fwa \
  --gaata-mode original-rerun \
  --attack-preset none \
  --max-images 1 \
  --output results/gaata2022_original_rerun
```

For all paper attacks, run with a matching preset, but note that exact paper equality still requires the same six cover images, the same six binary logos, the exact chaotic-map equations, and the same metric normalization.
