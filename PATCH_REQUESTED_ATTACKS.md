# Patch: requested robustness attack suite

This package now includes the requested paper-style attack preset:

```bash
PYTHONPATH=src python main.py --methods proposal --attack-preset requested --proposal-repeat full
```

The preset contains:

- Blur 1
- Sharpening 1 and 1.5
- Salt & pepper noise 0.05 and 0.10
- Gaussian noise variance 0.003 and 0.005 on normalized [0,1] images
- Median filter 3x3
- Average filter 3x3
- Lowpass filter 5x5
- JPEG QF 90, 70, 50
- JPEG2000 approximate ratios 3:1, 5:1, 10:1
- Rotation 5, 10, 45 degrees
- Crop-resize 95%, 90%, 75%
- Gamma 0.75, 1.0, 1.2, 1.5
- Histogram equalization
- Combined attacks:
  - JPEG70 + blur1
  - JPEG70 + salt pepper 0.05
  - crop90 + JPEG70
  - rotate5 + crop95
  - gamma1.2 + JPEG70 + blur1
  - histogram + JPEG70

The normal benchmark flag and optimization-phase attack flag both accept:

```text
--attack-preset requested
--proposal-optimizer-attack-preset requested
```

The new attack groups are:

- `gaussian_noise_var`: Gaussian noise by normalized-domain variance.
- `combined`: deterministic sequential attack composition.
