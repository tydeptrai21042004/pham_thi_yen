# Proposal attack-sensitivity plotting command

Added a new CLI phase:

```bash
PYTHONPATH=src python main.py --phase proposal-plot \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --output results/proposal_plot_wm \
  --proposal-param-mode auto \
  --proposal-param-file results/proposal_optimized_params.json
```

This generates a 3x2 figure and detailed CSV files for the proposal method.

Default plotted host images:
- airplane.bmp
- Girl.bmp (resolved to `lenna.bmp` if `Girl.bmp` is not present)
- house.bmp
- milkdrop.bmp
- safari.bmp
- tiffany.bmp

Default plotted attack-parameter sweeps:
- JPEG: 90, 80, 70, 60, 50, 40, 30
- Salt & pepper density: 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20
- Median filter size: 3, 5, 7, 9
- Resize factor: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
- JPEG2000 quality layer: 3, 5, 7, 10, 13, 15
- Rotation angle: 5, 10, 15, 30, 45

Optional overrides:

```bash
--plot-watermarks data/watermark/wm.png,data/watermark/Cc.logo.circle.png
--proposal-plot-hosts airplane.bmp,Girl.bmp,house.bmp,milkdrop.bmp,safari.bmp,tiffany.bmp
--proposal-plot-jpeg-values 90,80,70,60,50,40,30
--proposal-plot-salt-pepper-values 0.01,0.02,0.03,0.04,0.05,0.10,0.15,0.20
--proposal-plot-median-sizes 3,5,7,9
--proposal-plot-resize-values 0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0
--proposal-plot-jpeg2000-values 3,5,7,10,13,15
--proposal-plot-rotate-values 5,10,15,30,45
```

Outputs:
- `proposal_attack_sensitivity_<watermark>.png`
- `proposal_attack_sensitivity_<watermark>.pdf`
- `proposal_plot_detail_<watermark>.csv`
- `proposal_plot_detail_all_watermarks.csv`
- `proposal_plot_clean_summary.csv`


Update: plotting layout now defaults to `single`, which saves six separate figures (one figure for each attack family) instead of a 3x2 subplot grid. If you still want the old combined layout, use `--proposal-plot-layout grid`.

Exact requested default parameter sweeps are kept as:
- JPEG: 90, 80, 70, 60, 50, 40, 30
- Salt & pepper: 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20
- Median filter: 3, 5, 7, 9
- Resize: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
- JPEG2000: 3, 5, 7, 10, 13, 15
- Rotation: 5, 10, 15, 30, 45

The host resolver now prefers an exact file match first. So if your dataset really contains `Girl.bmp`, it will use `Girl.bmp`. The fallback alias to `lenna.bmp` is used only when `Girl.bmp` is absent.


## v3 update: denser default parameter grids

The proposal plotting phase now uses more values for smoother curves.

- JPEG: `100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20`
- Salt & pepper: `0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30`
- Median filter: `3,5,7,9,11,13,15`
- Resize: `0.25,0.33,0.40,0.50,0.60,0.75,0.90,1.00,1.10,1.25,1.50,1.75,2.00,2.50,3.00,4.00`
- JPEG2000: `1,2,3,4,5,6,7,8,10,12,13,15,18,20,25,30`
- Rotation: `1,2,3,5,7,10,12,15,20,25,30,35,40,45,60,75,90`

For dense grids, the code plots every measured point but thins the displayed x-axis ticks for readability.
