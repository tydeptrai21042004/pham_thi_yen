# Patch: Proposal host/watermark size sweep command

This patch adds a proposal-only experiment command that runs the proposal method over multiple host-image sizes and watermark sizes without modifying `data/`.

## New command

```bash
python main.py \
  --phase proposal-size-sweep \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --output results/proposal_size_sweep \
  --methods proposal \
  --attack-preset lite \
  --size-sweep-host-sizes 256,512,1024 \
  --size-sweep-watermark-sizes 32,64,128 \
  --proposal-param-mode ignore
```

Aliases are also supported:

```bash
--phase proposal_size_sweep
--phase size-sweep
```

## What it does

- Loads host images from `data/host`.
- Loads one or more watermark files.
- Resizes hosts and watermarks **in memory only**.
- Runs the proposal embedding/extraction for each host-size/watermark-size pair.
- Computes clean PSNR before attack.
- Computes NC after each attack.
- Writes CSV summaries and plots under the selected output directory.

## Important outputs

- `proposal_size_sweep_clean_results.csv`
- `proposal_size_sweep_attack_results.csv`
- `proposal_size_sweep_clean_summary.csv`
- `proposal_size_sweep_attack_summary_by_attack.csv`
- `proposal_size_sweep_attack_summary_by_size.csv`
- `proposal_size_sweep_summary_by_size.csv`
- `proposal_size_sweep_failures.json`
- `proposal_size_sweep_psnr_before_attack.png/.pdf`
- `proposal_size_sweep_nc_after_attack.png/.pdf`
- `proposal_size_sweep_psnr_before_attack_heatmap.png/.pdf`
- `proposal_size_sweep_nc_after_attack_heatmap.png/.pdf`

## Multiple watermark files

```bash
python main.py \
  --phase proposal-size-sweep \
  --size-sweep-watermarks data/watermark \
  --size-sweep-host-sizes 256,512 \
  --size-sweep-watermark-sizes 32,64 \
  --attack-preset lite \
  --output results/proposal_size_sweep_multi_wm
```

If `--size-sweep-watermarks` is empty, the command uses `--watermark`.

## Notes

- The original data folder is never changed.
- Host resizing uses bicubic interpolation.
- Watermark resizing uses nearest-neighbor interpolation followed by binary thresholding.
- Watermark sizes must be even square sizes when the proposal's watermark-DWT payload is enabled.
- The proposal implementation now supports arbitrary square even watermark sizes and arbitrary host sizes that are large enough for the DWT/block pipeline.
