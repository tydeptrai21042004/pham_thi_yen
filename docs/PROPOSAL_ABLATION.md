# Proposal ablation analysis

This repo now supports a proposal-only ablation phase for measuring the role of each internal component in the proposed watermarking method.

## What the ablation tests

| Variant | Component tested | Meaning |
|---|---|---|
| `full` | All components | Full proposal baseline. |
| `q4_only` | Q4/Hessenberg-Q branch | Disables the H-position branch and keeps only Q4-domain embedding. |
| `hpos_only` | H-position branch | Disables Q4 and keeps only H-position quantization. |
| `no_structured_repetition` | Repetition + majority voting | Uses only one block per payload bit. This shows the role of redundant embedding and voting. |
| `no_watermark_dwt` | Watermark DWT-LL payload | Embeds the full 64×64 binary watermark directly instead of only the DWT-LL payload. |
| `no_arnold` | Arnold scrambling | Keeps the same payload but disables spatial scrambling. |
| `mse_only_selection` | Candidate selection | Chooses the lowest-MSE valid candidate instead of weighted BSS/MSE ranking. |
| `bss_only_selection` | Candidate selection | Chooses by bit-survival score only. |
| `no_strength_filter` | Candidate filtering | Selects from all candidates without the strong-candidate survival/MSE filter. |

## Full ablation command

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python main.py \
  --phase proposal-ablation \
  --methods proposal \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --attack-preset full \
  --proposal-repeat full \
  --proposal-param-mode ignore \
  --guo-param-mode ignore \
  --gaata-param-mode ignore \
  --no-save-images \
  --output results/proposal_ablation_full
```

## Quick smoke command

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python main.py \
  --phase proposal-ablation \
  --methods proposal \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --max-images 1 \
  --attack-preset lite \
  --ablation-variants full,q4_only,hpos_only,no_structured_repetition \
  --proposal-param-mode ignore \
  --guo-param-mode ignore \
  --gaata-param-mode ignore \
  --no-save-images \
  --output results/proposal_ablation_smoke
```

## Output files

The ablation phase writes:

- `proposal_ablation_manifest.csv`: variant definitions and parameter switches.
- `proposal_ablation_per_image_attack_results.csv`: all raw rows from every variant.
- `proposal_ablation_summary_by_variant_phase.csv`: clean and attacked aggregate metrics per variant.
- `proposal_ablation_summary_by_variant_attack.csv`: per-attack aggregate robustness table, written when attacks are enabled.
- `proposal_ablation_delta_vs_full.csv`: metric changes relative to the full proposal.
- `proposal_ablation_failures.json`: variant-level failures, if any.

Use `proposal_ablation_delta_vs_full.csv` to discuss contribution of each component. For robustness, a negative `delta_nc_mean_vs_full` or positive `delta_ber_mean_vs_full` means the removed/changed component was useful.
