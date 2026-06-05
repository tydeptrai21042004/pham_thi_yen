# Patch: Proposal Tier-1 Hyperparameter Sweep Command

This patch adds a new benchmark phase:

```bash
--phase proposal-hparam-sweep
```

It sweeps the four proposal parameters that directly control the robustness / imperceptibility trade-off:

```text
q4_tau
q4_margin
h01_q
h01_margin
```

## Default grid

```text
q4_tau:      0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65
q4_margin:   0.04, 0.06, 0.08, 0.10, 0.12, 0.14
h01_q:       5, 6, 7, 8, 9, 10
h01_margin:  0.50, 0.70, 0.90, 1.10, 1.20
```

The full grid has `7 * 6 * 6 * 5 = 1260` combinations.

## Full command

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase proposal-hparam-sweep \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --output results/proposal_hparam_sweep \
  --attack-preset lite \
  --proposal-repeat full \
  --no-save-images \
  --proposal-param-mode ignore
```

## Quick smoke command

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --phase proposal-hparam-sweep \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --output results/proposal_hparam_sweep_smoke \
  --max-images 1 \
  --attack-preset none \
  --proposal-repeat 1 \
  --no-save-images \
  --hparam-sweep-q4-tau 0.40,0.50 \
  --hparam-sweep-q4-margin 0.08 \
  --hparam-sweep-h01-q 7 \
  --hparam-sweep-h01-margin 0.90 \
  --proposal-param-mode ignore
```

## Output files

```text
proposal_hparam_sweep_per_image_attack_results.csv
proposal_hparam_sweep_raw_summary_by_method_phase.csv
proposal_hparam_sweep_summary_by_combo_phase.csv
proposal_hparam_sweep_summary_by_combo.csv
proposal_hparam_sweep_ranking.csv
proposal_hparam_sweep_compare_before_after.csv
proposal_hparam_sweep_manifest.csv
proposal_hparam_sweep_best.json
proposal_hparam_sweep_failures.json
```

## Notes

- The embedding/extraction algorithm is unchanged.
- The phase intentionally does not load `proposal_optimized_params.json`; it studies the explicit Tier-1 grid.
- `h01_margin` is stored as both requested and effective because `ProposalParams.from_dict()` clips it to `h01_margin <= 0.49 * h01_q`.
- Use `--hparam-sweep-max-combinations N` only for debugging.
