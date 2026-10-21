# Manuscript 43 attacks and 64/128-watermark reproducibility patch

The original manuscript has **43 attacked conditions, plus a clean control**. The existing `full` suite contains 109 attacked cases and `requested` contains 35; neither is appropriate for the manuscript's main comparison tables. The new `--attack-preset manuscript43` lists the 43 in the exact order of the two robustness tables, with operation signatures saved to `manuscript43_protocol.json`.

## What was added

- `src/watermarklab/common/attack.py`: exact 43-case suite and 6 manuscript-group labels.
- `src/watermarklab/benchmark.py`: `manuscript43` CLI support, six-group ablation aggregation, and global-frozen proposal parameters preferred over per-host parameters (to avoid evaluation leakage).
- `src/watermarklab/experiments/bss_validation.py`: CLI-selectable `--attack-preset` (previously *always* `reviewer_stress`).
- `src/watermarklab/experiments/channel_ablation.py`: explicit `--watermark-size` to allow matching 128-watermark extensions if desired.
- `scripts/run_manuscript43_and_128.py`: independently runnable, named stages, strict host split/payload preflight, no overwriting of manuscript results.
- `scripts/report_manuscript43_resize.py`: produce summaries **only if every one of the 43 attacks is actually present** for all 12 hosts and both watermarks at both sizes.

## Requirements and assumptions

- A held-out folder of twelve 512x512 RGB hosts, excluding Lena/Lenna, and **separate** training images for parameter tuning.
- `wm.png` and `Cc.logo.circle.png` are resized in memory with nearest-neighbor interpolation to 64x64 and 128x128. The original files stay unchanged.
- 64x64 source watermarks -> 32x32 LL embedded bits = **1,024 unique bits**. 128x128 source watermarks -> 64x64 LL embedded bits = **4,096 unique bits**. Both keep the same 512x512 carrier and 43 attacks; redundancy need not stay equal.
- A separately trained **frozen global proposal parameter file**. Use `--proposal-params /path/to/proposal_optimized_params.json`. The runner rejects missing global params in a full run.
- Run in Python environment with `numpy`, `pandas`, `Pillow`, `scipy`, `PyWavelets`, `matplotlib`, and relevant baseline packages. The runner sets `WATERMARKLAB_STRICT_PYWT=1` to prevent accidental Haar fallback during the publication evaluation.
- Pillow/OpenJPEG must support JPEG2000. The code fails rather than substituting JPEG.
- Operations are reproducibly specified as Pillow-based RGB transformations, bicubic rotations/resizes with zero fill, variance normalized to [0,1], and RNG seed 123. The paper's tables do **not** fully state filtering, interpolation or library implementation details; these assumptions are not proof that the historical results used identical underlying operators.
- `average_3x3`/`lowpass_3x3`, and likewise the 5x5 counterparts, share the box-filter implementation as in the existing code. They count as separate manuscript rows but are *not* independent operators under this implementation.

## Validate the 43 attack definitions (cheap)

```bash
python scripts/run_manuscript43_and_128.py --stages preflight --output results/manuscript43
PYTHONPATH=src pytest -q tests/test_manuscript43_and_128_protocol.py
```

## Show every command before expensive Kaggle runs

```bash
python scripts/run_manuscript43_and_128.py \
 --hosts /kaggle/working/splits/test12 \
 --train-hosts /kaggle/working/splits/train9 \
 --proposal-params /kaggle/working/proposal_optimized_params.json \
 --output /kaggle/working/manuscript43 \
 --stages all --dry-run
```

## Run the additional 128x128 experiment (both watermarks)

```bash
python scripts/run_manuscript43_and_128.py \
 --hosts /kaggle/working/splits/test12 \
 --proposal-params /kaggle/working/proposal_optimized_params.json \
 --output /kaggle/working/manuscript43 \
 --stages resize128

python scripts/report_manuscript43_resize.py \
 --input /kaggle/working/manuscript43/resized64_vs_128
```

Outputs: `proposal_size_sweep_clean_results.csv`, `proposal_size_sweep_attack_results.csv`, `paper43_size_summary.csv`, `paper43_by_attack.csv`, `paper43_by_family.csv` and `paper43_paired_128_minus_64.csv`. These are *new* results; they do not modify or replace the original 64x64 manuscript tables.

## Fill existing blank tables (separate Kaggle sessions recommended)

Run with `--stages baseline64`, `--stages ablation64`, `--stages block64`, `--stages channel64`, and `--stages bss64`, with the same `--hosts`, `--proposal-params` and `--output`. The hyperparameter sweep uses **training** hosts and requires `--train-hosts`; use `--stages hparam64`. The 43-attack TrustMark spatial 4096-bit comparison is available with `--stages trustmark64` (native model required; Google/Kaggle installation and weights are not in this ZIP). It uses alpha-channel source bits for the circle logo, and erasure-aware reporting, not assigning a valid zero to a rejected chunk.

The stages correspond to the manuscript's blank items:

| Stage | Manuscript placeholders |
|---|---|
| baseline64, trustmark64 | TrustMark comparison and paired TrustMark statistics (coverage-aware scores are mandatory) |
| ablation64 | Main ablation + six attack-group ablation tables |
| block64 | 4/8/16 block size robustness |
| channel64 | Y/R/G/B channel comparison |
| bss64 | BSS versus other reliability statistics |
| hparam64 | Hyperparameter tradeoff figure (based on separate training hosts) |
| resize128 | Additional watermark-size test: 128x128 compared with 64x64, both W1 and W2 |

**Accuracy limitations:** No experimental figures have been manufactured. Individual stages may be lengthy on Kaggle and may fail from resource limits. `stage_status.json` and stage logs report this honestly. The current 128-bit-size smoke test is not a 12-host 43-attack result. TrustMark's 4096-bit implementation is a spatial extension of a 61-bit native model, not a native 4096-bit model. Never substitute its conditional-on-decoding NC into the paper as a full-payload NC when there are erased bits.

For the corrected 45-attack geometric extension use `--attack-preset reviewer45` with this runner and report tool; see `docs/REVIEWER45_PROTOCOL.md`. Keep a separate output directory.
