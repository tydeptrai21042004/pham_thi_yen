# Reviewer45 v1 — source-consistent W1/W2 geometric expansion

This **code-only patch** adds a fixed 45-attacked-condition set plus a clean control, shared identically between the `wm.png` (W1) and `Cc.logo.circle.png` (W2) source watermarks. It does not change the proposal algorithm or the submitted paper's historical measurements. Evaluate with the *same 12 held-out host images* for both watermarks. Both original source watermarks stay 64×64.

## Why 45 rather than 46

The manuscript's Tables 3 and 4 list 43 attacked conditions (plus a clean row). Reviewer #1 requests more geometric distortions and analysis at different attack strengths; Reviewer #3 asks specifically for high-variance noise and low-quality JPEG. Both were already in the 43 set. The requested missing geometry adds THREE conditions, not two:

- `resize_2p0`: Scale 512→1024→512 (bicubic; same image dimension after attack).
- `crop_resize_50`: Center 50%-width/height crop followed by bicubic resize to original canvas. **This keeps 25% of the original area**, not 50% of the image area. Specify this when presenting results.
- `shear_x_0p10`: Horizontal affine shear with coefficient +0.10, bicubic resampling, and black fill (uncompensated).

Simply appending all three would yield **46** attacks. To obtain exactly **45**, the new preset removes `lowpass_3x3`. In the supplied code it is an exact duplicate operation of `average_3x3` (`lowpass(size=3)` calls the same `average_filter(size=3)` function), so the coverage of *distinct implemented attack operators* does not decrease.

`manuscript43` remains available unchanged. `reviewer45` = manuscript43 minus duplicate Lowpass 3×3, plus the above three. It has group counts: 8 compression, 8 noise, 5 filtering, 13 geometric, 5 light/color, 6 combined = **45**; *clean is not counted*. The protocol manifest records ordered attack IDs, parameters, group, and a SHA256 digest.

## Exact reviewer45 comparison to the prior manuscript

| Content | Manuscript43 | Reviewer45 | Change |
|---|---:|---:|---|
| JPEG / JPEG2000 | 8 | 8 | Same parameters (JPEG Q=90,70,50,40,30; JPEG2000 ratios 3:1,5:1,10:1) |
| Noise | 8 | 8 | Same incl Gaussian normalized variance 0.05 and speckle 0.03 |
| Filtering | 6 | 5 | Drops redundant Lowpass 3×3; retains Average 3×3 and Lowpass 5×5 |
| Geometric | 10 | 13 | Adds scale ×2, crop-resize 50%, shear 0.10 |
| Light/color | 5 | 5 | Same |
| Combined | 6 | 6 | Same |
| Total attacked | **43** | **45** | 42 original entries kept + 3 new |

Do **not** say existing mean NC over 43 attacks was measured under the new 45 attacks. Change manuscript captions and report separately when the new measurements are complete. The protocol uses deterministic noise seeds and identical attack code for W1 and W2. Some operation details may differ from the historical original table if its software/interpolation conventions were not documented; a same-name attack is not sufficient to certify byte-identical legacy attacked images.

## Installation and full held-out evaluation

Apply each file from the ZIP over the same relative path in your current repository. Only modified/new files are included.

```bash
pip install numpy pandas pillow matplotlib PyWavelets pymupdf
python scripts/run_reviewer45_experiments.py --mode manifest --output results/reviewer45_v1
```

Use an independently trained, **frozen** proposal parameter file (selected without test-host access):

```bash
python scripts/run_reviewer45_experiments.py \
  --mode evaluate --test-hosts data/host \
  --watermark-w1 data/watermark/wm.png \
  --watermark-w2 data/watermark/Cc.logo.circle.png \
  --proposal-param-file results/proposal_optimized_params.json \
  --proposal-param-mode require --methods proposal \
  --repeat full --output results/reviewer45_v1
```

Replace `data/host` by the actual **held-out 12-host directory**; the repository's own `data/host` may contain other images. Keep W1/W2 outputs separately at `evaluation_W1/` and `evaluation_W2/`. A completeness checker requires each method×host to have one clean row and **exactly 45 unique after-attack rows with finite NC and BER**; missing results trigger failure, never automatic zero-filling. Interruptions can resume from matching fingerprints of the attack journal.

A quick code check (NOT for publication) may use `--max-images 1 --repeat 1 --proposal-param-mode ignore --allow-fallback`. Normal runs require PyWavelets and full structured repetition. The `--allow-fallback` flag is intentionally forbidden by default to avoid a silent DWT backend change.

## Firefly / hyperparameter figure using the **same 45 attacks**

Run the sweep on the **separate training set**, not the held-out test images:

```bash
python scripts/run_reviewer45_experiments.py \
  --mode sweep --train-hosts /kaggle/working/train_hosts \
  --watermark-w1 data/watermark/wm.png \
  --watermark-w2 data/watermark/Cc.logo.circle.png \
  --tau 0.4,0.5,0.6 --qh 6,7,8 \
  --tau-margins 0.06,0.10 --h-margins 0.7,0.9 \
  --repeat full --output results/reviewer45_v1
```

The above **36 settings per watermark** require many host×attack×parameter evaluations and can take hours on CPU. Start with `--max-combinations 1 --max-images 1 --repeat 1 --allow-fallback` for smoke tests; do not use those reduced settings for the paper figure. For a full 45-attack Figure 9, the script refuses missing attack-mean values; each parameter combination must have exactly `45 × number_of_training_hosts` attacked rows.

Create figure files **from measured sweep CSV files**, not hypothetical numerical values:

```bash
python scripts/plot_reviewer45_hparams.py \
  --w1 results/reviewer45_v1/hyperparams_W1/proposal_hparam_sweep_summary_by_combo.csv \
  --w2 results/reviewer45_v1/hyperparams_W2/proposal_hparam_sweep_summary_by_combo.csv \
  --reference-pdf '/path/to/hparam_tradeoff_tauQ_qH_consistent(1).pdf' \
  --output results/reviewer45_v1/hparam_figures
```

The script creates `hparam_tradeoff_tauQ_qH_consistent.pdf` (use unchanged LaTeX `\\includegraphics` filename), individual W1/W2 PDFs, plot-source CSV, and `hparam_comparison_original_vs_reviewer45.pdf`. In the new scatter, **marker shape = τ_Q** (circle 0.4, square 0.5, triangle 0.6); **color = q_H** (6 purple, 7 teal, 8 yellow) to match the supplied 1-page historical plot. For the combined figure, the axes use the mean of W1 and W2 evaluated at each identical hyperparameter combination. Axis ranges are *not frozen to the historical plot* because new metrics/attacks may be outside its range. The comparison figure displays the original unmodified PDF on the left and actual new values on the right, with distinct axis ranges and provenance; it is not a claim of numerical superiority.

Suggested updated figure caption (only after a full sweep):

> Hyperparameter sensitivity under the common 45-attack protocol. Each point represents a combination of the Q-branch threshold τ_Q, H-domain quantization step q_H, and safety margins. Marker shape represents τ_Q; color represents q_H. Means are computed over the training host images and both source watermarks. Test-host results are reported separately.

## Reproduction limits and reviewer scope

This patch addresses the *geometric protocol completeness and shared W1/W2 attack definitions*. It does **not** by itself establish geometric synchronization robustness or answer other reviewer concerns about false positives, key-space analysis, BSS correlation, TrustMark parity, or statistical significance. Those require independent measured experiments and a manuscript update. Keep the 43 and 45 protocols explicitly versioned in all tables, legends, and CSV exports.

## Final merge with the 2026-10-21 12:09 repository

This patch is cumulative **relative to `pham_thi_yen-main(20261021-120927).zip`**,
not a replacement copy of the whole repository. That repository already includes
previous fixes for Su (2016), Gaata (2022), proposal H/Q selection and TrustMark
coverage diagnostics; those algorithms are **not** overwritten by this update.

The patch additionally includes `scripts/run_manuscript43_and_128.py` and
`scripts/report_manuscript43_resize.py` from the earlier 64/128 experiment patch,
now supporting `--attack-preset reviewer45`. For the manuscript's original
43-condition grid, pass `--attack-preset manuscript43` instead. Use separate
output directories for each protocol.

**Run the 45-attack 64-vs-128 experiment on both watermarks:**

```bash
python scripts/run_manuscript43_and_128.py \
  --attack-preset reviewer45 \
  --hosts /kaggle/working/test12 \
  --proposal-params /kaggle/working/proposal_optimized_params.json \
  --output /kaggle/working/reviewer45_size \
  --stages resize128

python scripts/report_manuscript43_resize.py \
  --attack-preset reviewer45 \
  --input /kaggle/working/reviewer45_size/resized64_vs_128
```

This requires *both* source watermarks and one 12-host held-out split, excludes
Lena, and tests 64×64 (1,024 LL payload bits) and 128×128 (4,096 LL bits) against
the same 45 operations. The report refuses missing/duplicate/nonfinite outcomes.
The 128×128 extension is for the **proposed method**, not automatic full-capacity
support by TrustMark or the historical classical baselines.

**Run complete 45-attack evidence for manuscript placeholders:**

```bash
python scripts/run_manuscript43_and_128.py \
  --attack-preset reviewer45 \
  --hosts /kaggle/working/test12 \
  --train-hosts /kaggle/working/train9 \
  --proposal-params /kaggle/working/proposal_optimized_params.json \
  --output /kaggle/working/reviewer45_full \
  --stages baseline64,ablation64,block64,channel64,bss64
```

TrustMark (`--stages trustmark64`) requires independently installed neural weights
and should be reported with payload erasures. The optional `hparam64` stage uses
training hosts only. The dedicated reviewer45 sweep runner above is recommended
for Figure 9 since it outputs W1/W2 summaries suitable for the plotting script.

**Validation status:** 43 and 45 preset unit tests passed; a diagnostic
one-host W1/W2 45-attack run and a one-host W1 128×128 45-attack run
completed with repetition 1. These are *not* paper measurements. PyWavelets is
required for strict paper runs; the diagnostic environment used the documented
Haar fallback.
