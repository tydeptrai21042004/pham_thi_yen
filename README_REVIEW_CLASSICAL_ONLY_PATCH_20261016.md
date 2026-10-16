# Complete classical/reviewer experiment support — code-only patch

**Input base**: `pham_thi_yen-main (1)(3).zip` (16 October 2026).  The ZIP in this delivery contains **only changed/new source, runner, tests, and instructions**. Merge its `pham_thi_yen-main/...` paths into that base. Do not treat the patch as a standalone repository: keep the supplied image datasets and existing baseline implementations.

**Preserved unchanged**: `src/watermarklab/methods/proposal_qh_dwt_hess.py` — SHA-256 `10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d`. No embedding/extraction, mathematical equations, optimization objective, or baseline methods were modified by this patch.

## Kaggle usage

1. Add the original project ZIP and `WatermarkLab_REVIEW_CLASSICAL_ONLY_PATCH_20261016.zip` as Kaggle inputs; also attach `watermark_all_results_9train_12test.zip` if resuming the previous run. Internet ON for Python dependencies; GPU **not** required. Paste the complete contents of `KAGGLE_CLASSICAL_REVIEWER_ONE_CELL.py` into one Kaggle notebook cell.
2. Default `STEP = 'all'`: train the proposal and two baselines **on nine distinct training images**, benchmark 13 classical configurations on 12 test images with 109 attacks, run five reviewer-specific experiment families, derive statistical/reviewer tables, sensitivity sweeps, visual watermark gallery, and then the **separate second-watermark cohort**.
3. Alternatively set `STEP` in succession to `optimize`, `heldout`, `reviewer`, `reports`, `extras`, `secondary` (or `verify`). Complete steps are reused when their contract allows. Use `--extended-sensitivity` / `EXTENDED_SENSITIVITY=True` to run sensitivity studies across all hosts rather than predeclared resource-saving subsets (size sweep defaults to 2 test hosts; hyperparameter sweep defaults to 1 training host). These are sensitivity experiments, not the main 12-host benchmark.
4. When a Kaggle session stops by budget, download `/kaggle/working/watermark_all_results_9train_12test.zip` and add it as an input in the next session. The script restores successful baseline optimization and any checkpointed proposal optimization progress. The previous failed ZIP has Guo and Gaata optimized parameters but **no completed proposal optimization**, so proposal work needs to run again. For the second watermark a separate ZIP is contained in the primary checkpoint.
5. Inspect `completeness.json`, `recovery_status.json`, `classical/reviewer_analysis/review_coverage.json`, and each job log. Incomplete/missing results stay **false** and are never interpreted as favorable measurements. The `verify` step exits nonzero if complete evidence is absent.

**Native TrustMark 61-bit and native WAM 32-bit are never launched.** The classical-only `all` mode also excludes the 4096-bit neural adaptations. Only an *explicit* `--step neural4096` or `--include-neural4096` invokes the adapted neural experiments. The Kaggle cell supplied here has no neural flag and does not install either model.

## Evidence coverage

| Reviewer experiment | Production evidence |
|---|---|
| Fixed independent train/test split | `host_split.json` + per-image SHA/strict recheck; 9 training, 12 test; original `lenna.bmp` excluded |
| 13-method full comparison | `classical/heldout_full/per_image_attack_results.csv`, strict 17,160 unique expected clean/attack rows |
| Full strong/geometric attack suite | 109 configurations, with severity and geometric breakdown in `classical/reviewer_analysis/` |
| PSNR/SSIM/NC/BER and embedding/extraction time | Per-host measurements and aggregate tables, host-bootstrap confidence intervals; `hardware_environment.json` |
| Nine-variant ablation | Nine `classical/ablation/<variant>/per_image_attack_results.csv` tables; new validator audits all nine |
| Color channels | Y, R, G, B independent comparisons with frozen parameters, attack-wise tables |
| Security | Wrong-key trials, unrelated host, side-information tamper, metadata size, separate keyspace/threat-model report |
| BSS and reliability | Cross-attack survival/error data; fail-closed nonempty BSS validator and exported raw measurement tables |
| Block-size 4, 8, 16 | Real embed/attack/extract experiments, payload capacity/coverage and timing recorded |
| Sensitivity sweeps | Size and hyperparameter sweep and proposal figures; limited or extended host coverage explicitly flagged |
| Extracted watermark gallery | Fixed first held-out host, five predeclared attacks and three predeclared methods; no cherry-picking |
| Second watermark | `Cc.logo.circle.png` alpha converted to reproducible 64x64 binary mask; independent cohort / separate optimized parameters/results |

The paper must accurately describe the proposed method as **semi-blind** when it requires side information, and distinguish the embedded DWT-LL payload from the original 64x64 watermark and from any 4096-bit neural adaptation. The theoretical BSS claims and security/key-space limitations need scientific argument beyond numerical tables; this patch does **not** claim to prove them.

## Files changed in this patch

- `scripts/recover_missing_results.py`: classical default, optional neural only, second watermark independent cohort, correct per-host validators, strict nine-variant validation, independent statistics stage, hardware profile, and fail-closed completion.
- `scripts/build_review_tables.py`: report generator with measured clean/attack/geometry/channel/block/BSS/security tables and paired host-level bootstrap confidence intervals; refuses incomplete grids.
- `KAGGLE_CLASSICAL_REVIEWER_ONE_CELL.py`: deterministic base+patch overlay, source-hash verification, Kaggle resumption without native neural execution.
- `tests/test_reviewer_experiment_orchestration_2026.py`: synthetic independent-grid tests for completeness, duplicate rows, missing ablation and source integrity.

## Testing vs. long-running work

Locally tested source parsing, the new report generator on synthetic complete/incomplete grids, focused existing reviewer/recovery regression tests, and an actual `--step verify --restore-zip <prior-results>` smoke. The previous ZIP is correctly identified as incomplete and its Guo/Gaata files restore. PyWavelets cannot be installed in this offline execution environment, so **the full 109-attack Kaggle benchmark has not been executed here**. Full-run performance and strict completion must be checked in Kaggle before claiming manuscript results.
