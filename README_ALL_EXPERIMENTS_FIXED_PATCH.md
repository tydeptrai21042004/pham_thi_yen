# Full experiment recovery patch — 15 October 2026

**Exact input:** `pham_thi_yen-main(20261015-090635).zip`.

**Output:** `WatermarkLab_ALL_EXPERIMENTS_FIXED_PATCH.zip` contains **only added/modified source, runner and test files**, relative to the specified input archive. It does not contain the datasets or pretrained neural weights. The proposed watermarking algorithm source (`src/watermarklab/methods/proposal_qh_dwt_hess.py`) is **unchanged**. No native 61/32-bit TrustMark/WAM benchmark is launched by the patched Kaggle runner.

## Kaggle usage

1. Create a Kaggle notebook; turn **Internet ON** and enable a **GPU** for the neural `4096` step.
2. Add these Kaggle inputs as ZIP files: the **exact original project ZIP**, the **new changes-only patch ZIP**, and optionally the previous `watermark_all_results_9train_12test.zip` (or the **most recent checkpoint ZIP** from a later session).
3. Paste `KAGGLE_ALL_4096_AND_CLASSICAL.py` into a single notebook cell. Set `STEP = 'all'` (default) to run the entire workflow; or select `optimize`, `heldout`, `reviewer`, `extras`, `neural4096` individually. `STEP='all'` automatically stops at the Kaggle time budget and can be rerun on the next session.
4. Each new Kaggle session must receive the **latest** `watermark_all_results_9train_12test.zip` as input, or set `RESUME_ZIP` to its exact path. An interrupted attack uses its verified saved marked-image cache. Old 61/32-bit native results are excluded from the new checkpoint.
5. Inspect `/kaggle/working/watermark_all_results_9train_12test/completeness.json`, `recovery_status.json`, and detailed `/logs/`. Do **not** declare the complete grid until all required checks are true.

### Intended experiment coverage

| Step | Output |
|:--|:--|
| `optimize` | Train-only proposal optimizer with checkpointed nine host optimizations and all 81 candidate/host objective evaluations; validated frozen Guo & Gaata parameters |
| `heldout` | 13 methods × 12 *held-out* hosts × (1 clean + 109 attacks): **17,160 expected rows**, strict uniqueness and numeric validity |
| `reviewer` | Security controls, block-size comparisons, BSS validation, channel ablation, and all proposal ablation variants, host-by-host |
| `extras` | Size/hyperparameter sweeps and proposal visualization PDFs and CSVs |
| `neural4096` | TrustMark/WAM *spatial chunk adaptation only* × 12 hosts × (1 clean + 1 negative control + 109 attacks): **2,664 expected rows** |

Both experimental neural extensions use the original **64×64 = 4,096-bit** watermark. They each split it into multiple native-sized messages; **this is not a claim that the official pretrained networks natively support 4,096 information bits**. The `--force-full` option makes every attacked image a measured trial even when clean watermark recovery fails; those failures and erasures remain visible and are **not** scored as successful extraction.

This runner benchmarks `data/watermark/wm.png`. `Cc.logo.circle.png` has a distinct alpha-only image representation and is **not** automatically evaluated in the same fixed benchmark. If reporting results for two watermarks, the second requires a separately defined, consistent binary preprocessing and an independent protocol.

### Additional safeguards

- A new Kaggle session extracts a clean copy of the exact original ZIP, preventing stale checkout files from contaminating results.
- Dataset split is fixed; proposal/Guo/Gaata tuning is restricted to the nine training images, no test-image tuning.
- Expensive proposal optimization checkpoints after each completed train image and candidate/host objective evaluation; restarts verify image+watermark+settings fingerprints.
- Held-out batches are keyed to host, code and trained-parameter checksums, not assumed complete from filenames alone.
- Reviewer host batches are validated for expected output/host and absence of reported audit errors before being aggregated.
- Marked-image cache is hashed and required for resuming neural attack metrics; failed attack-processing rows are retried, not counted as measured rows.
- Atomic WAM weight download avoids keeping partially downloaded model checkpoints. The official WAM model/config must be present and usable.
- Failed GPU/model dependencies, corrupt results, or missing metrics cause visible failures and a non-complete manifest.

### Testing performed **locally**

- **104 focused offline tests passed**. These include per-attack neural restart, full-row force evaluation with zero clean coverage, fixed 9/12 split, model wrappers, optimization checkpoint recovery, evidence smoke tests, and new checkpoint/restore safeguards.
- Executed both **Su et al. 2016** and **Abodena et al. 2022** adapted baselines on an actual 512×512 host with a 64×64 watermark, no attacks; both returned normal exit code, with CSV outputs.
- Restored the prior result ZIP and verified that Guo/Gaata parameters survive and incomplete optimization is explicitly detected.
- The repository-wide test attempt exceeded the local execution window, so a complete full-suite pass is **not** claimed. The complete 109-attack 12-host multi-model **GPU run has not been performed** in this environment. These require Kaggle compute and may need several sessions.

**This patch is an execution/recovery correction, not a guarantee of perfect capacity, attack robustness, or completion in one Kaggle session.**
