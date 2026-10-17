# Experiment reliability patch (17 October 2026)

This is an **experiment-only** update. It does not change any files under `src/watermarklab/methods/` and does not change the proposed embedding/extraction equations. Copy the paths in this patch ZIP into the **root** of the GitHub repository, preserving directories, and commit. Do not upload the ZIP as one file or delete the other project files.

## Files to commit

- `src/watermarklab/benchmark.py`: optional source/input-fingerprinted per-attack recovery in normal benchmarks, restartable size-sweep per host/size/attack, and restartable hparam sweep.
- `src/watermarklab/experiments/partial_results.py`: atomic checkpoint store with validation and marked-image hash to prevent mixing different embeddings.
- `scripts/recover_missing_results.py`: use attack recovery for each held-out job, require frozen proposal parameters for size-sweep, and strictly validate extra result grids before marking them complete.
- `tests/test_incremental_evidence_2026.py`: simulated interrupted attacks, resume, parameter mismatch and incomplete-result tests.
- `KAGGLE_RESUMABLE_EVIDENCE_2026.py`: Kaggle entry cell with Git commit/payload protection and separate primary/secondary cohorts.

## Run on Kaggle

1. Upload and commit the patch to `tydeptrai21042004/pham_thi_yen` on GitHub.
2. Create a Kaggle notebook. Enable Internet. Use no accelerator for classical-only experiments. A GPU is strongly recommended for neural comparisons.
3. Copy **all** of `KAGGLE_RESUMABLE_EVIDENCE_2026.py` into one notebook cell and run it.
4. The default is **primary** `wm.png` and **pilot** 1 firefly × 2 generations. For a strict 10 × 10 manuscript optimization, **change `FIREFLIES=10` and `GENERATIONS=10` before the very first session**; do not call 1 × 2 results a reproduction of the manuscript's 10 × 10 experiment.
5. The pipeline cannot be guaranteed to finish in one Kaggle CPU session. It periodically saves `watermark_all_results_9train_12test.zip` (primary) or `watermark_2_Cc.zip` (secondary) into Kaggle Output. When a session ends, save that ZIP as a private Kaggle Dataset and add it as the **only matching input ZIP** next session. Keep the same Git commit, cohort, optimizer settings and sensitivity protocol. The new launcher refuses inconsistent checkpoint provenance.
6. To evaluate the second watermark independently, change `COHORT='secondary'`. Its results are placed into its **own** directory and ZIP; do not combine rows with the first watermark.
7. To evaluate TrustMark/WAM with a forced 4,096-bit grid, set `INCLUDE_NEURAL_4096=True` and use a suitable accelerator. These are adaptations and must not be claimed to be native 61-bit TrustMark or native 32-bit WAM.
8. Review `recovery_status.json` and `completeness.json` before using results in the paper. A ZIP alone does not demonstrate completion.

**Important scientific mismatches**: the PDF describes watermark W1 of 64×64 and W2 of 128×128 pixels, while the repository's `wm.png` and `Cc.logo.circle.png` are both 64×64. The standalone size-sweep contains 128×128 resized-watermark cases, but the automatic second-watermark cohort is **not** a direct 128×128 reproduction. The PDF reports 10 fireflies and 10 iterations, unlike the pilot Kaggle settings. WAVES-style MS-COCO evidence and manuscript-specific statistical significance must be checked against their actual raw outputs; this patch does not certify them automatically.

## Testing performed before delivery

- 72 targeted regression checks passed (dataset, reviewer, strict completion, hparam/size sweeps, recovery).
- One real end-to-end normal CLI with a clean `Girl.bmp` image passed; a second **lite** full attack family run yielded 11 valid CSV rows (1 clean + 10 attacked) using one repetition and the local Haar fallback.
- Re-running the same lite configuration reused 10/10 completed attack rows and took about **2.33 seconds** locally; an identical embedding was still recomputed before cached attack rows were trusted.
- Mock interruption tests confirmed an already completed attack is not re-executed, an incomplete attack is retried, and changing the watermark or marked pixels invalidates the previous cache.
- The complete 12-host × 13-method × 109-attack and 128×128 manuscript protocols have **not** been executed end-to-end during this local patch validation. Run on Kaggle and inspect its raw result archive before making scientific claims.

## Special note on checkpoints

Old experiments produced using a different Git commit or without this new session provenance should not be mixed automatically with the patched result grids. The launcher intentionally requires a new patch-compatible checkpoint. An out-of-budget message with `exit=0` means **paused**, not **all results passed**.
