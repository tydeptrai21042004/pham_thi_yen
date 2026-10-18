# Resume only missing experiments: both watermarks

## Exactly what is changed
- `scripts/recover_missing_results.py`: fixes strict neural CSV validation of a *control* row with mathematical `PSNR=+inf` when image equals original. Other PSNR/SSIM/coverage/BER metrics must remain valid; no fabricated model results.
- `scripts/kaggle_resume_both_2026.py`: consumes **both existing result ZIPs**, checks provenance and 156 existing heldout contracts per cohort, restores them, completes A's extras, and resumes neural jobs for each watermark separately. A failed job is never misrepresented as complete.
- `KAGGLE_RESUME_BOTH_ONE_CELL.py`: copy/paste into one fresh Kaggle notebook cell, after the patch is pushed to GitHub.
- `tests/test_finish_both_recovery_2026.py`: offline validator/regression tests.

## Kaggle setup
1. Upload changed files to the same paths in `tydeptrai21042004/pham_thi_yen` and push the commit.
2. Attach TWO output ZIPs from the prior separate Kaggle notebooks as datasets to this NEW notebook: `watermark_all_results_9train_12test.zip` and `watermark_2_Cc.zip`. They are not included in this patch; the two provided text logs alone cannot reconstruct result CSVs.
3. Switch **Internet ON**. A GPU is recommended for TrustMark/WAM. Run the entire `KAGGLE_RESUME_BOTH_ONE_CELL.py` file in **one** Kaggle Python cell.
4. Read final `completeness.json` in both ZIPs. If not both complete, save the **two newly created ZIPs** as Kaggle outputs, attach them as inputs in a new session, and execute the same cell again. A full neural grid may exceed one Kaggle session, especially on CPU.

## Scientific limitations intentionally NOT hidden
- Both original checkpoints used 1 firefly and 2 generations, not a 10x10 optimizer. Changing these values would require retraining and invalidate all cached comparisons; this recovery patch does NOT silently change them.
- TrustMark/WAM are spatial 4096-bit extensions of native 61-/32-bit pretrained models, not native 4096-bit models.
- The existing hyperparameter sweep tests 24 prefix-of-grid combinations. Fixing the *design* later requires a separately identified supplemental experiment; changing `benchmark.py` now would invalidate every heldout contract and defeat checkpoint recovery.
- No model-inference or Kaggle end-to-end validation was performed here because pretrained weights, live GPU environment, and the two large result archives were not provided. Unit tests cover the validation bug and safe source contract checks.
