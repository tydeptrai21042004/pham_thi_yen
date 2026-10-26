# Hessenberg baseline recovery (W1 and W2) — patch and independent Kaggle cells

## Apply patch

Copy **only** the five files below from the ZIP to the same repository-relative paths on GitHub `tydeptrai21042004/pham_thi_yen`, replacing matching files, then commit to `main`:

- `scripts/recover_missing_results.py`
- `scripts/run_remaining_fresh.py`
- `scripts/plot_remaining_fresh.py`
- `src/watermarklab/experiments/reviewer_checks.py`
- `tests/test_hessenberg_repair_runner.py` (new)

Do not upload the `kaggle_cells/` folder to GitHub unless you want to retain notebook copies there; it is not required by the runner.

## Confirmed fixes (from W1 and W2 Kaggle logs)

1. **Held-out `UnboundLocalError: hashlib`**: remove the inner `import hashlib` from the `heldout()` function. `hashlib` is imported at module scope; the inner import wrongly makes all earlier uses of the name local and unbound.
2. **Fresh-run `IndexError`**: change the `recovery_step` log tag builder to handle a one-element list `['--extended-sensitivity']` safely; retain the full command options unchanged.
3. **Channel and BSS `validated=False` despite PASS manifests**: parameterize each validator by the **executed** attack preset (`reviewer45`, 45 attacks), rather than comparing against the default `reviewer_stress` 29-attack manifest.
4. **Priority**: `--focus-failed-baselines` runs Su 2016, Nha 2023, Gaata 2022, Abodena 2022 on all 12 test hosts under 109 attacks; DWT–WHT–SVD on the two previously failed hosts; the two previously failed 45-attack method groups; the remaining channel and BSS studies; available extras and result-derived plots. It avoids rerunning the already completed block-size, ablation and security studies.
5. **Selective figures**: plot only validated select-cohort measurements, label 12-host and 2-host cohorts separately. No inference of a complete 13-method comparison.

## Run

Start TWO **independent** Kaggle notebooks, one for `KAGGLE_W1_HESSENBERG_FIXED_NO_ATTACH.py`, one for `KAGGLE_W2_HESSENBERG_FIXED_NO_ATTACH.py`.

Settings: Internet ON, Accelerator None/CPU. Each cell clones `main` directly. No previous-results ZIP attachments are needed. The same 9 training / 12 test split applies to both, with Lenna/Lena excluded. W1 uses `wm.png`; W2 derives the 64×64 binary watermark from `Cc.logo.circle.png` alpha.

Each output goes to `/kaggle/working/yen_w1_hessenberg_fixed.zip` or `/kaggle/working/yen_w2_hessenberg_fixed.zip` and includes `fresh_remaining_status.json` plus detailed logs. Incomplete jobs are not silently promoted to PASS; an incomplete run may still create a usable checkpoint ZIP.

**No TrustMark, WAM, or other neural baselines are run.** The proposed Q/H algorithm, h22/h21 selection, and existing Firefly objective are unchanged. Training uses 2 fireflies and 2 generations, which is only a runtime choice.

## Scientific scope

**Su 2016 is the existing 64×64 adapted box-LP reproduction of the Contourlet–Hessenberg paper**, not an exact recreation of its unavailable LP/DFB filters. These patches fix execution and validation, not original-paper filter equivalence. Do not label the Su output as an exact native Contourlet reproduction. Missing/undefined NC should remain null/NaN with valid BER, not made-up zeroes.

The new run is a **fresh selective experiment**, not an automatic merge with the previously uploaded W1/W2 archives. A full 13-method table requires separately verified compatible results from the successful methods. The previous completed block-size, ablation and security results remain in the user's earlier result archives; the new sessions do not contain them.

## Local verification completed

- 19 fast regression tests passed on patched code.
- Actual W2 Girl channel and BSS outputs from failed archive validated under the corrected reviewer45 criteria.
- One-host Su/Nha clean smoke tests for both W1 and W2 executed, but full 109-attack Kaggle jobs have NOT run locally.

## Test commands

```bash
PYTHONPATH=src python -m pytest -q tests/test_hessenberg_repair_runner.py tests/test_recovery_resume_2026.py tests/test_undefined_nc_checkpoint_fix.py tests/test_full_reviewer_kaggle_contract_2026.py
```
