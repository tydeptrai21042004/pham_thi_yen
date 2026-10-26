# DWT-Hessenberg: fresh remaining experiments (no Kaggle Inputs)

## Authoritative source and scope
Patch built by comparing `pham_thi_yen-main(4).zip` against the known Kaggle failures. This package changes only:

1. `src/watermarklab/experiments/partial_results.py` — serialize non-finite correlations as JSON `null`; resume an actually measured all-zero extraction with finite BER.
2. `scripts/recover_missing_results.py` — permit explicitly undefined NC/NCC in a complete attack row, while still rejecting missing rows and non-finite BER, PSNR, SSIM or timing; support independent reviewer kinds and selective held-out baselines/hosts; support the 45-attack reviewer protocol.
3. `scripts/run_remaining_fresh.py` — **new** no-attachment orchestrator for W1 or W2. Trains on 9 fixed hosts, tests on 12 fixed hosts excluding Lenna, and prioritizes missing studies.
4. `scripts/plot_remaining_fresh.py` — **new**, figures derived only from measured data. Writes a manifest for missing sources.
5. `tests/test_undefined_nc_checkpoint_fix.py` — **new** regression tests.
6. `tests/test_full_reviewer_kaggle_contract_2026.py` — adjust obsolete test: undefined NC is admissible for a completed extraction with finite BER; non-finite BER remains a failure.

No changes to the proposed Q/H embedding algorithm, its h22/h21 selection, the existing Firefly objective, datasets, or watermark images.

## Installation

1. Extract the files at the *root of* `tydeptrai21042004/pham_thi_yen` and commit to `main`.
2. Open two *separate* Kaggle notebooks, Accelerator=None, Internet=On.
3. Paste the full source from `kaggle_cells/KAGGLE_W1_NO_ATTACH_REMAINING.py` into notebook A; paste `kaggle_cells/KAGGLE_W2_NO_ATTACH_REMAINING.py` into notebook B.
4. **Do not add any files under Kaggle Inputs.** Both cells clone the newly committed `main` branch independently.
5. New output archives are `/kaggle/working/yen_w1_remaining_fresh.zip` and `/kaggle/working/yen_w2_remaining_fresh.zip`. The archives are *new* outputs, not restored old archives.

## What is executed

- Optimization on 9 training hosts for Proposal, Guo, and Gaata; original objective remains unchanged; 2 fireflies and 2 generations for the proposal.
- On 12 held-out 512x512 RGB BMP test hosts (no Lena), W1 `wm.png` and W2 alpha-derived binary `Cc.logo.circle.png`, each separately 64x64.
- Reviewer experiments **independently**, in order: 4x4/8x8/16x16 block comparison; channel Y/R/G/B; BSS; security controls; full ablation. For block/channel/BSS/ablation, `reviewer45` is requested with 45 attacks; security runs its separate seven-attack negative-control procedure.
- Previously failing 45-attack baselines Gaata and DWT-WHT-SVD, both evaluated from scratch on all 12 hosts.
- Previously failing 109-attack heldout baseline subsets: Abodena, Su (existing adapted approximation), Gaata, Nha, 12 hosts each, plus DWT-WHT-SVD on Girl and milkdrop (50 selected method-host combinations).
- If CPU budget remains, size/hyperparameter sweeps, existing extraction-image gallery, and reports. No TrustMark and no WAM are run.
- Figures generated from real completed measurements in `fresh_remaining_figures/` with `FIGURE_MANIFEST.json`. Missing panels are recorded, not invented.

## Scientific and operational boundaries

- **Without the previous W1/W2 ZIP archives, none of the old successful Proposed, Guo or other baseline rows is available in this new output.** Their previous results are not destroyed or rerun, but they cannot automatically be merged by this notebook. A complete 13-method statistical table requires an evidence-preserving merge performed later using matching protocols and raw results.
- The Su method in the source repository remains the pre-existing adapted box-LP/Contourlet approximation. This patch fixes failure handling but does not establish paper-exact Su implementation.
- Undefined NC and NCC stay undefined (`NaN` in CSV / `null` in JSON); corresponding BER is preserved. Undefined must not be imputed as NC=0 or NC=1. Aggregates must disclose valid/undefined counts.
- A method with all-zero output may legitimately have undefined NC; this does not establish robust detection.
- Full 45-attack reviewer analyses and 109-attack baseline subsets are computationally expensive. **One Kaggle session may terminate before finishing all jobs.** `fresh_remaining_status.json`, `recovery_status.json`, `completeness.json`, and the output ZIP describe what actually finished. Without uploaded archives, another fresh session cannot reuse work from a previous session.
- The runner packs results at regular checkpoints as well as in `finally`; log and manifest files are included.

## Verification done before delivery

- The modified files compile as Python.
- 36 targeted tests passed (checkpoint, recovery, reviewer protocol, dataset split).
- Both generated standalone Python Kaggle cells parse as valid Python and reference files in the patch.
- Full Kaggle results and every baseline's paper fidelity are **not** claimed as verified by local tests.
