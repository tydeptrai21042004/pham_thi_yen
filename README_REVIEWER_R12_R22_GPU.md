# Reviewer R1.2 + R2.2 experiments — GPU Kaggle patch, proposal unchanged

## Scope (READ)

**Do not modify `src/watermarklab/methods/proposal_qh_dwt_hess.py`.** This patch only updates the neural adapter and 4096-bit experiment reporting and adds a separate full-resolution **security audit using the already implemented proposal**, frozen published training parameters, and a restartable Kaggle driver. An independent source SHA-256 assertion protects this boundary.

**R2.2** preserves the experimental **4096-bit source watermark** exactly: 68 TrustMark calls of 61 BCH_5 information bits each (4148 slots, 52 padding bits), 8 columns, 256×256 resized tiles, 512×512 host, one official pretrained Q-model on CUDA, 12 test hosts, 109 attacks per host, both source patterns. This is a **spatial *adaptation* of the pretrained model**, NOT its native 4096-bit capacity. Native TrustMark baseline would be a separate study; no inference is fabricated.

**R1.2** reuses `review_audit.audit()` (embedding/extraction core unchanged) on the same 12 test hosts, 512×512, 64×64 source, with the `global_best` training parameters from the earlier classical 9-train/12-test archives. Tests marked, original unmarked, unrelated hosts, deterministic and random wrong schedule keys, position/side-metadata tampering and external HMAC integrity. Includes fixed-threshold false positives, key-space assumptions and storage cost. This is **not a validated cryptographic ownership detector**. Unmarked-original high NC and wrong-key high NC must remain in the manuscript as negative findings. The HMAC is an *external benchmark integrity control*, not a newly added part of the proposal.

### Changed files (only these)

- `src/watermarklab/experiments/neural_4096.py` — failure-aware full-4096-bit metrics; explicit alpha channel for logo; no source-bit/layout change.
- `src/watermarklab/experiments/neural_baselines.py` — pass `device='cuda'` to official TrustMark and fail when GPU requested but unavailable.
- `scripts/reviewer_4096_report.py` — already verified old six-archive report; optional instant baseline comparison.
- `scripts/reviewer_r12_security_batch.py` — restartable 12-host security evaluation using frozen params (core unchanged).
- `scripts/reviewer_r12_r22_tables.py` — new independent evidence summaries, keeps erased-bit scores separate from conditional NC.
- `configs/reviewer_r12_frozen_A.json` and `configs/reviewer_r12_frozen_B.json` — frozen `NB0215` training-only parameter vectors, copied from earlier classical archives; no new tuning.
- `tests/test_reviewer_4096_only.py`, `tests/test_r12_r22_gpu_evidence.py` — offline regression checks.
- `KAGGLE_R12_R22_GPU_ONE_CELL.py` — the entire one-cell Kaggle launcher.

### How to use

1. Extract the ZIP patch over the **root of your GitHub repository** preserving file paths. Commit/push.
2. Kaggle notebook: Settings → GPU enabled; Internet ON. The notebook must be allowed to download pretrained model weights from the official package's source.
3. Paste entire `KAGGLE_R12_R22_GPU_ONE_CELL.py` into ONE Kaggle Python cell.
4. The cell clones GitHub at a recorded SHA, checks that the proposal method is byte-for-byte unchanged (`SHA256 10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d`), installs dependencies, verifies CUDA and runs the offline regression tests.
5. It then runs security R1.2 (CPU) and actual TrustMark R2.2 (CUDA) on watermarks A and B. It creates `/kaggle/working/R12_R22_GPU_CHECKPOINT.zip` periodically and after jobs, plus `/kaggle/working/R12_R22_RESULTS.zip` after each launch. **If the 11h15min budget is reached, the output will be explicitly PARTIAL**; download the checkpoint and attach it as Input in the next session with the same updated code and selected cohorts. Restore resumes from validated marked-image caches and per-attack records.
6. Optionally attach the previous **six experimental ZIPs** as Kaggle Input to generate the older archived baseline-vs-TrustMark comparison immediately without expensive inference. The main GPU experiments do not require those archives.

For parallel sessions, set `COHORTS = ('A',)` in the first notebook and `COHORTS = ('B',)` in the second. They work independently; do not share a mutable checkpoint across sessions. Default is `('A','B')`, but a single GPU session may not finish all 24 TrustMark host runs. Each session should keep only its own checkpoint.

### Output files

- `R12_R22_GPU/SECURITY/A|B/security_all_cases.csv` and `security_summary_by_case.csv` — actual wrong-key/negative controls, raw and aggregate.
- `R12_R22_GPU/NEURAL/A|B/<host>/trustmark/<host>_results.csv` — clean + unmarked control + 109 attack rows.
- `R12_R22_GPU/REVIEW_TABLES/trustmark_4096_summary.csv`, `trustmark_per_host.csv`, `trustmark_all_cases.csv`, `security_aggregate.csv`, `security_all_cases.csv`, `report.json` — new measured summaries.
- `R12_R22_GPU/OLD_DATA_REPORT/` (if all six archived datasets attached) — previous already-completed original experiment report.
- `R12_R22_RESULTS.zip` — compact tables. `R12_R22_GPU_CHECKPOINT.zip` — cached marked images and all raw progress, needed for resume.

### Correct numerical comparison

On a TrustMark attack, failed BCH chunks produce missing/erased bits; they are NOT a valid ordinary NC prediction. Numerical full-payload score:

`correct_fraction_all_4096_bits = correctly_decoded_bits / 4096 = 1 - erasure_inclusive_BER`

Use conditional NC only alongside **coverage** and **number of decoded cases**; it is undefined when no bits decode. Never average this conditional NC with reconstructed NC from the proposal.

**Crucial fairness limitation:** The proposal accepts a 4096-pixel source watermark but embeds only the **1024 LL bits**, storing other wavelet details in side information. TrustMark's 4096 spatial adaptation physically attempts to encode **all 4096 source bits**. Same sources/images/109 attacks is a common benchmark, *not equal embedded payload capacity*, so you cannot declare a rigorously equal-payload baseline without a separate payload-matched comparison. The patch does not change the proposed algorithm to manufacture parity.

### What was actually validated here

- Verified 109 named attack specifications, 12 supported test hosts, 4096 binary bits in both source watermarks.
- Offline regression tests exercise the original 68×61 chunk logic, failed-chunk accounting, codec resume behavior, security summary and CUDA adapter call with a mock TrustMark class.
- Inspected both training-only parameter records and verified their frozen values.
- **No actual CUDA-backed pretrained TrustMark run was executed in this local environment**; the real GPU invocation is provided for Kaggle and must be considered unverified until Kaggle outputs exist.
- A full 512×512 security smoke run with repeated extraction proved expensive on the local CPU; 12-host security testing is resumable, not guaranteed to finish within a single Kaggle session.

No claims about false positive improvements, cryptographic security, native 4096-bit TrustMark support, or improved proposal-method performance are made by this patch.
