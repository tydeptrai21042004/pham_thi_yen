# Reviewer-focused patch — Workstream A + Workstream B

This archive is a **file overlay** for the latest `pham_thi_yen-main(20261014-040036).zip` codebase. It does **not** contain the complete project, existing results, model weights, or datasets. Unzip over the existing project root, preserving `src/`, `scripts/`, and `tests/` paths. Apply this overlay to the **specified source snapshot** only.

## Nonnegotiable constraints

- The proposed method `src/watermarklab/methods/proposal_qh_dwt_hess.py` is **not included and must not be modified**.
- Existing experiments/results remain untouched. Send new outputs to a new `results/reviewer_final/` directory.
- All matched-input comparisons use the same RGB host (512×512) and original binary watermark (64×64 = **4,096 source bits before any DWT**). The proposal's own transformations and side information remain as implemented.
- Extended TrustMark/WAM are **experimental, spatially chunked derivatives**, NOT official 4096-bit pretrained baselines. They need real checkpoint-backed tests; small regions may fail to decode even without attacks. Do not label untested models as successful.

## Changed files

| File | Why | Reviewer |
|---|---|---|
| `src/watermarklab/experiments/bss_validation.py` | Decode current dynamically encoded H positions and count validation errors/levels | R1.6, R2.4 |
| `src/watermarklab/experiments/review_audit.py` | Report side-information component size and full extraction-contract JSON byte count, while retaining wrong-key/unmarked controls | R1.1, R1.2, R1.7, R2.9 |
| `scripts/export_reviewer_evidence.py` | Export attack-wise ablations, host-bootstrap CIs, timing, security and existing block/channel results from saved experiment files | R1.4, R1.7, R2.1, R2.5, R2.10, R3.2 |
| `src/watermarklab/experiments/neural_4096.py` | 68×61-bit TrustMark and 128×32-bit WAM regional wrappers; one host and one original watermark; explicit erasures, 4096-bit reconstruction, clean gate and restart | R2.2, R3.4 |
| `tests/test_reviewer_neural_4096.py` | Offline contracts for 4096-bit bit-order, chunks, reconstruction, failure-aware scoring and output CSV | Reproducibility |

## Workstream A: existing saved results and corrected BSS

Unzip `pham_yen_groupwise_FULL_RESULTS(2).zip` and point the exporter to its extracted root:

```bash
export PYTHONPATH=src
python scripts/export_reviewer_evidence.py \
  --results-dir /path/to/pham_yen_groupwise_FULL_RESULTS \
  --output results/reviewer_final/existing_evidence
```

Run the **corrected BSS** on the held-out hosts only, using the same frozen optimization parameters as the original experiment. For an initial local smoke test:

```bash
python -m watermarklab.experiments.bss_validation \
  --host data/host/Girl.bmp --watermark data/watermark/wm.png \
  --output results/reviewer_final/bss_smoke \
  --host-size 128 --watermark-size 16 --max-attacks 1
```

For a full rerun, use the frozen held-out split (not all hosts including tuning), use `--param-file /path/to/proposal_optimized_params.json`, and choose the **same** sizes and attack preset as the existing BSS experiment. Do not interpret undefined BSS rank correlations as 1.0 or 0.0.

Run the *optional additional* side-information audit, without modifying the proposal:

```bash
python -m watermarklab.experiments.review_audit \
  --host-dir data/host --watermark data/watermark/wm.png \
  --output results/reviewer_final/security_audit \
  --max-images 2 --host-size 128 --watermark-size 16 \
  --param-file /path/to/proposal_optimized_params.json --max-attacks 3
```

For publication, match this audit's host/watermark sizes and host split to the manuscript protocol. It estimates uncompressed UTF-8 JSON metadata bytes, **not** optimized serialized or compressed overhead. Side-information-only success/false positives remain reported as limitations, not security improvements.

## Workstream B: 4096-bit neural feasibility and comparison

Install project requirements first. TrustMark requires `pip install trustmark`. For WAM, obtain the official `facebookresearch/watermark-anything` checkout, dependencies and MIT checkpoint; set `WAM_REPO_DIR` and `WAM_CHECKPOINT`. Use a GPU if available; the 128 independent WAM calls per image can be extremely expensive.

The existing split generator can create three tuning and nine held-out hosts:

```bash
python -m watermarklab.experiments.splits \
  --host-dir data/host --output results/reviewer_final/host_split.json \
  --seed 3101 --create-dirs
```

**Feasibility first on tuning hosts**. Adaptation choices (region grid, `--resize-to`) should be chosen on tuning images only. The wrapper initially uses an 8-column grid and resizing of local tiles to 256×256 for native inference:

```bash
python -m watermarklab.experiments.neural_4096 \
  --methods trustmark,wam \
  --host-dir results/reviewer_final/splits/train \
  --watermark data/watermark/wm.png \
  --output results/reviewer_final/neural_4096_tuning \
  --attack-preset full --max-attacks 2 --resize-to 256 --save-images
```

If clean decoded-bit coverage falls below 95% or clean erasure-as-error BER exceeds 5%, the tool records the failure and **skips attacks for that host**. Use `--force-full` only to deliberately evaluate an already failing capacity extension, not to imply successful 4096-bit embedding. This is an experiment, not a capacity proof.

After testing clean feasibility and freezing settings on tuning hosts, run on held-out hosts:

```bash
python -m watermarklab.experiments.neural_4096 \
  --methods trustmark,wam \
  --host-dir results/reviewer_final/splits/test \
  --watermark data/watermark/wm.png \
  --output results/reviewer_final/neural_4096_test \
  --attack-preset full --max-attacks 0 --resize-to 256 --save-images
```

Output: `per_image_attack_results.csv`, `comparison_4096.csv`, per-host CSV/contract files, `manifest.json`, optionally final marked images. Clean failure is documented, not silently scored as success. Host image attacks are applied **once to the assembled 512×512 image**, never separately to individual fragments.

### Interpretation of NC and BER

- `nc`, `ncc`, and `ber` are **conditional on successfully decoded source-bit positions**; if nothing is decoded, they are N/A.
- `nc_zero_fill` / `ncc_zero_fill` are diagnostic metrics after inserting zeros for unavailable bits; those zeros are **not neural predictions**.
- `ber_erasure_as_error` counts every missing bit as wrong, separately from observed bit errors.
- `bit_coverage`, `decoded_chunks`, `total_chunks`, and `exact_recovery` expose failure in the 68/128 independent invocations.
- A decoded 64×64 bitmap may be generated from the bits only if labels distinguish unknown/zero-filled pixels; no original watermark pixels are used to fill missing data.

TrustMark is processed in **68 chunks** (67 full 61-bit messages plus 9 data bits and 52 zero-pad bits). WAM uses **128 chunks** (32 bits each). They are not demonstrated pretrained 4096-bit models. If clean recovery fails, training redesigned 4096-bit networks would be a separate, extensive research effort **not contained in this patch**. That outcome must be reported rather than concealing it behind a 4096-bit input interface.

## Verification performed while preparing this patch

- Offline integration focused suite: 17 tests passed.
- BSS smoke: 912 valid blocks for one host/attack with zero indexing errors (measured correlation was small, not automatically favorable).
- Existing-results evidence exporter: loaded all six expected evidence groups and generated CSVs.
- Security/side-information audit: produced additional sizes; original-unmarked false-positive risk remains present.
- Full repository test suite did not finish within the available execution window. **Neither TrustMark nor WAM official checkpoints were run in this environment**; 4096-bit real-model performance is therefore unverified.

**Do not insert mock-test output as manuscript results.**
