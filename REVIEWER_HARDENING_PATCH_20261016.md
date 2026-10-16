# Reviewer numerical-code corrections — changes-only overlay (16 October 2026)

**Input baseline:** uploaded GitHub tree at commit `98dd1d4580ca447d9f9267b3d61d74d4baeca24b`.

This patch includes only the paths listed in `PATCH_DIFF_MANIFEST.json`. It does NOT replace the proposed DWT–Hessenberg algorithm (`src/watermarklab/methods/proposal_qh_dwt_hess.py`), the baseline algorithms, the host dataset, or the original source watermark. No numerical results have been invented.

## Changes

- `channel_ablation.py`: one `before_attack/no_attack` row per host/channel, with clean NC, BER, PSNR, SSIM; attacked rows are explicitly labeled `after_attack`.
- `bss_validation.py`: independent normalized decoder decision-margin comparison in addition to BSS/negative MSE; strict per-attack sample, nonfinite and read-error checks. Constant-response attacks retain *undefined* Spearman correlations rather than fabricated zeroes.
- `review_audit.py`: separate unrelated held-out host, 8 randomly generated wrong keys per host, integrity-threat controls (flags, H positions, contract truncation), public keyspace/entropy caveats, and demonstrative **external HMAC-SHA256** checks on the side information. HMAC is **not implemented as a security feature of the original embedding algorithm**. Keys are not written to results.
- `second_watermark.py`: converts the alpha channel in `Cc.logo.circle.png` to a separate, deterministic, 64×64 black/white image; refuses constant masks; preserves original SHA-256 provenance. The unmodified source image appears black when converted from RGB, so this conversion is essential.
- `reviewer_checks.py`: strict validators for 4-channel clean/attacked grids, sampled BSS blocks, randomized wrong-key controls/unrelated host/tampering, and a visual comparison gallery.
- `recover_missing_results.py`: executes the gallery in `extras`; checks per-host evidence rather than file existence; all requested checks are included in `completeness.json`. `--step all` also executes a **separate full second-watermark cohort** after the first is complete: fresh 9-train-image optimization, 13-method × 12-test-image × 109 attacks, reviewer experiments, figures/gallery, and capacity-matched 4,096-bit TrustMark/WAM experiments. Each cohort has an independent checkpoint ZIP and frozen parameters. `--step secondary` runs the secondary cohort separately if desired. No default project algorithm parameters were modified.
- `tests/test_reviewer_completion_extended_2026.py`: specific regression tests for those changes.

## Apply after cloning GitHub (Kaggle Internet ON)

```python
!git clone --depth 1 https://github.com/tydeptrai21042004/pham_thi_yen.git /kaggle/working/pham_thi_yen
!unzip -o /kaggle/input/YOUR_PATCH_DATASET/REVIEWER_HARDENING_CHANGED_FILES_20261016.zip -d /kaggle/working/pham_thi_yen
%cd /kaggle/working/pham_thi_yen
!pip install -q -r requirements.txt
!PYTHONPATH=src python -m pytest -q tests/test_reviewer_completion_extended_2026.py tests/test_operational_smoke_2026.py
```

Use the GitHub-clone Kaggle cell from the previous conversation, but **apply this overlay immediately after cloning and before running the tests/runner**. It MUST use the current repository, not the obsolete requirement for `WatermarkLab_ALL_EXPERIMENTS_FIXED_PATCH.zip`. For standalone CLI invocation:

```bash
export PYTHONPATH=src
export WATERMARKLAB_STRICT_PYWT=1
python scripts/recover_missing_results.py --step all --out /kaggle/working/watermark_all_results_9train_12test --max-jobs 0 --max-seconds 27000 --checkpoint-seconds 600
```

**Full TrustMark/WAM execution requires GPU, their official dependencies/checkpoints and their own setup** as described in your existing Kaggle cell. This patch does not download those models, bypass validation, or fabricate unavailable results.

For resumed notebook sessions, add the newest `watermark_all_results_9train_12test.zip` as Kaggle input and pass `--restore-zip /kaggle/input/.../watermark_all_results_9train_12test.zip`. The parent ZIP contains the independent second-cohort checkpoint `watermark_2_Cc.zip` once that stage has begun.

## Completion contracts

Do not claim the full project is complete unless the parent `completeness.json` reports **all** of the following true:

- `classical_13_methods_12_hosts_109_attacks`
- `reviewer_all_hosts_validated` (including all Y/R/G/B clean extractions, strict BSS, wrong-key/metadata security, independent unrelated host)
- `extras_sensitivity_and_figures`
- `visual_extracted_gallery_valid`
- `neural_4096_both_models_full_grid`
- `secondary_watermark_separate_full_grid`
- `all_core_sections_complete`

The two watermark sets must be analyzed as **separate** cohorts; do not pool their optimized parameter files. The proposed method embeds an LL-derived 1,024-bit payload from the original 64×64 watermark and needs side information; the experimental TrustMark/WAM modes are spatially tiled 4,096-bit extensions, NOT official native 4,096-bit model modes. Side-information integrity control does not turn the proposed method into authenticated encryption. Claims of cryptographic key strength are bounded by secret entropy, not SHA-256 output length.

## Validation performed locally

- `70 passed` across targeted reviewer, recovery, algorithm contract, smoke and dataset tests.
- `6 passed` in new security/BSS/second-watermark regression tests (included in the 70).
- PyWavelets was not available in this execution environment, so local source-method tests used the package's explicitly labeled fallback. Run Kaggle with `WATERMARKLAB_STRICT_PYWT=1` and PyWavelets installed.
- The repository-wide test run exceeded the local time limit after 87 tests. A successful full benchmark/official GPU run is **not** claimed.

