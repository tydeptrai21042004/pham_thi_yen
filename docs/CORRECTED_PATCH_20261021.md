# Focused October 21 source patch — DWT/Hessenberg baselines and TrustMark reporting

Apply this ZIP **over a fresh checkout of the October 21 `pham_thi_yen-main` repository**, preserving the same paths. The ZIP includes changed/new files only. It deliberately contains no original images, model weights, source-paper PDFs, result checkpoints, or fictional new experiment results.

## Corrected proposal

`ProposalParams.h_selection_policy="manuscript"` (now the default) implements the manuscript's two-stage comparison: score both eligible H candidates, select the eligible H candidate with minimum local MSE, and compare that candidate against Q using the chosen BSS/MSE ablation score. An alternate `"independent"` setting preserves the former three-way Q-versus-two-H competition for explicit historical comparisons. Keys record the policy. The Givens search omits row pairs that cannot change the Q statistic. Candidate acceptance continues to require actual clean Hessenberg re-extraction, so invalid Q candidates cannot be selected solely from their intended rotation statistic.

The reference manuscript uses an LL-derived **1024-bit** payload from the 64x64 source image; the repository also supports a separate 4096-bit direct-payload ablation. Do not substitute outcomes across those configurations. The code intentionally retains semi-blind side information. Watermark reconstruction NC is **not** a reliable ownership-authentication test unless negative controls succeed.

## Gaata (2022)

`--gaata-mode adapt` selects the existing 8-bit-aware QIM adaptation. `--gaata-mode original-rerun` selects the fragile original-style decimal parity rule, but it no longer silently changes chaotic key amplitude or automatically executes FWA on a held-out host. FWA can be fitted in the repository's **separate training-phase optimization workflow**, and the benchmark now favors the globally frozen result over per-host parameters. For decimal-rule runs the result CSV contains `gaata_clean_ber_float` and `gaata_clean_ber_uint8`, so the float/PNG discrepancy cannot be hidden.

FWA's key-selection objective is now computed on the reconstructed 8-bit image rather than on pre-quantization floating-point pixels. This changes optimization results; retrain on training images and never mix old FWA checkpoints with revised scoring.

The paper omits sufficient details to guarantee exact reproduction of its numerical tables. Both implementations must be reported as paper-guided reproductions/adaptations, not source-code-identical replications.

## Su (2016)

The repository provides a 4096-bit, blue-channel, invertible **box-Laplacian-pyramid lowpass approximation**, with the paper's H-domain dominant-coefficient QIM rule. This is **not an exact Contourlet analysis/synthesis filter bank**. `--su2016-mode original-rerun` now fails explicitly with a clear message rather than silently producing adapted results labeled as an exact reproduction. Use `--su2016-mode adapt` for the supported baseline and label it as adapted.

## TrustMark (4096-bit spatial extension)

`neural_4096.py` now records `decode_status`, `rejected_chunks`, `exception_chunks`, `invalid_chunks`, and `missing_chunks`; `nc` and `ber` remain undefined for zero bit coverage. All missing bits are counted as errors in `ber_erasure_as_error`. Native 61-bit TrustMark capacity is **not** misrepresented as native 4096-bit capacity. No model weights or TrustMark algorithms were changed.

**Do not overwrite completed R12/R22 checkpoints with this changed module:** its source hash and result schema have changed, and the cache-contract guard is supposed to invalidate incompatible resume state. Run new tests into a **new output folder**. Analyze existing outputs without rerunning networks using:

```bash
python scripts/audit_trustmark_erasures.py path/to/trustmark_all_cases.csv --output results/trustmark_erasure_audit.json
```

## Installation and tests

```bash
python -m pip install -r requirements.txt
python -m pip install pytest PyWavelets
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 pytest -q tests/test_corrected_watermark_patch_20261021.py tests/test_h_energy_top_two.py tests/test_paper_hess_baselines_2016_2022.py tests/test_neural4096_only_cache_and_retry.py
```

To run a small real-image **smoke test** (not publication evidence):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python main.py \
  --methods su2016_contourlet_hess,gaata2022_dwt_hess_fwa,proposal \
  --max-images 1 --attack-preset none --no-save-images \
  --proposal-repeat 1 --proposal-param-mode ignore \
  --gaata-param-mode ignore --output results/patch_smoke_20261021
```

The first smoke test in this environment passed on Girl.bmp. It used **repeat=1** only for quick validation and showed nonzero clean BER for the proposal, so **do not present it as the full proposal's clean NC**. Full structured repetition and attack studies have not been rerun in this patch. PyWavelets was missing in the smoke environment, causing the explicitly warned Haar fallback. For final paper runs install PyWavelets and enable `WATERMARKLAB_STRICT_PYWT=1` to prohibit fallback.

The changes deliberately do **not** fabricate improvements, fill manuscript blank cells, or imply that original results are automatically reproduced by revised math. Re-run the fixed protocol and use output method IDs, payload sizes, parameter hashes, and host-level data when filling manuscript tables.
