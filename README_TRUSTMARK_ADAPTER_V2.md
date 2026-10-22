# TrustMark 4096-bit spatial adapter — reviewer45 controlled revision

This is a **new experimental adapter setting**, **not a change to pretrained TrustMark weights or native 61-bit capacity**. Do not overwrite published results from the old adapter, and do not report conditional NC over only recovered bits as full-payload accuracy.

## Files to upload at their exact GitHub paths

- `src/watermarklab/experiments/neural_baselines.py` (official `WM_STRENGTH` forwarding)
- `src/watermarklab/experiments/neural_4096.py` (residual projection, validated fallback resampling, accounting, run-contract, CLI)
- `tests/test_trustmark4096_adapter_revision.py` (offline regression tests)

These source files are based on the user's uploaded `pham_thi_yen-main(20261021-120927).zip`, which already includes separate 2026 fixes. **Inspect differences with your live GitHub branch if newer modifications were committed since that snapshot.**

## Old settings reproduced by default

`--wm-strength 1.0 --projection legacy --residual-gain 1.0 --decode-views bicubic`

## New fixed, exploratory adapter setting

`--wm-strength 1.5 --projection residual --residual-gain 1.25 --decode-views bicubic,lanczos`

`WM_STRENGTH` is an upstream TrustMark encoding parameter. Residual projection leaves original high-frequency image detail in place and transfers the resampled watermark residual, with one *fixed* scalar gain. Second-view LANCZOS decoding is attempted only after a first BCH-valid decode fails. An accepted payload still requires TrustMark's normal `present == True` and 61 binary information bits. No per-attack rules, retraining, test-set optimization, side channel or fake recovered bits are introduced.

## Steps

1. Upload the two changed Python files and the test file into their respective GitHub paths, then commit.
2. Run the separately provided `KAGGLE_TRUSTMARK_4096_ADAPTER_V2_ONE_CELL.ipynb` in Kaggle with GPU and Internet enabled.
3. Download `/kaggle/working/TRUSTMARK_4096_V2_45_RESULTS.zip`.
4. Compare both clean bit coverage **and** attacked mean bit coverage, erasure-aware BER, PSNR, full recovery rate, unmarked controls, and zero-coverage cases against the old adapter before claiming an improvement.

## Reliability caveat

Embedding 4096 bits as 68 distinct 61-bit messages into a single 512×512 image means each encoded region is only approximately 64×57 pixels. Many attacks destroy those tiny watermarks, regardless of software correctness. This revision **does not guarantee removal of zeros or complete recovery**. Failed chunk detections remain erasures and are retained in the output. If clean 4096-bit recovery remains weak, present it as an unsuccessful pretrained-model capacity extension, not as a matched native 4096-bit TrustMark result.

The new `fallback_chunks` field shows how many chunks were rescued by alternative interpolation. A zero-coverage case has `nc` undefined and `nc_zero_fill` equal to zero. The control phase is excluded from the attack mean.

Offline tests: 11 TrustMark capacity-adapter tests passed in the local environment. Real pretrained-model testing and GPU 12-host ×45-attack execution have **not** been performed locally.
