# Patch: proposal method aligned with uploaded Python script

This package was patched so that `src/watermarklab/methods/proposal_qh_dwt_hess.py` uses the uploaded Python script as the single source of truth for the proposal method.

## Algorithmic changes

- Host embedding domain changed from RGB all-channel embedding to OpenCV-style BGR -> YCrCb conversion with embedding only in the Y channel.
- Watermark payload changed from full 64x64 binary watermark bits to one-level Haar DWT watermark preprocessing with only the LL subband embedded.
- Payload length changed from 4096 bits to 1024 bits.
- Extraction reconstructs the 64x64 binary watermark from extracted LL bits plus saved watermark DWT detail subbands, matching the source script metadata-assisted reconstruction.
- Structured repetition default changed to full capacity, giving 16 repetitions per payload bit for 512x512 hosts with four DWT bands and one Y channel.
- Q4/H-position candidate construction, Bit Survival Score screening, MSE filtering, and majority voting were kept aligned with the uploaded script.
- Old ZIP-only quick random optimizer behavior was removed from the proposal method; `use_optimizer=True` no longer switches to a different algorithm.

## Compatibility notes

- The benchmark package still loads/saves arrays as RGB through Pillow. The proposal method internally bridges this to the source script's OpenCV BGR/YCrCb convention.
- CLI default `--proposal-repeat` was changed to `full`. Use `--proposal-repeat 1` or `--proposal-repeat 3` only for quick smoke tests.
- `opencv-python` was added as a dependency because the source-script-faithful color conversion uses OpenCV.

## Smoke checks performed

- `python3 -m py_compile` on the patched proposal, validation, and benchmark files.
- `PYTHONPATH=src pytest -q` passed: 4 tests passed.
- Proposal quick smoke run with `--proposal-repeat 1` completed successfully.
