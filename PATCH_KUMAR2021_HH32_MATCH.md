# Patch: Kumar 2021 HH-subband / 32x32 block correction

This patch changes the `kumar2021` baseline to follow the Kumar & Singh 2021 paper more closely for the two key points requested:

1. **Watermark preprocessing**
   - Before: the 64x64 binary watermark was embedded directly.
   - Now: the 64x64 watermark is transformed by one-level Haar DWT and only the watermark `HH` subband is embedded.
   - Resulting payload size: `32x32`.

2. **Cover block size**
   - Before: the cover-image `HH` subband was divided into `64x64` blocks.
   - Now: the cover-image `HH` subband is divided into `32x32` blocks, matching the watermark `HH` payload size.

3. **Extraction/reconstruction**
   - The extracted payload is now treated as the recovered watermark `HH` subband.
   - The final 64x64 watermark is reconstructed by inverse DWT using the original watermark `LL`, `LH`, `HL` side information stored in the method key.

4. **Alpha recalibration**
   - After embedding only a 32x32 HH payload, the previous default `alpha=0.97` made the clean PSNR unrealistically high compared with the paper.
   - The default was changed to `alpha=0.65` to bring the clean PSNR back near the paper-reported range for the bundled sample data.

Verification:

```text
PYTHONPATH=src pytest -q
4 passed
```

Clean benchmark smoke test on bundled host images:

```text
method: kumar2021
attack preset: none
mean PSNR: 51.7901 dB
mean NC: 1.0000
mean NCC: 1.0000
mean BER: 0.0000
```

Note: the paper reports average PSNR 51.6145 dB on its own seven host images. The bundled ZIP uses a different host-image set and a different watermark image, so exact reproduction still requires matching the paper dataset and watermark.
