# Patch: add Hess-Nha2023 64x64 baseline and occlusion percentage attacks

## Added baseline

New method ID:

```text
hess_nha2023
```

Implementation file:

```text
src/watermarklab/methods/hess_nha2023.py
```

Main behavior:

- accepts the project-standard `64x64` binary watermark;
- uses Arnold scrambling;
- embeds into 4x4 blocks of the blue channel;
- applies Hessenberg decomposition and quantizes `H(2,2)` using the Nha et al. Eq. (18)/(19) logic;
- extracts blindly using Eq. (20);
- uses 2D tiling over the 128x128 block grid of a 512x512 host, so each 64x64 watermark bit has 2x2 = 4 repeated copies;
- recovers the watermark by majority voting and inverse Arnold transform.

Default mode:

```text
--hess-nha-mode adapt
```

uses the uploaded notebook's 64x64 adapter setting:

```text
T = 15
alpha = 3.2
```

Paper-parameter mode:

```text
--hess-nha-mode paper
```

uses:

```text
T = 65
alpha = 3.2
```

but still remains a 64x64 project adapter because the original paper used 32x32 watermarks.

## Added attacks

The attack registry now includes percentage occlusion:

```text
occlusion_25pct
occlusion_50pct
```

These are available in the `requested`, `full`, `script`, and `grid` attack presets through the new `occlusion_fraction` attack group.

## Validation

Smoke tests were run with:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
```

Result:

```text
4 passed
```

A clean one-image Hess-Nha smoke run gave:

```text
BER = 0.0
NC = 1.0
NCC = 1.0
PSNR ≈ 50.665 dB
```
