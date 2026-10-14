# Clean baseline methods

This project keeps the requested paper baselines plus the proposal method.

## Method IDs

| ID | Paper baseline | Implementation file |
|---|---|---|
| `kumar2021` | Kumar & Singh 2021, DWT maximum-entropy alpha blending | `src/watermarklab/methods/kumar2021_dwt_entropy.py` |
| `guo2017_dwt_qr_fa` | Guo et al. 2017, DWT-QR with Firefly Algorithm | `src/watermarklab/methods/guo2017_dwt_qr_fa.py` |
| `gaata2022_dwt_hess_fwa` | Gaata et al. 2022, DWT-Hessenberg with Firework Algorithm | `src/watermarklab/methods/gaata2022_dwt_hess_fwa.py` |
| `dwt_hd_svd_2025` | Dong et al. 2024/2025, DWT-HD-SVD with logistic chaos | `src/watermarklab/methods/dwt_hd_svd2025.py` |
| `hess_nha2023` | Nha et al. 2023, blind Hessenberg H(2,2) quantization, adapted to 64x64 watermark | `src/watermarklab/methods/hess_nha2023.py` |
| `roy2018_dwt_svd` | Roy & Pal 2018, YCbCr-Y blockwise 3-level DWT-SVD side-information baseline | `src/watermarklab/methods/roy2018_dwt_svd.py` |
| `proposal` | Your proposed Q/H DWT-Hessenberg method | `src/watermarklab/methods/proposal_qh_dwt_hess.py` |

Removed from the old project: the obsolete old `roy2018` implementation, `iwt_hess_svd_2024`, `mahto2022_firefly_dual`, old smoke-result folders, old refactor changelogs, and obsolete tests.

## Why reproduced results may differ from the paper

The paper-reported numbers and this unified benchmark are not the same experiment. Differences come from:

1. different cover images,
2. different watermark size or payload,
3. grayscale vs. color carrier adaptation,
4. MATLAB vs. Python/SciPy/Pillow numerical behavior,
5. unspecified wavelet/optimizer/chaotic-map details in some papers,
6. unified attack parameters that may not match the original paper exactly.

Use `docs/baseline_paper_reported_results.csv` for the original-paper reported scale, and use `results/.../*.csv` for local unified-benchmark results.

## Fair table rule for the manuscript

Do not mix these in one table without labels:

- **Paper-reported result**: copied from the original paper.
- **Paper-guided reproduction**: this implementation following the paper algorithm.
- **Unified benchmark**: same host set, watermark, metrics, and attacks for all methods.
- **Proposal result**: your method under the same benchmark.


## Roy 2018 DWT-SVD side-information baseline

Method id: `roy2018_dwt_svd`. This baseline follows Roy and Pal's YCbCr-Y blockwise DWT-SVD procedure: 512x512 RGB host images are converted to YCbCr, the Y channel is divided into 32x32 blocks, a three-level Haar DWT is applied to each block, and 4x4 watermark blocks are embedded into the singular-value matrix with `alpha=0.02`. Extraction stores and reuses `Sp`, `UWp`, and `VWp` from the embedding phase, so the method is a side-information baseline rather than a fully blind method.

## Added baselines: DWT-WHT-SVD2024 and Zhang2022 QWT-QSVD

### `dwt_wht_svd_2024`

- Paper: entropy-based adaptive DWT-WHT-SVD color watermarking in YCbCr space.
- Host/payload compatibility: native 512x512 color host and 64x64 watermark.
- Local payload: binary 64x64 watermark is used as a grayscale `{0,255}` watermark, preserving 4096 payload symbols.
- Extraction type: semi-blind, matching the original paper. The key stores `HPC`, `UW`, `VW^T`, adaptive `alpha`, and Arnold parameters.
- Main table placement: semi-blind / side-information table.

### `qwt_qsvd_zhang2022_blind`

- Paper: robust image watermarking based on QWT and QSVD using 2D Chebyshev-Logistic map.
- Host/payload compatibility: color host is converted to YCbCr; low-frequency QWT/Q1 domain is split into 4x4 blocks. For a 512x512 host this gives 4096 blocks, exactly matching a 64x64 binary watermark.
- Extraction type: blind.
- Main table placement: fair blind table.

### `qwt_qsvd_zhang2022_semiblind`

- Same embedding structure as `qwt_qsvd_zhang2022_blind`.
- Extraction type: semi-blind, preserving the paper's second extraction mode using selector information.
- Main table placement: semi-blind / side-information table.

## Added paper baselines (2022 LWT-DCT-HD, 2016 Contourlet-HD)

Two *blind*, paper-guided baselines were added under separate method IDs. No
proposal code or proposal parameters were changed.

| ID | Original paper (DOI) | Original protocol | This project's **adapt** protocol |
|---|---|---|---|
| `abodena2022_lwt_dct_hess` | Abodena (2022), 10.51984/JOPAS.V21I2.1934 | grayscale 1024x1024, two-level LWT -> DCT -> 4x4 Hessenberg H(1,1), 64x64 watermark, entropy scale | RGB 512x512 -> Y, ONE Haar-lifting level to give 4096 4x4 blocks; keyed ordering, entropy-based quantization with uint8-safe minimum, Y->RGB. 4096 real bits (not 1024). |
| `su2016_contourlet_hess` | Su et al. (2016), 10.1007/s11042-016-3522-z | RGB 512x512, 32x32 color watermark (24,576 bits), one-level Contourlet LP low-pass, 4x4 Hessenberg max-energy coefficient, keyed MD5 block selection, Arnold scrambling, T=16 | RGB 512x512, *single blue channel* LP low-pass 256x256 and 4096 blocks, 64x64 binary watermark, Arnold + MD5 ordering; T=16. |

**Method-fidelity boundaries:** Neither paper releases the complete filter,
quantization, and key-generation specifications needed for a bit-for-bit MATLAB
reproduction. Abodena does not specify the chosen LWT lifting filter or the
entropy histogram binning; the paper inconsistently labels its second-level band as both a detail
and an approximation, and printed Eqs. 6–12 have a candidate/parity
ambiguity. Here the diagonal detail of the second LWT level is selected. The adapted implementation uses a consistent ceil-parity target and
an empirically documented minimum step of 30 for uint8 stability. Su does not
specify the exact Contourlet LP filters; the implementation uses a 2x2 box
Laplacian pyramid with explicit highpass residual, which ensures perfect
reconstruction before uint8 rounding. Because embedding is confined to the
lowpass band, the Contourlet directional detail bank is *not computed*; this is
not an exact NSCT/DFB reproduction. These two algorithms MUST be labelled
**paper-guided adaptations**, not original implementations or the authors' code.

Neither method requires the original host or watermark at extraction. The
Abodena extraction key stores one quantization strength scalar in addition to
the seed and shape; Su stores T, seed, and Arnold rounds. Neither key contains
image-dependent matrices or the original logo. Disclose key/metadata bytes in
side-information reporting when applicable.

For paper-original numbers use `--baseline-mode original`, which writes a
separate `original_reported/` export and never mixes it with rerun results.
Do not pool 2016's 24,576-bit *color* watermark metrics or 2022's 1024x1024
grayscale metrics with this project's native 4096-bit RGB benchmark numbers.

### Run only the two new baselines

```bash
python main.py --methods abodena2022_lwt_dct_hess,su2016_contourlet_hess --attack-preset lite --output results/new_hessenberg_baselines
python -m pytest -q tests/test_paper_hess_baselines_2016_2022.py
```

The algorithms use the original 64x64 watermark pixels as 4096 input bits;
**do not calculate payload size after transforming the watermark**. All NC,
BER, PSNR, SSIM, and attack-result files come from the benchmark's unchanged
metric/export pipeline.
