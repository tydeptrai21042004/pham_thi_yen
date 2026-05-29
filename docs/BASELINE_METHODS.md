# Clean baseline methods

This project keeps only the four paper baselines requested by the user plus the proposal method.

## Method IDs

| ID | Paper baseline | Implementation file |
|---|---|---|
| `kumar2021` | Kumar & Singh 2021, DWT maximum-entropy alpha blending | `src/watermarklab/methods/kumar2021_dwt_entropy.py` |
| `guo2017_dwt_qr_fa` | Guo et al. 2017, DWT-QR with Firefly Algorithm | `src/watermarklab/methods/guo2017_dwt_qr_fa.py` |
| `gaata2022_dwt_hess_fwa` | Gaata et al. 2022, DWT-Hessenberg with Firework Algorithm | `src/watermarklab/methods/gaata2022_dwt_hess_fwa.py` |
| `dwt_hd_svd_2025` | Dong et al. 2024/2025, DWT-HD-SVD with logistic chaos | `src/watermarklab/methods/dwt_hd_svd2025.py` |
| `proposal` | Your proposed Q/H DWT-Hessenberg method | `src/watermarklab/methods/proposal_qh_dwt_hess.py` |

Removed from the old project: `roy2018`, `iwt_hess_svd_2024`, `mahto2022_firefly_dual`, old smoke-result folders, old refactor changelogs, and obsolete tests.

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
