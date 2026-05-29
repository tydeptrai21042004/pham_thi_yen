# Cleanup manifest

## Kept methods

- `kumar2021`
- `guo2017_dwt_qr_fa`
- `gaata2022_dwt_hess_fwa`
- `dwt_hd_svd_2025`
- `proposal`

## Removed methods

- `roy2018`
- `iwt_hess_svd_2024`
- `mahto2022_firefly_dual`

## Removed clutter

- old refactor changelog files,
- stale smoke/stress result folders,
- obsolete tests referring to removed methods,
- unused Mahto vendor code.

## Validation run

```text
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
4 passed
```
