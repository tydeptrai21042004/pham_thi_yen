# Patch: fix DWT-WHT-SVD2024 and QWT-QSVD Zhang2022 baselines

This patch fixes the issues found in the lite-attack sanity check for the newly added baselines.

## 1. DWT-WHT-SVD2024 inverse SVD reconstruction

Fixed the embedding reconstruction step:

```python
marked_coeff = uw @ sw_diag @ vwt
```

instead of incorrectly reusing the pre-embedding `vt` matrix. This matches the SVD of the embedded principal component:

```python
uw, sw, vwt = np.linalg.svd(embedded_pc, full_matrices=True)
```

## 2. QWT-QSVD Zhang2022 Q1 branch

The previous version used a zero-imaginary quaternion wrapper:

```python
QuaternionBlock(r=ll_r, i=zeros, j=zeros, k=zeros)
```

The corrected version now uses a deterministic in-repo QWT-style low-frequency quaternion approximation:

- real branch: `LL(Y)`;
- i branch: low-frequency phase response from horizontally shifted Y;
- j branch: low-frequency phase response from vertically shifted Y;
- k branch: low-frequency phase response from diagonally shifted Y.

This is still dependency-free and should be described as an adapted QWT implementation unless a full dual-tree/Hilbert QWT library is added.

## 3. QWT-QSVD semi-blind selector now affects extraction

The previous `_extract_block()` ignored the selector and always read `s[0]`, so blind and semi-blind outputs were identical. The corrected version now:

- stores selector information only in semi-blind mode;
- uses selector-coded QIM residue classes;
- lets blind extraction infer the selector from the attacked block;
- lets semi-blind extraction use the stored selector from the key.

Therefore, the semi-blind branch is now actually different from the blind branch and can benefit from side information, as intended by the paper's blind/semi-blind distinction.

## 4. Added test coverage

Added `test_qwt_qsvd_semiblind_selector_changes_extraction_after_attack`, which verifies that:

- blind and semi-blind embedding produce the same watermarked image;
- only the semi-blind key stores selector side information;
- selector values are nontrivial;
- after an attack, blind and semi-blind extraction outputs are not identical.

## Verified tests

```bash
PYTHONPATH=src pytest -q \
  tests/test_new_baselines_2024_2022.py \
  tests/test_clean_registry.py \
  tests/test_baseline_smoke.py
```

Result:

```text
9 passed
```

## Verified CLI checks

Clean and lite checks were run for:

```text
dwt_wht_svd_2024
qwt_qsvd_zhang2022_blind
qwt_qsvd_zhang2022_semiblind
```

The corrected QWT-QSVD semi-blind branch now gives different results from blind mode and usually higher NC under selector-sensitive attacks.
