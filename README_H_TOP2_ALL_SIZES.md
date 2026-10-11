# Dynamic Top-2 Hessenberg energy selection at 4x4 / 8x8 / 16x16

This update changes the **H-domain coefficient candidate-selection policy**, not the Q-domain implementation, quantization rule, Firefly objective, Arnold transform, structured repetition, BSS definition, or downstream attack tests.

## New rule

For each *unmodified* host DWT block A (n in {4, 8, 16}):

1. Compute its Hessenberg matrix H (with the existing scipy.linalg.hessenberg routine).
2. Consider every structurally valid coefficient h[i,j], i <= j + 1, in zero-based indexing.
3. Rank coefficients by E[i,j] = abs(h[i,j])**2, descending. Break energy ties by ascending (i,j).
4. Generate *two* H-domain embedding candidates using the highest- and second-highest-energy positions. Existing safe-zone quantization, bit-survival evaluation, MSE limits, and H-versus-Q scoring are unchanged.
5. Store the chosen H-coordinate in the existing key's hpos_list. Extraction uses the stored coordinate, NEVER re-ranks positions from an attacked image.

This procedure applies equally to **4x4, 8x8 and 16x16**. The old special case returning fixed h21/h22 at 4x4 has been removed. Constants HPOS_CANDIDATES and numeric key codes 0/1 remain *only* to read pre-update side information.

## Key encoding and backward compatibility

New code: `key.hpos_list[k] = 2 + 16*i + j` for the selected H position `(i,j)` (zero-based), for **all block sizes**. Values `0` and `1` are never produced for new watermarks and are reserved for legacy h21 `(1,0)` and h22 `(1,1)` decoding. No key schema change is required.

The metadata includes `h_candidate_policy=per_block_top_two_H_energy`, `h_energy_definition`, `h_position_encoding`, and `h_energy_top2_count` at every size.

## Tests completed locally

- `PYTHONPATH=src python -m pytest -q tests --disable-warnings`: **66 passed** (45 fallback/import warnings), including 14 tests in `test_h_energy_top_two.py`.
- End-to-end smoke tests: embedding and extraction execute at **4x4, 8x8, 16x16** with H-only embedding.
- Comparing the same seeded 128x128 RGB host and 16x16 watermark against the **previous** `H_top2_energy_only_8x8_16x16` patch (64 H blocks/size):
  - **4x4**: output differs as intended; **34/64** H selections were outside old h21/h22 positions.
  - **8x8**: embedded image, H-position list, extracted watermark exactly identical.
  - **16x16**: embedded image, H-position list, extracted watermark exactly identical.
  - The new extractor reproduced the earlier extractor's output using legacy keys at every size.

**Local limitation:** PyWavelets is not installed in this test environment. The package used its explicit orthonormal Haar fallback. Validate the final experiments on Kaggle with PyWavelets installed before replacing manuscript tables. Passing the tests does NOT establish a higher NC, lower BER, or improved security.

## Apply to repository

Extract the ZIP over your repository root (only paths `src/watermarklab/methods/proposal_qh_dwt_hess.py` and `tests/test_h_energy_top_two.py` are code files). Commit these changes to GitHub, then rerun Kaggle with a fresh clone/reset.

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests/test_h_energy_top_two.py
PYTHONPATH=src pytest -q tests
```

Since the **4x4 algorithm has materially changed**, *old* 4x4 PSNR/SSIM/NC/BER, ablation, optimization, and security results do not represent the new proposal; rerun relevant evaluation and tuning on the unchanged train/test split. Existing 8x8/16x16 results should also be revalidated in the target environment. Classical baselines are unaffected by the H-policy change.

Manuscript description must replace statements saying that every H candidate comes from a fixed set `{h21, h22}` with the per-block Top-2 energy rule, explain the final BSS/MSE candidate decision, and disclose that the selected H position is stored as side information (semi-blind).
