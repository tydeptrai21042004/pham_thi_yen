# Reviewer-oriented code revision — implementation guide

This is a **change-only patch** for `pham_thi_yen-main`. It preserves the original code for paper comparison, adds explicit diagnostics, experimental checks, and two optional *unverified-in-this-environment* neural adapters. **It does not claim that the watermarking method is now verified or reviewer-ready.** Regenerate all affected experimental tables only after a verified production environment is available.

## What was changed

| Area | New behavior | Reviewer link | Status |
|---|---|---|---|
| Candidate BSS | Exact decoded bit must be correct before empirical BSS is accepted, including `no_strength_filter` | R1.6, R2.4 | Unit tested |
| Q-domain | Skips row-pair `(1,2)` (zero based): its rotation preserves the two-component squared norm | Mathematical consistency | Static proof; code changed |
| Decode diagnostics | Per-payload valid votes, ties, erasures, confidence and captured decoding failures | R1.1, R1.4 | Smoke tested |
| Metadata | Keeps original watermark detail bands for legacy mode; records BSS/MSE per scheduled candidate and exceptions | AE, R1.1, R1.7 | Reproduction preserved |
| Contrastive embedding | Optional `require_contrastive_embedding=True` skips naturally matching unmarked blocks | False-positive risk | **Experimental, not a verified fix** |
| Strict block size | Rejects `block_size != 4` with an explicit explanation instead of silently implying generalized support | R2.1 | Guard in place |
| JPEG2000 | Explicit failure if codec unavailable instead of substituting ordinary JPEG | Experimental validity | Tested |
| Strong attacks | `reviewer_stress` preset with exact Gaussian/speckle variance, JPEG 40/30, geometry, crops and scales | R1.5, R3.2 | Smoke tested |
| Independent split | Deterministic image-level train/test split and input hashes | R1.3 | Tested, not yet wired to FA trainer |
| Presence audit | Original-host and different-host negative controls; payload BER, watermark BER/NC, color-conversion-only PSNR, timings, side info | AE, R1.1–R1.4 | Smoke tested |
| BSS validation | Local BSS vs per-block actual attack survival with Spearman correlation | R1.6, R2.4 | Smoke tested |
| Optional neural | Native-payload adapters for TrustMark and ChunkySeal; checkpoints downloaded by upstream libs | R2.2 | API wired; **inference not tested locally** |

## Installation

```bash
cd pham_thi_yen-main
python -m pip install -r requirements.txt
export PYTHONPATH=src               # Windows PowerShell: $env:PYTHONPATH="src"
python -m pytest -q tests/test_reviewer_revision.py
```

**Mandatory for publication**: install **PyWavelets**, pin exact dependency versions, and set `WATERMARKLAB_STRICT_PYWT=1`. In the container where this patch was tested, PyWavelets was not installed; the existing pure NumPy orthonormal Haar transform was used with explicit `RuntimeWarning`. That is suitable for code smoke tests but **not** a validated reproduction of PyWavelets results. Do not mix these outputs.

## Run reviewer evidence

```bash
# 1. Reproducible training/test split (this does NOT automatically run FA).
python -m watermarklab.experiments.splits \
  --host-dir data/host --output results/reviewer_split.json --seed 3101

# 2. Negative controls + small method smoke test. Use --host-size 512
# --watermark-size 64 for the manuscript configuration after smoke testing.
python -m watermarklab.experiments.review_audit \
  --host-dir data/host --watermark data/watermark/wm.png \
  --output results/reviewer_audit --max-images 2 \
  --host-size 128 --watermark-size 16 --repeat 1 \
  --attack-preset reviewer_stress --max-attacks 5

# 3. Direct binary 4096-bit mode; same experiment with no original WM DWT details.
# The full 64x64 / 512x512 setting has reduced repetition capacity.
python -m watermarklab.experiments.review_audit \
  --host-dir data/host --watermark data/watermark/wm.png \
  --output results/reviewer_direct --host-size 128 \
  --watermark-size 16 --repeat 1 --direct-binary

# 4. OPTIONAL experimental contrastive mode: often rejects many blocks.
python -m watermarklab.experiments.review_audit \
  --host-dir data/host --watermark data/watermark/wm.png \
  --output results/reviewer_contrast --host-size 128 \
  --watermark-size 16 --repeat 1 --contrastive

# 5. Does local BSS predict real attacks?
python -m watermarklab.experiments.bss_validation \
  --host data/host/Girl.bmp --watermark data/watermark/wm.png \
  --output results/reviewer_bss --host-size 128 \
  --watermark-size 16 --max-attacks 5
```

### Optional neural baselines (native payload, not directly payload-matched)

```bash
python -m pip install trustmark
python -m watermarklab.experiments.neural_baselines \
  --method trustmark --host data/host/Girl.bmp \
  --output results/neural_trustmark --max-attacks 3
```

ChunkySeal requires a separate installation of the official `facebookresearch/videoseal` repository and PyTorch. Once installed:

```bash
python -m watermarklab.experiments.neural_baselines \
  --method chunkyseal --host data/host/Girl.bmp \
  --output results/neural_chunkyseal --max-attacks 3
```

These adapters require upstream checkpoints and internet on first use. They **were not executed** in the provided offline environment; update the adapter if the upstream API differs. Never label their native-message results as equivalent to the proposed 1024- or 4096-bit payload without a payload-matched protocol.

## Important remaining issues — do not claim resolved

1. **False positives are still present in the original algorithm**. In one *small smoke experiment*, the watermarked and original unmarked host had identical recovered watermark NC. The audit now makes this visible. Removing naturally matching blocks reduces embedding capacity and did not solve this issue in the small test.
2. Original watermark DWT detail coefficients remain in legacy `payload_meta`; these make watermark reconstruction **semi-blind with original-watermark-dependent side information**. Use `watermark_dwt_enabled=False` for direct binary payload, but expect a substantially different trade-off and rerun results.
3. Q-domain candidate validity after a full image round trip is **not proved**; the exact candidate check occurs in floating point. The encoder needs a full round-trip acceptance/rebuild policy before robust ownership claims.
4. BSS correlation on a tiny 8x8 watermark / 64x64 host was **negative** in the smoke test; no robustness-prediction claim is justified yet. Use representative data and held-out attacks.
5. `splits.py` creates a disjoint manifest only. Existing FA tuning scripts must be made to **consume and enforce** that split; do not manually use test images for parameter tuning.
6. Current FA fitness differs from the LaTeX description; the optimization script requires alignment with the manuscript before reporting optimized results.
7. Existing classical baseline implementations were preserved, **not fully audited against their original algorithms**. The neural adapters are not installed/tested locally.
8. Do not claim tested 8x8 or 16x16 blocks: those require changing Q/H coordinate definitions and validated selection mathematics.
9. The paper and code have a known Arnold/DWT order mismatch. Align the revised manuscript with the implemented code or explicitly change and re-evaluate the code.
10. The short development key `KB123` is not a production-strength secret. Secrets should not be stored in published manifests.

## Validation summary

- Eight new targeted tests passed.
- The focused mixed regression subset passed **17 tests**.
- The reviewer-stress attack catalogue and BSS smoke run generated output files.
- The neural adapters are unverified without optional dependencies.
- **Full project suite not certified**. A previous attempt was interrupted due to long experiment tests; run it in the target environment with PyWavelets installed and adequate execution time.

## Evidence file meanings

- `per_case_results.csv` contains raw payload BER (all and valid positions), reconstructed watermark BER/NC/centered NCC, coverage, erasures, ties, negative controls, runtimes, metadata bytes and color conversion PSNR.
- `bss_per_block.csv` measures whether selected blocks decode correctly after each **whole-image** attack.
- `bss_summary.json` contains survival rates and Spearman correlation (null when undefined).
- `split.json` records disjoint image partitions and source-image hashes.
- `manifest.json` documents the configuration; the development secret is redacted.

**Scientific rule:** Do not copy smoke-test values into the manuscript's tables. Re-run the final method and every comparison under the same validated environment, fixed parameters, independent test images and documented payload accounting.
