# Runnable tests added — 14 October 2026

This update adds **33 offline regression tests** in `tests/test_operational_smoke_2026.py` and three small operational safeguards in `src/watermarklab/experiments/neural_4096.py`. The original proposed method (`src/watermarklab/methods/proposal_qh_dwt_hess.py`) is **unchanged**. No model architecture, image payload definition, scoring method, or previously saved experiment result is changed.

## Install and execute (Windows, macOS, Linux)

From the project root, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_operational_smoke_2026.py
```

To run all the newly added tests together with the existing reviewer/neural tests:

```bash
python -m pytest -q \
  tests/test_operational_smoke_2026.py \
  tests/test_reviewer_neural_4096.py \
  tests/test_reviewer_revision.py \
  tests/test_reviewer_patch_2026.py \
  tests/test_neural_integration.py
```

On Windows PowerShell, put the same file names on one line, rather than using `\` to join lines. Pytest's `pythonpath = ["src"]` in `pyproject.toml` makes `PYTHONPATH` optional for these tests.

## Test coverage

| Area | Checks |
|---|---|
| Original payload | Original 64x64 watermark is precisely 4096 source bits; rejects wrong sizes and degenerate inputs |
| Native chunk boundaries | TrustMark 61-bit messages create 68 chunks (last 9 real bits); WAM 32-bit messages create 128 chunks |
| Spatial assembly | 512x512 RGB image regions do not overlap; embed/detect reconstruction; host input remains unmodified |
| Failure accounting | Missing and invalid decoded chunks cannot be silently counted as recovered; erasure-as-error BER and conditional BER are distinct |
| Clean feasibility gate | Failed clean recovery skips attacks unless explicitly forced; failed experiments never become successful paper data |
| Whole-image attacks | Attack is invoked once on the assembled 512x512 image; per-attack CSV, summary and manifest are consistent |
| Crash/error reporting | Embedding failures record the native root cause; all-failure summary files retain parseable CSV column headers |
| Resume contract | Reusing a matching host/watermark is safe; changed watermark, model variant, force-full setting and other configuration invalidate old results |
| Real proposed-method pipelines | Small-image **real** BSS validation and security/side-information audit, including their CLI entry points and negative controls |
| Reviewer data exporter | Executes actual exporter script on saved-result fixtures; verifies host-bootstrap and missing-evidence reports |
| CLI entry points | `--help` works for the BSS, audit, and neural modules without installing pretrained neural models |

**Test result in the supplied local environment:** 33/33 added tests pass; 58/58 pass when combined with the four existing reviewer/neural test modules. `python -m compileall` also passes. These tests use synthetic, reversible adapters to validate chunk accounting. They do **not** claim that official TrustMark or WAM can recover 4096 bits in the real setting.

## Optional manual reproduction with original files

The small CLI runs use the **existing proposed method** and project images. They are functional checks, not manuscript-scale results.

```bash
# Windows PowerShell: first run `$env:PYTHONPATH = "src"`
# Linux/macOS: first run `export PYTHONPATH=src`
python -m watermarklab.experiments.bss_validation --host data/host/Girl.bmp --watermark data/watermark/wm.png --output results/bss_smoke_new --host-size 64 --watermark-size 8 --repeat 1 --max-attacks 1
python -m watermarklab.experiments.review_audit --host-dir data/host --watermark data/watermark/wm.png --output results/audit_smoke_new --max-images 1 --host-size 64 --watermark-size 8 --repeat 1 --attack-preset none --max-attacks 0
```

## Scope and remaining verification

- The official TrustMark package and WAM checkpoint are **not** provided by this package. Real pretrained inference must be tested separately with their upstream dependencies and weights; mock-adapter success is not pretrained-model success.
- **PyWavelets (`pywt`) was missing** in the local test environment. The project issued warnings and used its explicit orthonormal Haar fallback. Install PyWavelets and use the project's strict mode for publication-quality reproduction.
- The entire pre-existing test suite was **not** certified: an earlier unrestricted run exceeded 200 seconds. The fast focused suite above completed successfully.
- Existing experiments, previous results, and the proposed algorithm are kept unchanged. The limited runner changes affect only diagnostics, CSV handling, and detection of incompatible cached results.
