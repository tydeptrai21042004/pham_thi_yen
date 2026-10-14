# Final integrated code — comparison against uploaded source (14 October 2026)

**Base supplied by user:** `pham_thi_yen-main(20261014-081931).zip`.
**Prior updates reconciled:** 33 operational checks, Abodena 2022 + Su et al. 2016 paper-guided baselines, and a 9-new-training / 12-original-testing host split.

## Change inventory

Compared against the user ZIP (ignoring Python bytecode caches), this release has **17 added files and 15 modified files**, plus this report. **106 original source/assets files** are byte-for-byte identical. The original ZIP contained `.pyc`/`__pycache__` artifacts; these are intentionally excluded from the deliverables. No substantive original source file was deleted.

### New files (excluding this report)
- `TEST_ADDITIONS_20261014.md`
- `data/host_train/NB0001.bmp`
- `data/host_train/NB0025.bmp`
- `data/host_train/NB0196.bmp`
- `data/host_train/NB0215.bmp`
- `data/host_train/NB0232.bmp`
- `data/host_train/NB0688.bmp`
- `data/host_train/NB1353.bmp`
- `data/host_train/fruits.bmp`
- `data/host_train/seaport.bmp`
- `docs/DATASET_SPLIT_9TRAIN_12TEST.md`
- `src/watermarklab/methods/contourlet_hess_su2016.py`
- `src/watermarklab/methods/lwt_dct_hess_abodena2022.py`
- `src/watermarklab/methods/paper_hessenberg_common.py`
- `tests/test_dataset_9_train_12_test.py`
- `tests/test_operational_smoke_2026.py`
- `tests/test_paper_hess_baselines_2016_2022.py`

### Modified existing files
- `README.md`
- `README_REVIEWER_TWO_WORKSTREAMS.md`
- `docs/BASELINE_METHODS.md`
- `docs/baseline_paper_reported_results.csv`
- `main.py`
- `run_all.py`
- `scripts/kaggle_neural_one_cell.py`
- `scripts/kaggle_review_run.py`
- `src/watermarklab/benchmark.py`
- `src/watermarklab/experiments/neural_4096.py`
- `src/watermarklab/experiments/splits.py`
- `src/watermarklab/methods/__init__.py`
- `src/watermarklab/paper_reported.py`
- `tests/test_clean_registry.py`
- `tests/test_revised_protocol.py`

## Requested feature comparison

| Item | Source ZIP | Final result |
|---|---|---|
| Additional operational regression tests | Absent | `tests/test_operational_smoke_2026.py`, 33 added cases, with runner diagnostic safeguards |
| 2022 LWT–DCT–Hessenberg paper | Absent | `abodena2022_lwt_dct_hess` registered and benchmarkable |
| 2016 Contourlet–Hessenberg paper | Absent | `su2016_contourlet_hess` registered and benchmarkable |
| New RGB training hosts | No `data/host_train/` | Nine 512×512 RGB BMP images present |
| Old 3 training hosts in test | Seed-based old split | `baboon.bmp`, `safari.bmp`, and `sailboat.bmp` are now in the 12-host test split |
| Watermark bit count | 64×64=4096 | Exactly preserved, measured from original watermark |
| Proposed watermarking algorithm | Original | `proposal_qh_dwt_hess.py` SHA256 unchanged |
| Fresh-checkout runnable CLI | `python main.py --help` failed without PYTHONPATH | `main.py` and `run_all.py` add only a local `src` import path |

## Final host assignment

**Train (nine newly supplied):** `NB0001.bmp`, `NB0025.bmp`, `NB0196.bmp`, `NB0215.bmp`, `NB0232.bmp`, `NB0688.bmp`, `NB1353.bmp`, `fruits.bmp`, `seaport.bmp`.

**Test (12 original):** `Girl.bmp`, `airplane.bmp`, `athens.bmp`, `baboon.bmp`, `butfly.bmp`, `house.bmp`, `manhatan.bmp`, `milkdrop.bmp`, `pepper.bmp`, `safari.bmp`, `sailboat.bmp`, `tiffany.bmp`.

`lenna.bmp` remains stored in the original host directory but **excluded**. `fruits.bmp` is re-encoded as true BMP without altering the decoded RGB pixels. No training/tuning host is evaluated as held-out test, and both groups contain different decoded images. The word **training** here refers to tuning/optimizing benchmark parameters, not training the TrustMark or WAM pretrained neural model weights.

## Verification completed on the merged files

- **99/99** focused operational, dataset, paper-baseline, registry and CLI tests passed.
- **25/25** reviewer/neural integration tests passed after standalone CLI import repair.
- **124/124** total selected checks passed; test groups have no repeated modules.
- `python -m compileall -q src scripts main.py run_all.py tests`: pass.
- `python main.py --help` from fresh source without PYTHONPATH: pass.
- Programmatic `run_all.py` import from fresh source: pass (the computationally expensive benchmark was not started).
- Created a concrete 9-train/12-test split and ran both new paper-guided baselines on 2 held-out hosts (4 output rows). Exported CSV and `failures.json` with **no execution failures**.
- All 21 selected images validated as 512×512 RGB BMP.
- The proposal-method source and both original 64×64 watermark assets remain unchanged from the uploaded ZIP.

## Quick start

From the extracted `pham_thi_yen-main` directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_operational_smoke_2026.py tests/test_dataset_9_train_12_test.py tests/test_paper_hess_baselines_2016_2022.py
python main.py --help
```

To generate the frozen split with **bash/Linux/macOS**:

```bash
PYTHONPATH=src python -m watermarklab.experiments.splits --host-dir data/host --train-dir data/host_train --output results/final_9_12/split.json --create-dirs
```

On **Windows PowerShell**, use `$env:PYTHONPATH = "src"` once in that shell and then run `python -m watermarklab.experiments.splits ...` with the same arguments.

Execute a two-baseline fast check on the 12-host held-out directory (checks one image; no tuning):

```bash
python main.py --methods abodena2022_lwt_dct_hess,su2016_contourlet_hess --host-dir results/final_9_12/splits/test --max-images 1 --attack-preset none --no-save-images --output results/final_9_12/smoke_baselines
```

Full parameter optimization, classical evaluation and reviewer exports (long-running; use Kaggle/adequate CPU time):

```bash
python scripts/kaggle_review_run.py --out results/reviewer_9train_12test
```

Optional neural baseline runs require actual **pretrained TrustMark and WAM weights**; the mock-adapter tests do not establish real pretrained performance.

## Scientific limitations

1. The new baselines are **paper-guided adaptations**, not exact MATLAB replications. Abodena's grayscale 1024×1024 / two-level design was adapted to 512×512 RGB with 4096 payload bits; Su's Contourlet stage is a documented Laplacian-pyramid low-pass approximation, not an exact directional Contourlet implementation. Do **not** label local benchmark metrics as original-paper results.
2. A complete new 9-host optimization plus 12-host, full-attack benchmark has **not** run here; old 3/9 benchmark scores cannot be reused as held-out results for this split.
3. The local environment lacked **PyWavelets**; tests used the project's explicit Haar fallback. Install dependencies before strict publication-quality verification.
4. Testing passed for the named suites above, not for every long-running optimization, pretrained checkpoint, or external-dependency path.
