# 21-host image protocol: 9 training + 12 testing

## Requested assignment (October 2026)

The three existing *training/tuning* hosts were **baboon.bmp, safari.bmp, and sailboat.bmp** under the previous shuffled `seed=3101` protocol. All three are now assigned to **testing**, alongside the other nine original images.

**Training (9 newly supplied images)** — in `data/host_train/`:

`fruits.bmp`, `NB0001.bmp`, `NB0025.bmp`, `NB0196.bmp`, `NB0215.bmp`, `NB0232.bmp`, `NB0688.bmp`, `NB1353.bmp`, `seaport.bmp`.

**Testing (12 originals)** — selected from `data/host/`:

`Girl.bmp`, `airplane.bmp`, `athens.bmp`, `baboon.bmp`, `butfly.bmp`, `house.bmp`, `manhatan.bmp`, `milkdrop.bmp`, `pepper.bmp`, `safari.bmp`, `sailboat.bmp`, `tiffany.bmp`.

`lenna.bmp` is kept in the original host directory but excluded, consistent with the earlier manuscript protocol. There is no random new split and no crossover between training and testing. This *changes the experimental protocol*; prior 3/9 results cannot be reported as results from the new 9/12 evaluation without rerunning experiments.

All images must be actual **512×512 RGB BMP** files. The newly uploaded `fruits.bmp` was PNG-encoded despite its `.bmp` extension; the packaged copy is losslessly re-encoded as a proper BMP (same RGB pixel values).

The proposed embedding/extraction algorithm, baseline algorithms, and the watermark payload are **not changed**. "Training" means tuning/optimizing any selected parameters on the nine training hosts, **not** retraining pretrained TrustMark or WAM model weights.

## Create split + manifest

Run at the project root (after `pip install -r requirements.txt`):

```bash
python -m watermarklab.experiments.splits --host-dir data/host --train-dir data/host_train --output results/final_9_12/split.json --create-dirs
```

The script validates filenames, 512×512 RGB/BMP format, no duplicate decoded images, and generates reproducibility SHA-256 hashes. It creates:

- `results/final_9_12/split.json` (including `train_dir` and `test_dir`)
- `results/final_9_12/splits/train/` (9 images)
- `results/final_9_12/splits/test/` (12 images)

## Run benchmark only on the test set

**First tune all parameters exclusively using the train split**; do not reuse parameters optimized on the former training images (now test hosts). The end-to-end `scripts/kaggle_review_run.py` handles this: its optimization phases use the 9-image train directory, while full classical and optional neural evaluation phases use the 12-image test directory.

A quick, *not-yet-tuned* functional check for the two new paper baselines:

```bash
python main.py --methods abodena2022_lwt_dct_hess,su2016_contourlet_hess \
  --host-dir results/final_9_12/splits/test --max-images 1 \
  --attack-preset none --no-save-images --output results/final_9_12/smoke_baselines
```

For a full benchmark with optimized parameters, run:

```bash
python scripts/kaggle_review_run.py --out results/reviewer_9train_12test
```

`run_all.py` also creates this named split and evaluates the 12 held-out hosts. Without a new 9-host-tuned parameter file it warns that the run is only a functional check; it does not load the old optimization file.

The full reviewer script may be computationally expensive. The separate one-cell Kaggle neural runner **clones GitHub** rather than reading this ZIP; commit the updated `data/host_train/` and split code to GitHub before using it.

Run dataset regression tests:

```bash
python -m pytest -q tests/test_dataset_9_train_12_test.py tests/test_revised_protocol.py tests/test_reviewer_revision.py
```
