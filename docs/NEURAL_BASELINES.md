# Run pretrained ML / DL watermark baselines inside this repository

Your classical `python main.py --methods all` **does not** run neural methods. Neural inference is optional and separated deliberately, so your 4,096-bit watermark is not directly compared against lower native payload capacities.

## 1. Install your project

```bash
python -m pip install -r requirements.txt
```

Install TrustMark (uses pretrained model weights, downloaded on first run):

```bash
python -m pip install trustmark
```

Optionally install ChunkySeal (high-capacity VideoSeal model). A compatible `torch` and `torchvision` installation is required. Check the official instructions at https://github.com/facebookresearch/videoseal (the upstream package currently lists `decord` and additional ML dependencies):

```bash
python -m pip install "git+https://github.com/facebookresearch/videoseal.git"
```

The official pretrained weights are downloaded on first use. **If optional dependencies are missing, the runner records a failure and exits nonzero instead of inventing results.** On Kaggle enable Internet and GPU (recommended for ChunkySeal).

## 2. Run via *your existing main.py*

```bash
# Quick validation, one image, two image attacks
PYTHONPATH=src python main.py --phase neural --methods trustmark \
  --host-dir data/host --max-images 1 --attack-preset lite \
  --neural-max-attacks 2 --no-save-images --output results/neural_trustmark_smoke

# Full native-payload TrustMark + ChunkySeal on the entire host folder
PYTHONPATH=src python main.py --phase neural --methods trustmark,chunkyseal \
  --host-dir data/host --attack-preset full --no-save-images \
  --neural-device auto --output results/neural_all
```

Windows PowerShell: set `$env:PYTHONPATH="src"` and then run `python main.py ...`. Or use `python -m pip install -e .` to make `PYTHONPATH` unnecessary.

You can also run a *held-out* folder from the earlier Kaggle notebook:

```bash
PYTHONPATH=src python main.py --phase neural --methods trustmark,chunkyseal \
  --host-dir /kaggle/working/pham_yen_kaggle_results/splits/test \
  --attack-preset full --no-save-images \
  --output /kaggle/working/pham_yen_kaggle_results/heldout_neural
```

The earlier notebook excluded `splits/` from its ZIP archive: run neural evaluation on the same held-out folders **within the same session**, or recreate the split from `experiment_manifest.json` / seed `3101` before running another notebook. Never evaluate training hosts as an unseen test set.

## 3. Reuse this in your existing Kaggle scheduler

Immediately after `pip_install` (and **before** long `size_sweep`), add the following stages to the existing Kaggle script. Each `main(...)` call is already handled by your autosave system:

```python
run_step('install_trustmark', [sys.executable, '-m', 'pip', 'install', '-q', 'trustmark'])
# Optional: may require extra dependencies; use a compatible torch environment.
run_step('install_videoseal', [sys.executable, '-m', 'pip', 'install', '-q',
                             'git+https://github.com/facebookresearch/videoseal.git'])
main('heldout_neural_trustmark', [
    '--phase', 'neural', '--methods', 'trustmark', '--host-dir', test_dir,
    '--attack-preset', 'full', '--no-save-images',
    '--output', OUT / 'heldout_neural_trustmark',
])
main('heldout_neural_chunkyseal', [
    '--phase', 'neural', '--methods', 'chunkyseal', '--host-dir', test_dir,
    '--attack-preset', 'full', '--no-save-images',
    '--output', OUT / 'heldout_neural_chunkyseal',
])
```

Alternatively use `scripts/kaggle_neural_one_cell.py` in a new Kaggle notebook after pushing this patch to the GitHub repo; paste the entire script into a single cell.

## 4. Outputs and paper reporting

All outputs are under the selected `--output` directory:

- `per_image_attack_results.csv`: image × attack; same main metrics as your classical experiment plus `native_capacity_bits`, `detected`, `decoded`, `ber_if_missing_one`, and `model_device`.
- `summary_by_method_phase.csv`: clean/attacked mean metrics and **decode success rate**.
- `compare_psnr_nc_ber_ncc_before_after_attack.csv`: wide format for plots/tables.
- `manifest.json`: requested methods, host names, attack names, seed and native-capacity warning.
- `failures.json`: unavailable models, failed image processing, failed attacks/decodes.

`BER` is **blank** when valid bits could not be decoded. `ber_if_missing_one` conservatively counts a failure as BER=1; never silently average only successful decoding. Unmarked-image negative controls have phase `control` and are *excluded* from robustness means. `detected` is meaningful for TrustMark; ChunkySeal's presence threshold is *not* treated as calibrated.

**Do not merge native-capacity neural results with your 4,096-bit capacity-matched benchmark table.** TrustMark BCH_5 is normally 61 information bits; ChunkySeal is 1,024 bits. Include model/checkpoint versions, device, image resolution and native capacity in the paper. This is **pretrained inference**, not training or fine-tuning.
