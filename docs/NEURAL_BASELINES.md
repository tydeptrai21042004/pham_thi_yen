# Neural baselines: TrustMark + WAM (no ChunkySeal)

Only **TrustMark** (61-bit BCH_5 information payload) and **WAM (ICLR 2025)**
(32-bit MIT checkpoint) are configured as optional pretrained comparisons.
Neither is comparable as an equal-capacity test with the proposed 1024-bit LL
payload. Keep `decode_success_rate`, `nc` (conditional on a successful decode),
`ber_if_missing_one`, payload capacity, clean PSNR and runtime separate.

On Kaggle, use GPU and Internet ON. Install the full project dependencies,
`pip install trustmark`, clone the **official**
`https://github.com/facebookresearch/watermark-anything.git`, and install
its `requirements.txt` in a compatible PyTorch/CUDA environment. Download the
MIT checkpoint from `https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth`
into `checkpoints/wam_mit.pth`, leaving upstream `checkpoints/params.json` and
`configs/` intact. Set environment variables `WAM_REPO_DIR` and
`WAM_CHECKPOINT`. Then use:

```bash
PYTHONPATH=src python main.py --phase neural --methods trustmark,wam \
  --host-dir /path/to/frozen/test --attack-preset full --no-save-images \
  --output results/neural_comparison
```

Or use `scripts/kaggle_neural_one_cell.py` after committing the updated files.
On a machine without GPU or official weights, the WAM real-model integration
**cannot be validated**; there is no mock fallback during production runs.
Test fixture adapters are only allowed in offline unit tests.

The WAM adapter uses the official full-image watermark embedder and its
32-bit message detector. The pixel mask threshold is a segmentation rule,
not a statistically calibrated watermark ownership decision. Always report
unwatermarked-image controls separately.
