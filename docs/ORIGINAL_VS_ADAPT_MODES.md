# Baseline modes: original vs adapt

The project separates two tasks that should not be mixed in one manuscript table.

## `--baseline-mode original`

Writes the published paper numbers to:

```text
<output>/original_reported/paper_reported_results.csv
<output>/original_reported/paper_reported_summary.csv
```

This mode is for **paper-reported results**. It is the only mode that can match a paper table exactly, because it copies the reported values from the papers. It does not claim a bit-for-bit rerun.

## `--baseline-mode adapt`

Runs the implemented methods on your selected host images and watermark. This is the mode for the unified benchmark against your proposal. It keeps the original algorithm idea and hyperparameters as much as possible, but changes the minimum required parts to accept your common input type and common attack suite.

## `--baseline-mode original-rerun`

Development mode for stricter local reruns. Do not claim a 100% paper match from this mode unless the original authors' exact images, watermark, code, random seeds, attack operators, software versions, and numerical libraries are available.

## Per-baseline overrides

```bash
python main.py --methods baselines --baseline-mode original --output results/original_reported
python main.py --methods baselines,proposal --baseline-mode adapt --attack-preset lite --output results/adapt_lite
python main.py --methods kumar2021,proposal --baseline-mode adapt --kumar-mode original --output results/kumar_reported_plus_proposal
```

Available override flags:

```text
--kumar-mode
--guo-mode
--gaata-mode
--dwt-hd-svd-mode
```
