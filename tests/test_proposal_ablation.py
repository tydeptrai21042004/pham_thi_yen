import csv
import os
import subprocess
import sys
from pathlib import Path


def test_proposal_ablation_phase_smoke(tmp_path):
    """Run the ablation phase in a subprocess so the smoke test stays isolated and fast."""
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_ablation"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    command = [
        sys.executable,
        str(root / "main.py"),
        "--phase",
        "proposal-ablation",
        "--host-dir",
        str(root / "data/host"),
        "--watermark",
        str(root / "data/watermark/wm.png"),
        "--output",
        str(out),
        "--ablation-variants",
        "full,no_structured_repetition",
        "--max-images",
        "1",
        "--no-save-images",
        "--attack-preset",
        "none",
        "--proposal-param-mode",
        "ignore",
        "--proposal-repeat",
        "1",
    ]

    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    assert (out / "proposal_ablation_manifest.csv").exists()
    assert (out / "proposal_ablation_per_image_attack_results.csv").exists()
    assert (out / "proposal_ablation_summary_by_variant_phase.csv").exists()
    assert (out / "proposal_ablation_delta_vs_full.csv").exists()
    assert (out / "proposal_ablation_failures.json").exists()

    with (out / "proposal_ablation_per_image_attack_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {r["proposal_ablation"] for r in rows} == {"full", "no_structured_repetition"}
