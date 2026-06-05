from pathlib import Path
import os
import subprocess
import sys

from watermarklab.benchmark import run_proposal_hparam_sweep_phase


def test_proposal_hparam_sweep_phase_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_hparam_sweep"

    result = run_proposal_hparam_sweep_phase(
        host_dir=root / "data/host",
        watermark_path=root / "data/watermark/wm.png",
        output_dir=out,
        q4_tau_values=[0.40, 0.50],
        q4_margin_values=[0.08],
        h01_q_values=[7.0],
        h01_margin_values=[0.90],
        max_images=1,
        save_outputs=False,
        attack_preset="none",
        repeat=1,
    )

    assert not result["results"].empty
    assert not result["summary"].empty
    assert not result["summary_by_combo"].empty
    assert not result["ranking"].empty
    assert result["failures"] == []
    assert len(result["manifest"]) == 2
    assert set(result["manifest"]["q4_tau_effective"]) == {0.40, 0.50}
    assert (out / "proposal_hparam_sweep_per_image_attack_results.csv").exists()
    assert (out / "proposal_hparam_sweep_summary_by_combo_phase.csv").exists()
    assert (out / "proposal_hparam_sweep_summary_by_combo.csv").exists()
    assert (out / "proposal_hparam_sweep_ranking.csv").exists()
    assert (out / "proposal_hparam_sweep_best.json").exists()
    assert (out / "proposal_hparam_sweep_failures.json").exists()


def test_proposal_hparam_sweep_cli_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_hparam_sweep_cli"
    cmd = [
        sys.executable,
        str(root / "main.py"),
        "--phase",
        "proposal-hparam-sweep",
        "--host-dir",
        str(root / "data/host"),
        "--watermark",
        str(root / "data/watermark/wm.png"),
        "--output",
        str(out),
        "--max-images",
        "1",
        "--no-save-images",
        "--attack-preset",
        "none",
        "--proposal-repeat",
        "1",
        "--hparam-sweep-q4-tau",
        "0.40,0.50",
        "--hparam-sweep-q4-margin",
        "0.08",
        "--hparam-sweep-h01-q",
        "7",
        "--hparam-sweep-h01-margin",
        "0.90",
        "--proposal-param-mode",
        "ignore",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True, env=env)
    assert "Saved proposal hparam sweep ranking" in completed.stdout
    assert (out / "proposal_hparam_sweep_per_image_attack_results.csv").exists()
    assert (out / "proposal_hparam_sweep_summary_by_combo.csv").exists()
    assert (out / "proposal_hparam_sweep_ranking.csv").exists()
