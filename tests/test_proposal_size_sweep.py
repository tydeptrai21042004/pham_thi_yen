from pathlib import Path
import subprocess
import sys

from watermarklab.benchmark import run_proposal_size_sweep_phase


def test_proposal_size_sweep_phase_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_size_sweep"

    result = run_proposal_size_sweep_phase(
        host_dir=root / "data/host",
        watermark_paths=[root / "data/watermark/wm.png"],
        output_dir=out,
        host_sizes=[128],
        watermark_sizes=[16],
        max_images=1,
        save_outputs=False,
        attack_preset="lite",
        proposal_options={"params": {"repeat": None, "dwt_mode": "pywt"}},
        dpi=80,
    )

    assert not result["clean_results"].empty
    assert not result["attack_results"].empty
    assert not result["summary_by_size"].empty
    assert result["failures"] == []
    assert set(result["clean_results"]["host_size"]) == {128}
    assert set(result["clean_results"]["watermark_size"]) == {16}
    assert (out / "proposal_size_sweep_clean_results.csv").exists()
    assert (out / "proposal_size_sweep_attack_results.csv").exists()
    assert (out / "proposal_size_sweep_summary_by_size.csv").exists()
    assert (out / "proposal_size_sweep_psnr_before_attack.png").exists()
    assert (out / "proposal_size_sweep_nc_after_attack.png").exists()
    assert (out / "proposal_size_sweep_failures.json").exists()


def test_proposal_size_sweep_cli_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "proposal_size_sweep_cli"
    cmd = [
        sys.executable,
        str(root / "main.py"),
        "--phase",
        "proposal-size-sweep",
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
        "lite",
        "--size-sweep-host-sizes",
        "128",
        "--size-sweep-watermark-sizes",
        "16",
        "--size-sweep-dpi",
        "80",
        "--proposal-param-mode",
        "ignore",
    ]
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True, env=env)
    assert "Saved proposal size-sweep summary" in completed.stdout
    assert (out / "proposal_size_sweep_clean_results.csv").exists()
    assert (out / "proposal_size_sweep_attack_results.csv").exists()
    assert (out / "proposal_size_sweep_nc_after_attack.png").exists()
