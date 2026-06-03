import csv
import os
import subprocess
import sys
from pathlib import Path


def test_main_cli_quick_run_creates_result_files(tmp_path):
    """Run the real CLI entry point on one image with no attack.

    This catches packaging/argument/import problems that unit-level tests can miss.
    """
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "cli_smoke"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    command = [
        sys.executable,
        str(root / "main.py"),
        "--methods",
        "kumar2021",
        "--max-images",
        "1",
        "--no-save-images",
        "--attack-preset",
        "none",
        "--output",
        str(output_dir),
        "--guo-param-mode",
        "ignore",
        "--gaata-param-mode",
        "ignore",
        "--proposal-param-mode",
        "ignore",
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

    per_image_csv = output_dir / "per_image_attack_results.csv"
    summary_csv = output_dir / "summary_by_method_phase.csv"
    comparison_csv = output_dir / "compare_psnr_nc_ber_ncc_before_after_attack.csv"
    failures_json = output_dir / "failures.json"

    assert per_image_csv.exists()
    assert summary_csv.exists()
    assert comparison_csv.exists()
    assert failures_json.exists()

    with per_image_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows, "CLI run should produce at least one result row"
    assert rows[0]["method_id"] == "kumar2021"
    assert rows[0]["attack"] == "no_attack"
    assert rows[0]["phase"] == "before_attack"
    assert float(rows[0]["psnr"]) > 30.0
    assert float(rows[0]["ber"]) == 0.0
