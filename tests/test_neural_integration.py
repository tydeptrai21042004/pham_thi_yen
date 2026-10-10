"""Offline contracts for neural integration (no large checkpoint downloads)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from watermarklab.experiments.neural_baselines import run_dataset, _select_methods


class DummyNeuralAdapter:
    name = "DummyPretrained"
    capacity = 24
    device = "cpu-test"

    def __init__(self):
        self.bits = None

    def embed(self, rgb, bits):
        self.bits = bits.copy()
        return np.asarray(rgb, dtype=np.uint8).copy()

    def detect(self, rgb):
        # Simulated 'detector' intentionally returns an observed bit vector
        return self.bits.copy(), None


class MissingAdapter(DummyNeuralAdapter):
    def detect(self, rgb):
        return None, False


def _images(tmp_path):
    folder = tmp_path / "hosts"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for label in ["a", "b"]:
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(folder / (label + ".png"))
    return folder


def test_two_neural_models_share_csv_schema_and_controls(tmp_path):
    hosts = _images(tmp_path)
    output = tmp_path / "out"
    rows = run_dataset(methods="all", host_dir=hosts, output=output, attack_preset="none",
                       adapter_factory=lambda n, m, d: DummyNeuralAdapter())
    assert len(rows) == 8  # two models, two hosts, clean + unmarked control
    df = pd.read_csv(output / "per_image_attack_results.csv")
    assert set(df.phase) == {"before_attack", "control"}
    assert set(df.method_id) == {"dummypretrained"}
    assert (df.query("phase == 'before_attack'")["ber"] == 0).all()
    summary = pd.read_csv(output / "summary_by_method_phase.csv")
    assert set(summary.phase) == {"before_attack"}
    assert not (output / "marked_images").exists()
    assert json.loads((output / "manifest.json").read_text())["capacity_matched_4096_bits"] is False


def test_lite_attack_outputs_rows_and_ber_fail_as_one(tmp_path):
    hosts = _images(tmp_path)
    output = tmp_path / "out"
    rows = run_dataset(methods="trustmark", host_dir=hosts, output=output,
                       attack_preset="lite", max_images=1, max_attacks=2,
                       adapter_factory=lambda n, m, d: MissingAdapter())
    assert len(rows) == 4
    df = pd.read_csv(output / "per_image_attack_results.csv")
    failed = df.query("phase != 'control'")
    assert failed.ber.isna().all()
    assert (failed.ber_if_missing_one == 1.0).all()
    assert len(df.query("phase == 'after_attack'")) == 2
    assert (output / "compare_psnr_nc_ber_ncc_before_after_attack.csv").is_file()


def test_missing_model_fails_nonzero_and_writes_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        run_dataset(methods="trustmark", host_dir=_images(tmp_path), output=tmp_path / "out",
                    attack_preset="none", adapter_factory=lambda n, m, d: 1 / 0)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    failures = json.loads((tmp_path / "out" / "failures.json").read_text())
    assert manifest["status"] == "partial_or_failed"
    assert failures[0]["stage"] == "model_load"


def test_neural_selector_does_not_change_classical_default():
    assert _select_methods("all") == ["trustmark", "chunkyseal"]
    assert _select_methods("trustmark") == ["trustmark"]
    with pytest.raises(ValueError):
        _select_methods("proposal")


def test_main_cli_has_neural_phase(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "main.py", "--help"], cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert completed.returncode == 0
    assert "neural-max-attacks" in completed.stdout
    assert "neural" in completed.stdout
