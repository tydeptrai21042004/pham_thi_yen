"""Contract/regression tests for two paper-guided blind baselines.

Includes end-to-end genuine 4096-bit embed/extract, attack, wrong-key,
registry/CLI, native-size constraints, and numerical transform invariants.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from watermarklab.common.attack import AttackConfig, apply_attack
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import psnr, ssim, ber, nc
from watermarklab.methods import BASELINE_METHOD_IDS, build_methods
from watermarklab.methods.paper_hessenberg_common import arnold, md5_permutation
from watermarklab.methods.lwt_dct_hess_abodena2022 import (
    Abodena2022LWTDCTHessenberg, _lifting2, _inverse_lifting2, _candidate_parity,
)
from watermarklab.methods.contourlet_hess_su2016 import (
    Su2016ContourletHessenberg, lp_analysis, lp_synthesis,
    _qim_hmax, _extract_hmax,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    return (load_host_rgb(ROOT / "data/host/airplane.bmp"),
            load_watermark_binary(ROOT / "data/watermark/wm.png"))


@pytest.fixture(scope="module")
def paper_runs(example):
    host, wm = example
    runs = {}
    for mid, m in build_methods(["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"]).items():
        marked, key = m.embed(host, wm)
        decoded = m.extract(marked, key)
        runs[mid] = m, marked, key, decoded
    return runs


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_4096_bit_clean_roundtrip(example, paper_runs, mid):
    host, wm = example
    m, marked, key, decoded = paper_runs[mid]
    assert wm.shape == decoded.shape == (64, 64)
    assert np.isin(decoded, [0, 255]).all()
    assert marked.shape == host.shape and marked.dtype == np.uint8
    assert psnr(host, marked) > 40.0
    assert 0.0 < ssim(host, marked) <= 1.0
    assert ber(wm, decoded) < 0.02
    assert nc(wm, decoded) > 0.97
    assert key.watermark_shape == (64, 64)
    assert m.is_blind and not m.requires_side_information
    # Keys must not store original host pixels or a watermark copy.
    assert not any(isinstance(v, np.ndarray) for v in vars(key).values())


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_no_original_host_required(paper_runs, mid):
    m, marked, key, decoded = paper_runs[mid]
    np.testing.assert_array_equal(decoded, m.extract(marked, key, host_rgb=None))


def test_registry_and_all_selector():
    names = ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"]
    assert all(m in BASELINE_METHOD_IDS for m in names)
    assert list(build_methods(names)) == names
    assert all(m in build_methods(["all"]) for m in names)
    assert all(m in build_methods(["baselines"]) for m in names)


@pytest.mark.parametrize("shape", [(12, 14), (32, 32), (64, 64)])
def test_lifting_invertibility(shape):
    rng = np.random.default_rng(123)
    original = rng.normal(size=shape)
    bands = _lifting2(original)
    np.testing.assert_allclose(_inverse_lifting2(*bands), original, atol=1e-12)


def test_contourlet_lp_invertibility():
    rng = np.random.default_rng(345)
    original = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    low, resid = lp_analysis(original)
    assert low.shape == (24, 24)
    np.testing.assert_allclose(lp_synthesis(low, resid), original.astype(float), atol=1e-12)
    np.testing.assert_allclose(lp_analysis(lp_synthesis(low+3, resid))[0], low+3, atol=1e-12)


@pytest.mark.parametrize("rounds", [0, 1, 10, 32])
def test_arnold_roundtrip(rounds):
    rng = np.random.default_rng(2016)
    wm = rng.integers(0, 2, (64, 64), dtype=np.uint8)
    np.testing.assert_array_equal(arnold(arnold(wm, rounds), rounds, inverse=True), wm)


def test_keyed_block_selection_no_collision():
    p = md5_permutation(4096, 2016)
    assert len(p) == 4096 and len(set(p.tolist())) == 4096
    np.testing.assert_array_equal(p, md5_permutation(4096, 2016))
    assert not np.array_equal(p, md5_permutation(4096, 2017))


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("value", [-50.1, -17.3, 0.0, 4.7, 18.0, 33.2])
def test_abodena_parity_consistency(value, bit):
    coeff = _candidate_parity(value, bit, 16.0)
    assert int(np.ceil(coeff/16)) % 2 == bit


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("value", [-42.2, -0.5, 0, 16.2, 80.6])
def test_su_qim_parity(value, bit):
    assert _extract_hmax(_qim_hmax(value, bit, 16), 16) == bit


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_64x64_watermark_input_rejected(example, mid):
    host, wm = example
    with pytest.raises(ValueError, match="64x64"):
        build_methods([mid])[mid].embed(host, wm[:32, :32])


def test_2022_native_mode_does_not_silently_crop_or_resize(example):
    host, wm = example
    with pytest.raises(ValueError, match="1024x1024"):
        Abodena2022LWTDCTHessenberg(mode="original-rerun").embed(host, wm)


def test_2016_nonstandard_size_is_explicit_error(example):
    host, wm = example
    with pytest.raises(ValueError, match="512x512"):
        Su2016ContourletHessenberg().embed(host[:256, :256], wm)


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_wrong_key_is_not_equivalent(paper_runs, mid):
    m, marked, key, decoded = paper_runs[mid]
    wrong = replace(key, seed=key.seed+1)
    wrong_bits = m.extract(marked, wrong)
    assert np.mean(wrong_bits != decoded) > 0.05


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_no_attack_and_jpeg_attack_outputs(paper_runs, example, mid):
    host, wm = example
    method, marked, key, decoded = paper_runs[mid]
    jpeg = apply_attack(marked, AttackConfig("jpeg_qf80", "jpeg", {"quality": 80}))
    attacked_decoded = method.extract(jpeg, key)
    assert jpeg.shape == host.shape
    assert attacked_decoded.shape == wm.shape
    assert np.isfinite(psnr(host, jpeg))
    assert np.isfinite(ber(wm, attacked_decoded))
    assert np.isfinite(nc(wm, attacked_decoded))


def test_both_baselines_cli_generated_exports(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OPENBLAS_NUM_THREADS": "1"}
    output = tmp_path / "new_baselines"
    p = subprocess.run([
        sys.executable, str(ROOT / "main.py"),
        "--methods", "abodena2022_lwt_dct_hess,su2016_contourlet_hess",
        "--attack-preset", "none", "--max-images", "1",
        "--no-save-images", "--output", str(output),
        "--guo-param-mode", "ignore", "--gaata-param-mode", "ignore",
        "--proposal-param-mode", "ignore",
    ], cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    import pandas as pd
    frame = pd.read_csv(output / "per_image_attack_results.csv")
    assert set(frame.method_id) == {"abodena2022_lwt_dct_hess", "su2016_contourlet_hess"}
    assert len(frame) == 2
    # The first alphabetically selected host is Girl.bmp; paper-guided Su2016
    # may have nonzero quantization error there. Test reporting, not invented BER.
    assert frame.ber.between(0.0, 1.0).all()
    assert (frame.psnr > 40).all()
    assert (output / "compare_psnr_nc_ber_ncc_before_after_attack.csv").exists()
    assert (output / "summary_by_method_phase.csv").exists()
    assert json.loads((output / "failures.json").read_text()) == []


@pytest.mark.parametrize("mid", ["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
def test_second_real_watermark_retains_all_4096_bits(example, mid):
    host, _ = example
    wm = load_watermark_binary(ROOT / "data/watermark/Cc.logo.circle.png")
    assert wm.size == 4096
    method = build_methods([mid])[mid]
    marked, key = method.embed(host, wm)
    extracted = method.extract(marked, key)
    assert extracted.shape == wm.shape
    assert psnr(host, marked) > 40.0
    assert ber(wm, extracted) < 0.02


def test_2022_two_levels_native_size_roundtrip(example):
    host, wm = example
    native = np.repeat(np.repeat(host, 2, axis=0), 2, axis=1)
    method = Abodena2022LWTDCTHessenberg(mode="original-rerun")
    marked, key = method.embed(native, wm)
    assert key.levels == 2
    assert marked.shape == (1024, 1024, 3)
    assert ber(wm, method.extract(marked, key)) < 0.02


def test_original_reported_paper_results_separated():
    from watermarklab.paper_reported import paper_reported_rows
    rows = paper_reported_rows(["abodena2022_lwt_dct_hess", "su2016_contourlet_hess"])
    assert set(r["method_id"] for r in rows) == {
        "abodena2022_lwt_dct_hess", "su2016_contourlet_hess"
    }
    assert all(r["mode"] == "original_reported" for r in rows)
    assert any(r["method_id"] == "abodena2022_lwt_dct_hess" and r["value"] == 44.880 for r in rows)
    assert any(r["method_id"] == "su2016_contourlet_hess" and r["value"] == 48.3033 for r in rows)
