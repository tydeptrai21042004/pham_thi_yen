import json
import os
import subprocess
import sys

import numpy as np

from watermarklab.common.iwt import iwt2, iiwt2
from watermarklab.methods import build_methods
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD


def test_iwt_roundtrip_exact_on_integer_block():
    rng = np.random.default_rng(123)
    block = rng.integers(0, 256, size=(8, 8)).astype(np.float64)
    ll, lh, hl, hh = iwt2(block)
    rec = iiwt2(ll, lh, hl, hh)
    assert np.array_equal(rec.astype(np.int64), block.astype(np.int64))


def test_zhu2021_adapted_clean_roundtrip(tmp_path):
    """Run the 4096-block SVD roundtrip in a subprocess to avoid SVD-state leakage."""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    report = tmp_path / "zhu2021_roundtrip.json"
    code = f"""
import json
from pathlib import Path
import numpy as np
from watermarklab.common.metrics import ber, psnr
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD, Zhu2021IWTSVDKey
rng = np.random.default_rng(2021)
host = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
watermark = (rng.integers(0, 2, size=(64, 64), dtype=np.uint8) * 255).astype(np.uint8)
method = Zhu2021IWTSVD(delta=16.0)
watermarked, key = method.embed(host, watermark)
extracted = method.extract(watermarked, key)
out = dict(
    key_class=isinstance(key, Zhu2021IWTSVDKey),
    watermarked_shape=list(watermarked.shape),
    extracted_shape=list(extracted.shape),
    ber=float(ber(watermark, extracted)),
    psnr=float(psnr(host, watermarked)),
)
Path({str(report)!r}).write_text(json.dumps(out), encoding='utf-8')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["key_class"] is True
    assert data["watermarked_shape"] == [512, 512, 3]
    assert data["extracted_shape"] == [64, 64]
    assert data["ber"] == 0.0
    assert data["psnr"] > 35.0


def test_zhu2021_registered_and_baselines_proposal_expansion():
    methods = build_methods(["zhu2021_iwt_svd_adapted"])
    assert list(methods.keys()) == ["zhu2021_iwt_svd_adapted"]
    assert isinstance(methods["zhu2021_iwt_svd_adapted"], Zhu2021IWTSVD)

    expanded = build_methods(["baselines", "proposal"])
    assert "zhu2021_iwt_svd_adapted" in expanded
    assert "proposal" in expanded
