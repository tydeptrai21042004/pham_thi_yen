from pathlib import Path
import json
import os
import subprocess
import sys

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import ber, nc, psnr
from watermarklab.methods import build_methods
from watermarklab.methods.dwt_wht_svd2024 import DWTWHTSVD2024, DWTWHTSVD2024Key
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022, QWTQSVDZhang2022Key


def _sample_data():
    root = Path(__file__).resolve().parents[1]
    host = load_host_rgb(root / "data/host/airplane.bmp")
    wm = load_watermark_binary(root / "data/watermark/wm.png")
    return host, wm


def test_dwt_wht_svd2024_matches_requested_payload_and_semiblind_contract():
    host, wm = _sample_data()
    method = DWTWHTSVD2024(lambda_value=0.03)
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert isinstance(key, DWTWHTSVD2024Key)
    assert method.is_blind is False
    assert method.requires_side_information is True
    assert key.hpc.shape == (64, 64)
    assert key.uw.shape == (64, 64)
    assert key.vwt.shape == (64, 64)
    assert key.watermark_shape == (64, 64)
    assert watermarked.shape == host.shape
    assert extracted.shape == wm.shape
    assert psnr(host, watermarked) > 35.0
    assert ber(wm, extracted) < 0.01
    assert nc(wm, extracted) > 0.99


def _qwt_subprocess_report(tmp_path, extraction_mode: str, attack: bool = False):
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / f"qwt_{extraction_mode}_{'attack' if attack else 'clean'}.json"
    code = f"""
import json
from pathlib import Path
import numpy as np
from watermarklab.common.attack import median_filter
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import ber, nc, psnr
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022, QWTQSVDZhang2022Key
root = Path({str(root)!r})
host = load_host_rgb(root / 'data/host/airplane.bmp')
wm = load_watermark_binary(root / 'data/watermark/wm.png')
method = QWTQSVDZhang2022(extraction_mode={extraction_mode!r}, delta=4.0)
watermarked, key = method.embed(host, wm)
source = median_filter(watermarked, size=3) if {attack!r} else watermarked
extracted = method.extract(source, key, host_rgb=host)
out = dict(
    key_class=isinstance(key, QWTQSVDZhang2022Key),
    is_blind=bool(method.is_blind),
    requires_side_information=bool(method.requires_side_information),
    selector_is_none=key.selector is None,
    selector_shape=None if key.selector is None else list(key.selector.shape),
    selector_values=[] if key.selector is None else sorted(np.unique(key.selector).astype(int).tolist()),
    watermark_shape=list(key.watermark_shape),
    permutation_shape=list(key.permutation.shape),
    watermarked_shape=list(watermarked.shape),
    extracted_shape=list(extracted.shape),
    psnr=float(psnr(host, watermarked)),
    ber=float(ber(wm, extracted)),
    nc=float(nc(wm, extracted)),
)
Path({str(report)!r}).write_text(json.dumps(out), encoding='utf-8')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=120)
    return json.loads(report.read_text(encoding="utf-8"))

def test_qwt_qsvd_zhang2022_blind_uses_4096_bits_on_512_color_host(tmp_path):
    data = _qwt_subprocess_report(tmp_path, "blind")
    assert data["key_class"] is True
    assert data["is_blind"] is True
    assert data["requires_side_information"] is False
    assert data["selector_is_none"] is True
    assert data["watermark_shape"] == [64, 64]
    assert data["permutation_shape"] == [4096]
    assert data["watermarked_shape"] == [512, 512, 3]
    assert data["extracted_shape"] == [64, 64]
    assert data["psnr"] > 38.0
    assert data["ber"] < 0.01
    assert data["nc"] > 0.99

def test_qwt_qsvd_zhang2022_semiblind_keeps_selector_side_information(tmp_path):
    data = _qwt_subprocess_report(tmp_path, "semi-blind")
    assert data["is_blind"] is False
    assert data["requires_side_information"] is True
    assert data["selector_is_none"] is False
    assert data["selector_shape"] == [4096]
    assert set(data["selector_values"]).issubset({0, 1, 2})
    assert data["psnr"] > 38.0
    assert data["ber"] < 0.01

def test_new_baselines_registered_by_id():
    methods = build_methods(["dwt_wht_svd_2024", "qwt_qsvd_zhang2022_blind", "qwt_qsvd_zhang2022_semiblind"])
    assert list(methods.keys()) == ["dwt_wht_svd_2024", "qwt_qsvd_zhang2022_blind", "qwt_qsvd_zhang2022_semiblind"]


def test_qwt_qsvd_semiblind_selector_changes_extraction_after_attack(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / "qwt_attack_compare.json"
    code = f"""
import json
from pathlib import Path
import numpy as np
from watermarklab.common.attack import median_filter
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import nc
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022
root = Path({str(root)!r})
host = load_host_rgb(root / 'data/host/airplane.bmp')
wm = load_watermark_binary(root / 'data/watermark/wm.png')
blind = QWTQSVDZhang2022(extraction_mode='blind', delta=4.0)
semiblind = QWTQSVDZhang2022(extraction_mode='semi-blind', delta=4.0)
watermarked_blind, key_blind = blind.embed(host, wm)
watermarked_semiblind, key_semiblind = semiblind.embed(host, wm)
attacked = median_filter(watermarked_semiblind, size=3)
extracted_blind = blind.extract(attacked, key_blind, host_rgb=host)
extracted_semiblind = semiblind.extract(attacked, key_semiblind, host_rgb=host)
out = dict(
    same_watermarked=bool(np.array_equal(watermarked_blind, watermarked_semiblind)),
    blind_selector_none=key_blind.selector is None,
    semiblind_selector_shape=list(key_semiblind.selector.shape),
    semiblind_selector_unique=int(len(set(key_semiblind.selector.tolist()))),
    extracted_equal=bool(np.array_equal(extracted_blind, extracted_semiblind)),
    blind_nc=float(nc(wm, extracted_blind)),
    semiblind_nc=float(nc(wm, extracted_semiblind)),
)
Path({str(report)!r}).write_text(json.dumps(out), encoding='utf-8')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["same_watermarked"] is True
    assert data["blind_selector_none"] is True
    assert data["semiblind_selector_shape"] == [4096]
    assert data["semiblind_selector_unique"] >= 2
    assert data["extracted_equal"] is False
    assert data["semiblind_nc"] >= data["blind_nc"] - 0.05
