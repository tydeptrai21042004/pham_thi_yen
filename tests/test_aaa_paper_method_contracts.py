from pathlib import Path

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import ber, nc, psnr
from watermarklab.methods.dwt_wht_svd2024 import DWTWHTSVD2024, DWTWHTSVD2024Key
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022, QWTQSVDZhang2022Key


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sample_payload(name: str = "wm.png"):
    root = _repo_root()
    host = load_host_rgb(root / "data/host/airplane.bmp")
    wm = load_watermark_binary(root / f"data/watermark/{name}")
    assert host.shape == (512, 512, 3)
    assert wm.shape == (64, 64)
    assert set(np.unique(wm).tolist()).issubset({0, 255})
    return host, wm


def _checkerboard(size: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return (((xx + yy) % 2) * 255).astype(np.uint8)


@pytest.mark.parametrize("watermark_name", ["wm.png", "Cc.logo.circle.png"])
def test_dwt_wht_svd2024_native_64x64_color_contract_and_clean_extract(watermark_name):
    host, wm = _sample_payload(watermark_name)
    method = DWTWHTSVD2024(lambda_value=0.03, dwt_mode="average")

    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key)  # host is intentionally not needed after key creation

    assert isinstance(key, DWTWHTSVD2024Key)
    assert method.is_blind is False
    assert method.requires_side_information is True
    assert 0.0 < key.alpha < key.lambda_value
    assert key.host_shape == (512, 512)
    assert key.watermark_shape == (64, 64)
    assert key.hpc.shape == (64, 64)
    assert key.uw.shape == (64, 64)
    assert key.vwt.shape == (64, 64)
    assert watermarked.shape == host.shape
    assert extracted.shape == wm.shape
    assert psnr(host, watermarked) > 35.0
    assert ber(wm, extracted) < 0.01
    assert nc(wm, extracted) > 0.99


def test_dwt_wht_svd2024_rejects_non_adapted_payload_shapes():
    host, wm = _sample_payload()
    method = DWTWHTSVD2024()

    with pytest.raises(ValueError, match="64x64"):
        method.embed(host, wm[:32, :32])

    with pytest.raises(ValueError, match="512x512"):
        method.embed(host[:256, :256], wm)

    with pytest.raises(ValueError, match="RGB"):
        method.embed(host[:, :, 0], wm)


@pytest.mark.parametrize("extraction_mode", ["blind", "semi-blind"])
def test_qwt_qsvd_zhang2022_4096_bit_contract_clean_extract_without_host(extraction_mode, tmp_path):
    root = _repo_root()
    report = tmp_path / f"qwt_{extraction_mode.replace('-', '_')}_report.json"
    code = f"""
import json
from pathlib import Path
import numpy as np
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import ber, nc, psnr
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022, QWTQSVDZhang2022Key
root = Path({str(root)!r})
host = load_host_rgb(root / 'data/host/airplane.bmp')
wm = load_watermark_binary(root / 'data/watermark/wm.png')
method = QWTQSVDZhang2022(extraction_mode={extraction_mode!r}, delta=4.0)
watermarked, key = method.embed(host, wm)
extracted = method.extract(watermarked, key)
out = dict(
    key_class=isinstance(key, QWTQSVDZhang2022Key),
    is_blind=bool(method.is_blind),
    requires_side_information=bool(method.requires_side_information),
    host_shape=list(key.host_shape),
    watermark_shape=list(key.watermark_shape),
    block_size=int(key.block_size),
    permutation_len=int(key.permutation.shape[0]),
    permutation_min=int(key.permutation.min()),
    permutation_max=int(key.permutation.max()),
    selector_is_none=key.selector is None,
    selector_shape=None if key.selector is None else list(key.selector.shape),
    selector_values=[] if key.selector is None else sorted(np.unique(key.selector).astype(int).tolist()),
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
    data = json.loads(report.read_text(encoding="utf-8"))

    assert data["key_class"] is True
    assert data["host_shape"] == [512, 512]
    assert data["watermark_shape"] == [64, 64]
    assert data["block_size"] == 4
    assert data["permutation_len"] == 4096
    assert data["permutation_min"] == 0
    assert data["permutation_max"] == 4095
    if extraction_mode == "blind":
        assert data["is_blind"] is True
        assert data["requires_side_information"] is False
        assert data["selector_is_none"] is True
    else:
        assert data["is_blind"] is False
        assert data["requires_side_information"] is True
        assert data["selector_shape"] == [4096]
        assert set(data["selector_values"]).issubset({0, 1, 2})
    assert data["watermarked_shape"] == [512, 512, 3]
    assert data["extracted_shape"] == [64, 64]
    assert data["psnr"] > 38.0
    assert data["ber"] < 0.01
    assert data["nc"] > 0.99

def test_qwt_qsvd_zhang2022_rejects_wrong_shape_and_wrong_block_size():
    host, wm = _sample_payload()

    with pytest.raises(ValueError, match="4x4"):
        QWTQSVDZhang2022(block_size=8)

    method = QWTQSVDZhang2022()
    with pytest.raises(ValueError, match="64x64"):
        method.embed(host, wm[:32, :32])

    with pytest.raises(ValueError, match="512x512"):
        method.embed(host[:256, :256], wm)


def test_zhu2021_iwt_svd_adapted_64x64_y_channel_contract_and_key_only_extract(tmp_path):
    """Check the adapted Zhu method in a clean subprocess.

    This keeps the full pytest run stable after the heavier QSVD tests while
    still validating the key paper/adaptation contract and exact clean recovery.
    """
    root = _repo_root()
    report = tmp_path / "zhu_adapt_report.json"
    code = f"""
import json
from pathlib import Path
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import ber, psnr
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD, Zhu2021IWTSVDKey
root = Path({str(root)!r})
host = load_host_rgb(root / 'data/host/airplane.bmp')
wm = load_watermark_binary(root / 'data/watermark/wm.png')
method = Zhu2021IWTSVD(mode='adapt', delta=16.0)
watermarked, key = method.embed(host, wm)
extracted = method.extract(watermarked, key)
out = dict(
    key_class=isinstance(key, Zhu2021IWTSVDKey),
    color_mode=key.color_mode,
    host_shape=list(key.host_shape),
    watermark_shape=list(key.watermark_shape),
    block_size=key.block_size,
    delta=key.delta,
    blocks_total=key.stats['blocks_total'],
    iwt_ll_shape_per_block=key.stats['iwt_ll_shape_per_block'],
    adapter_watermark=key.stats['adapter_watermark'],
    failed_blocks=key.stats['failed_blocks'],
    watermarked_shape=list(watermarked.shape),
    extracted_shape=list(extracted.shape),
    ber=float(ber(wm, extracted)),
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
    assert data["color_mode"] == "ycbcr_y"
    assert data["host_shape"] == [512, 512]
    assert data["watermark_shape"] == [64, 64]
    assert data["block_size"] == 8
    assert data["delta"] == 16.0
    assert data["blocks_total"] == 4096
    assert data["iwt_ll_shape_per_block"] == [4, 4]
    assert data["adapter_watermark"] == "64x64"
    assert data["failed_blocks"] == 0
    assert data["watermarked_shape"] == [512, 512, 3]
    assert data["extracted_shape"] == [64, 64]
    assert data["ber"] == 0.0
    assert data["psnr"] > 35.0

def test_zhu2021_original_rerun_uses_native_32x32_gray_contract_by_default(tmp_path):
    root = _repo_root()
    report = tmp_path / "zhu_original_report.json"
    code = f"""
import json
from pathlib import Path
import numpy as np
from watermarklab.common.metrics import ber
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD
rng = np.random.default_rng(202104)
gray_host = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
yy, xx = np.indices((32, 32))
wm32 = (((xx + yy) % 2) * 255).astype(np.uint8)
method = Zhu2021IWTSVD(mode='original-rerun', delta=18.0)
watermarked, key = method.embed(gray_host, wm32)
extracted = method.extract(watermarked, key)
out = dict(
    color_mode=key.color_mode,
    watermark_shape=list(key.watermark_shape),
    block_size=key.block_size,
    paper_native_host=key.stats['paper_native_host'],
    paper_native_watermark=key.stats['paper_native_watermark'],
    watermarked_shape=list(watermarked.shape),
    extracted_shape=list(extracted.shape),
    ber=float(ber(wm32, extracted)),
)
Path({str(report)!r}).write_text(json.dumps(out), encoding='utf-8')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))

    assert data["color_mode"] == "gray_mean"
    assert data["watermark_shape"] == [32, 32]
    assert data["block_size"] == 16
    assert data["paper_native_host"] == "512x512 grayscale"
    assert data["paper_native_watermark"] == "32x32"
    assert data["watermarked_shape"] == [512, 512, 3]
    assert data["extracted_shape"] == [32, 32]
    assert data["ber"] == 0.0
