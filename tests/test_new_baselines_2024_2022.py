from pathlib import Path

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


def test_qwt_qsvd_zhang2022_blind_uses_4096_bits_on_512_color_host():
    host, wm = _sample_data()
    method = QWTQSVDZhang2022(extraction_mode="blind", delta=10.0)
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert isinstance(key, QWTQSVDZhang2022Key)
    assert method.is_blind is True
    assert method.requires_side_information is False
    assert key.selector is None
    assert key.watermark_shape == (64, 64)
    assert key.permutation.shape == (4096,)
    assert watermarked.shape == host.shape
    assert extracted.shape == wm.shape
    assert psnr(host, watermarked) > 38.0
    assert ber(wm, extracted) < 0.01
    assert nc(wm, extracted) > 0.99


def test_qwt_qsvd_zhang2022_semiblind_keeps_selector_side_information():
    host, wm = _sample_data()
    method = QWTQSVDZhang2022(extraction_mode="semi-blind", delta=10.0)
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert method.is_blind is False
    assert method.requires_side_information is True
    assert key.mode == "semi-blind"
    assert key.selector is not None
    assert key.selector.shape == (4096,)
    assert psnr(host, watermarked) > 38.0
    assert ber(wm, extracted) < 0.01


def test_new_baselines_registered_by_id():
    methods = build_methods(["dwt_wht_svd_2024", "qwt_qsvd_zhang2022_blind", "qwt_qsvd_zhang2022_semiblind"])
    assert list(methods.keys()) == ["dwt_wht_svd_2024", "qwt_qsvd_zhang2022_blind", "qwt_qsvd_zhang2022_semiblind"]
