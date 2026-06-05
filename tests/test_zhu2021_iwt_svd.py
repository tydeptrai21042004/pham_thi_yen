import numpy as np

from watermarklab.common.iwt import iwt2, iiwt2
from watermarklab.common.metrics import ber, psnr
from watermarklab.methods import build_methods
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD, Zhu2021IWTSVDKey


def test_iwt_roundtrip_exact_on_integer_block():
    rng = np.random.default_rng(123)
    block = rng.integers(0, 256, size=(8, 8)).astype(np.float64)
    ll, lh, hl, hh = iwt2(block)
    rec = iiwt2(ll, lh, hl, hh)
    assert np.array_equal(rec.astype(np.int64), block.astype(np.int64))


def test_zhu2021_adapted_clean_roundtrip():
    rng = np.random.default_rng(2021)
    host = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    watermark = (rng.integers(0, 2, size=(64, 64), dtype=np.uint8) * 255).astype(np.uint8)
    method = Zhu2021IWTSVD(delta=16.0)
    watermarked, key = method.embed(host, watermark)
    extracted = method.extract(watermarked, key)
    assert isinstance(key, Zhu2021IWTSVDKey)
    assert watermarked.shape == host.shape
    assert extracted.shape == watermark.shape
    assert ber(watermark, extracted) == 0.0
    assert psnr(host, watermarked) > 35.0


def test_zhu2021_registered_and_baselines_proposal_expansion():
    methods = build_methods(["zhu2021_iwt_svd_adapted"])
    assert list(methods.keys()) == ["zhu2021_iwt_svd_adapted"]
    assert isinstance(methods["zhu2021_iwt_svd_adapted"], Zhu2021IWTSVD)

    expanded = build_methods(["baselines", "proposal"])
    assert "zhu2021_iwt_svd_adapted" in expanded
    assert "proposal" in expanded
