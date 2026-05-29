from pathlib import Path

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import psnr, ber
from watermarklab.methods import build_methods


def test_four_requested_baselines_clean_smoke():
    root = Path(__file__).resolve().parents[1]
    host = load_host_rgb(root / "data/host/airplane.bmp")
    wm = load_watermark_binary(root / "data/watermark/wm.png")
    methods = build_methods(["baselines"])

    for method_id, method in methods.items():
        watermarked, key = method.embed(host, wm)
        extracted = method.extract(watermarked, key, host_rgb=host)
        assert watermarked.shape == host.shape, method_id
        assert extracted.shape == wm.shape, method_id
        assert psnr(host, watermarked) > 30.0, method_id
        # Gaata's published decimal/parity rule is quantization-sensitive, so its
        # Python reproduction has a looser clean BER threshold than the others.
        assert ber(wm, extracted) < (0.10 if method_id == "gaata2022_dwt_hess_fwa" else 0.02), method_id
