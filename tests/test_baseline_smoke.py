from pathlib import Path

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import psnr, ber
from watermarklab.methods import build_methods
from watermarklab.methods.roy2018_dwt_svd import Roy2018DWTSVD, Roy2018Key


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
        assert ber(wm, extracted) < (0.10 if method_id == "gaata2022_dwt_hess_fwa" else 0.03 if method_id == "roy2018_dwt_svd" else 0.02), method_id


def test_roy2018_matches_paper_structure_and_clean_result():
    root = Path(__file__).resolve().parents[1]
    host = load_host_rgb(root / "data/host/airplane.bmp")
    wm = load_watermark_binary(root / "data/watermark/wm.png")
    method = Roy2018DWTSVD(alpha=0.02, dwt_mode="average")

    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert isinstance(key, Roy2018Key)
    assert method.is_blind is False
    assert method.requires_side_information is True
    assert key.alpha == 0.02
    assert key.y_block_shape == (32, 32)
    assert key.watermark_block_shape == (4, 4)
    assert key.dwt_levels == 3

    # 512x512 host -> 16x16 = 256 Y blocks.
    # 64x64 watermark -> 16x16 = 256 watermark blocks of size 4x4.
    assert len(key.original_s_mats) == 256
    assert len(key.uw_mats) == 256
    assert len(key.vtw_mats) == 256
    assert key.original_s_mats[0].shape == (4, 4)
    assert key.uw_mats[0].shape == (4, 4)
    assert key.vtw_mats[0].shape == (4, 4)

    assert watermarked.shape == host.shape
    assert extracted.shape == wm.shape
    assert psnr(host, watermarked) > 38.0
    # Roy paper reports small no-attack BER around 0.006--0.008. The local Haar implementation should
    # recover the watermark cleanly under no attack.
    assert ber(wm, extracted) < 0.02
