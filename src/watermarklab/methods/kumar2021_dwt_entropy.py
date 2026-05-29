from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2
from watermarklab.common.embedding_math import alpha_blend_cover_weight, extract_from_alpha_blend_cover_weight
from watermarklab.common.entropy import visual_entropy, select_max_entropy_score_block


@dataclass
class Kumar2021Key:
    """Side information for the non-blind Kumar & Singh 2021 baseline.

    The paper embeds in the Y channel of YCbCr after DWT and extracts with the
    cover subband.  The selected block index and alpha are therefore stored.
    """

    alpha: float
    block_index: tuple[int, int]
    block_size: int
    dwt_mode: str
    watermark_shape: tuple[int, int]


class Kumar2021DWTEntropy:
    """Paper-guided Kumar & Singh 2021 DWT + entropy + alpha blending baseline.

    Baseline path used here:
        RGB -> YCbCr -> Y channel -> one-level Haar DWT -> HH subband
        -> select the highest-entropy 64x64 block -> alpha blending with a
        64x64 binary watermark -> inverse DWT -> RGB reconstruction.

    Notes:
        * The paper reports a 512x512 color cover image and 64x64 gray
          watermark.  This implementation keeps that contract.
        * Extraction is non-blind because the alpha-blending equation uses the
          original cover HH block.
    """

    name = "Kumar2021_DWT_Entropy"

    def __init__(self, alpha: float = 0.97, block_size: int = 64, dwt_mode: str = "orthonormal", mode: str = "adapt"):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.dwt_mode = str(dwt_mode)
        self.mode = str(mode)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        wm = np.asarray(watermark_binary, dtype=np.uint8)
        if wm.shape != (64, 64):
            raise ValueError(f"Kumar2021 expects a 64x64 watermark, got {wm.shape}")

        y, cb, cr = rgb_to_ycbcr(host_rgb)
        ll, lh, hl, hh = dwt2(y, mode=self.dwt_mode)

        (bi, bj), _score = select_max_entropy_score_block(hh, block_size=self.block_size)
        r = bi * self.block_size
        c = bj * self.block_size
        cover_block = hh[r : r + self.block_size, c : c + self.block_size]
        if cover_block.shape != wm.shape:
            raise ValueError(f"Selected cover block {cover_block.shape} does not match watermark {wm.shape}")

        hh_marked = hh.copy()
        hh_marked[r : r + self.block_size, c : c + self.block_size] = alpha_blend_cover_weight(
            cover_block, wm.astype(np.float64), self.alpha
        )
        y_marked = idwt2(ll, lh, hl, hh_marked, mode=self.dwt_mode)
        watermarked = ycbcr_to_rgb(y_marked, cb, cr)
        key = Kumar2021Key(
            alpha=self.alpha,
            block_index=(int(bi), int(bj)),
            block_size=self.block_size,
            dwt_mode=self.dwt_mode,
            watermark_shape=wm.shape,
        )
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Kumar2021Key, host_rgb: np.ndarray | None = None):
        if host_rgb is None:
            raise ValueError("Kumar2021 extraction is non-blind and requires host_rgb")

        bi, bj = key.block_index
        r = bi * key.block_size
        c = bj * key.block_size

        y_wm, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        y_host, _, _ = rgb_to_ycbcr(host_rgb)
        _, _, _, hh_wm = dwt2(y_wm, mode=key.dwt_mode)
        _, _, _, hh_host = dwt2(y_host, mode=key.dwt_mode)

        rec = extract_from_alpha_blend_cover_weight(
            hh_wm[r : r + key.block_size, c : c + key.block_size],
            hh_host[r : r + key.block_size, c : c + key.block_size],
            key.alpha,
        )
        return np.where(np.clip(rec, 0, 255) >= 127, 255, 0).astype(np.uint8)


def shannon_entropy(block: np.ndarray, bins: int = 256) -> float:
    return visual_entropy(block, bins=bins)


def select_max_entropy_block(hh: np.ndarray, block_size: int = 64):
    return select_max_entropy_score_block(hh, block_size=block_size)


__all__ = ["Kumar2021DWTEntropy", "Kumar2021Key", "shannon_entropy", "select_max_entropy_block"]
