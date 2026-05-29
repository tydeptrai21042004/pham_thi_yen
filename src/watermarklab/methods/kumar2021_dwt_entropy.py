from __future__ import annotations
from dataclasses import dataclass
from typing import Any
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
    watermark_ll: Any | None = None
    watermark_lh: Any | None = None
    watermark_hl: Any | None = None


class Kumar2021DWTEntropy:
    """Paper-guided Kumar & Singh 2021 DWT + entropy + alpha blending baseline.

    Baseline path used here, following the Kumar & Singh 2021 paper more closely:
        RGB -> YCbCr -> Y channel -> one-level Haar DWT -> cover HH subband
        -> divide cover HH into 32x32 blocks -> select the highest-entropy block
        -> one-level Haar DWT on the 64x64 watermark -> embed watermark HH
        subband, which is 32x32 -> inverse DWT -> RGB reconstruction.

    Notes:
        * The paper reports a 512x512 color cover image and 64x64 gray
          watermark.  After one-level DWT, the watermark HH payload is 32x32.
        * Extraction is non-blind because the alpha-blending equation uses the
          original cover HH block and the key stores the non-embedded watermark
          DWT subbands needed for IDWT reconstruction.
    """

    name = "Kumar2021_DWT_Entropy"

    def __init__(self, alpha: float = 0.65, block_size: int = 32, dwt_mode: str = "orthonormal", mode: str = "adapt"):
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

        # Paper-consistent watermark preprocessing:
        # 64x64 watermark -> one-level DWT -> embed only the HH payload (32x32).
        wm_f = wm.astype(np.float64)
        wm_ll, wm_lh, wm_hl, wm_hh = dwt2(wm_f, mode=self.dwt_mode)
        payload = wm_hh.astype(np.float64)
        if payload.shape != (32, 32):
            raise ValueError(f"Expected 32x32 watermark HH payload, got {payload.shape}")

        if self.block_size != payload.shape[0]:
            raise ValueError(
                f"Kumar2021 paper mode requires a {payload.shape[0]}x{payload.shape[1]} cover block, "
                f"got block_size={self.block_size}"
            )

        y, cb, cr = rgb_to_ycbcr(host_rgb)
        ll, lh, hl, hh = dwt2(y, mode=self.dwt_mode)

        # Paper-consistent cover block selection: divide cover HH into 32x32 blocks
        # and select the block with maximum entropy.
        (bi, bj), _score = select_max_entropy_score_block(hh, block_size=self.block_size)
        r = bi * self.block_size
        c = bj * self.block_size
        cover_block = hh[r : r + self.block_size, c : c + self.block_size]
        if cover_block.shape != payload.shape:
            raise ValueError(f"Selected cover block {cover_block.shape} does not match watermark HH payload {payload.shape}")

        hh_marked = hh.copy()
        hh_marked[r : r + self.block_size, c : c + self.block_size] = alpha_blend_cover_weight(
            cover_block, payload, self.alpha
        )
        y_marked = idwt2(ll, lh, hl, hh_marked, mode=self.dwt_mode)
        watermarked = ycbcr_to_rgb(y_marked, cb, cr)
        key = Kumar2021Key(
            alpha=self.alpha,
            block_index=(int(bi), int(bj)),
            block_size=self.block_size,
            dwt_mode=self.dwt_mode,
            watermark_shape=wm.shape,
            watermark_ll=wm_ll.copy(),
            watermark_lh=wm_lh.copy(),
            watermark_hl=wm_hl.copy(),
        )
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Kumar2021Key, host_rgb: np.ndarray | None = None):
        if host_rgb is None:
            raise ValueError("Kumar2021 extraction is non-blind and requires host_rgb")

        if key.watermark_ll is None or key.watermark_lh is None or key.watermark_hl is None:
            raise ValueError("Kumar2021 paper-mode extraction requires watermark LL/LH/HL side information in the key")

        bi, bj = key.block_index
        r = bi * key.block_size
        c = bj * key.block_size

        y_wm, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        y_host, _, _ = rgb_to_ycbcr(host_rgb)
        _, _, _, hh_wm = dwt2(y_wm, mode=key.dwt_mode)
        _, _, _, hh_host = dwt2(y_host, mode=key.dwt_mode)

        # Recover the embedded watermark HH payload using the paper's
        # alpha-blending inverse, then reconstruct the 64x64 watermark by IDWT.
        rec_hh = extract_from_alpha_blend_cover_weight(
            hh_wm[r : r + key.block_size, c : c + key.block_size],
            hh_host[r : r + key.block_size, c : c + key.block_size],
            key.alpha,
        )
        wm_rec = idwt2(
            np.asarray(key.watermark_ll, dtype=np.float64),
            np.asarray(key.watermark_lh, dtype=np.float64),
            np.asarray(key.watermark_hl, dtype=np.float64),
            rec_hh,
            mode=key.dwt_mode,
        )
        wm_rec = wm_rec[: key.watermark_shape[0], : key.watermark_shape[1]]
        return np.where(np.clip(wm_rec, 0, 255) >= 127, 255, 0).astype(np.uint8)


def shannon_entropy(block: np.ndarray, bins: int = 256) -> float:
    return visual_entropy(block, bins=bins)


def select_max_entropy_block(hh: np.ndarray, block_size: int = 32):
    return select_max_entropy_score_block(hh, block_size=block_size)


__all__ = ["Kumar2021DWTEntropy", "Kumar2021Key", "shannon_entropy", "select_max_entropy_block"]
