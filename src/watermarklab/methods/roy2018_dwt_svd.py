from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import multilevel_dwt_ll, multilevel_idwt_ll
from watermarklab.common.linalg_utils import diag_from_s


@dataclass
class Roy2018Key:
    """Side information required by Roy and Pal 2018 extraction.

    The original paper stores Sp, UWp and VWp for each block.  These are
    needed in the extraction equations:

        MSWp = UWp * S1p * VWp^T
        ReWp = (MSWp - Sp) / alpha

    Therefore this baseline is a side-information / non-blind baseline.
    """

    alpha: float
    original_s_mats: list[np.ndarray]
    uw_mats: list[np.ndarray]
    vtw_mats: list[np.ndarray]
    host_shape: tuple[int, int]
    watermark_shape: tuple[int, int]
    y_block_shape: tuple[int, int]
    watermark_block_shape: tuple[int, int]
    dwt_levels: int
    dwt_mode: str
    threshold_output: bool


class Roy2018DWTSVD:
    """Roy and Pal 2018 DWT-SVD color image watermarking baseline.

    Paper-faithful core:
      * RGB -> YCbCr and embed in Y component.
      * Split Y into 32x32 non-overlapping blocks.
      * Apply 3-level 2D Haar DWT to each Y block, giving 4x4 LL block.
      * Split 64x64 watermark into 4x4 blocks.
      * Embed each watermark block into the singular-value matrix of one
        DWT-transformed Y block using Sp + alpha*Wbp.
      * Store Sp, UWp, VWp from the embedding phase for extraction.

    References in the paper:
      Eq. (5) SVD(Y_bdwtp) = Up * Sp * Vp^T
      Eq. (6) Sp + alpha * Wbp = SWp
      Eq. (7) SVD(SWp) = UWp * SWp * VWp^T
      Eq. (8) WTp = Up * SWp * Vp^T
      Eq. (15) MSWp = UWp * S1p * VWp^T
      Eq. (16) ReWp = (MSWp - Sp) / alpha
    """

    name = "Roy2018_DWT_SVD"
    is_blind = False
    requires_side_information = True
    side_information = "Sp,UWp,VWp"

    def __init__(
        self,
        alpha: float = 0.02,
        mode: str = "adapt",
        dwt_mode: str = "average",
        threshold_output: bool = True,
    ):
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.dwt_mode = str(dwt_mode)
        self.threshold_output = bool(threshold_output)
        self.y_block_shape = (32, 32)
        self.watermark_block_shape = (4, 4)
        self.dwt_levels = 3

        if self.alpha <= 0:
            raise ValueError("Roy2018 alpha must be positive")
        if self.dwt_mode not in {"average", "orthonormal"}:
            raise ValueError("dwt_mode must be 'average' or 'orthonormal'")

    @staticmethod
    def _split_blocks(x: np.ndarray, block_shape: tuple[int, int]) -> list[np.ndarray]:
        arr = np.asarray(x, dtype=np.float64)
        bh, bw = block_shape
        h, w = arr.shape
        if h % bh != 0 or w % bw != 0:
            raise ValueError(f"Array shape {arr.shape} is not divisible by block shape {block_shape}")
        return [arr[r:r + bh, c:c + bw].copy() for r in range(0, h, bh) for c in range(0, w, bw)]

    @staticmethod
    def _merge_blocks(blocks: list[np.ndarray], image_shape: tuple[int, int], block_shape: tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        bh, bw = block_shape
        expected = (h // bh) * (w // bw)
        if len(blocks) != expected:
            raise ValueError(f"Expected {expected} blocks for image shape {image_shape}, got {len(blocks)}")
        out = np.empty((h, w), dtype=np.float64)
        idx = 0
        for r in range(0, h, bh):
            for c in range(0, w, bw):
                block = np.asarray(blocks[idx], dtype=np.float64)
                if block.shape != (bh, bw):
                    raise ValueError(f"Block {idx} shape {block.shape} does not match {(bh, bw)}")
                out[r:r + bh, c:c + bw] = block
                idx += 1
        return out

    def _validate_shapes(self, host_y: np.ndarray, watermark: np.ndarray) -> None:
        hy, wy = host_y.shape
        wh, ww = watermark.shape
        ybh, ybw = self.y_block_shape
        wbh, wbw = self.watermark_block_shape

        if hy % ybh != 0 or wy % ybw != 0:
            raise ValueError(
                f"Roy2018 requires Y shape divisible by {self.y_block_shape}, got {host_y.shape}"
            )
        if wh % wbh != 0 or ww % wbw != 0:
            raise ValueError(
                f"Roy2018 requires watermark shape divisible by {self.watermark_block_shape}, got {watermark.shape}"
            )

        n_host_blocks = (hy // ybh) * (wy // ybw)
        n_wm_blocks = (wh // wbh) * (ww // wbw)
        if n_host_blocks != n_wm_blocks:
            raise ValueError(
                "Roy2018 requires one watermark block per Y block: "
                f"host blocks={n_host_blocks}, watermark blocks={n_wm_blocks}. "
                "For a 512x512 host this means a 64x64 watermark."
            )

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        host_rgb = np.asarray(host_rgb)
        if host_rgb.ndim != 3 or host_rgb.shape[2] != 3:
            raise ValueError(f"Roy2018 expects RGB host image with shape HxWx3, got {host_rgb.shape}")

        y, cb, cr = rgb_to_ycbcr(host_rgb)
        wm = np.asarray(watermark_binary, dtype=np.float64)
        if wm.ndim != 2:
            raise ValueError(f"Roy2018 expects 2D grayscale/binary watermark, got {wm.shape}")

        self._validate_shapes(y, wm)

        y_blocks = self._split_blocks(y, self.y_block_shape)
        wm_blocks = self._split_blocks(wm, self.watermark_block_shape)

        marked_y_blocks: list[np.ndarray] = []
        original_s_mats: list[np.ndarray] = []
        uw_mats: list[np.ndarray] = []
        vtw_mats: list[np.ndarray] = []

        for y_block, wm_block in zip(y_blocks, wm_blocks):
            # Paper Step 3: 3-level 2D DWT on each 32x32 Y block -> 4x4 LL.
            ll3, details = multilevel_dwt_ll(y_block, levels=self.dwt_levels, mode=self.dwt_mode)
            if ll3.shape != self.watermark_block_shape:
                raise RuntimeError(f"Unexpected 3-level DWT LL shape {ll3.shape}; expected {self.watermark_block_shape}")

            # Paper Eq. (5): SVD(Y_bdwtp) = Up * Sp * Vp^T.
            up, s, vtp = np.linalg.svd(ll3, full_matrices=True)
            sp = diag_from_s(s, ll3.shape)

            # Paper Eq. (6): Sp + alpha*Wbp = SWp.
            embedded_s_matrix = sp + self.alpha * wm_block

            # Paper Eq. (7): SVD(SWp) = UWp * SWp * VWp^T.
            uw, sw_singular_values, vtw = np.linalg.svd(embedded_s_matrix, full_matrices=True)
            sw_diag = diag_from_s(sw_singular_values, ll3.shape)

            # Paper Eq. (8): WTp = Up * SWp * Vp^T.
            marked_ll3 = up @ sw_diag @ vtp

            marked_y_block = multilevel_idwt_ll(marked_ll3, details, mode=self.dwt_mode)
            marked_y_blocks.append(marked_y_block)

            original_s_mats.append(sp)
            uw_mats.append(uw)
            vtw_mats.append(vtw)

        y_marked = self._merge_blocks(marked_y_blocks, y.shape, self.y_block_shape)
        watermarked_rgb = ycbcr_to_rgb(y_marked, cb, cr)

        key = Roy2018Key(
            alpha=self.alpha,
            original_s_mats=original_s_mats,
            uw_mats=uw_mats,
            vtw_mats=vtw_mats,
            host_shape=tuple(y.shape),
            watermark_shape=tuple(wm.shape),
            y_block_shape=self.y_block_shape,
            watermark_block_shape=self.watermark_block_shape,
            dwt_levels=self.dwt_levels,
            dwt_mode=self.dwt_mode,
            threshold_output=self.threshold_output,
        )
        return watermarked_rgb, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Roy2018Key, host_rgb: np.ndarray | None = None):
        attacked_rgb = np.asarray(possibly_attacked_rgb)
        if attacked_rgb.ndim != 3 or attacked_rgb.shape[2] != 3:
            raise ValueError(f"Roy2018 expects RGB image with shape HxWx3, got {attacked_rgb.shape}")

        y, _, _ = rgb_to_ycbcr(attacked_rgb)
        if tuple(y.shape) != tuple(key.host_shape):
            raise ValueError(f"Attacked image Y shape {y.shape} does not match embedded host shape {key.host_shape}")

        y_blocks = self._split_blocks(y, key.y_block_shape)
        if len(y_blocks) != len(key.original_s_mats):
            raise ValueError("Number of attacked Y blocks does not match stored Roy2018 side information")

        recovered_wm_blocks: list[np.ndarray] = []
        for idx, y_block in enumerate(y_blocks):
            # Paper extraction Step 3: 3-level 2D DWT on attacked/modified block.
            ll3, _details = multilevel_dwt_ll(y_block, levels=key.dwt_levels, mode=key.dwt_mode)

            # Paper Eq. (14): SVD(Y_mbdwtp) = U1p * S1p * V1p^T.
            _, s1, _ = np.linalg.svd(ll3, full_matrices=True)
            s1_diag = diag_from_s(s1, ll3.shape)

            # Paper Eq. (15): MSWp = UWp * S1p * VWp^T.
            mswp = key.uw_mats[idx] @ s1_diag @ key.vtw_mats[idx]

            # Paper Eq. (16): ReWp = (MSWp - Sp) / alpha.
            recovered = (mswp - key.original_s_mats[idx]) / key.alpha
            recovered_wm_blocks.append(recovered)

        recovered_wm = self._merge_blocks(
            recovered_wm_blocks,
            key.watermark_shape,
            key.watermark_block_shape,
        )
        recovered_wm = np.clip(recovered_wm, 0, 255)

        if key.threshold_output:
            return np.where(recovered_wm >= 127.0, 255, 0).astype(np.uint8)
        return np.clip(np.rint(recovered_wm), 0, 255).astype(np.uint8)
