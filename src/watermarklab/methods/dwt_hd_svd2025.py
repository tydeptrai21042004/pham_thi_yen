from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2
from watermarklab.common.chaos import chaotic_encrypt_uint8, chaotic_decrypt_uint8
from watermarklab.common.linalg_utils import hess_decompose, hess_reconstruct, diag_from_s
from watermarklab.common.embedding_math import alpha_blend_payload_weight, extract_from_alpha_blend_payload_weight

@dataclass
class DWTHDSVDKey:
    alpha: float
    uw: np.ndarray
    vtw: np.ndarray
    sh_original: np.ndarray
    chaotic_indices: np.ndarray
    chaotic_mask: np.ndarray
    ll_roi_shape: tuple[int, int]
    wm_shape_out: tuple[int, int]


class DWTHDSVD2025:
    name = "DWT_HD_SVD_2025"

    def __init__(self, alpha: float = 0.045, mode: str = "adapt"):
        # Common-benchmark adaptation: use a 64x64 ROI in the DWT-LL band so the
        # method can accept the user's 64x64 binary watermark without expensive
        # 256x256 SVDs. The original-paper mode used a 256x256 watermark.
        self.alpha = float(alpha)
        self.mode = str(mode)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        y, cb, cr = rgb_to_ycbcr(host_rgb)
        ll, lh, hl, hh = dwt2(y, mode="orthonormal")  # 512 -> 256, one-level Haar DWT
        wm = np.asarray(watermark_binary, dtype=np.uint8)
        expected_shape = (256, 256) if self.mode == "original-rerun" else (64, 64)
        if wm.shape != expected_shape:
            raise ValueError(f"DWT-HD-SVD {self.mode} mode expects a {expected_shape[0]}x{expected_shape[1]} binary watermark, got {wm.shape}")
        roi = ll[:expected_shape[0], :expected_shape[1]].copy()
        wm_enc, idx, mask = chaotic_encrypt_uint8(wm)
        uw, sw, vtw = np.linalg.svd(wm_enc, full_matrices=True)
        q, h = hess_decompose(roi)
        uh, sh, vth = np.linalg.svd(h, full_matrices=True)
        sh_prime = alpha_blend_payload_weight(sh, sw, self.alpha)  # Eq. (17): S_H' = alpha*S_w + (1-alpha)*S_H
        h_prime = uh @ diag_from_s(sh_prime, h.shape) @ vth
        roi_prime = hess_reconstruct(q, h_prime)
        ll2 = ll.copy()
        ll2[:expected_shape[0], :expected_shape[1]] = roi_prime
        y2 = idwt2(ll2, lh, hl, hh, mode="orthonormal")
        watermarked = ycbcr_to_rgb(y2, cb, cr)
        key = DWTHDSVDKey(self.alpha, uw, vtw, sh, idx, mask, roi.shape, wm.shape)
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: DWTHDSVDKey, host_rgb: np.ndarray | None = None):
        y, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        ll, _, _, _ = dwt2(y, mode="orthonormal")
        roi = ll[:key.ll_roi_shape[0], :key.ll_roi_shape[1]]
        _, h_wm = hess_decompose(roi)
        _, swm, _ = np.linalg.svd(h_wm, full_matrices=True)
        sw_ext = extract_from_alpha_blend_payload_weight(swm, key.sh_original, key.alpha)
        n = key.uw.shape[0]
        enc = key.uw @ diag_from_s(sw_ext[:n], (n, n)) @ key.vtw
        dec = chaotic_decrypt_uint8(np.clip(np.rint(enc), 0, 255).astype(np.uint8), key.chaotic_indices, key.chaotic_mask)
        return np.where(np.clip(dec, 0, 255) >= 127, 255, 0).astype(np.uint8)
