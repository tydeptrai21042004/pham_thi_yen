from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from watermarklab.common.chaos import arnold_scramble, arnold_unscramble
from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2, DWTLevel
from watermarklab.common.entropy import visual_entropy, edge_entropy
from watermarklab.common.hadamard import wht2, iwht2
from watermarklab.common.linalg_utils import diag_from_s


@dataclass
class DWTWHTSVD2024Key:
    """Side information for Kumar/Verma/Singh/Kumar/Chandra/Barde 2024.

    The paper uses a semi-blind extraction.  Extraction needs the original host
    principal component H_PC, the U_W/V_W matrices from SVD of the embedded
    principal component, the adaptive alpha value, and the Arnold key.
    """

    alpha: float
    lambda_value: float
    hpc: np.ndarray
    uw: np.ndarray
    vwt: np.ndarray
    host_shape: tuple[int, int]
    watermark_shape: tuple[int, int]
    arnold_iterations: int
    dwt_mode: str
    threshold_output: bool


class DWTWHTSVD2024:
    """Entropy-based adaptive DWT-WHT-SVD color watermarking baseline, 2024.

    Paper-faithful setting for the requested benchmark:
      * 512x512 RGB host -> YCbCr, embed in Y.
      * Three-level DWT is applied along the HH branch: Y -> HH1 -> HH2 -> HH3.
        For a 512x512 image, HH3 is exactly 64x64.
      * Apply WHT to HH3 and SVD to the WHT coefficients.
      * Compute principal component H_PC = U * S.
      * Arnold-scramble the 64x64 watermark and embed it in H_PC.
      * Keep the paper's semi-blind side information for extraction.

    The method accepts a binary 64x64 watermark and uses it as a grayscale
    {0,255} image, preserving the same payload size as the paper's 64x64
    grayscale watermark.
    """

    name = "DWT_WHT_SVD_2024"
    is_blind = False
    requires_side_information = True
    side_information = "HPC,UW,VW,alpha,Arnold key"

    def __init__(
        self,
        lambda_value: float = 0.03,
        arnold_iterations: int = 17,
        mode: str = "adapt",
        dwt_mode: str = "average",
        threshold_output: bool = True,
    ):
        self.lambda_value = float(lambda_value)
        self.arnold_iterations = int(arnold_iterations)
        self.mode = str(mode)
        self.dwt_mode = str(dwt_mode)
        self.threshold_output = bool(threshold_output)
        if self.lambda_value <= 0:
            raise ValueError("DWT-WHT-SVD2024 lambda_value must be positive")
        if self.dwt_mode not in {"average", "orthonormal"}:
            raise ValueError("dwt_mode must be 'average' or 'orthonormal'")

    def _adaptive_alpha(self, y: np.ndarray) -> float:
        ei = max(visual_entropy(y), 1e-12)
        ee = max(edge_entropy(y), 1e-12)
        return float(self.lambda_value * (1.0 / (1.0 + np.exp(-(ee / ei)))))

    def _dwt_hh3(self, y: np.ndarray):
        ll1, lh1, hl1, hh1 = dwt2(y, mode=self.dwt_mode)
        ll2, lh2, hl2, hh2 = dwt2(hh1, mode=self.dwt_mode)
        ll3, lh3, hl3, hh3 = dwt2(hh2, mode=self.dwt_mode)
        return hh3, (DWTLevel(lh=lh1, hl=hl1, hh=hh1), DWTLevel(lh=lh2, hl=hl2, hh=hh2), DWTLevel(lh=lh3, hl=hl3, hh=hh3), ll1, ll2, ll3)

    def _idwt_hh3(self, hh3_new: np.ndarray, saved) -> np.ndarray:
        level1, level2, level3, ll1, ll2, ll3 = saved
        hh2_new = idwt2(ll3, level3.lh, level3.hl, hh3_new, mode=self.dwt_mode)
        hh1_new = idwt2(ll2, level2.lh, level2.hl, hh2_new, mode=self.dwt_mode)
        y_new = idwt2(ll1, level1.lh, level1.hl, hh1_new, mode=self.dwt_mode)
        return y_new

    def _validate_shapes(self, host_rgb: np.ndarray, watermark: np.ndarray) -> None:
        if host_rgb.ndim != 3 or host_rgb.shape[2] != 3:
            raise ValueError(f"DWT-WHT-SVD2024 expects RGB host HxWx3, got {host_rgb.shape}")
        if host_rgb.shape[0] != 512 or host_rgb.shape[1] != 512:
            raise ValueError(f"DWT-WHT-SVD2024 is configured for 512x512 hosts, got {host_rgb.shape[:2]}")
        if watermark.shape != (64, 64):
            raise ValueError(f"DWT-WHT-SVD2024 requires a 64x64 watermark, got {watermark.shape}")

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        host_rgb = np.asarray(host_rgb)
        wm = np.asarray(watermark_binary, dtype=np.float64)
        self._validate_shapes(host_rgb, wm)

        y, cb, cr = rgb_to_ycbcr(host_rgb)
        alpha = self._adaptive_alpha(y)

        hh3, saved = self._dwt_hh3(y)
        if hh3.shape != wm.shape:
            raise RuntimeError(f"HH3 shape {hh3.shape} does not match watermark shape {wm.shape}")

        coeff = wht2(hh3)
        u, s, vt = np.linalg.svd(coeff, full_matrices=True)
        s_diag = diag_from_s(s, coeff.shape)
        hpc = u @ s_diag

        wm_bits = (wm >= 127).astype(np.uint8)
        scrambled = arnold_scramble(wm_bits, iterations=self.arnold_iterations).astype(np.float64) * 255.0

        embedded_pc = hpc + alpha * scrambled
        uw, sw, vwt = np.linalg.svd(embedded_pc, full_matrices=True)
        sw_diag = diag_from_s(sw, embedded_pc.shape)

        marked_coeff = uw @ sw_diag @ vwt
        marked_hh3 = iwht2(marked_coeff)
        marked_y = self._idwt_hh3(marked_hh3, saved)
        watermarked_rgb = ycbcr_to_rgb(marked_y, cb, cr)

        key = DWTWHTSVD2024Key(
            alpha=alpha,
            lambda_value=self.lambda_value,
            hpc=hpc,
            uw=uw,
            vwt=vwt,
            host_shape=tuple(y.shape),
            watermark_shape=tuple(wm.shape),
            arnold_iterations=self.arnold_iterations,
            dwt_mode=self.dwt_mode,
            threshold_output=self.threshold_output,
        )
        return watermarked_rgb, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: DWTWHTSVD2024Key, host_rgb: np.ndarray | None = None):
        attacked_rgb = np.asarray(possibly_attacked_rgb)
        if attacked_rgb.ndim != 3 or attacked_rgb.shape[2] != 3:
            raise ValueError(f"DWT-WHT-SVD2024 expects RGB image HxWx3, got {attacked_rgb.shape}")
        y, _, _ = rgb_to_ycbcr(attacked_rgb)
        if tuple(y.shape) != tuple(key.host_shape):
            raise ValueError(f"Attacked Y shape {y.shape} does not match embedded host shape {key.host_shape}")

        hh3, _saved = self._dwt_hh3(y)
        coeff = wht2(hh3)
        _u1, s1, _vt1 = np.linalg.svd(coeff, full_matrices=True)
        s1_diag = diag_from_s(s1, coeff.shape)

        # Paper-style semi-blind extraction from stored UW/VW and original H_PC.
        hex_pc = key.uw @ s1_diag @ key.vwt
        recovered = (hex_pc - key.hpc) / max(float(key.alpha), 1e-12)
        recovered = np.clip(recovered, 0, 255)
        recovered_bits = (recovered >= 127.0).astype(np.uint8)
        descrambled = arnold_unscramble(recovered_bits, iterations=key.arnold_iterations)
        out = descrambled.astype(np.uint8) * 255
        if key.threshold_output:
            return out
        return out.astype(np.uint8)
