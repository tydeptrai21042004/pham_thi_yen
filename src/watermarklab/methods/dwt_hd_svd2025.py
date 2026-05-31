from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2
from watermarklab.common.linalg_utils import hess_decompose, hess_reconstruct, diag_from_s
from watermarklab.common.embedding_math import (
    alpha_blend_payload_weight,
    extract_from_alpha_blend_payload_weight,
)


# =========================================================
# DWT-HD-SVD 2025 Logistic chaotic mapping
# =========================================================

PAPER_REPORTED_LOGISTIC_X0 = 0.5
PAPER_LOGISTIC_X0 = 0.517
PAPER_LOGISTIC_MU = 4.0
PAPER_BETA = 0.95
WM_THRESHOLD = 127

# Paper Eq. (8): lambda = 0.04.
PAPER_ALPHA_LAMBDA = 0.04


def _paper_logistic_sequence(
    n: int,
    x0: float = PAPER_LOGISTIC_X0,
    mu: float = PAPER_LOGISTIC_MU,
) -> np.ndarray:
    """Generate a deterministic Logistic key stream."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    x = float(x0)
    mu = float(mu)

    if not (0.0 < x < 1.0):
        raise ValueError("Logistic x0 must be in (0, 1)")
    if not (3.5699456 < mu <= 4.0):
        raise ValueError("Logistic mu should be in the chaotic range (3.5699456, 4]")

    # The paper reports x0 = 0.5, but x0 = 0.5 and mu = 4 degenerates:
    # 0.5 -> 1 -> 0 -> 0 ...
    # Keep the existing corrected-key behavior.
    if x in (0.5,):
        x = PAPER_LOGISTIC_X0

    seq = np.empty(n, dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for i in range(n):
        x = mu * x * (1.0 - x)

        if x <= 0.0:
            x = eps
        elif x >= 1.0:
            x = 1.0 - eps

        seq[i] = x

    return seq


def _paper_logistic_mask(
    shape: tuple[int, ...],
    x0: float = PAPER_LOGISTIC_X0,
    mu: float = PAPER_LOGISTIC_MU,
) -> np.ndarray:
    seq = _paper_logistic_sequence(int(np.prod(shape)), x0=x0, mu=mu)

    mask = np.floor(seq * 256.0).astype(np.uint16)
    mask = np.clip(mask, 0, 255).astype(np.uint8)

    return mask.reshape(shape)


def _paper_logistic_xor_encrypt_uint8(
    mat: np.ndarray,
    x0: float = PAPER_LOGISTIC_X0,
    mu: float = PAPER_LOGISTIC_MU,
):
    data = np.clip(np.rint(mat), 0, 255).astype(np.uint8)
    mask = _paper_logistic_mask(data.shape, x0=x0, mu=mu)

    encrypted = np.bitwise_xor(data, mask).astype(np.uint8)

    return encrypted.astype(np.float64), mask


def _paper_logistic_xor_decrypt_uint8(encrypted: np.ndarray, mask: np.ndarray):
    data = np.clip(np.rint(encrypted), 0, 255).astype(np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)

    if data.shape != mask.shape:
        raise ValueError(
            f"Encrypted watermark shape {data.shape} and mask shape {mask.shape} do not match"
        )

    decrypted = np.bitwise_xor(data, mask).astype(np.uint8)

    return decrypted.astype(np.float64)


def _singular_value_beta_correction(
    singular_values: np.ndarray,
    beta: float = PAPER_BETA,
) -> np.ndarray:
    """Apply the paper's singular-value correction S^beta, beta=0.95."""
    s = np.asarray(singular_values, dtype=np.float64)
    return np.power(np.maximum(s, 0.0), float(beta))


# =========================================================
# DWT-HD-SVD 2025 adaptive entropy alpha
# =========================================================

def _pixel_probabilities_u8(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Return non-zero gray-level probabilities p_i.

    The paper computes Ev and Ee from pixel-value probabilities p_i.
    Since the watermark is embedded in the Y component, this function should
    be applied to the Y channel of the cover image.
    """
    arr = np.clip(np.rint(image), 0, bins - 1).astype(np.uint8)
    counts = np.bincount(arr.ravel(), minlength=bins).astype(np.float64)

    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("Cannot compute entropy from an empty image")

    p = counts / total
    return p[p > 0.0]


def compute_visual_entropy(image: np.ndarray) -> float:
    """Paper Eq. (9): Ev = -sum_i p_i log(p_i).

    The paper does not state the logarithm base clearly. For 8-bit image
    entropy, log2 is the standard Shannon entropy convention and gives alpha
    values in the same range as the paper's Table 1.
    """
    p = _pixel_probabilities_u8(image)
    return float(-np.sum(p * np.log2(p)))


def compute_edge_entropy(image: np.ndarray) -> float:
    """Paper Eq. (10): Ee = sum_i p_i * e^(1 - p_i).

    The printed equation depends on gray-level probabilities p_i, not on an
    explicit edge detector. This follows the printed equation directly.
    """
    p = _pixel_probabilities_u8(image)
    return float(np.sum(p * np.exp(1.0 - p)))


def compute_adaptive_alpha(
    cover_y: np.ndarray,
    lambda_value: float = PAPER_ALPHA_LAMBDA,
) -> tuple[float, float, float]:
    """Compute DWT-HD-SVD 2025 adaptive embedding factor.

    Paper Eq. (8):

        alpha = lambda / (1 + exp(-(Ee / Ev)))

    Returns:
        alpha, visual_entropy, edge_entropy
    """
    ev = compute_visual_entropy(cover_y)
    ee = compute_edge_entropy(cover_y)

    if ev <= 0.0:
        ratio = 0.0
    else:
        ratio = ee / ev

    alpha = float(lambda_value) / (1.0 + np.exp(-ratio))
    return float(alpha), float(ev), float(ee)


@dataclass
class DWTHDSVDKey:
    alpha: float
    beta: float
    uw: np.ndarray
    vtw: np.ndarray
    sh_original: np.ndarray
    sh_marked_reference: np.ndarray
    chaotic_mask: np.ndarray
    ll_roi_shape: tuple[int, int]
    wm_shape_out: tuple[int, int]
    logistic_x0: float = PAPER_LOGISTIC_X0
    logistic_mu: float = PAPER_LOGISTIC_MU

    # Added for adaptive-alpha tracking.
    alpha_mode: str = "adaptive_entropy"
    lambda_value: float = PAPER_ALPHA_LAMBDA
    visual_entropy: float | None = None
    edge_entropy: float | None = None


class DWTHDSVD2025:
    name = "DWT_HD_SVD_2025"

    def __init__(
        self,
        alpha: float = 0.045,
        mode: str = "adapt",
        beta: float = PAPER_BETA,
        logistic_x0: float = PAPER_LOGISTIC_X0,
        logistic_mu: float = PAPER_LOGISTIC_MU,
        beta_correction_mode: str = "auto",
        beta_auto_threshold: float = 1e-3,
        alpha_mode: str = "adaptive_entropy",
        lambda_value: float = PAPER_ALPHA_LAMBDA,
    ):
        # alpha_mode:
        #   "adaptive_entropy" = paper Eq. (8)
        #   "fixed"            = old fixed-alpha behavior for ablation only
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.beta = float(beta)
        self.logistic_x0 = float(logistic_x0)
        self.logistic_mu = float(logistic_mu)

        self.alpha_mode = str(alpha_mode).lower().strip()
        if self.alpha_mode not in {
            "adaptive",
            "adaptive_entropy",
            "paper",
            "fixed",
            "constant",
        }:
            raise ValueError(
                "alpha_mode must be one of: adaptive_entropy, adaptive, paper, fixed, constant"
            )

        self.lambda_value = float(lambda_value)
        if self.lambda_value <= 0.0:
            raise ValueError("lambda_value must be positive")

        self.beta_correction_mode = str(beta_correction_mode).lower().strip()
        if self.beta_correction_mode not in {"auto", "always", "off", "none", "false"}:
            raise ValueError("beta_correction_mode must be one of: auto, always, off")

        self.beta_auto_threshold = float(beta_auto_threshold)

    def _expected_watermark_shape(self) -> tuple[int, int]:
        return (256, 256) if self.mode == "original-rerun" else (64, 64)

    def _select_alpha(self, cover_y: np.ndarray) -> tuple[float, float | None, float | None]:
        """Return alpha used in Eq. (17), plus Ev/Ee if adaptive."""
        if self.alpha_mode in {"adaptive", "adaptive_entropy", "paper"}:
            alpha, ev, ee = compute_adaptive_alpha(
                cover_y,
                lambda_value=self.lambda_value,
            )
            return alpha, ev, ee

        return float(self.alpha), None, None

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        y, cb, cr = rgb_to_ycbcr(host_rgb)
        ll, lh, hl, hh = dwt2(y, mode="orthonormal")

        wm = np.asarray(watermark_binary, dtype=np.uint8)
        expected_shape = self._expected_watermark_shape()

        if wm.shape != expected_shape:
            raise ValueError(
                f"DWT-HD-SVD {self.mode} mode expects a "
                f"{expected_shape[0]}x{expected_shape[1]} binary watermark, got {wm.shape}"
            )

        # Adapted mode: 64x64 top-left LL ROI.
        # original-rerun mode: 256x256, which equals full LL for 512x512 host.
        roi = ll[: expected_shape[0], : expected_shape[1]].copy()

        # Corrected part:
        # Paper Eq. (8), alpha is now image-dependent.
        alpha_used, ev, ee = self._select_alpha(y)

        wm_enc, mask = _paper_logistic_xor_encrypt_uint8(
            wm,
            x0=self.logistic_x0,
            mu=self.logistic_mu,
        )

        uw, sw, vtw = np.linalg.svd(wm_enc, full_matrices=True)

        q, h = hess_decompose(roi)
        uh, sh, vth = np.linalg.svd(h, full_matrices=True)

        # Paper Eq. (17):
        # S'_H = alpha*S_w + (1-alpha)*S_H
        sh_prime = alpha_blend_payload_weight(sh, sw, alpha_used)

        h_prime = uh @ diag_from_s(sh_prime, h.shape) @ vth
        roi_prime = hess_reconstruct(q, h_prime)

        ll2 = ll.copy()
        ll2[: expected_shape[0], : expected_shape[1]] = roi_prime

        y2 = idwt2(ll2, lh, hl, hh, mode="orthonormal")
        watermarked = ycbcr_to_rgb(y2, cb, cr)

        key = DWTHDSVDKey(
            alpha=alpha_used,
            beta=self.beta,
            uw=uw,
            vtw=vtw,
            sh_original=sh,
            sh_marked_reference=sh_prime,
            chaotic_mask=mask,
            ll_roi_shape=roi.shape,
            wm_shape_out=wm.shape,
            logistic_x0=self.logistic_x0,
            logistic_mu=self.logistic_mu,
            alpha_mode=self.alpha_mode,
            lambda_value=self.lambda_value,
            visual_entropy=ev,
            edge_entropy=ee,
        )

        return watermarked, key

    def extract(
        self,
        possibly_attacked_rgb: np.ndarray,
        key: DWTHDSVDKey,
        host_rgb: np.ndarray | None = None,
    ):
        y, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        ll, _, _, _ = dwt2(y, mode="orthonormal")

        roi = ll[: key.ll_roi_shape[0], : key.ll_roi_shape[1]]

        _, h_wm = hess_decompose(roi)
        _, swm, _ = np.linalg.svd(h_wm, full_matrices=True)

        mode = self.beta_correction_mode
        apply_beta = False

        if mode == "always":
            apply_beta = True

        elif mode == "auto":
            ref = np.asarray(getattr(key, "sh_marked_reference", []), dtype=np.float64)

            if ref.shape == swm.shape and ref.size > 0:
                denom = max(float(np.linalg.norm(ref)), 1e-12)
                drift = float(np.linalg.norm(swm - ref) / denom)
                apply_beta = drift > float(self.beta_auto_threshold)
            else:
                apply_beta = True

        if apply_beta:
            beta = float(getattr(key, "beta", self.beta))
            swm_for_extract = _singular_value_beta_correction(swm, beta=beta)
        else:
            swm_for_extract = swm

        # Paper Eq. (25):
        # S'_W = (S_WM - (1-alpha)*S_H) / alpha
        sw_ext = extract_from_alpha_blend_payload_weight(
            swm_for_extract,
            key.sh_original,
            key.alpha,
        )

        n = key.uw.shape[0]
        enc = key.uw @ diag_from_s(sw_ext[:n], (n, n)) @ key.vtw

        dec = _paper_logistic_xor_decrypt_uint8(
            np.clip(np.rint(enc), 0, 255).astype(np.uint8),
            key.chaotic_mask,
        )

        return np.where(np.clip(dec, 0, 255) >= WM_THRESHOLD, 255, 0).astype(np.uint8)
