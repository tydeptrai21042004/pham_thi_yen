"""Abodena (2022), LWT-DCT-Hessenberg, paper-GUIDED local reproduction.

Paper DOI 10.51984/JOPAS.V21I2.1934.  The paper's grayscale 1024x1024
carrier uses two LWT levels to fit 64x64 bits in 4x4 DCT blocks.  The 512x512
RGB benchmark mode uses Y luminance and ONE LWT level to keep all 4096 bits
without changing/resizing the original watermark or reducing its payload.

Underspecified in paper: lifting prediction/update, histogram binning for
entropy, and exact pseudorandom key generation. We use reversible Haar lifting,
a deterministic entropy statistic, and MD5-based ordering. In the paper's
printed Eqs. (6)-(12), the two 'nearest' candidates can have *opposite bit
parities*; adapted mode uses the candidate consistent with the desired parity.
Those differences are documented; DO NOT describe this as exact paper code.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.fft import dctn, idctn
from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.methods.paper_hessenberg_common import (
    validate_rgb_watermark, md5_permutation, hess,
)


def _lifting2(a: np.ndarray):
    """Separable predict/update Haar lifting; return approximation, H, V, D."""
    a = np.asarray(a, dtype=np.float64)
    even, odd = a[:, ::2], a[:, 1::2]
    d = odd - even
    s = even + 0.5 * d
    even, odd = s[::2], s[1::2]
    dv = odd - even
    av = even + 0.5 * dv
    even, odd = d[::2], d[1::2]
    dd = odd - even
    ah = even + 0.5 * dd
    return av, ah, dv, dd


def _inverse_lifting2(av, ah, dv, dd):
    def undo(a, d, axis):
        even = a - 0.5 * d
        odd = d + even
        shape = list(a.shape)
        shape[axis] *= 2
        out = np.empty(shape, dtype=np.float64)
        idx0 = [slice(None)] * out.ndim
        idx1 = idx0.copy()
        idx0[axis] = slice(0, None, 2)
        idx1[axis] = slice(1, None, 2)
        out[tuple(idx0)] = even
        out[tuple(idx1)] = odd
        return out
    s = undo(av, dv, 0)
    d = undo(ah, dd, 0)
    return undo(s, d, 1)


def _get_diag(y, levels):
    a = y
    paths = []
    for _ in range(levels):
        a, h, v, d = _lifting2(a)
        paths.append((h, v, d))
    return a, paths


def _put_diag(a, paths):
    for h, v, d in reversed(paths):
        a = _inverse_lifting2(a, h, v, d)
    return a


def _entropy_quantizer(coeff, block=4):
    """Average 4x4 Shannon entropy of uniformly quantized DCT coefficients.

    Histogram details are absent from the paper; here bins are fixed to integer
    coefficient values so embed/extract can use a stable published parameter.
    """
    vals = np.rint(np.asarray(coeff)).astype(np.int64)
    rows, cols = vals.shape
    entropies = []
    for r in range(0, rows, block):
        for c in range(0, cols, block):
            _, counts = np.unique(vals[r:r+block, c:c+block], return_counts=True)
            p = counts / counts.sum()
            entropies.append(-float(np.sum(p * np.log2(p))))
    return float(np.mean(entropies))


def _candidate_parity(h11, bit, alpha):
    """Smallest-change shift to the parity interval used in Eq. (12)."""
    k0 = int(np.ceil(h11 / alpha))
    ks = np.arange(k0-3, k0+4, dtype=np.int64)
    # Choose interior of each bucket: ceil(value/alpha) % 2 is bit.
    candidates = (ks - 0.5) * alpha
    valid = (ks % 2) == int(bit)
    return float(candidates[valid][np.argmin(np.abs(candidates[valid] - h11))])


@dataclass(frozen=True)
class Abodena2022Key:
    watermark_shape: tuple[int, int]
    levels: int
    alpha: float
    seed: int
    channel: str = "Y"
    block_size: int = 4
    mode: str = "adapt"
    is_blind: bool = True


class Abodena2022LWTDCTHessenberg:
    name = "Abodena2022_LWT_DCT_Hessenberg_4096_adapt"
    is_blind = True
    requires_side_information = False

    def __init__(self, mode="adapt", seed=1208, quantization_multiplier=9.0):
        self.mode = str(mode).strip().lower()
        if self.mode not in {"adapt", "original-rerun", "paper"}:
            raise ValueError("Use adapt or original-rerun (1024 grayscale native size)")
        self.seed = int(seed)
        self.quantization_multiplier = float(quantization_multiplier)
        if not np.isfinite(self.quantization_multiplier) or self.quantization_multiplier <= 0:
            raise ValueError("quantization_multiplier must be positive and finite")

    def _levels(self, n):
        if self.mode == "adapt":
            if n != 512:
                raise ValueError("Adapted Abodena2022 mode requires 512x512 host")
            return 1
        if n != 1024:
            raise ValueError("Paper-native mode requires 1024x1024 grayscale-derived RGB host")
        return 2

    def embed(self, host_rgb, watermark_binary):
        host, bits = validate_rgb_watermark(host_rgb, watermark_binary)
        levels = self._levels(host.shape[0])
        y, cb, cr = rgb_to_ycbcr(host)
        a, paths = _get_diag(y, levels)
        coeff = dctn(paths[-1][2], norm="ortho")
        blocks = (coeff.shape[0] // 4) * (coeff.shape[1] // 4)
        if blocks != bits.size:
            raise ValueError(f"Capacity mismatch: {blocks} 4x4 blocks vs {bits.size} bits")
        alpha = max(30.0, _entropy_quantizer(coeff) * self.quantization_multiplier) if self.mode == "adapt" else max(0.1, _entropy_quantizer(coeff) * self.quantization_multiplier)
        block_order = md5_permutation(blocks, self.seed)
        changed = coeff.copy()
        for bit_idx, block_id in enumerate(block_order):
            r, c = divmod(int(block_id), coeff.shape[1] // 4)
            sub = changed[r*4:r*4+4, c*4:c*4+4]
            hh, q = hess(sub)
            hh[0, 0] = _candidate_parity(float(hh[0, 0]), int(bits.flat[bit_idx]), alpha)
            changed[r*4:r*4+4, c*4:c*4+4] = q @ hh @ q.T
        updated_paths = list(paths)
        h, v, _ = updated_paths[-1]
        updated_paths[-1] = (h, v, idctn(changed, norm="ortho"))
        new_y = _put_diag(a, updated_paths)
        marked = ycbcr_to_rgb(new_y, cb, cr)
        return marked, Abodena2022Key(bits.shape, levels, alpha, self.seed, mode=self.mode)

    def extract(self, possibly_attacked_rgb, key: Abodena2022Key, host_rgb=None):
        host = np.asarray(possibly_attacked_rgb)
        if host.ndim != 3 or host.shape[2] != 3 or host.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB watermarked host")
        if key.levels != self._levels(host.shape[0]):
            raise ValueError("Key/host transform level mismatch")
        y, _, _ = rgb_to_ycbcr(host)
        _, paths = _get_diag(y, key.levels)
        coeff = dctn(paths[-1][2], norm="ortho")
        count = key.watermark_shape[0] * key.watermark_shape[1]
        if (coeff.shape[0] // 4) * (coeff.shape[1] // 4) != count:
            raise ValueError("Key/payload capacity mismatch")
        decoded = np.zeros(count, dtype=np.uint8)
        for bit_idx, block_id in enumerate(md5_permutation(count, key.seed)):
            r, c = divmod(int(block_id), coeff.shape[1] // 4)
            hh, _ = hess(coeff[r*4:r*4+4, c*4:c*4+4])
            decoded[bit_idx] = int(np.ceil(float(hh[0, 0]) / key.alpha)) % 2
        return (decoded.reshape(key.watermark_shape) * 255).astype(np.uint8)
