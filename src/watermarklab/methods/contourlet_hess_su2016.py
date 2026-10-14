"""Su et al. (2016), Contourlet-Hessenberg blind baseline, 4096-bit adapter.

DOI 10.1007/s11042-016-3522-z. The original method uses a Contourlet
Laplacian-pyramid (LP) lowpass, 4x4 Hessenberg blocks, MD5 keyed selection,
Arnold scrambling and QIM on the greatest-energy H element at T=16.

The paper uses 32x32 *color* logos (24,576 bits) over three 512x512 channels.
For THIS project's 64x64 *binary* logo (4096 bits), the adapted mode uses
only the B channel LP lowpass (256x256, 4096 blocks), with one bit per block.

The exact LP analysis/synthesis filters are not specified in the paper. We
use a one-level 2x2 box-filter Laplacian pyramid with EXACT-RECONSTRUCTION
residual and no directional-detail modification (the original Contourlet DFB
only affects high-frequency detail, which the published algorithm does not
embed into). This is a paper-guided reproduction, not the authors' MATLAB.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from watermarklab.methods.paper_hessenberg_common import (
    validate_rgb_watermark, arnold, md5_permutation, hess, quantize_uint8,
)


def lp_analysis(im):
    """One-level box LP. Returns lowpass, highpass residual."""
    a = np.asarray(im, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] % 2 or a.shape[1] % 2:
        raise ValueError("LP requires even-sided 2D image")
    low = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).mean(axis=(1, 3))
    high = a - np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)
    return low, high


def lp_synthesis(low, high):
    return np.repeat(np.repeat(np.asarray(low), 2, axis=0), 2, axis=1) + np.asarray(high)


def _qim_hmax(value: float, bit: int, step: float) -> float:
    """Su et al. (2016), Eq. (12): 0.25T for bit 0, 0.75T for bit 1."""
    return float(value - np.mod(value, step) + (0.25 if bit == 0 else 0.75) * step)


def _extract_hmax(value: float, step: float) -> int:
    """Su et al. (2016), Eq. (14)."""
    return int(np.mod(value, step) >= 0.5 * step)


@dataclass(frozen=True)
class Su2016Key:
    watermark_shape: tuple[int, int]
    seed: int
    arnold_rounds: int
    step: float
    channel: int = 2
    mode: str = "adapt"
    block_size: int = 4
    is_blind: bool = True


class Su2016ContourletHessenberg:
    name = "Su2016_Contourlet_Hessenberg_4096_adapt"
    is_blind = True
    requires_side_information = False

    def __init__(self, mode="adapt", seed=2016, arnold_rounds=10,
                 quantization_step=16.0, channel=2):
        self.mode = str(mode).strip().lower()
        if self.mode not in {"adapt", "original-rerun"}:
            raise ValueError("Only the 4096-bit adapted Su2016 variant is implemented")
        self.seed = int(seed)
        self.arnold_rounds = int(arnold_rounds)
        self.quantization_step = float(quantization_step)
        self.channel = int(channel)
        if self.arnold_rounds < 0 or self.channel not in range(3):
            raise ValueError("Invalid Arnold rounds or channel")
        if not np.isfinite(self.quantization_step) or self.quantization_step <= 0:
            raise ValueError("quantization_step must be positive and finite")

    def _validate_size(self, host, wm_shape):
        if host.shape[:2] != (512, 512) or wm_shape != (64, 64):
            raise ValueError("Su2016 4096-bit adapter requires 512x512 host and 64x64 watermark")

    def embed(self, host_rgb, watermark_binary):
        host, bits = validate_rgb_watermark(host_rgb, watermark_binary)
        self._validate_size(host, bits.shape)
        low, high = lp_analysis(host[..., self.channel])
        nr, nc = low.shape[0] // 4, low.shape[1] // 4
        if nr * nc != bits.size:
            raise ValueError("LP lowpass block capacity must equal 4096 bits")
        perm = md5_permutation(nr*nc, self.seed)
        scrambled = arnold(bits, self.arnold_rounds)
        changed = low.copy()
        for bit_index, block_id in enumerate(perm):
            r, c = divmod(int(block_id), nc)
            block = changed[4*r:4*r+4, 4*c:4*c+4]
            hh, q = hess(block)
            where = np.unravel_index(np.argmax(np.abs(hh)), hh.shape)
            hh[where] = _qim_hmax(float(hh[where]), int(scrambled.flat[bit_index]), self.quantization_step)
            changed[4*r:4*r+4, 4*c:4*c+4] = q @ hh @ q.T
        out = host.copy()
        out[..., self.channel] = quantize_uint8(lp_synthesis(changed, high))
        return out, Su2016Key(bits.shape, self.seed, self.arnold_rounds,
                              self.quantization_step, self.channel, self.mode)

    def extract(self, possibly_attacked_rgb, key: Su2016Key, host_rgb=None):
        host = np.asarray(possibly_attacked_rgb)
        if host.ndim != 3 or host.shape[2] != 3 or host.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB watermarked host")
        self._validate_size(host, key.watermark_shape)
        low, _ = lp_analysis(host[..., key.channel])
        count = int(np.prod(key.watermark_shape))
        perm = md5_permutation(count, key.seed)
        decoded = np.zeros(count, dtype=np.uint8)
        for bit_index, block_id in enumerate(perm):
            r, c = divmod(int(block_id), low.shape[1] // 4)
            hh, _ = hess(low[4*r:4*r+4, 4*c:4*c+4])
            where = np.unravel_index(np.argmax(np.abs(hh)), hh.shape)
            decoded[bit_index] = _extract_hmax(float(hh[where]), key.step)
        descrambled = arnold(decoded.reshape(key.watermark_shape), key.arnold_rounds, inverse=True)
        return (descrambled * 255).astype(np.uint8)
