"""Small, explicit shared primitives for the two NEW paper-guided baselines.

This module is deliberately independent of the proposal and of its parameters.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.linalg import hessenberg


def validate_rgb_watermark(host_rgb, watermark_binary, *, size=64):
    rgb = np.asarray(host_rgb)
    wm = np.asarray(watermark_binary)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB host (height,width,3)")
    if rgb.shape[0] != rgb.shape[1] or rgb.shape[0] % 8:
        raise ValueError("Expected square host with dimension divisible by 8")
    if wm.shape != (size, size):
        raise ValueError(f"Expected {size}x{size} binary watermark, got {wm.shape}")
    if not np.isin(wm, [0, 1, 255]).all():
        raise ValueError("Binary watermark values must be in {0,1,255}")
    return rgb, (wm > 0).astype(np.uint8)


def arnold(img: np.ndarray, rounds: int, inverse: bool = False) -> np.ndarray:
    """Arnold cat map, bijective even for a 64x64 watermark."""
    out = np.asarray(img).copy()
    n = out.shape[0]
    if out.ndim != 2 or out.shape[1] != n:
        raise ValueError("Arnold permutation requires square image")
    y, x = np.indices((n, n))
    # A^-1 = [[2,-1],[-1,1]] mod n
    for _ in range(rounds):
        z = np.empty_like(out)
        if inverse:
            z[(2*y-x) % n, (-y+x) % n] = out[y, x]
        else:
            z[(y+x) % n, (y+2*x) % n] = out[y, x]
        out = z
    return out


def md5_permutation(count: int, seed: int) -> np.ndarray:
    """Deterministic key-dependent, non-repeating MD5-ranked block ordering.

    MD5 here follows the *block-selector* concept of Su et al. (2016), NOT a
    claim of cryptographic security. Does not use random-replacement collisions.
    """
    keys = np.fromiter(
        (int.from_bytes(hashlib.md5(f"{seed}:{i}".encode()).digest()[:8], "big") for i in range(count)),
        dtype=np.uint64, count=count,
    )
    return np.argsort(keys, kind="stable")


def hess(block):
    return hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)


def quantize_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)
