"""Image and watermark evaluation metrics."""
from __future__ import annotations

import math
import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return float(np.mean((x - y) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / m))


def _bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) > 0).astype(np.uint8).ravel()


def ber(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Bit error rate for binary watermarks."""
    a = _bits(original_bits)
    b = _bits(recovered_bits)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a != b))


def nc(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Normalized correlation used in many watermarking papers.

    For binary {0,1} images this is sum(W*W_hat)/sum(W^2).  The result is 1
    for perfect recovery and approaches 0 when the extracted watermark contains
    no positive overlap with the original foreground.
    """
    a = _bits(original_bits).astype(np.float64)
    b = _bits(recovered_bits).astype(np.float64)
    denom = float(np.sum(a * a))
    if denom == 0:
        return 1.0 if np.sum(b) == 0 else 0.0
    return float(np.sum(a * b) / denom)


def ncc(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Pearson normalized cross-correlation.

    Returns 1 for identical non-constant binary watermarks.  If both inputs are
    constant and equal, the function returns 1; if one is constant and the other
    is not equal, it returns 0.
    """
    a = _bits(original_bits).astype(np.float64)
    b = _bits(recovered_bits).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a0, b0) / denom)
