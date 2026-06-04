from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.linalg import hadamard


@lru_cache(maxsize=16)
def orthonormal_hadamard_matrix(n: int) -> np.ndarray:
    """Return an orthonormal Walsh-Hadamard matrix of order n.

    scipy.linalg.hadamard requires n to be a power of two.  The normalized
    transform is self-inverse, so H @ X @ H.T is both the forward and inverse
    2-D WHT when applied twice.
    """
    n = int(n)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard order must be a positive power of two, got {n}")
    return hadamard(n, dtype=np.float64) / np.sqrt(float(n))


def wht2(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"2-D WHT expects a square matrix, got {arr.shape}")
    h = orthonormal_hadamard_matrix(arr.shape[0])
    return h @ arr @ h.T


def iwht2(coeffs: np.ndarray) -> np.ndarray:
    # The orthonormal Walsh-Hadamard transform is its own inverse.
    return wht2(coeffs)
