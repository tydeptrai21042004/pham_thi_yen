"""Small dependency-free 2D Haar DWT/IDWT utilities.

The paper states that a 2D-DWT is applied to each RGB channel and produces
LL, LH, HL, HH bands.  The wavelet family is not specified in the article, so
this reproduction uses the orthonormal Haar transform, which is the simplest
and most common default in image watermarking baselines.
"""
from __future__ import annotations

import numpy as np


def ensure_even_2d(x: np.ndarray) -> np.ndarray:
    """Crop a 2D array to even height and width."""
    h, w = x.shape
    return x[: h - (h % 2), : w - (w % 2)]


def haar_dwt2(channel: np.ndarray) -> dict[str, np.ndarray]:
    """Orthonormal 1-level Haar DWT for one image channel.

    Returns a dict with keys LL, LH, HL, HH. All returned bands have shape
    (H/2, W/2) when the input dimensions are even.
    """
    x = ensure_even_2d(np.asarray(channel, dtype=np.float64))
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


def haar_idwt2(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`haar_dwt2` for one image channel."""
    ll = np.asarray(bands["LL"], dtype=np.float64)
    lh = np.asarray(bands["LH"], dtype=np.float64)
    hl = np.asarray(bands["HL"], dtype=np.float64)
    hh = np.asarray(bands["HH"], dtype=np.float64)
    h, w = ll.shape
    x = np.empty((2 * h, 2 * w), dtype=np.float64)
    x[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    x[0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    x[1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    x[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return x
