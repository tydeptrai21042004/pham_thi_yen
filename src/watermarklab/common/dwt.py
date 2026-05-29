from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class DWTLevel:
    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray


def dwt2_haar_average(x: np.ndarray):
    """One-level 2D Haar transform using average/difference normalization.

    Returns LL, LH, HL, HH. This exact pair with idwt2_haar_average is perfectly
    reconstructive in floating point and matches the lightweight implementations
    used in several image-watermarking papers.
    """
    x = np.asarray(x, dtype=np.float64)
    even_cols = x[:, 0::2]
    odd_cols = x[:, 1::2]
    low_cols = (even_cols + odd_cols) / 2.0
    high_cols = (even_cols - odd_cols) / 2.0

    even_rows_l = low_cols[0::2, :]
    odd_rows_l = low_cols[1::2, :]
    even_rows_h = high_cols[0::2, :]
    odd_rows_h = high_cols[1::2, :]

    ll = (even_rows_l + odd_rows_l) / 2.0
    lh = (even_rows_l - odd_rows_l) / 2.0
    hl = (even_rows_h + odd_rows_h) / 2.0
    hh = (even_rows_h - odd_rows_h) / 2.0
    return ll, lh, hl, hh


def idwt2_haar_average(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray):
    ll, lh, hl, hh = [np.asarray(a, dtype=np.float64) for a in (ll, lh, hl, hh)]
    low_even = ll + lh
    low_odd = ll - lh
    high_even = hl + hh
    high_odd = hl - hh
    rows, cols = ll.shape
    low_cols = np.empty((rows * 2, cols), dtype=np.float64)
    high_cols = np.empty((rows * 2, cols), dtype=np.float64)
    low_cols[0::2, :] = low_even
    low_cols[1::2, :] = low_odd
    high_cols[0::2, :] = high_even
    high_cols[1::2, :] = high_odd
    out = np.empty((rows * 2, cols * 2), dtype=np.float64)
    out[:, 0::2] = low_cols + high_cols
    out[:, 1::2] = low_cols - high_cols
    return out


def dwt2_haar_orthonormal(x: np.ndarray):
    """One-level orthonormal Haar transform compatible with common pywt ordering."""
    x = np.asarray(x, dtype=np.float64)
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a + b - c - d) / 2.0
    hl = (a - b + c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return ll, lh, hl, hh


def idwt2_haar_orthonormal(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray):
    ll, lh, hl, hh = [np.asarray(a, dtype=np.float64) for a in (ll, lh, hl, hh)]
    a = (ll + lh + hl + hh) / 2.0
    b = (ll + lh - hl - hh) / 2.0
    c = (ll - lh + hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float64)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out


def dwt2(x: np.ndarray, mode: str = "average"):
    if mode == "average":
        return dwt2_haar_average(x)
    if mode == "orthonormal":
        return dwt2_haar_orthonormal(x)
    raise ValueError(f"Unknown DWT mode: {mode}")


def idwt2(ll, lh, hl, hh, mode: str = "average"):
    if mode == "average":
        return idwt2_haar_average(ll, lh, hl, hh)
    if mode == "orthonormal":
        return idwt2_haar_orthonormal(ll, lh, hl, hh)
    raise ValueError(f"Unknown DWT mode: {mode}")


def multilevel_dwt_ll(x: np.ndarray, levels: int, mode: str = "average"):
    current = np.asarray(x, dtype=np.float64)
    details: list[DWTLevel] = []
    for _ in range(levels):
        ll, lh, hl, hh = dwt2(current, mode=mode)
        details.append(DWTLevel(lh=lh, hl=hl, hh=hh))
        current = ll
    return current, details


def multilevel_idwt_ll(ll: np.ndarray, details: list[DWTLevel], mode: str = "average"):
    current = np.asarray(ll, dtype=np.float64)
    for level in reversed(details):
        current = idwt2(current, level.lh, level.hl, level.hh, mode=mode)
    return current


def split_4bands(x: np.ndarray, mode: str = "average") -> dict[str, np.ndarray]:
    ll, lh, hl, hh = dwt2(x, mode=mode)
    return {"LL": ll.copy(), "LH": lh.copy(), "HL": hl.copy(), "HH": hh.copy()}


def merge_4bands(bands: dict[str, np.ndarray], mode: str = "average") -> np.ndarray:
    return idwt2(bands["LL"], bands["LH"], bands["HL"], bands["HH"], mode=mode)
