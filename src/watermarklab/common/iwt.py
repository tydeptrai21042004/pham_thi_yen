from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class IWTLevel:
    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray


def _haar_lift_1d_forward(x: np.ndarray, axis: int):
    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    high = odd - even
    low = even + high / 2.0
    return np.moveaxis(low, -1, axis), np.moveaxis(high, -1, axis)


def _haar_lift_1d_inverse(low: np.ndarray, high: np.ndarray, axis: int):
    low = np.moveaxis(np.asarray(low, dtype=np.float64), axis, -1)
    high = np.moveaxis(np.asarray(high, dtype=np.float64), axis, -1)
    even = low - high / 2.0
    odd = high + even
    out_shape = list(even.shape)
    out_shape[-1] = even.shape[-1] + odd.shape[-1]
    out = np.empty(out_shape, dtype=np.float64)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return np.moveaxis(out, -1, axis)


def iwt2(image: np.ndarray):
    lo_cols, hi_cols = _haar_lift_1d_forward(image, axis=1)
    ll, lh = _haar_lift_1d_forward(lo_cols, axis=0)
    hl, hh = _haar_lift_1d_forward(hi_cols, axis=0)
    return ll, lh, hl, hh


def iiwt2(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray):
    lo_cols = _haar_lift_1d_inverse(ll, lh, axis=0)
    hi_cols = _haar_lift_1d_inverse(hl, hh, axis=0)
    return _haar_lift_1d_inverse(lo_cols, hi_cols, axis=1)


def multilevel_iwt(image: np.ndarray, levels: int):
    current = np.asarray(image, dtype=np.float64)
    saved: list[IWTLevel] = []
    for _ in range(levels):
        ll, lh, hl, hh = iwt2(current)
        saved.append(IWTLevel(lh=lh, hl=hl, hh=hh))
        current = ll
    return current, saved


def multilevel_iiwt(ll: np.ndarray, levels: list[IWTLevel]):
    current = np.asarray(ll, dtype=np.float64)
    for level in reversed(levels):
        current = iiwt2(current, level.lh, level.hl, level.hh)
    return current
