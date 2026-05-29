from __future__ import annotations
import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray):
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.clip(np.rint(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)
