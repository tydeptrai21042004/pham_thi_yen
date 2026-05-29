from __future__ import annotations
import numpy as np


def additive_embed(cover: np.ndarray, payload: np.ndarray, alpha: float) -> np.ndarray:
    """Additive watermark equation: marked = cover + alpha * payload."""
    return np.asarray(cover, dtype=np.float64) + float(alpha) * np.asarray(payload, dtype=np.float64)


def alpha_blend_cover_weight(cover: np.ndarray, payload: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha blending with alpha assigned to cover: alpha*C + (1-alpha)*W."""
    return float(alpha) * np.asarray(cover, dtype=np.float64) + (1.0 - float(alpha)) * np.asarray(payload, dtype=np.float64)


def alpha_blend_payload_weight(cover: np.ndarray, payload: np.ndarray, alpha: float) -> np.ndarray:
    """DWT-HD-SVD Eq. (17): S'_H = alpha*S_w + (1-alpha)*S_H."""
    return (1.0 - float(alpha)) * np.asarray(cover, dtype=np.float64) + float(alpha) * np.asarray(payload, dtype=np.float64)


def extract_from_alpha_blend_payload_weight(marked: np.ndarray, cover: np.ndarray, alpha: float) -> np.ndarray:
    """Inverse of alpha_blend_payload_weight for the payload term."""
    if abs(float(alpha)) < 1e-15:
        raise ZeroDivisionError("alpha must be non-zero")
    return (np.asarray(marked, dtype=np.float64) - (1.0 - float(alpha)) * np.asarray(cover, dtype=np.float64)) / float(alpha)


def extract_from_alpha_blend_cover_weight(marked: np.ndarray, cover: np.ndarray, alpha: float) -> np.ndarray:
    """Inverse of alpha_blend_cover_weight for the payload term."""
    den = 1.0 - float(alpha)
    if abs(den) < 1e-15:
        raise ZeroDivisionError("1-alpha must be non-zero")
    return (np.asarray(marked, dtype=np.float64) - float(alpha) * np.asarray(cover, dtype=np.float64)) / den
