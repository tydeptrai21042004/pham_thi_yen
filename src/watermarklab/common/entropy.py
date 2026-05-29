from __future__ import annotations
import numpy as np


def probability_mass(block: np.ndarray, bins: int = 256, value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Return non-zero probability masses for entropy-style formulas.

    The watermarking papers define visual/information entropy in terms of pixel
    probabilities p_i. This helper makes the histogram normalization explicit so
    all methods and tests use the same math.
    """
    arr = np.asarray(block, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    if value_range is None:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if np.isclose(lo, hi):
            return np.asarray([1.0], dtype=np.float64)
        value_range = (lo, hi)
    hist, _ = np.histogram(arr, bins=int(bins), range=value_range)
    p = hist.astype(np.float64)
    total = float(np.sum(p))
    if total <= 0:
        return np.asarray([], dtype=np.float64)
    p = p[p > 0] / total
    return p


def visual_entropy(block: np.ndarray, bins: int = 256, log_base: float = 2.0) -> float:
    """Visual/information entropy: E_v = -sum_i p_i log(p_i)."""
    p = probability_mass(block, bins=bins)
    if p.size == 0:
        return 0.0
    if np.isclose(log_base, np.e):
        logp = np.log(p)
    else:
        logp = np.log(p) / np.log(float(log_base))
    return float(-np.sum(p * logp))


def edge_entropy(block: np.ndarray, bins: int = 256) -> float:
    """Edge entropy used in Kumar-style block selection: E_e = sum_i p_i exp(1-p_i)."""
    p = probability_mass(block, bins=bins)
    if p.size == 0:
        return 0.0
    return float(np.sum(p * np.exp(1.0 - p)))


def block_entropy_score(block: np.ndarray, bins: int = 256) -> float:
    """Kumar et al. score B_E = E_v - E_e.

    The selected block is the one maximizing this score.
    """
    return float(visual_entropy(block, bins=bins) - edge_entropy(block, bins=bins))


def select_max_entropy_score_block(arr: np.ndarray, block_size: int = 32, bins: int = 256):
    h, w = np.asarray(arr).shape[:2]
    best = (0, 0)
    best_score = -np.inf
    for bi, r in enumerate(range(0, h, block_size)):
        for bj, c in enumerate(range(0, w, block_size)):
            block = np.asarray(arr)[r:r + block_size, c:c + block_size]
            if block.shape != (block_size, block_size):
                continue
            score = block_entropy_score(block, bins=bins)
            if score > best_score:
                best = (bi, bj)
                best_score = score
    return best, float(best_score)
