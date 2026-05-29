from __future__ import annotations
import numpy as np
from scipy.linalg import hessenberg


def diag_from_s(s: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if shape is None:
        shape = (s.size, s.size)
    out = np.zeros(shape, dtype=np.float64)
    k = min(shape[0], shape[1], s.size)
    out[np.arange(k), np.arange(k)] = s[:k]
    return out


def svd_pc(mat: np.ndarray):
    u, s, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64), full_matrices=True)
    return u @ diag_from_s(s, (u.shape[0], vt.shape[0])), vt


def hess_decompose(mat: np.ndarray):
    h, q = hessenberg(np.asarray(mat, dtype=np.float64), calc_q=True)
    return q, h


def hess_reconstruct(q: np.ndarray, h: np.ndarray):
    return q @ h @ q.T
