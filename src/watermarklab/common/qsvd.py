from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QuaternionBlock:
    """Small real representation of a quaternion matrix.

    q = r + i*i + j*j + k*k where each field is a real matrix with the same
    shape.  The helper functions below use the standard complex adjoint
    representation of quaternion matrices so numpy/scipy SVD can be used in a
    deterministic way without adding a heavyweight quaternion dependency.
    """

    r: np.ndarray
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.r, self.i, self.j, self.k)


def quaternion_to_complex_adjoint(q: QuaternionBlock) -> np.ndarray:
    r, i, j, k = [np.asarray(x, dtype=np.float64) for x in q.as_tuple()]
    if not (r.shape == i.shape == j.shape == k.shape):
        raise ValueError("All quaternion components must have the same shape")
    a = r + 1j * i
    b = j + 1j * k
    top = np.concatenate([a, b], axis=1)
    bottom = np.concatenate([-np.conjugate(b), np.conjugate(a)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def complex_adjoint_to_quaternion(c: np.ndarray, shape: tuple[int, int]) -> QuaternionBlock:
    c = np.asarray(c, dtype=np.complex128)
    m, n = shape
    if c.shape != (2 * m, 2 * n):
        raise ValueError(f"Complex adjoint shape {c.shape} does not match quaternion shape {shape}")

    # Project the possibly numerically imperfect complex matrix back onto the
    # valid quaternion-adjoint structure.
    a1 = c[:m, :n]
    b1 = c[:m, n:]
    b2 = -np.conjugate(c[m:, :n])
    a2 = np.conjugate(c[m:, n:])
    a = 0.5 * (a1 + a2)
    b = 0.5 * (b1 + b2)
    return QuaternionBlock(r=np.real(a), i=np.imag(a), j=np.real(b), k=np.imag(b))


def qsvd_complex(q: QuaternionBlock):
    """QSVD through the complex adjoint representation.

    Quaternion singular values appear twice in the complex representation.  For
    embedding/reconstruction we keep the complex matrices directly, then project
    back to quaternion components after inverse reconstruction.
    """
    c = quaternion_to_complex_adjoint(q)
    return np.linalg.svd(c, full_matrices=True)
