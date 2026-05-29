from __future__ import annotations
import numpy as np


def logistic_sequence(n: int, x0: float = 0.517, mu: float = 3.999999, burn_in: int = 128) -> np.ndarray:
    x = float(x0)
    for _ in range(burn_in):
        x = mu * x * (1.0 - x)
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = mu * x * (1.0 - x)
        seq[i] = x
    return seq


def logistic_permutation(n: int, x0: float = 0.517, mu: float = 3.999999) -> np.ndarray:
    return np.argsort(logistic_sequence(n, x0=x0, mu=mu), kind="mergesort")


def chaotic_encrypt_uint8(mat: np.ndarray, x0: float = 0.517, mu: float = 3.999999):
    original_shape = mat.shape
    data = np.clip(np.rint(mat), 0, 255).astype(np.uint8).ravel()
    seq = logistic_sequence(data.size, x0=x0, mu=mu)
    idx = np.argsort(seq, kind="mergesort")
    mask = np.floor(seq * 256.0).astype(np.uint8)
    encrypted = np.bitwise_xor(data[idx], mask)
    return encrypted.reshape(original_shape).astype(np.float64), idx, mask


def chaotic_decrypt_uint8(encrypted: np.ndarray, idx: np.ndarray, mask: np.ndarray):
    flat = np.clip(np.rint(encrypted), 0, 255).astype(np.uint8).ravel()
    unmasked = np.bitwise_xor(flat, mask.astype(np.uint8))
    restored = np.empty_like(unmasked)
    restored[idx] = unmasked
    return restored.reshape(encrypted.shape).astype(np.float64)


def arnold_scramble(bits2d: np.ndarray, iterations: int = 17) -> np.ndarray:
    bits2d = np.asarray(bits2d, dtype=np.uint8)
    if bits2d.ndim != 2 or bits2d.shape[0] != bits2d.shape[1]:
        raise ValueError("Arnold scrambling requires a square 2-D array")
    n = bits2d.shape[0]
    out = bits2d.copy()
    for _ in range(int(iterations)):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                temp[(x + y) % n, (x + 2 * y) % n] = out[x, y]
        out = temp
    return out


def arnold_unscramble(bits2d: np.ndarray, iterations: int = 17) -> np.ndarray:
    bits2d = np.asarray(bits2d, dtype=np.uint8)
    if bits2d.ndim != 2 or bits2d.shape[0] != bits2d.shape[1]:
        raise ValueError("Arnold unscrambling requires a square 2-D array")
    n = bits2d.shape[0]
    out = bits2d.copy()
    for _ in range(int(iterations)):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                # Inverse of [[1,1],[1,2]] modulo n is [[2,-1],[-1,1]].
                temp[(2 * x - y) % n, (-x + y) % n] = out[x, y]
        out = temp
    return out
