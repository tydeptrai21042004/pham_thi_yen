from __future__ import annotations
import math
import numpy as np
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None


def mse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def psnr(a, b, data_range: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / m))


def ssim(a, b, data_range: float = 255.0) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if _ssim is not None:
        channel_axis = -1 if a.ndim == 3 else None
        return float(_ssim(a, b, data_range=data_range, channel_axis=channel_axis))
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = af.mean(), bf.mean()
    var_a, var_b = af.var(), bf.var()
    cov = ((af - mu_a) * (bf - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def to_bits(wm, threshold: int = 127) -> np.ndarray:
    return (np.asarray(wm, dtype=np.uint8) >= threshold).astype(np.uint8).ravel()


def nc(a, b) -> float:
    va = to_bits(a).astype(np.float64)
    vb = to_bits(b).astype(np.float64)
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    if den == 0:
        return 1.0 if np.array_equal(va, vb) else 0.0
    return float(np.clip(np.dot(va, vb) / den, -1.0, 1.0))


def ncc(a, b) -> float:
    va = to_bits(a).astype(np.float64)
    vb = to_bits(b).astype(np.float64)
    va = va - va.mean()
    vb = vb - vb.mean()
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    if den == 0:
        return 1.0 if np.array_equal(va, vb) else 0.0
    return float(np.clip(np.dot(va, vb) / den, -1.0, 1.0))


def ber(a, b) -> float:
    va = to_bits(a)
    vb = to_bits(b)
    if va.shape != vb.shape:
        raise ValueError(f"BER shape mismatch: {va.shape} vs {vb.shape}")
    return float(np.mean(va != vb))


def all_metrics(host, watermarked, watermark, extracted) -> dict[str, float]:
    return {
        "psnr": psnr(host, watermarked),
        "ssim": ssim(host, watermarked),
        "nc": nc(watermark, extracted),
        "ncc": ncc(watermark, extracted),
        "ber": ber(watermark, extracted),
    }
