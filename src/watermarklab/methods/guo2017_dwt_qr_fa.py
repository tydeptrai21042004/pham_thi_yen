from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2


@dataclass
class Guo2017Key:
    """Side information for Guo et al. 2017 DWT-QR extraction.

    The original method is blind with respect to the original cover image but uses
    the sort-position vector P and a random vector K as secret keys.
    """

    order: np.ndarray
    k_vector: np.ndarray
    lambda_strength: float
    block_size: int
    watermark_shape: tuple[int, int]
    dwt_mode: str
    color_mode: str


def _stable_qr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR with a deterministic sign convention for reproducible embedding/extraction."""
    q, r = np.linalg.qr(np.asarray(a, dtype=np.float64))
    diag = np.diag(r)
    signs = np.where(diag < 0, -1.0, 1.0)
    q = q * signs[None, :]
    r = signs[:, None] * r
    return q, r


def _corrcoef_sign(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


class Guo2017DWTQRFA:
    """Paper-guided Guo et al. 2017 DWT-QR-FA blind watermarking baseline.

    Baseline path:
        Y channel -> vector sorting/scrambling -> one-level DWT -> LL subband
        -> 4x4 blocks -> QR decomposition -> embed each binary watermark bit in
        the first row of R using +/- lambda*K -> inverse DWT -> inverse sorting.

    Extraction path:
        Re-apply the saved sorting vector, DWT and QR; decide each bit from the
        sign of corrcoef(R'(1,:), K).

    Practical notes:
        * The paper uses grayscale 512x512 cover images.  This benchmark adapts it
          to color covers by embedding in the Y component and preserving Cb/Cr.
        * The paper obtains lambda by Firefly Algorithm.  For fast reproducible
          benchmarking, a calibrated lambda is used by default.  The class name
          keeps FA because this is the Guo DWT-QR-FA baseline family.
    """

    name = "Guo2017_DWT_QR_FA"

    def __init__(
        self,
        lambda_strength: float = 4.0,
        block_size: int = 4,
        seed: int = 2017,
        dwt_mode: str = "orthonormal",
        color_mode: str = "ycbcr_y",
        mode: str = "adapt",
    ):
        self.lambda_strength = float(lambda_strength)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.dwt_mode = str(dwt_mode)
        self.mode = str(mode)
        if self.mode == "original-rerun" and color_mode == "ycbcr_y":
            color_mode = "gray_mean"
        self.color_mode = str(color_mode)
        if self.block_size != 4:
            raise ValueError("Guo2017 paper-guided mode expects 4x4 QR blocks")

    def _get_carrier(self, host_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.color_mode == "ycbcr_y":
            y, cb, cr = rgb_to_ycbcr(host_rgb)
            return y, cb, cr
        if self.color_mode == "gray_mean":
            x = np.asarray(host_rgb, dtype=np.float64)
            return x.mean(axis=2), None, None
        raise ValueError(f"Unsupported color_mode: {self.color_mode}")

    def _merge_carrier(self, y: np.ndarray, cb: np.ndarray | None, cr: np.ndarray | None, host_rgb: np.ndarray) -> np.ndarray:
        if self.color_mode == "ycbcr_y":
            assert cb is not None and cr is not None
            return ycbcr_to_rgb(y, cb, cr)
        gray = np.clip(np.rint(y), 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _make_k(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        k = rng.choice([-1.0, 1.0], size=(self.block_size,))
        # Avoid zero-variance vectors because extraction uses correlation.
        if np.all(k == k[0]):
            k[0] *= -1.0
        return k.astype(np.float64)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        wm = (np.asarray(watermark_binary, dtype=np.uint8) >= 127).astype(np.uint8)
        if wm.shape != (64, 64):
            raise ValueError(f"Guo2017 expects a 64x64 binary watermark, got {wm.shape}")

        carrier, cb, cr = self._get_carrier(host_rgb)
        h, w = carrier.shape
        if h != w or h % 8 != 0:
            raise ValueError("Guo2017 common mode expects a square cover with side divisible by 8")
        expected_n = h // 8
        if wm.shape != (expected_n, expected_n):
            raise ValueError(f"For {h}x{w} cover, watermark must be {expected_n}x{expected_n}")

        flat = carrier.reshape(-1)
        order = np.argsort(flat, kind="mergesort")
        scrambled = flat[order].reshape(h, w)

        ll, lh, hl, hh = dwt2(scrambled, mode=self.dwt_mode)
        ll_marked = ll.copy()
        k_vec = self._make_k()

        for i in range(wm.shape[0]):
            for j in range(wm.shape[1]):
                r0 = i * self.block_size
                c0 = j * self.block_size
                block = ll[r0 : r0 + self.block_size, c0 : c0 + self.block_size]
                q, r = _stable_qr(block)
                r2 = r.copy()
                delta = self.lambda_strength * k_vec
                if int(wm[i, j]) == 1:
                    r2[0, :] = r2[0, :] + delta
                else:
                    r2[0, :] = r2[0, :] - delta
                ll_marked[r0 : r0 + self.block_size, c0 : c0 + self.block_size] = q @ r2

        scrambled_marked = idwt2(ll_marked, lh, hl, hh, mode=self.dwt_mode)
        marked_flat_scrambled = scrambled_marked.reshape(-1)
        marked_flat = np.empty_like(marked_flat_scrambled)
        marked_flat[order] = marked_flat_scrambled
        marked_carrier = marked_flat.reshape(h, w)
        watermarked = self._merge_carrier(marked_carrier, cb, cr, host_rgb)
        key = Guo2017Key(
            order=order,
            k_vector=k_vec,
            lambda_strength=self.lambda_strength,
            block_size=self.block_size,
            watermark_shape=wm.shape,
            dwt_mode=self.dwt_mode,
            color_mode=self.color_mode,
        )
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Guo2017Key, host_rgb: np.ndarray | None = None):
        # Blind extraction: host_rgb is intentionally unused.
        if key.color_mode == "ycbcr_y":
            carrier, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        elif key.color_mode == "gray_mean":
            carrier = np.asarray(possibly_attacked_rgb, dtype=np.float64).mean(axis=2)
        else:
            raise ValueError(f"Unsupported color_mode in key: {key.color_mode}")

        h, w = carrier.shape
        scrambled = carrier.reshape(-1)[key.order].reshape(h, w)
        ll, _, _, _ = dwt2(scrambled, mode=key.dwt_mode)
        wm_out = np.zeros(key.watermark_shape, dtype=np.uint8)

        for i in range(key.watermark_shape[0]):
            for j in range(key.watermark_shape[1]):
                r0 = i * key.block_size
                c0 = j * key.block_size
                block = ll[r0 : r0 + key.block_size, c0 : c0 + key.block_size]
                _q, r = _stable_qr(block)
                score = _corrcoef_sign(r[0, :], key.k_vector)
                wm_out[i, j] = 255 if score >= 0.0 else 0
        return wm_out


__all__ = ["Guo2017DWTQRFA", "Guo2017Key"]
