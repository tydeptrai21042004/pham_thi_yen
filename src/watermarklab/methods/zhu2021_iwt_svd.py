from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.iwt import iwt2, iiwt2


@dataclass
class Zhu2021IWTSVDKey:
    """Key/metadata for the Zhu et al. 2021 IWT-SVD adapted baseline.

    The original Zhu et al. method is a grayscale IWT-SVD watermarking scheme
    with QIM on the first singular value and an optimized quantization step.
    This project adapter keeps the blind extraction rule and adapts the carrier
    to the Y channel of 512x512 RGB hosts so a 64x64 watermark can be compared
    with the other color-image baselines.
    """

    watermark_shape: tuple[int, int]
    host_shape: tuple[int, int]
    block_size: int
    delta: float
    affine_params: tuple[int, int, int, int, int, int]
    affine_iterations: int
    color_mode: str
    mode: str = "adapt"
    stats: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small number-theory / affine scrambling helpers
# -----------------------------------------------------------------------------

def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return abs(a), 1 if a >= 0 else -1, 0
    g, x1, y1 = _egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def _mod_inverse(a: int, n: int) -> int:
    a = int(a) % int(n)
    g, x, _ = _egcd(a, n)
    if g != 1:
        raise ValueError(f"{a} has no modular inverse modulo {n}")
    return int(x % n)


def _affine_scramble(bits2d: np.ndarray, params: tuple[int, int, int, int, int, int], iterations: int) -> np.ndarray:
    """Apply the paper-style affine watermark encryption map.

    Coordinates are mapped as
        [x'; y'] = [[A, B], [C, D]] [x; y] + [E; F] mod N.
    The default params (1, 1, 1, 2, 0, 0) are the Arnold cat map, which is a
    special invertible affine transform with determinant 1.
    """
    src = np.asarray(bits2d, dtype=np.uint8)
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"Affine scrambling requires a square 2D watermark, got {src.shape}")
    n = int(src.shape[0])
    a, b, c, d, e, f = (int(v) for v in params)
    det = (a * d - b * c) % n
    _mod_inverse(det, n)  # validate invertibility

    out = src.copy()
    for _ in range(int(iterations)):
        tmp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                xp = (a * x + b * y + e) % n
                yp = (c * x + d * y + f) % n
                tmp[xp, yp] = out[x, y]
        out = tmp
    return out


def _affine_unscramble(bits2d: np.ndarray, params: tuple[int, int, int, int, int, int], iterations: int) -> np.ndarray:
    src = np.asarray(bits2d, dtype=np.uint8)
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"Affine unscrambling requires a square 2D watermark, got {src.shape}")
    n = int(src.shape[0])
    a, b, c, d, e, f = (int(v) for v in params)
    det = (a * d - b * c) % n
    inv_det = _mod_inverse(det, n)
    ia = (d * inv_det) % n
    ib = (-b * inv_det) % n
    ic = (-c * inv_det) % n
    id_ = (a * inv_det) % n

    out = src.copy()
    for _ in range(int(iterations)):
        tmp = np.zeros_like(out)
        for xp in range(n):
            for yp in range(n):
                # First remove translation, then multiply by inverse matrix.
                u = (xp - e) % n
                v = (yp - f) % n
                x = (ia * u + ib * v) % n
                y = (ic * u + id_ * v) % n
                tmp[x, y] = out[xp, yp]
        out = tmp
    return out


# -----------------------------------------------------------------------------
# QIM and SVD helpers
# -----------------------------------------------------------------------------

def _watermark_bits(watermark_binary: np.ndarray) -> np.ndarray:
    return (np.asarray(watermark_binary, dtype=np.uint8) >= 127).astype(np.uint8)


def _nearest_qim_value(value: float, bit: int, delta: float) -> float:
    """Move value to the nearest QIM bin center for bit 0/1.

    Extraction is thresholded by ``value mod delta``:
      residue < delta/2 -> 0, residue >= delta/2 -> 1.
    Therefore we embed at residues delta/4 and 3delta/4 to maximize distance
    from the decision boundary while changing the singular value as little as
    possible.
    """
    d = float(delta)
    if d <= 0:
        raise ValueError("QIM delta must be positive")
    target = 0.25 * d if int(bit) == 0 else 0.75 * d
    base = math.floor(float(value) / d) * d
    candidates = [base + target, base - d + target, base + d + target]
    candidates = [c for c in candidates if c >= 0.0]
    if not candidates:
        return target
    return float(min(candidates, key=lambda c: abs(c - float(value))))


def _extract_qim_bit(value: float, delta: float) -> tuple[int, float]:
    d = float(delta)
    residue = float(np.mod(float(value), d))
    bit = 0 if residue < 0.5 * d else 1
    confidence = abs(residue - 0.5 * d)
    return int(bit), float(confidence)


def _embed_bit_in_ll_svd(ll: np.ndarray, bit: int, delta: float) -> tuple[np.ndarray, float, float]:
    """Apply SVD to an IWT LL block and QIM-embed one bit in S(0,0)."""
    u, s, vt = np.linalg.svd(np.asarray(ll, dtype=np.float64), full_matrices=False)
    old_s0 = float(s[0])
    s2 = s.copy()
    s2[0] = _nearest_qim_value(old_s0, int(bit), float(delta))
    marked = (u * s2[None, :]) @ vt
    return marked, old_s0, float(s2[0])


def _extract_bit_from_ll_svd(ll: np.ndarray, delta: float) -> tuple[int, float, float]:
    _u, s, _vt = np.linalg.svd(np.asarray(ll, dtype=np.float64), full_matrices=False)
    bit, conf = _extract_qim_bit(float(s[0]), float(delta))
    return int(bit), float(conf), float(s[0])


class Zhu2021IWTSVD:
    """Zhu et al. 2021 IWT-SVD baseline adapted to this 64x64 color benchmark.

    Paper core:
        carrier image -> non-overlapping blocks -> IWT per block -> SVD on LL
        -> QIM on the first singular value -> inverse SVD -> inverse IWT.

    Project adaptation:
        * Native paper setting: 512x512 grayscale host and 32x32 watermark.
        * Benchmark setting here: 512x512 RGB host, use Y channel in YCbCr,
          64x64 binary watermark, one 8x8 host block per watermark bit.
        * Extraction is blind/key-based: it needs only the affine scrambling key,
          block size and quantization step, not the original host image.

    The genetic algorithm in the paper optimizes the quantization step.  To keep
    this baseline deterministic and lightweight inside the unified benchmark,
    ``delta`` is exposed as a parameter and can be tuned externally/with sweeps.
    """

    name = "Zhu2021_IWT_SVD_adapted"
    is_blind = True
    requires_side_information = False
    side_information = "key-only: affine scrambling parameters, block size, QIM step"

    def __init__(
        self,
        mode: str = "adapt",
        delta: float | None = None,
        block_size: int | None = None,
        affine_params: tuple[int, int, int, int, int, int] = (1, 1, 1, 2, 0, 0),
        affine_iterations: int = 10,
        color_mode: str = "ycbcr_y",
    ):
        self.mode = str(mode).lower().strip()
        # Native Zhu is 32x32 grayscale.  The default benchmark adapter is 64x64
        # on YCbCr-Y.  original-rerun is provided for manual experiments with a
        # 32x32 watermark, but the project CLI normally loads a 64x64 watermark.
        if self.mode in {"paper", "original", "original-rerun"}:
            self.native_watermark_shape = (32, 32)
            self.default_block_size = 16
            self.default_color_mode = "gray_mean"
            self.default_delta = 18.0
        else:
            self.native_watermark_shape = (64, 64)
            self.default_block_size = 8
            self.default_color_mode = "ycbcr_y"
            self.default_delta = 16.0

        self.delta = float(self.default_delta if delta is None else delta)
        self.block_size = int(self.default_block_size if block_size is None else block_size)
        self.affine_params = tuple(int(v) for v in affine_params)
        self.affine_iterations = int(affine_iterations)
        self.color_mode = str(color_mode if color_mode != "auto" else self.default_color_mode)
        if color_mode == "ycbcr_y" and self.mode in {"paper", "original", "original-rerun"}:
            # Keep explicit user choice if provided; otherwise use gray_mean.
            self.color_mode = str(color_mode)
        if self.delta <= 0:
            raise ValueError("Zhu2021 IWT-SVD delta must be positive")
        if self.block_size % 2 != 0 or self.block_size < 4:
            raise ValueError("Zhu2021 IWT-SVD block_size must be an even integer >= 4")
        if self.color_mode not in {"ycbcr_y", "gray_mean"}:
            raise ValueError("color_mode must be 'ycbcr_y' or 'gray_mean'")

    # ------------------------------------------------------------------
    # Carrier handling
    # ------------------------------------------------------------------
    def _get_carrier(self, host_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        host = np.asarray(host_rgb)
        if self.color_mode == "ycbcr_y":
            if host.ndim != 3 or host.shape[2] != 3:
                raise ValueError(f"YCbCr-Y mode expects RGB host HxWx3, got {host.shape}")
            y, cb, cr = rgb_to_ycbcr(host)
            return y, cb, cr
        if host.ndim == 2:
            return host.astype(np.float64), None, None
        if host.ndim == 3:
            return host.astype(np.float64).mean(axis=2), None, None
        raise ValueError(f"Unsupported host shape: {host.shape}")

    def _merge_carrier(self, marked_carrier: np.ndarray, cb: np.ndarray | None, cr: np.ndarray | None) -> np.ndarray:
        marked = np.clip(np.rint(marked_carrier), 0, 255)
        if self.color_mode == "ycbcr_y":
            assert cb is not None and cr is not None
            return ycbcr_to_rgb(marked, cb, cr)
        gray = marked.astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _validate_payload(self, carrier: np.ndarray, wm: np.ndarray) -> None:
        if carrier.ndim != 2:
            raise ValueError(f"Zhu2021 carrier must be 2D, got {carrier.shape}")
        h, w = carrier.shape
        if h != 512 or w != 512:
            raise ValueError(f"Zhu2021 adapter expects a 512x512 host carrier, got {carrier.shape}")
        expected = (h // self.block_size, w // self.block_size)
        if wm.shape != expected:
            raise ValueError(
                f"For host {h}x{w} and block_size={self.block_size}, watermark must be {expected}, got {wm.shape}. "
                "Use adapt mode for 64x64 or original-rerun with a 32x32 watermark."
            )

    # ------------------------------------------------------------------
    # Public API used by the benchmark
    # ------------------------------------------------------------------
    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        carrier, cb, cr = self._get_carrier(host_rgb)
        wm01 = _watermark_bits(watermark_binary)
        self._validate_payload(carrier, wm01)

        wm_scr = _affine_scramble(wm01, self.affine_params, self.affine_iterations)
        marked_carrier = carrier.astype(np.float64).copy()
        bs = int(self.block_size)
        wm_h, wm_w = wm01.shape

        abs_s_change_sum = 0.0
        max_s_change = 0.0
        changed_blocks = 0
        failed_blocks = 0

        for i in range(wm_h):
            y0 = i * bs
            for j in range(wm_w):
                x0 = j * bs
                block = carrier[y0 : y0 + bs, x0 : x0 + bs].astype(np.float64)
                try:
                    ll, lh, hl, hh = iwt2(block)
                    ll_marked, old_s0, new_s0 = _embed_bit_in_ll_svd(ll, int(wm_scr[i, j]), self.delta)
                    rec = iiwt2(ll_marked, lh, hl, hh)
                    marked_carrier[y0 : y0 + bs, x0 : x0 + bs] = rec
                    ds = abs(new_s0 - old_s0)
                    abs_s_change_sum += ds
                    max_s_change = max(max_s_change, ds)
                    if ds > 1e-12:
                        changed_blocks += 1
                except np.linalg.LinAlgError:
                    failed_blocks += 1
                    # Leave block unchanged if SVD fails; this is extremely rare
                    # for 4x4 LL blocks but keeps the benchmark robust.
                    continue

        watermarked = self._merge_carrier(marked_carrier, cb, cr)
        total_blocks = int(wm_h * wm_w)
        stats = {
            "paper_native_host": "512x512 grayscale",
            "paper_native_watermark": "32x32",
            "adapter_host": "512x512 RGB using YCbCr-Y" if self.color_mode == "ycbcr_y" else "512x512 grayscale/mean RGB",
            "adapter_watermark": f"{wm_h}x{wm_w}",
            "blocks_total": total_blocks,
            "block_size": int(bs),
            "iwt_ll_shape_per_block": [int(bs // 2), int(bs // 2)],
            "changed_blocks": int(changed_blocks),
            "failed_blocks": int(failed_blocks),
            "mean_abs_s0_change": float(abs_s_change_sum / max(total_blocks - failed_blocks, 1)),
            "max_abs_s0_change": float(max_s_change),
        }
        key = Zhu2021IWTSVDKey(
            watermark_shape=tuple(int(v) for v in wm01.shape),
            host_shape=tuple(int(v) for v in carrier.shape),
            block_size=int(bs),
            delta=float(self.delta),
            affine_params=tuple(int(v) for v in self.affine_params),
            affine_iterations=int(self.affine_iterations),
            color_mode=str(self.color_mode),
            mode=str(self.mode),
            stats=stats,
        )
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Zhu2021IWTSVDKey, host_rgb: np.ndarray | None = None):
        # Blind extraction: host_rgb is intentionally unused.
        old_color_mode = self.color_mode
        try:
            self.color_mode = key.color_mode
            carrier, _cb, _cr = self._get_carrier(possibly_attacked_rgb)
        finally:
            self.color_mode = old_color_mode

        bs = int(key.block_size)
        wm_h, wm_w = (int(v) for v in key.watermark_shape)
        if carrier.shape[0] < wm_h * bs or carrier.shape[1] < wm_w * bs:
            raise ValueError(
                f"Attacked image carrier {carrier.shape} is too small for watermark {key.watermark_shape} with block_size={bs}"
            )

        wm_scr_rec = np.zeros((wm_h, wm_w), dtype=np.uint8)
        confidences = np.zeros((wm_h, wm_w), dtype=np.float64)
        failed_blocks = 0
        for i in range(wm_h):
            y0 = i * bs
            for j in range(wm_w):
                x0 = j * bs
                block = carrier[y0 : y0 + bs, x0 : x0 + bs].astype(np.float64)
                try:
                    ll, _lh, _hl, _hh = iwt2(block)
                    bit, conf, _s0 = _extract_bit_from_ll_svd(ll, key.delta)
                    wm_scr_rec[i, j] = 1 if bit else 0
                    confidences[i, j] = conf
                except np.linalg.LinAlgError:
                    failed_blocks += 1
                    wm_scr_rec[i, j] = 0
                    confidences[i, j] = 0.0

        wm01 = _affine_unscramble(wm_scr_rec, key.affine_params, key.affine_iterations)
        return (wm01 * 255).astype(np.uint8)


__all__ = [
    "Zhu2021IWTSVD",
    "Zhu2021IWTSVDKey",
]
