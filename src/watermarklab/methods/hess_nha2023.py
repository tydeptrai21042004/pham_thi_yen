from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
from scipy import linalg


BLOCK_SIZE = 4
WM_SIZE = 64
ADAPT_T = 15.0
PAPER_T = 65.0
ALPHA = 3.2
ARNOLD_ITER = 10


def _funit8_cast(mat: np.ndarray) -> np.ndarray:
    """Paper Eq. (6): round-half-up with saturation to uint8."""
    x = np.asarray(mat, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.uint8)
    mid = (x > 0.0) & (x < 255.0)
    out[mid] = np.floor(x[mid] + 0.5).astype(np.uint8)
    out[x >= 255.0] = 255
    return out


def _arnold_transform(mat: np.ndarray, iterations: int) -> np.ndarray:
    """Arnold cat-map: (x,y) -> ((x+y) mod N, (x+2y) mod N)."""
    src = np.asarray(mat).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"Arnold transform requires a square 2D array, got {src.shape}")
    n = src.shape[0]
    k = int(iterations)
    if k == 0:
        return src
    # Small watermark size, explicit indexing is clearer and deterministic.
    for _ in range(k):
        dst = np.zeros_like(src)
        for x in range(n):
            for y in range(n):
                dst[(x + y) % n, (x + 2 * y) % n] = src[x, y]
        src = dst
    return src


def _arnold_period(n: int, max_iter: int = 4096) -> int:
    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)
    cur = idx.copy()
    for p in range(1, max_iter + 1):
        cur = _arnold_transform(cur, 1)
        if np.array_equal(cur, idx):
            return p
    raise RuntimeError(f"Cannot find Arnold period up to {max_iter} for n={n}")


def _scramble(wm01: np.ndarray, arnold_iter: int, period: int) -> np.ndarray:
    return _arnold_transform(wm01, int(arnold_iter) % int(period))


def _descramble(wm_scr01: np.ndarray, arnold_iter: int, period: int) -> np.ndarray:
    k = int(arnold_iter) % int(period)
    return _arnold_transform(wm_scr01, (int(period) - k) % int(period))


def _embed_update_h22(h22: float, bit: int, quantization_step: float, alpha: float) -> tuple[float, str]:
    """Nha et al. Eq. (18) and Eq. (19) on H(2,2)."""
    t = float(quantization_step)
    a = float(alpha)
    modv = float(np.mod(float(h22), t))

    if int(bit) == 0 and modv >= 0.5 * t:
        if h22 <= t:
            return 0.5 * t - a, "eq18_const"
        return h22 + (0.5 * t - modv) - a, "eq18_shift"

    if int(bit) == 1 and modv < 0.5 * t:
        if h22 < t:
            return 0.5 * t + a, "eq19_const"
        return h22 + (0.5 * t - modv) + a, "eq19_shift"

    return float(h22), "none"


def _extract_bit_h22(h22_star: float, quantization_step: float) -> tuple[int, float]:
    """Nha et al. Eq. (20)."""
    t = float(quantization_step)
    modv = float(np.mod(float(h22_star), t))
    bit = 0 if modv < 0.5 * t else 1
    confidence = abs(modv - 0.5 * t)
    return int(bit), float(confidence)


@dataclass
class HessNha2023Key:
    """Key/metadata for the blind Hess-Nha2023 baseline.

    Extraction is blind with respect to host and original watermark.  The key only
    stores public algorithm parameters needed to undo Arnold scrambling and to
    interpret the repeated 2D watermark tiling.
    """

    watermark_shape: tuple[int, int]
    block_size: int = BLOCK_SIZE
    wm_size: int = WM_SIZE
    quantization_step: float = ADAPT_T
    alpha: float = ALPHA
    arnold_iter: int = ARNOLD_ITER
    arnold_period: int = 48
    embed_channels: tuple[int, ...] = (2,)
    mode: str = "adapt"
    stats: dict[str, Any] = field(default_factory=dict)


class HessNha2023Hessenberg:
    """Nha et al. 2023 Hessenberg blind watermarking baseline, adapted to 64x64.

    The paper embeds a binary watermark by Arnold scrambling, splitting the host
    image into 4x4 blocks, applying Hessenberg decomposition on the blue channel,
    and quantizing H(2,2).  This implementation keeps that structure but uses the
    project-wide 64x64 binary watermark.  For a 512x512 host, the 128x128 block
    grid repeats each 64x64 watermark bit in 2x2 positions, and extraction uses
    majority voting over these repeated copies.

    Parameters
    ----------
    mode:
        ``adapt`` uses the user's notebook setting T=15 for the 64x64 adapter.
        ``paper``/``original-rerun`` uses the paper's T=65 with the same 64x64
        tiling adapter, because the original paper used 32x32 watermarks.
    """

    name = "HessNha2023_Hessenberg_64x64"

    def __init__(
        self,
        mode: str = "adapt",
        quantization_step: float | None = None,
        alpha: float = ALPHA,
        arnold_iter: int = ARNOLD_ITER,
        block_size: int = BLOCK_SIZE,
        wm_size: int = WM_SIZE,
        embed_channels: tuple[int, ...] | list[int] | None = None,
    ):
        self.mode = str(mode).lower().strip()
        if quantization_step is None:
            self.quantization_step = PAPER_T if self.mode in {"paper", "original", "original-rerun"} else ADAPT_T
        else:
            self.quantization_step = float(quantization_step)
        self.alpha = float(alpha)
        self.arnold_iter = int(arnold_iter)
        self.block_size = int(block_size)
        self.wm_size = int(wm_size)
        # Project arrays are RGB, so blue channel is index 2.  The notebook uses
        # OpenCV BGR index 0; this is the same physical B channel.
        self.embed_channels = tuple(int(c) for c in (embed_channels if embed_channels is not None else (2,)))
        if self.block_size != 4:
            raise ValueError("Hess-Nha2023 baseline uses 4x4 blocks")
        if self.wm_size != 64:
            raise ValueError("This project adapter expects a 64x64 watermark")
        if not self.embed_channels:
            raise ValueError("At least one embed channel must be selected")
        for ch in self.embed_channels:
            if ch not in (0, 1, 2):
                raise ValueError(f"RGB channel index must be 0, 1, or 2; got {ch}")
        self.arnold_period = _arnold_period(self.wm_size)

    def _bit_for_block(self, br: int, bc: int, wm_scr01: np.ndarray) -> int:
        return int(wm_scr01[br % self.wm_size, bc % self.wm_size])

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        host = np.asarray(host_rgb, dtype=np.uint8)
        if host.ndim != 3 or host.shape[2] != 3:
            raise ValueError(f"Host must be RGB HxWx3, got {host.shape}")
        wm = np.asarray(watermark_binary, dtype=np.uint8)
        if wm.shape != (self.wm_size, self.wm_size):
            raise ValueError(f"Hess-Nha2023 adapter expects {self.wm_size}x{self.wm_size} watermark, got {wm.shape}")

        wm01 = (wm > 127).astype(np.uint8)
        wm_scr01 = _scramble(wm01, self.arnold_iter, self.arnold_period)

        out = host.copy()
        h_img, w_img = out.shape[:2]
        br_count = h_img // self.block_size
        bc_count = w_img // self.block_size

        action_counts = {"none": 0, "eq18_const": 0, "eq18_shift": 0, "eq19_const": 0, "eq19_shift": 0}
        changed_blocks = 0
        unchanged_blocks = 0
        linerr_blocks = 0
        ok_after_fail = 0
        sat_elems_low = 0
        sat_elems_high = 0

        for br in range(br_count):
            y = br * self.block_size
            for bc in range(bc_count):
                x = bc * self.block_size
                bit = self._bit_for_block(br, bc, wm_scr01)
                block_changed = False

                for ch in self.embed_channels:
                    block = out[y : y + self.block_size, x : x + self.block_size, ch].astype(np.float64)
                    try:
                        hess, q = linalg.hessenberg(block, calc_q=True)
                    except linalg.LinAlgError:
                        linerr_blocks += 1
                        continue

                    h22_old = float(hess[1, 1])
                    h22_new, action = _embed_update_h22(h22_old, bit, self.quantization_step, self.alpha)
                    action_counts[action] += 1

                    mod_new = float(np.mod(h22_new, self.quantization_step))
                    if (bit == 0 and mod_new >= 0.5 * self.quantization_step) or (bit == 1 and mod_new < 0.5 * self.quantization_step):
                        ok_after_fail += 1

                    if action != "none":
                        hess[1, 1] = h22_new
                        rec = q @ hess @ q.T
                        sat_elems_low += int(np.sum(rec <= 0.0))
                        sat_elems_high += int(np.sum(rec >= 255.0))
                        out[y : y + self.block_size, x : x + self.block_size, ch] = _funit8_cast(rec)
                        block_changed = True

                if block_changed:
                    changed_blocks += 1
                else:
                    unchanged_blocks += 1

        stats = {
            "blocks_total": int(br_count * bc_count),
            "block_grid": [int(br_count), int(bc_count)],
            "votes_per_bit_min": int(min(max(br_count // self.wm_size, 1), max(bc_count // self.wm_size, 1)) ** 2) if br_count >= self.wm_size and bc_count >= self.wm_size else 1,
            "changed_blocks": int(changed_blocks),
            "unchanged_blocks": int(unchanged_blocks),
            "unchanged_pct": float(100.0 * unchanged_blocks / max(br_count * bc_count, 1)),
            "linerr_blocks": int(linerr_blocks),
            "ok_after_fail": int(ok_after_fail),
            "sat_elems_low": int(sat_elems_low),
            "sat_elems_high": int(sat_elems_high),
            "action_counts": dict(action_counts),
        }
        key = HessNha2023Key(
            watermark_shape=wm.shape,
            block_size=self.block_size,
            wm_size=self.wm_size,
            quantization_step=self.quantization_step,
            alpha=self.alpha,
            arnold_iter=self.arnold_iter,
            arnold_period=self.arnold_period,
            embed_channels=self.embed_channels,
            mode=self.mode,
            stats=stats,
        )
        return out, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: HessNha2023Key, host_rgb: np.ndarray | None = None):
        img = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Image must be RGB HxWx3, got {img.shape}")

        bs = int(key.block_size)
        wm_size = int(key.wm_size)
        h_img, w_img = img.shape[:2]
        br_count = h_img // bs
        bc_count = w_img // bs

        bits_block = np.zeros((br_count, bc_count), dtype=np.uint8)
        conf_block = np.zeros((br_count, bc_count), dtype=np.float64)
        linerr_blocks = 0

        for br in range(br_count):
            y = br * bs
            for bc in range(bc_count):
                x = bc * bs
                ch_bits: list[int] = []
                ch_confs: list[float] = []
                for ch in key.embed_channels:
                    block = img[y : y + bs, x : x + bs, ch].astype(np.float64)
                    try:
                        hess, _q = linalg.hessenberg(block, calc_q=True)
                        bit, conf = _extract_bit_h22(float(hess[1, 1]), key.quantization_step)
                    except linalg.LinAlgError:
                        linerr_blocks += 1
                        bit, conf = 0, 0.0
                    ch_bits.append(bit)
                    ch_confs.append(conf)

                ones = int(sum(1 for b in ch_bits if b == 1))
                zeros = len(ch_bits) - ones
                if ones > zeros:
                    bits_block[br, bc] = 1
                elif zeros > ones:
                    bits_block[br, bc] = 0
                else:
                    c1 = float(sum(c for b, c in zip(ch_bits, ch_confs) if b == 1))
                    c0 = float(sum(c for b, c in zip(ch_bits, ch_confs) if b == 0))
                    bits_block[br, bc] = 1 if c1 >= c0 else 0
                conf_block[br, bc] = float(np.mean(ch_confs)) if ch_confs else 0.0

        wm_scr_rec = np.zeros((wm_size, wm_size), dtype=np.uint8)
        for i in range(wm_size):
            for j in range(wm_size):
                votes = bits_block[i::wm_size, j::wm_size].ravel()
                confs = conf_block[i::wm_size, j::wm_size].ravel()
                if votes.size == 0:
                    wm_scr_rec[i, j] = 0
                    continue
                ones = int(np.sum(votes == 1))
                zeros = int(np.sum(votes == 0))
                if ones > zeros:
                    wm_scr_rec[i, j] = 1
                elif zeros > ones:
                    wm_scr_rec[i, j] = 0
                else:
                    c1 = float(np.sum(confs[votes == 1]))
                    c0 = float(np.sum(confs[votes == 0]))
                    wm_scr_rec[i, j] = 1 if c1 >= c0 else 0

        wm01 = _descramble(wm_scr_rec, key.arnold_iter, key.arnold_period)
        out = (wm01 * 255).astype(np.uint8)
        return out[: key.watermark_shape[0], : key.watermark_shape[1]]


__all__ = [
    "HessNha2023Hessenberg",
    "HessNha2023Key",
]
