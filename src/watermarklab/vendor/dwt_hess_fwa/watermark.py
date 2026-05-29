"""DWT + Hessenberg watermark embed/extract implementation.

This module follows the algorithmic path described by Gaata et al. (2022):
RGB split -> 2D-DWT -> discard LL -> build embedding matrix from detail bands
-> add generated keys -> 4x4 blocks -> Hessenberg factorization -> parity
embedding in one H-matrix coefficient -> inverse Hessenberg -> IDWT.

Important reproducibility note
------------------------------
The article does not fully specify the wavelet family, chaotic-map equations,
selected decimal digit, exact attack parameters, or exact FWA iteration count.
Those are exposed as configuration values here.  Defaults are chosen so that
algorithmic tests are deterministic and the pipeline is paper-faithful.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Tuple
import json
from pathlib import Path
import numpy as np
from scipy.linalg import hessenberg

from .dwt import haar_dwt2, haar_idwt2
from .metrics import mse, psnr


DETAIL_ORDER = ("HH", "LH", "HL")


@dataclass
class WatermarkConfig:
    block_size: int = 4
    h_position: Tuple[int, int] = (3, 3)  # zero-based version of paper's example (4,4)
    decimal_position: int = 3             # digit after decimal point used for parity
    key_strength: float = 0.020           # chaotic key amplitude added to EM
    key_params: Tuple[float, float, float, float] = (0.21, 0.37, 4.90, 0.18)
    use_min_shift: bool = True
    clip_output: bool = True

    def to_json(self) -> str:
        d = asdict(self)
        d["h_position"] = list(self.h_position)
        d["key_params"] = list(self.key_params)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "WatermarkConfig":
        d = json.loads(s)
        d["h_position"] = tuple(d["h_position"])
        d["key_params"] = tuple(d["key_params"])
        return WatermarkConfig(**d)


@dataclass
class EmbedMetadata:
    config: WatermarkConfig
    original_shape: Tuple[int, int, int]
    cropped_shape: Tuple[int, int, int]
    watermark_shape: Tuple[int, int]
    embedding_matrix_shape: Tuple[int, int]
    capacity_bits: int
    used_bits: int
    min_shift: float
    changed_blocks: int
    total_blocks_used: int
    changed_rate_percent: float
    psnr_float: float
    mse_float: float

    def to_json(self) -> str:
        d = asdict(self)
        d["config"] = asdict(self.config)
        d["config"]["h_position"] = list(self.config.h_position)
        d["config"]["key_params"] = list(self.config.key_params)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "EmbedMetadata":
        d = json.loads(s)
        d["config"]["h_position"] = tuple(d["config"]["h_position"])
        d["config"]["key_params"] = tuple(d["config"]["key_params"])
        cfg = WatermarkConfig(**d.pop("config"))
        return EmbedMetadata(config=cfg, **d)


def _crop_rgb_even(image_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(image_rgb, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("cover image must be RGB with shape HxWx3")
    h, w, _ = img.shape
    return img[: h - (h % 2), : w - (w % 2), :]


def split_rgb_dwt(image_rgb: np.ndarray) -> list[dict[str, np.ndarray]]:
    img = _crop_rgb_even(image_rgb)
    return [haar_dwt2(img[..., c]) for c in range(3)]


def build_embedding_matrix(bands_by_channel: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Stack/concatenate detail bands as described in Section 3.3."""
    channel_rows = []
    for bands in bands_by_channel:
        channel_rows.append(np.concatenate([bands[k] for k in DETAIL_ORDER], axis=1))
    return np.concatenate(channel_rows, axis=0).astype(np.float64)


def put_embedding_matrix_back(
    bands_by_channel: list[dict[str, np.ndarray]], em: np.ndarray
) -> list[dict[str, np.ndarray]]:
    """Inverse of build_embedding_matrix; LL bands are preserved."""
    h2, w2 = bands_by_channel[0]["LL"].shape
    out = []
    for c in range(3):
        row = em[c * h2 : (c + 1) * h2, :]
        new_bands = {"LL": bands_by_channel[c]["LL"].copy()}
        for j, key in enumerate(DETAIL_ORDER):
            new_bands[key] = row[:, j * w2 : (j + 1) * w2]
        out.append(new_bands)
    return out


def reconstruct_rgb_from_bands(bands_by_channel: list[dict[str, np.ndarray]]) -> np.ndarray:
    chans = [haar_idwt2(bands) for bands in bands_by_channel]
    return np.stack(chans, axis=-1)


def hybrid_chaotic_keys(shape: Tuple[int, int], params: Tuple[float, float, float, float], strength: float) -> np.ndarray:
    """Generate deterministic hybrid Gaussian/exponential chaotic keys.

    Assumed equations due to missing equations in the paper:
    Gaussian map:    x[n+1] = exp(-a*x[n]^2) + b   (fractional part)
    Exponential map: y[n+1] = r*y[n]*exp(-y[n]) + c (fractional part)
    key[n] = strength * (0.5*x[n] + 0.5*y[n])
    """
    x0, y0, r, b = params
    n = int(np.prod(shape))
    keys = np.empty(n, dtype=np.float64)
    x = float(x0 % 1.0)
    y = float(y0 % 1.0)
    a = 4.0 + 0.5 * abs(float(r))
    c = float(b % 1.0)
    for i in range(n):
        x = (np.exp(-a * x * x) + c) % 1.0
        y = (r * y * np.exp(-y) + c) % 1.0
        keys[i] = strength * (0.5 * x + 0.5 * y)
    return keys.reshape(shape)


def _decimal_digit(value: float, decimal_position: int) -> int:
    scale = 10 ** decimal_position
    return int(np.floor(abs(value) * scale + 1e-8)) % 10


def _desired_parity_ok(value: float, bit: int, decimal_position: int) -> bool:
    return (_decimal_digit(value, decimal_position) % 2) == int(bit)


def _force_parity(value: float, bit: int, decimal_position: int) -> float:
    """Change one decimal digit by +/-1 unit at the chosen decimal position."""
    if _desired_parity_ok(value, bit, decimal_position):
        return value
    delta = 1.0 / (10 ** decimal_position)
    sign = 1.0 if value >= 0 else -1.0
    # Paper wording: for secret bit=1 and even selected digit, add one; for
    # secret bit=0 and odd selected digit, subtract one.  Here "one" means one
    # unit in the selected decimal digit.
    if int(bit) == 1:
        return value + sign * delta
    return value - sign * delta


def _iter_block_slices(shape: Tuple[int, int], block_size: int, limit_bits: int | None = None):
    h, w = shape
    count = 0
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            if limit_bits is not None and count >= limit_bits:
                return
            yield count, slice(r, r + block_size), slice(c, c + block_size)
            count += 1


def capacity_bits_for_em(shape: Tuple[int, int], block_size: int) -> int:
    return (shape[0] // block_size) * (shape[1] // block_size)


def embed_bits_in_em(em_keyed: np.ndarray, bits: np.ndarray, config: WatermarkConfig) -> tuple[np.ndarray, int]:
    """Apply Hessenberg block embedding to a keyed embedding matrix.

    A practical detail is included for numerical reproducibility: after a block
    is reconstructed as Q @ H @ Q.T, SciPy's Hessenberg factorization of that
    reconstructed block may choose slightly different signs in Q.  Therefore the
    implementation verifies the parity after re-factorization and, if needed,
    tries a few one-digit increments until the extraction rule reads the desired
    bit.  This keeps the observable extraction rule identical to the paper while
    making the tests deterministic.
    """
    out = np.array(em_keyed, dtype=np.float64, copy=True)
    flat_bits = np.asarray(bits, dtype=np.uint8).ravel()
    b = config.block_size
    pos = config.h_position
    changed_blocks = 0

    if pos[0] >= b or pos[1] >= b:
        raise ValueError("h_position must lie inside the block")
    if len(flat_bits) > capacity_bits_for_em(out.shape, b):
        raise ValueError("watermark too large for cover image and block size")

    delta = 1.0 / (10 ** config.decimal_position)

    for idx, rs, cs in _iter_block_slices(out.shape, b, len(flat_bits)):
        bit = int(flat_bits[idx])
        block = out[rs, cs]
        H, Q = hessenberg(block, calc_q=True)

        # If extraction from this block already gives the desired bit, leave it
        # unchanged exactly as the paper describes.
        if _desired_parity_ok(H[pos], bit, config.decimal_position):
            continue

        old_value = float(H[pos])
        sign = 1.0 if old_value >= 0 else -1.0
        paper_direction = sign if bit == 1 else -sign
        directions = [paper_direction, -paper_direction]
        best_block = None
        best_error = float("inf")

        # Try one-decimal-digit edits, starting with the direction prescribed by
        # the article.  The loop is rarely longer than one or two steps.
        for direction in directions:
            for k in range(1, 41):
                H_try = H.copy()
                H_try[pos] = old_value + direction * delta * k
                candidate = Q @ H_try @ Q.T
                H_check, _ = hessenberg(candidate, calc_q=True)
                if _desired_parity_ok(H_check[pos], bit, config.decimal_position):
                    err = float(np.linalg.norm(candidate - block))
                    if err < best_error:
                        best_error = err
                        best_block = candidate
                    break

        if best_block is None:
            # Last-resort fallback: direct paper edit. This should not normally
            # happen, but keeps the function total for pathological matrices.
            H[pos] = _force_parity(old_value, bit, config.decimal_position)
            best_block = Q @ H @ Q.T

        changed_blocks += 1
        out[rs, cs] = best_block
    return out, changed_blocks


def extract_bits_from_em(em_keyed: np.ndarray, num_bits: int, config: WatermarkConfig) -> np.ndarray:
    bits = np.empty(num_bits, dtype=np.uint8)
    b = config.block_size
    pos = config.h_position
    for idx, rs, cs in _iter_block_slices(em_keyed.shape, b, num_bits):
        block = em_keyed[rs, cs]
        H, _ = hessenberg(block, calc_q=True)
        bits[idx] = _decimal_digit(H[pos], config.decimal_position) % 2
    return bits


def embed_watermark(
    image_rgb: np.ndarray,
    watermark_bits_2d: np.ndarray,
    config: WatermarkConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, EmbedMetadata]:
    """Embed a binary watermark into an RGB image.

    Returns (watermarked_float, watermarked_uint8, metadata).
    """
    config = config or WatermarkConfig()
    original_shape = tuple(np.asarray(image_rgb).shape)
    cover = _crop_rgb_even(image_rgb)
    watermark_bits = (np.asarray(watermark_bits_2d) > 0).astype(np.uint8)

    bands = split_rgb_dwt(cover)
    em = build_embedding_matrix(bands)
    capacity = capacity_bits_for_em(em.shape, config.block_size)
    if watermark_bits.size > capacity:
        raise ValueError(f"watermark has {watermark_bits.size} bits but capacity is {capacity}")

    min_shift = float(abs(np.min(em))) if config.use_min_shift else 0.0
    em_positive = em + min_shift
    keys = hybrid_chaotic_keys(em.shape, config.key_params, config.key_strength)
    em_keyed = em_positive + keys
    em_mod_keyed, changed_blocks = embed_bits_in_em(em_keyed, watermark_bits.ravel(), config)

    # Reverse process: subtract keys and minimum shift, rebuild DWT detail bands,
    # then apply inverse 2D-DWT on each channel.
    em_mod = em_mod_keyed - keys - min_shift
    mod_bands = put_embedding_matrix_back(bands, em_mod)
    watermarked_float = reconstruct_rgb_from_bands(mod_bands)
    if config.clip_output:
        watermarked_float = np.clip(watermarked_float, 0, 255)
    watermarked_uint8 = np.rint(np.clip(watermarked_float, 0, 255)).astype(np.uint8)

    m = mse(cover, watermarked_float)
    p = psnr(cover, watermarked_float)
    metadata = EmbedMetadata(
        config=config,
        original_shape=original_shape,
        cropped_shape=tuple(cover.shape),
        watermark_shape=tuple(watermark_bits.shape),
        embedding_matrix_shape=tuple(em.shape),
        capacity_bits=capacity,
        used_bits=int(watermark_bits.size),
        min_shift=min_shift,
        changed_blocks=int(changed_blocks),
        total_blocks_used=int(watermark_bits.size),
        changed_rate_percent=100.0 * float(changed_blocks) / float(watermark_bits.size),
        psnr_float=p,
        mse_float=m,
    )
    return watermarked_float, watermarked_uint8, metadata


def extract_watermark(
    watermarked_rgb: np.ndarray,
    watermark_shape: Tuple[int, int],
    config: WatermarkConfig,
    min_shift: float | None = None,
) -> np.ndarray:
    """Extract a binary watermark.

    By default, this follows the article's extraction description and recomputes
    the minimum-shift from the watermarked embedding matrix.  For exact floating
    validation the caller may pass the original min_shift from the metadata.
    """
    img = _crop_rgb_even(watermarked_rgb)
    bands = split_rgb_dwt(img)
    em = build_embedding_matrix(bands)
    shift = float(abs(np.min(em))) if min_shift is None and config.use_min_shift else float(min_shift or 0.0)
    keys = hybrid_chaotic_keys(em.shape, config.key_params, config.key_strength)
    em_keyed = em + shift + keys
    bits = extract_bits_from_em(em_keyed, int(np.prod(watermark_shape)), config)
    return bits.reshape(watermark_shape)
