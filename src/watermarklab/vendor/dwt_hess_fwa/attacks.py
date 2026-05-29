"""Simple attack operators for robustness testing."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(img, 0, 255)).astype(np.uint8)


def jpeg_compression(img: np.ndarray, quality: int = 70) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), "RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def scaling(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), "RGB")
    w, h = pil.size
    small = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
    back = small.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(back, dtype=np.uint8)


def rotation(img: np.ndarray, angle: float = 3.0) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), "RGB")
    rot = pil.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return np.asarray(rot, dtype=np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float = 2.0, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.asarray(img, dtype=np.float64) + rng.normal(0, sigma, np.asarray(img).shape)
    return to_uint8(noisy)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), "RGB")
    ycbcr = pil.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_eq = ImageOps.equalize(y)
    return np.asarray(Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB"), dtype=np.uint8)


def image_adjustment(img: np.ndarray, brightness: float = 1.05, contrast: float = 1.05) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), "RGB")
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    return np.asarray(pil, dtype=np.uint8)


def all_attacks(img: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "none_uint8": to_uint8(img),
        "jpeg_q70": jpeg_compression(img, quality=70),
        "scaling_075": scaling(img, scale=0.75),
        "rotation_3deg": rotation(img, angle=3.0),
        "gaussian_sigma2": gaussian_noise(img, sigma=2.0),
        "hist_equalization": histogram_equalization(img),
        "image_adjustment": image_adjustment(img),
    }
