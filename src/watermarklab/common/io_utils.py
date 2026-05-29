from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

HOST_SIZE = 512
WM_SIZE = 64


def _alpha_composite_rgba_on_white(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def load_host_rgb(path: str | Path, size: int = HOST_SIZE) -> np.ndarray:
    """Load a host image as exact 24-bit RGB uint8 with the expected size."""
    path = Path(path)
    img = Image.open(path)
    if img.mode == "RGBA":
        img = _alpha_composite_rgba_on_white(img)
    else:
        img = img.convert("RGB")
    if img.size != (size, size):
        raise ValueError(f"Host image must be {size}x{size}; got {img.size} for {path}")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Host image must be 24-bit RGB: {path}")
    return arr


def load_watermark_binary(path: str | Path, size: int = WM_SIZE, threshold: int = 127, invert: bool = False) -> np.ndarray:
    """Load watermark as exact binary {0,255} uint8 image.

    RGBA watermarks are composited on white first to avoid losing transparent logos.
    """
    path = Path(path)
    img = Image.open(path)
    if img.mode == "RGBA":
        img = _alpha_composite_rgba_on_white(img)
    else:
        img = img.convert("RGB")
    img = img.resize((size, size), Image.Resampling.NEAREST).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    out = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    if invert:
        out = 255 - out
    return out


def save_image(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(path)
    else:
        raise ValueError(f"Unsupported image shape for saving: {arr.shape}")


def list_image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
