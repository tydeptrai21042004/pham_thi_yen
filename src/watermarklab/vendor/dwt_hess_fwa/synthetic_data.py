"""Generate ad-hoc cover images and binary logos similar to the paper figures.

The original article uses six standard cover images (house, car, Lenna,
peppers, woman, baboon) and six binary logos.  Those exact copyrighted/test-set
images are not bundled here; this module creates deterministic stand-ins with
similar visual structure so the algorithm can be validated end-to-end.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw


def _gradient_bg(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    r = 80 + 80 * x / size + 20 * np.sin(y / 13)
    g = 90 + 90 * y / size + 25 * np.cos(x / 17)
    b = 110 + 60 * np.sin((x + y) / 31)
    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(0, 5, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def make_cover(name: str, size: int = 128, seed: int = 123) -> np.ndarray:
    img = _gradient_bg(size, seed)
    d = ImageDraw.Draw(img)

    if name == "house":
        d.rectangle([28, 60, 100, 108], fill=(210, 180, 125), outline=(70, 55, 40), width=2)
        d.polygon([(22, 60), (64, 25), (106, 60)], fill=(150, 50, 45), outline=(80, 20, 20))
        d.rectangle([55, 78, 73, 108], fill=(90, 55, 35))
        d.rectangle([36, 70, 50, 84], fill=(95, 150, 200))
        d.rectangle([79, 70, 93, 84], fill=(95, 150, 200))
    elif name == "car":
        d.rectangle([18, 78, 108, 97], fill=(35, 110, 190), outline=(10, 40, 80), width=2)
        d.polygon([(38, 78), (55, 57), (84, 57), (99, 78)], fill=(70, 150, 220), outline=(10, 50, 90))
        d.ellipse([29, 91, 45, 107], fill=(25, 25, 25))
        d.ellipse([83, 91, 99, 107], fill=(25, 25, 25))
    elif name == "lena_like":
        d.ellipse([36, 24, 92, 91], fill=(215, 155, 125), outline=(90, 50, 45))
        d.polygon([(22, 43), (64, 10), (110, 48), (83, 56), (50, 38)], fill=(165, 80, 95))
        d.ellipse([51, 50, 57, 56], fill=(25, 25, 25))
        d.ellipse([75, 50, 81, 56], fill=(25, 25, 25))
        d.arc([52, 62, 83, 78], 15, 165, fill=(120, 40, 55), width=2)
    elif name == "peppers":
        for box, color in [([22, 36, 70, 102], (220, 30, 35)), ([58, 28, 108, 99], (40, 170, 55)), ([44, 58, 95, 114], (230, 200, 35))]:
            d.ellipse(box, fill=color, outline=(45, 70, 30), width=2)
        d.line([67, 29, 70, 18, 79, 12], fill=(40, 90, 35), width=3)
    elif name == "woman_like":
        d.ellipse([38, 22, 92, 83], fill=(190, 130, 95))
        d.arc([22, 10, 108, 118], 110, 435, fill=(45, 25, 20), width=15)
        d.rectangle([45, 82, 86, 122], fill=(130, 45, 75))
        d.ellipse([54, 48, 60, 54], fill=(10, 10, 10))
        d.ellipse([75, 48, 81, 54], fill=(10, 10, 10))
    elif name == "baboon_like":
        # High-texture animal-like image with strong red/blue face cues.
        arr = np.asarray(img).astype(np.float64)
        rng = np.random.default_rng(seed + 99)
        noise = rng.normal(0, 45, arr.shape[:2])
        arr[..., 0] += noise + 40 * np.sin(np.arange(size)[None, :] / 2)
        arr[..., 1] += noise * 0.4
        arr[..., 2] += -noise + 40 * np.cos(np.arange(size)[:, None] / 3)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img)
        d.ellipse([30, 30, 98, 112], fill=(125, 95, 80), outline=(50, 35, 35), width=2)
        d.ellipse([50, 55, 78, 106], fill=(210, 60, 70))
        d.ellipse([35, 45, 53, 63], fill=(35, 50, 110))
        d.ellipse([76, 45, 94, 63], fill=(35, 50, 110))
    else:
        raise ValueError(f"unknown cover name: {name}")

    return np.asarray(img, dtype=np.uint8)


def make_logo(index: int, size: int = 32) -> np.ndarray:
    img = Image.new("1", (size, size), 0)
    d = ImageDraw.Draw(img)
    s = size
    if index == 0:
        d.rectangle([5, 7, 21, 21], outline=1, width=3)
        d.line([13, 22, 10, 27, 24, 27], fill=1, width=2)
    elif index == 1:
        d.polygon([(s // 2, 4), (5, s - 5), (s - 5, s - 5)], fill=1)
        d.polygon([(s // 2, 10), (12, s - 8), (s - 12, s - 8)], fill=0)
    elif index == 2:
        for k in range(7):
            x0 = s // 2
            y0 = s - 4
            angle = -math.pi + k * math.pi / 6
            d.polygon([(x0, y0), (x0 + int(26 * math.cos(angle)), y0 + int(26 * math.sin(angle))), (x0 + int(5 * math.cos(angle+0.12)), y0 + int(5 * math.sin(angle+0.12)))], fill=1)
        d.ellipse([8, 18, 24, 30], outline=1, width=2)
    elif index == 3:
        d.ellipse([5, 5, s - 5, s - 5], fill=1)
        d.text((10, 10), "hp", fill=0)
    elif index == 4:
        d.polygon([(s // 2, 4), (5, s - 6), (s - 5, s - 6)], outline=1, width=2)
        d.line([s // 2, 4, s // 2, s - 6], fill=1, width=2)
        d.arc([8, 10, s - 8, s + 4], 200, 340, fill=1, width=2)
    elif index == 5:
        d.ellipse([6, 6, s - 6, s - 2], fill=1)
        d.rectangle([6, 16, s - 6, s - 2], fill=1)
        d.ellipse([11, 12, s - 11, 22], fill=0)
        d.rectangle([s // 2 - 2, 2, s // 2 + 2, 9], fill=1)
    else:
        raise ValueError("logo index must be 0..5")
    return np.asarray(img, dtype=np.uint8)


def save_demo_images(out_dir: str | Path, cover_size: int = 128, logo_size: int = 32) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    covers = ["house", "car", "lena_like", "peppers", "woman_like", "baboon_like"]
    for i, name in enumerate(covers):
        Image.fromarray(make_cover(name, cover_size, seed=100 + i)).save(out / f"cover_{i+1}_{name}.png")
    for i in range(6):
        Image.fromarray((make_logo(i, logo_size) * 255).astype(np.uint8)).save(out / f"logo_{i+1}.png")
