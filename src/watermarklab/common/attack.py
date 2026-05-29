from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
import io
import math
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

Array = np.ndarray


@dataclass(frozen=True)
class AttackConfig:
    """Deterministic attack configuration.

    name: unique attack id used in result CSV.
    group: function group key.
    params: keyword arguments for the attack function.
    """

    name: str
    group: str
    params: dict


def _u8(img: Array) -> Array:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Attack image must be HxWx3 RGB; got {img.shape}")
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def no_attack(image: Array, **kwargs) -> Array:
    return _u8(image).copy()


def jpeg(image: Array, quality: int = 70, **kwargs) -> Array:
    quality = int(np.clip(int(quality), 1, 100))
    buf = io.BytesIO()
    Image.fromarray(_u8(image), mode="RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def jpeg2000(image: Array, quality_layer: float = 7.0, quality: int | None = None, **kwargs) -> Array:
    """JPEG2000 compression attack.

    Pillow support for JPEG2000 varies by environment. When JP2 cannot be
    encoded/decoded, this falls back to JPEG at a comparable quality, keeping the
    benchmark deterministic and testable.
    """
    try:
        buf = io.BytesIO()
        Image.fromarray(_u8(image), mode="RGB").save(
            buf,
            format="JPEG2000",
            quality_layers=[float(quality_layer)],
        )
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    except Exception:
        q = int(quality) if quality is not None else int(np.clip(100 - 8 * float(quality_layer), 10, 95))
        return jpeg(image, quality=q)


def gaussian_noise(image: Array, sigma: float = 5.0, seed: int = 123, **kwargs) -> Array:
    rng = np.random.default_rng(int(seed))
    img = _u8(image).astype(np.float64)
    return _u8(img + rng.normal(0.0, float(sigma), img.shape))


def salt_pepper(image: Array, amount: float = 0.005, seed: int = 123, **kwargs) -> Array:
    rng = np.random.default_rng(int(seed))
    out = _u8(image).copy()
    amount = float(np.clip(amount, 0.0, 1.0))
    mask = rng.random(out.shape[:2])
    out[mask < amount / 2.0] = 0
    out[(mask >= amount / 2.0) & (mask < amount)] = 255
    return out


def speckle_noise(image: Array, variance: float = 0.005, seed: int = 123, **kwargs) -> Array:
    rng = np.random.default_rng(int(seed))
    img = _u8(image).astype(np.float64) / 255.0
    noisy = img + img * rng.normal(0.0, math.sqrt(max(float(variance), 0.0)), img.shape)
    return _u8(noisy * 255.0)


def poisson_noise(image: Array, peak: float = 255.0, seed: int = 123, **kwargs) -> Array:
    """Signal-dependent Poisson noise with deterministic RNG."""
    rng = np.random.default_rng(int(seed))
    peak = max(float(peak), 1.0)
    img = _u8(image).astype(np.float64) / 255.0
    noisy = rng.poisson(img * peak) / peak
    return _u8(noisy * 255.0)


def median_filter(image: Array, size: int = 3, **kwargs) -> Array:
    size = int(size)
    if size % 2 == 0 or size < 1:
        raise ValueError("Median filter size must be a positive odd integer")
    return np.asarray(Image.fromarray(_u8(image)).filter(ImageFilter.MedianFilter(size=size)), dtype=np.uint8)


def average_filter(image: Array, size: int = 3, **kwargs) -> Array:
    size = max(int(size), 1)
    return np.asarray(Image.fromarray(_u8(image)).filter(ImageFilter.BoxBlur(radius=(size - 1) / 2.0)), dtype=np.uint8)


def gaussian_blur(image: Array, radius: float = 1.0, **kwargs) -> Array:
    return np.asarray(Image.fromarray(_u8(image)).filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.uint8)


def lowpass(image: Array, size: int = 5, **kwargs) -> Array:
    return average_filter(image, size=size)


def sharpen(image: Array, factor: float = 2.0, **kwargs) -> Array:
    return np.asarray(ImageEnhance.Sharpness(Image.fromarray(_u8(image))).enhance(float(factor)), dtype=np.uint8)


def unsharp_mask(image: Array, radius: float = 2.0, percent: int = 150, threshold: int = 3, **kwargs) -> Array:
    pil = Image.fromarray(_u8(image)).filter(
        ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=int(threshold))
    )
    return np.asarray(pil, dtype=np.uint8)


def hist_equalization(image: Array, **kwargs) -> Array:
    pil = Image.fromarray(_u8(image)).convert("YCbCr")
    y, cb, cr = pil.split()
    out = Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")
    return np.asarray(out, dtype=np.uint8)


def clahe_like(image: Array, **kwargs) -> Array:
    """Local-contrast style attack without OpenCV dependency.

    This is not exact CLAHE, but it stresses luminance redistribution locally by
    autocontrast on the Y channel.
    """
    pil = Image.fromarray(_u8(image)).convert("YCbCr")
    y, cb, cr = pil.split()
    y = ImageOps.autocontrast(y)
    return np.asarray(Image.merge("YCbCr", (y, cb, cr)).convert("RGB"), dtype=np.uint8)


def rotate_keep_size(image: Array, degrees: float = 5.0, fill: int = 0, **kwargs) -> Array:
    pil = Image.fromarray(_u8(image))
    return np.asarray(
        pil.rotate(float(degrees), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(fill, fill, fill)),
        dtype=np.uint8,
    )


def translate(image: Array, shift_x: int = 5, shift_y: int = 5, fill: int = 0, **kwargs) -> Array:
    img = _u8(image)
    h, w = img.shape[:2]
    out = np.full_like(img, int(fill))
    sx, sy = int(shift_x), int(shift_y)
    src_x0 = max(0, -sx)
    src_x1 = min(w, w - sx) if sx >= 0 else w
    dst_x0 = max(0, sx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    src_y0 = max(0, -sy)
    src_y1 = min(h, h - sy) if sy >= 0 else h
    dst_y0 = max(0, sy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def resize_attack(image: Array, factor: float = 0.5, **kwargs) -> Array:
    img = _u8(image)
    h, w = img.shape[:2]
    factor = max(float(factor), 1e-3)
    small = Image.fromarray(img).resize(
        (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(small.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def crop_resize(image: Array, keep: float = 0.9, seed: int | None = None, random_crop: bool = False, **kwargs) -> Array:
    img = _u8(image)
    h, w = img.shape[:2]
    keep = float(np.clip(keep, 1e-3, 1.0))
    ch = max(1, int(round(h * keep)))
    cw = max(1, int(round(w * keep)))
    if random_crop:
        rng = np.random.default_rng(int(seed) if seed is not None else 123)
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
    else:
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
    crop = Image.fromarray(img[y0 : y0 + ch, x0 : x0 + cw])
    return np.asarray(crop.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def occlusion(image: Array, block: int = 64, seed: int = 123, value: int = 0, num_blocks: int = 1, **kwargs) -> Array:
    rng = np.random.default_rng(int(seed))
    out = _u8(image).copy()
    h, w = out.shape[:2]
    block = min(max(int(block), 1), h, w)
    for _ in range(max(int(num_blocks), 1)):
        y0 = int(rng.integers(0, h - block + 1))
        x0 = int(rng.integers(0, w - block + 1))
        out[y0 : y0 + block, x0 : x0 + block] = int(value)
    return out


def gamma_correction(image: Array, gamma: float = 1.2, **kwargs) -> Array:
    img = _u8(image).astype(np.float64) / 255.0
    return _u8(np.power(img, float(gamma)) * 255.0)


def brightness(image: Array, factor: float = 1.1, **kwargs) -> Array:
    return np.asarray(ImageEnhance.Brightness(Image.fromarray(_u8(image))).enhance(float(factor)), dtype=np.uint8)


def contrast(image: Array, factor: float = 1.1, **kwargs) -> Array:
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(_u8(image))).enhance(float(factor)), dtype=np.uint8)


def bit_depth_reduction(image: Array, bits: int = 5, **kwargs) -> Array:
    bits = int(np.clip(int(bits), 1, 8))
    levels = 2**bits - 1
    img = _u8(image).astype(np.float64) / 255.0
    return _u8(np.round(img * levels) / levels * 255.0)


def motion_blur(image: Array, size: int = 9, angle: str = "horizontal", **kwargs) -> Array:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float64)
    if str(angle).lower().startswith("v"):
        kernel[:, size // 2] = 1.0 / size
    else:
        kernel[size // 2, :] = 1.0 / size
    img = _u8(image).astype(np.float64)
    # Lightweight convolution with edge padding, no scipy dependency here.
    pad = size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(size):
        for x in range(size):
            if kernel[y, x] != 0:
                out += kernel[y, x] * padded[y : y + img.shape[0], x : x + img.shape[1]]
    return _u8(out)


def shear_affine(image: Array, shear_x: float = 0.08, shear_y: float = 0.0, fill: int = 0, **kwargs) -> Array:
    """Small affine shear attack, resized back to the original canvas.

    This stresses watermark synchronization without changing the benchmark
    image size. PIL uses inverse affine coefficients: x_in = a*x + b*y + c.
    """
    img = _u8(image)
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    coeffs = (1.0, -float(shear_x), 0.0, -float(shear_y), 1.0, 0.0)
    return np.asarray(
        pil.transform((w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=(fill, fill, fill)),
        dtype=np.uint8,
    )


def row_col_delete(image: Array, rows: int = 4, cols: int = 4, seed: int = 123, **kwargs) -> Array:
    """Delete random rows/columns and resize back, similar to line/column deletion attacks."""
    img = _u8(image)
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(seed))
    rows = int(np.clip(rows, 0, max(h - 1, 0)))
    cols = int(np.clip(cols, 0, max(w - 1, 0)))
    keep_rows = np.ones(h, dtype=bool)
    keep_cols = np.ones(w, dtype=bool)
    if rows > 0:
        keep_rows[rng.choice(h, size=rows, replace=False)] = False
    if cols > 0:
        keep_cols[rng.choice(w, size=cols, replace=False)] = False
    reduced = img[keep_rows][:, keep_cols]
    return np.asarray(Image.fromarray(reduced).resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def mosaic_pixelate(image: Array, factor: int = 8, **kwargs) -> Array:
    """Pixelation/mosaic attack by downsampling with nearest-neighbour then upsampling."""
    img = _u8(image)
    h, w = img.shape[:2]
    factor = max(1, int(factor))
    small = Image.fromarray(img).resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.NEAREST)
    return np.asarray(small.resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)


def posterize_attack(image: Array, bits: int = 4, **kwargs) -> Array:
    bits = int(np.clip(int(bits), 1, 8))
    return np.asarray(ImageOps.posterize(Image.fromarray(_u8(image)), bits), dtype=np.uint8)


def solarize_attack(image: Array, threshold: int = 128, **kwargs) -> Array:
    return np.asarray(ImageOps.solarize(Image.fromarray(_u8(image)), threshold=int(np.clip(threshold, 0, 255))), dtype=np.uint8)


def saturation_attack(image: Array, factor: float = 0.6, **kwargs) -> Array:
    return np.asarray(ImageEnhance.Color(Image.fromarray(_u8(image))).enhance(float(factor)), dtype=np.uint8)


def channel_dropout(image: Array, channel: int = 2, value: int = 0, **kwargs) -> Array:
    out = _u8(image).copy()
    ch = int(np.clip(int(channel), 0, 2))
    out[:, :, ch] = int(np.clip(int(value), 0, 255))
    return out


def checkerboard_cutout(image: Array, tile: int = 32, value: int = 0, **kwargs) -> Array:
    """Structured occlusion attack: replace every other tile with a constant."""
    out = _u8(image).copy()
    tile = max(1, int(tile))
    h, w = out.shape[:2]
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((y // tile) + (x // tile)) % 2 == 0:
                out[y:y+tile, x:x+tile] = int(np.clip(int(value), 0, 255))
    return out


def border_crop_pad(image: Array, pixels: int = 16, value: int = 0, **kwargs) -> Array:
    """Remove a border and pad with a constant to keep the original size."""
    img = _u8(image)
    h, w = img.shape[:2]
    p = int(np.clip(int(pixels), 0, min(h, w) // 2 - 1))
    if p <= 0:
        return img.copy()
    out = np.full_like(img, int(np.clip(value, 0, 255)))
    out[p:h-p, p:w-p] = img[p:h-p, p:w-p]
    return out


def color_quantization(image: Array, colors: int = 64, **kwargs) -> Array:
    """Palette quantization attack using PIL adaptive quantization."""
    colors = int(np.clip(int(colors), 2, 256))
    pil = Image.fromarray(_u8(image)).convert('RGB')
    q = pil.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return np.asarray(q.convert('RGB'), dtype=np.uint8)


_ATTACK_FUNCS: dict[str, Callable[..., Array]] = {
    "none": no_attack,
    "jpeg": jpeg,
    "jpeg2000": jpeg2000,
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper,
    "speckle_noise": speckle_noise,
    "poisson_noise": poisson_noise,
    "median_filter": median_filter,
    "average_filter": average_filter,
    "gaussian_blur": gaussian_blur,
    "lowpass": lowpass,
    "sharpen": sharpen,
    "unsharp_mask": unsharp_mask,
    "hist_equalization": hist_equalization,
    "clahe_like": clahe_like,
    "rotation": rotate_keep_size,
    "translation": translate,
    "resize": resize_attack,
    "crop_resize": crop_resize,
    "occlusion": occlusion,
    "gamma": gamma_correction,
    "brightness": brightness,
    "contrast": contrast,
    "bit_depth": bit_depth_reduction,
    "motion_blur": motion_blur,
    "shear": shear_affine,
    "row_col_delete": row_col_delete,
    "mosaic": mosaic_pixelate,
    "posterize": posterize_attack,
    "solarize": solarize_attack,
    "saturation": saturation_attack,
    "channel_dropout": channel_dropout,
    "checkerboard_cutout": checkerboard_cutout,
    "border_crop_pad": border_crop_pad,
    "color_quantization": color_quantization,
}


def available_attack_groups() -> tuple[str, ...]:
    return tuple(sorted(_ATTACK_FUNCS.keys()))


def apply_attack(image: Array, cfg: AttackConfig) -> Array:
    if cfg.group not in _ATTACK_FUNCS:
        raise ValueError(f"Unknown attack group: {cfg.group}")
    input_u8 = _u8(image)
    out = _ATTACK_FUNCS[cfg.group](input_u8, **dict(cfg.params))
    out = _u8(out)
    if out.shape != input_u8.shape:
        raise AssertionError(f"Attack changed shape: {input_u8.shape} -> {out.shape}")
    return out


def lite_attack_suite(include_none: bool = True) -> list[AttackConfig]:
    attacks = [
        AttackConfig("no_attack", "none", {}),
        AttackConfig("jpeg_q90", "jpeg", {"quality": 90}),
        AttackConfig("jpeg_q70", "jpeg", {"quality": 70}),
        AttackConfig("gaussian_noise_sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}),
        AttackConfig("salt_pepper_0p005", "salt_pepper", {"amount": 0.005, "seed": 123}),
        AttackConfig("median_3x3", "median_filter", {"size": 3}),
        AttackConfig("gaussian_blur_r1", "gaussian_blur", {"radius": 1.0}),
        AttackConfig("hist_equalization", "hist_equalization", {}),
        AttackConfig("rotation_2deg", "rotation", {"degrees": 2.0}),
        AttackConfig("crop_resize_90", "crop_resize", {"keep": 0.90}),
        AttackConfig("gamma_1p2", "gamma", {"gamma": 1.2}),
    ]
    return attacks if include_none else attacks[1:]


def full_attack_suite(include_none: bool = True) -> list[AttackConfig]:
    attacks = [
        AttackConfig("no_attack", "none", {}),
        # Compression
        AttackConfig("jpeg_q95", "jpeg", {"quality": 95}),
        AttackConfig("jpeg_q90", "jpeg", {"quality": 90}),
        AttackConfig("jpeg_q70", "jpeg", {"quality": 70}),
        AttackConfig("jpeg_q50", "jpeg", {"quality": 50}),
        AttackConfig("jpeg_q30", "jpeg", {"quality": 30}),
        AttackConfig("jpeg2000_q7", "jpeg2000", {"quality_layer": 7.0}),
        # Noise
        AttackConfig("gaussian_noise_sigma2", "gaussian_noise", {"sigma": 2.0, "seed": 123}),
        AttackConfig("gaussian_noise_sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}),
        AttackConfig("gaussian_noise_sigma10", "gaussian_noise", {"sigma": 10.0, "seed": 123}),
        AttackConfig("salt_pepper_0p002", "salt_pepper", {"amount": 0.002, "seed": 123}),
        AttackConfig("salt_pepper_0p005", "salt_pepper", {"amount": 0.005, "seed": 123}),
        AttackConfig("salt_pepper_0p01", "salt_pepper", {"amount": 0.01, "seed": 123}),
        AttackConfig("speckle_var_0p001", "speckle_noise", {"variance": 0.001, "seed": 123}),
        AttackConfig("speckle_var_0p005", "speckle_noise", {"variance": 0.005, "seed": 123}),
        AttackConfig("poisson_peak_64", "poisson_noise", {"peak": 64.0, "seed": 123}),
        # Filters and enhancement
        AttackConfig("median_3x3", "median_filter", {"size": 3}),
        AttackConfig("median_5x5", "median_filter", {"size": 5}),
        AttackConfig("average_3x3", "average_filter", {"size": 3}),
        AttackConfig("average_5x5", "average_filter", {"size": 5}),
        AttackConfig("lowpass_5x5", "lowpass", {"size": 5}),
        AttackConfig("gaussian_blur_r1", "gaussian_blur", {"radius": 1.0}),
        AttackConfig("gaussian_blur_r2", "gaussian_blur", {"radius": 2.0}),
        AttackConfig("motion_blur_h9", "motion_blur", {"size": 9, "angle": "horizontal"}),
        AttackConfig("sharpen_1p5", "sharpen", {"factor": 1.5}),
        AttackConfig("unsharp_mask", "unsharp_mask", {"radius": 2.0, "percent": 150, "threshold": 3}),
        AttackConfig("hist_equalization", "hist_equalization", {}),
        AttackConfig("clahe_like", "clahe_like", {}),
        # Geometric and cropping
        AttackConfig("rotation_1deg", "rotation", {"degrees": 1.0}),
        AttackConfig("rotation_2deg", "rotation", {"degrees": 2.0}),
        AttackConfig("rotation_5deg", "rotation", {"degrees": 5.0}),
        AttackConfig("rotation_45deg", "rotation", {"degrees": 45.0}),
        AttackConfig("translation_5_5", "translation", {"shift_x": 5, "shift_y": 5}),
        AttackConfig("resize_0p5", "resize", {"factor": 0.5}),
        AttackConfig("resize_0p75", "resize", {"factor": 0.75}),
        AttackConfig("resize_1p5", "resize", {"factor": 1.5}),
        AttackConfig("crop_resize_95", "crop_resize", {"keep": 0.95}),
        AttackConfig("crop_resize_90", "crop_resize", {"keep": 0.90}),
        AttackConfig("random_crop_resize_90", "crop_resize", {"keep": 0.90, "random_crop": True, "seed": 123}),
        AttackConfig("occlusion_32", "occlusion", {"block": 32, "num_blocks": 1, "seed": 123}),
        AttackConfig("occlusion_64", "occlusion", {"block": 64, "num_blocks": 1, "seed": 123}),
        AttackConfig("occlusion_50x3", "occlusion", {"block": 50, "num_blocks": 3, "seed": 123}),
        # Photometric and quantization
        AttackConfig("gamma_0p8", "gamma", {"gamma": 0.8}),
        AttackConfig("gamma_1p2", "gamma", {"gamma": 1.2}),
        AttackConfig("brightness_0p9", "brightness", {"factor": 0.9}),
        AttackConfig("brightness_1p1", "brightness", {"factor": 1.1}),
        AttackConfig("contrast_0p9", "contrast", {"factor": 0.9}),
        AttackConfig("contrast_1p1", "contrast", {"factor": 1.1}),
        AttackConfig("bit_depth_5", "bit_depth", {"bits": 5}),
        AttackConfig("posterize_bits4", "posterize", {"bits": 4}),
        AttackConfig("posterize_bits3", "posterize", {"bits": 3}),
        AttackConfig("solarize_128", "solarize", {"threshold": 128}),
        AttackConfig("saturation_0p5", "saturation", {"factor": 0.5}),
        AttackConfig("saturation_1p5", "saturation", {"factor": 1.5}),
        AttackConfig("channel_dropout_r", "channel_dropout", {"channel": 0, "value": 0}),
        AttackConfig("channel_dropout_b", "channel_dropout", {"channel": 2, "value": 0}),
        AttackConfig("mosaic_factor8", "mosaic", {"factor": 8}),
        AttackConfig("mosaic_factor16", "mosaic", {"factor": 16}),
        AttackConfig("shear_x_0p05", "shear", {"shear_x": 0.05}),
        AttackConfig("shear_x_0p10", "shear", {"shear_x": 0.10}),
        AttackConfig("row_col_delete_4_4", "row_col_delete", {"rows": 4, "cols": 4, "seed": 123}),
        AttackConfig("row_col_delete_8_8", "row_col_delete", {"rows": 8, "cols": 8, "seed": 123}),
        AttackConfig("checkerboard_cutout_64", "checkerboard_cutout", {"tile": 64, "value": 0}),
        AttackConfig("border_crop_pad_16", "border_crop_pad", {"pixels": 16, "value": 0}),
        AttackConfig("color_quantization_64", "color_quantization", {"colors": 64}),
        AttackConfig("color_quantization_32", "color_quantization", {"colors": 32}),
    ]
    return attacks if include_none else attacks[1:]


def stress_attack_suite(include_none: bool = True) -> list[AttackConfig]:
    """A compact but harsh attack suite for proposal stress testing.

    This preset is useful when you want to test synchronization/geometric
    robustness without running every full-suite attack.
    """
    attacks = [
        AttackConfig("no_attack", "none", {}),
        AttackConfig("jpeg_q50", "jpeg", {"quality": 50}),
        AttackConfig("jpeg_q30", "jpeg", {"quality": 30}),
        AttackConfig("gaussian_noise_sigma10", "gaussian_noise", {"sigma": 10.0, "seed": 123}),
        AttackConfig("salt_pepper_0p01", "salt_pepper", {"amount": 0.01, "seed": 123}),
        AttackConfig("median_5x5", "median_filter", {"size": 5}),
        AttackConfig("gaussian_blur_r2", "gaussian_blur", {"radius": 2.0}),
        AttackConfig("motion_blur_h9", "motion_blur", {"size": 9, "angle": "horizontal"}),
        AttackConfig("rotation_5deg", "rotation", {"degrees": 5.0}),
        AttackConfig("translation_8_8", "translation", {"shift_x": 8, "shift_y": 8}),
        AttackConfig("resize_0p5", "resize", {"factor": 0.5}),
        AttackConfig("crop_resize_90", "crop_resize", {"keep": 0.90}),
        AttackConfig("shear_x_0p10", "shear", {"shear_x": 0.10}),
        AttackConfig("row_col_delete_8_8", "row_col_delete", {"rows": 8, "cols": 8, "seed": 123}),
        AttackConfig("mosaic_factor16", "mosaic", {"factor": 16}),
        AttackConfig("posterize_bits3", "posterize", {"bits": 3}),
        AttackConfig("color_quantization_32", "color_quantization", {"colors": 32}),
    ]
    return attacks if include_none else attacks[1:]


def default_attack_suite(include_none: bool = True, preset: str = "lite") -> list[AttackConfig]:
    preset = str(preset).lower().strip()
    if preset in {"none", "clean", "no_attack"}:
        return [AttackConfig("no_attack", "none", {})] if include_none else []
    if preset in {"lite", "small", "quick"}:
        return lite_attack_suite(include_none=include_none)
    if preset in {"full", "all"}:
        return full_attack_suite(include_none=include_none)
    if preset in {"stress", "hard", "proposal_stress"}:
        return stress_attack_suite(include_none=include_none)
    raise ValueError(f"Unknown attack preset: {preset}")
