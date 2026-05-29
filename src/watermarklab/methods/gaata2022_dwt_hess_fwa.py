from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from watermarklab.vendor.dwt_hess_fwa.watermark import (
    WatermarkConfig,
    EmbedMetadata,
    embed_watermark,
    extract_watermark,
)
from watermarklab.vendor.dwt_hess_fwa.fwa import optimize_key_params, FWAResult


@dataclass
class Gaata2022HessFWAKey:
    """Side information required by the paper-guided Gaata et al. baseline.

    The original article describes chaotic keys and an extraction reverse path, but does
    not publish all numerical choices.  We therefore keep the chosen configuration and the
    minimum DWT-matrix shift in the key for deterministic reproduction.
    """

    config: WatermarkConfig
    metadata: EmbedMetadata
    exact_float_watermarked: np.ndarray | None = None
    fwa_result: FWAResult | None = None


class Gaata2022DWTHessFWA:
    """Gaata et al. 2022 DWT + Hessenberg + Firework Algorithm baseline.

    Paper path:
        RGB split -> one-level DWT -> omit LL detail-free bands -> assemble EM
        -> add hybrid Gaussian/exponential chaotic keys -> 4x4 Hessenberg blocks
        -> embed bit by parity of a selected H coefficient -> inverse transform.

    Reproduction modes:
        adapt:
            Fast common-benchmark mode. It keeps the package runnable for all baselines.
        original-rerun:
            Paper-faithful local rerun mode for this baseline. It switches on the
            decimal-digit Hessenberg rule, uses stronger chaotic keys, and runs the
            larger FWA-style search configured from the paper settings.
    """

    name = "Gaata2022_DWT_Hess_FWA"

    def __init__(
        self,
        block_size: int = 4,
        h_position: tuple[int, int] = (3, 3),
        decimal_position: int = -1,
        key_strength: float = 0.020,
        key_params: tuple[float, float, float, float] = (0.21, 0.37, 4.90, 0.18),
        use_fwa: bool = False,
        fwa_population: int = 6,
        fwa_iterations: int = 2,
        fwa_sparks: int = 2,
        seed: int = 2022,
        mode: str = "adapt",
    ):
        self.mode = str(mode)

        if self.mode == "original-rerun":
            # Paper-faithful correction requested by the user:
            #   * use the decimal digit after the floating point instead of the
            #     quantization-aware parity shortcut used by the quick adapted mode;
            #   * use a stronger chaotic-key contribution than 0.020;
            #   * run the larger Firework Algorithm setting reported in the paper
            #     (population N=100 and mutation-spark number=5).
            #
            # The paper does not state an exact iteration count, so we use at least
            # 10 optimizer iterations for a serious local rerun while still allowing
            # a caller to pass a larger value.
            decimal_position = 3
            key_strength = 1.0
            use_fwa = True
            fwa_population = max(int(fwa_population), 100)
            fwa_iterations = max(int(fwa_iterations), 10)
            fwa_sparks = max(int(fwa_sparks), 5)

        self.config = WatermarkConfig(
            block_size=int(block_size),
            h_position=tuple(h_position),
            decimal_position=int(decimal_position),
            key_strength=float(key_strength),
            key_params=tuple(float(x) for x in key_params),
            clip_output=True,
        )
        self.use_fwa = bool(use_fwa)
        self.fwa_population = int(fwa_population)
        self.fwa_iterations = int(fwa_iterations)
        self.fwa_sparks = int(fwa_sparks)
        self.seed = int(seed)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        wm_bits = (np.asarray(watermark_binary) >= 127).astype(np.uint8)
        cfg = self.config
        fwa_result = None
        if self.use_fwa:
            fwa_result = optimize_key_params(
                host_rgb,
                wm_bits,
                cfg,
                population_size=self.fwa_population,
                iterations=self.fwa_iterations,
                sparks_per_firework=self.fwa_sparks,
                seed=self.seed,
            )
            cfg = WatermarkConfig(**{**cfg.__dict__, "key_params": fwa_result.best_params})
        watermarked_float, watermarked_uint8, metadata = embed_watermark(host_rgb, wm_bits, cfg)
        key = Gaata2022HessFWAKey(
            config=cfg,
            metadata=metadata,
            exact_float_watermarked=watermarked_float,
            fwa_result=fwa_result,
        )
        return watermarked_uint8, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Gaata2022HessFWAKey, host_rgb: np.ndarray | None = None):
        bits = extract_watermark(
            possibly_attacked_rgb,
            key.metadata.watermark_shape,
            key.config,
            min_shift=key.metadata.min_shift,
        )
        return (bits.astype(np.uint8) * 255).astype(np.uint8)

    def extract_exact_float(self, key: Gaata2022HessFWAKey):
        if key.exact_float_watermarked is None:
            raise ValueError("exact_float_watermarked is not available in this key")
        bits = extract_watermark(
            key.exact_float_watermarked,
            key.metadata.watermark_shape,
            key.config,
            min_shift=key.metadata.min_shift,
        )
        return (bits.astype(np.uint8) * 255).astype(np.uint8)


__all__ = ["Gaata2022DWTHessFWA", "Gaata2022HessFWAKey"]
