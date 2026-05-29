"""Compact Firework-Algorithm-inspired optimizer for chaotic key parameters.

The original paper uses FWA to choose the initial values for chaotic maps, but
it does not provide enough implementation detail for exact bit-level matching.
This module implements the same intent: search for key parameters that minimize
watermarked-image MSE / maximize PSNR for the described embedding pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .watermark import WatermarkConfig, embed_watermark


@dataclass
class FWAResult:
    best_params: tuple[float, float, float, float]
    best_mse: float
    best_psnr: float
    history: list[dict]


def _map_bounds(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    width = high - low
    return low + np.mod(x - low, width)


def optimize_key_params(
    cover_rgb: np.ndarray,
    watermark_bits: np.ndarray,
    base_config: WatermarkConfig,
    population_size: int = 8,
    iterations: int = 3,
    sparks_per_firework: int = 3,
    seed: int = 7,
) -> FWAResult:
    """Search for chaotic-map initial parameters.

    The default values are intentionally small for a runnable reproduction.
    Increase population_size/iterations for a heavier optimization study.
    """
    rng = np.random.default_rng(seed)
    low = np.array([0.01, 0.01, 3.10, 0.01], dtype=np.float64)
    high = np.array([0.99, 0.99, 5.90, 0.99], dtype=np.float64)
    pop = rng.uniform(low, high, size=(population_size, 4))
    history: list[dict] = []

    def evaluate(params: np.ndarray) -> tuple[float, float]:
        cfg = WatermarkConfig(**{**base_config.__dict__, "key_params": tuple(float(v) for v in params)})
        _, _, meta = embed_watermark(cover_rgb, watermark_bits, cfg)
        return meta.mse_float, meta.psnr_float

    best_params = None
    best_mse = float("inf")
    best_psnr = -float("inf")

    for it in range(iterations):
        scored = []
        for p in pop:
            m, p_db = evaluate(p)
            scored.append((m, p_db, p.copy()))
            if m < best_mse:
                best_mse = m
                best_psnr = p_db
                best_params = p.copy()
        scored.sort(key=lambda t: t[0])
        history.append({"iteration": it, "best_mse": scored[0][0], "best_psnr": scored[0][1], "best_params": scored[0][2].tolist()})

        # Explosion: better fireworks get smaller amplitude local search.
        new_pop = [scored[0][2].copy()]
        max_rank = max(1, len(scored) - 1)
        for rank, (_, _, center) in enumerate(scored[: max(2, population_size // 2)]):
            amp = 0.25 * (1.0 - rank / max_rank) + 0.03
            for _ in range(sparks_per_firework):
                spark = center + rng.uniform(-amp, amp, size=4) * (high - low)
                new_pop.append(_map_bounds(spark, low, high))
            # Gaussian mutation spark.
            mut = center * rng.normal(1.0, 0.12, size=4)
            new_pop.append(_map_bounds(mut, low, high))
        rng.shuffle(new_pop)
        pop = np.array(new_pop[:population_size])

    assert best_params is not None
    return FWAResult(tuple(float(v) for v in best_params), float(best_mse), float(best_psnr), history)
