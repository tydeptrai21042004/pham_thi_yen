from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2, idwt2
from watermarklab.common.attack import apply_attack, AttackConfig
from watermarklab.common.metrics import ssim, ber


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


@dataclass
class GuoFAResult:
    """Firefly optimization result for the Guo 2017 embedding strength lambda."""

    lambda_strength: float
    objective: float
    clean_ssim: float
    mean_attack_ber: float
    history: list[dict[str, Any]]
    fa_params: dict[str, Any]


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------

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


def _watermark_bits(watermark_binary: np.ndarray) -> np.ndarray:
    return (np.asarray(watermark_binary, dtype=np.uint8) >= 127).astype(np.uint8)


class Guo2017DWTQRFA:
    """Guo et al. 2017 DWT-QR-FA blind watermarking baseline.

    Paper algorithm:
        cover image -> sort/permutation -> one-level db1/Haar DWT -> LL subband
        -> 4x4 blocks -> QR decomposition -> embed each binary watermark bit in
        first row of R using +/- lambda*K -> inverse QR -> inverse DWT -> inverse
        sorting.

    Extraction:
        Use the same sorting vector and random integral vector K.  Decide each bit
        from sign(corrcoef(R'(1,:), K)).

    Corrections made for paper reproduction:
        1. K is now generated as a random integral vector uniformly from {-1, 0, 1}
           with rejection of zero-variance vectors, instead of only +/-1.
        2. A real Firefly optimizer is implemented in ``optimize_lambda_firefly``
           using the paper objective:
               f(lambda) = [1 - SSIM(X, Xw)] + 30 * mean_i BER(w, w'_i).
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
        k_mode: str = "paper_integral",
    ):
        self.lambda_strength = float(lambda_strength)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.dwt_mode = str(dwt_mode)
        self.mode = str(mode)
        self.k_mode = str(k_mode)
        if self.mode == "original-rerun" and color_mode == "ycbcr_y":
            color_mode = "gray_mean"
        self.color_mode = str(color_mode)
        if self.block_size != 4:
            raise ValueError("Guo2017 paper-guided mode expects 4x4 QR blocks")

    # ------------------------------------------------------------------
    # Carrier handling
    # ------------------------------------------------------------------
    def _get_carrier(self, host_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.color_mode == "ycbcr_y":
            y, cb, cr = rgb_to_ycbcr(host_rgb)
            return y, cb, cr
        if self.color_mode == "gray_mean":
            x = np.asarray(host_rgb, dtype=np.float64)
            if x.ndim == 2:
                return x, None, None
            return x.mean(axis=2), None, None
        raise ValueError(f"Unsupported color_mode: {self.color_mode}")

    def _merge_carrier(self, y: np.ndarray, cb: np.ndarray | None, cr: np.ndarray | None, host_rgb: np.ndarray) -> np.ndarray:
        if self.color_mode == "ycbcr_y":
            assert cb is not None and cr is not None
            return ycbcr_to_rgb(y, cb, cr)
        gray = np.clip(np.rint(y), 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _make_k(self) -> np.ndarray:
        """Generate the paper-style random integral vector K.

        The paper states that K is a 1x4 random integral vector with values
        uniformly distributed in [-1, 1].  For integer values this corresponds to
        {-1, 0, 1}.  Because extraction uses correlation, zero-variance vectors
        are rejected and regenerated deterministically from the same RNG stream.
        """
        rng = np.random.default_rng(self.seed)
        for _ in range(1024):
            if self.k_mode in {"paper", "paper_integral", "integral"}:
                k = rng.integers(-1, 2, size=(self.block_size,)).astype(np.float64)
            elif self.k_mode in {"pm1", "plusminus1"}:
                k = rng.choice([-1.0, 1.0], size=(self.block_size,)).astype(np.float64)
            else:
                raise ValueError(f"Unsupported k_mode: {self.k_mode}")
            if np.linalg.norm(k - k.mean()) > 1e-12 and np.linalg.norm(k) > 1e-12:
                return k
        # Deterministic safe fallback, should almost never be reached.
        return np.asarray([-1.0, 0.0, 1.0, -1.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Embed/extract
    # ------------------------------------------------------------------
    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        wm = _watermark_bits(watermark_binary)
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

        delta = self.lambda_strength * k_vec
        for i in range(wm.shape[0]):
            for j in range(wm.shape[1]):
                r0 = i * self.block_size
                c0 = j * self.block_size
                block = ll[r0 : r0 + self.block_size, c0 : c0 + self.block_size]
                q, r = _stable_qr(block)
                r2 = r.copy()
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
            x = np.asarray(possibly_attacked_rgb, dtype=np.float64)
            carrier = x if x.ndim == 2 else x.mean(axis=2)
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

    # ------------------------------------------------------------------
    # Firefly optimization for lambda, following Guo et al. 2017
    # ------------------------------------------------------------------
    def _objective_for_lambda(
        self,
        lambda_strength: float,
        host_rgb: np.ndarray,
        watermark_binary: np.ndarray,
        attack_suite: list[AttackConfig],
        robustness_weight: float = 30.0,
    ) -> dict[str, float]:
        old_lambda = self.lambda_strength
        self.lambda_strength = float(lambda_strength)
        try:
            watermarked, key = self.embed(host_rgb, watermark_binary)
            clean_ssim = ssim(host_rgb, watermarked)
            attack_bers: list[float] = []
            for attack in attack_suite:
                attacked = apply_attack(watermarked, attack)
                extracted = self.extract(attacked, key, host_rgb=host_rgb)
                attack_bers.append(ber(watermark_binary, extracted))
            mean_attack_ber = float(np.mean(attack_bers)) if attack_bers else 0.0
            objective = float((1.0 - clean_ssim) + robustness_weight * mean_attack_ber)
            return {
                "objective": objective,
                "clean_ssim": float(clean_ssim),
                "mean_attack_ber": mean_attack_ber,
                "min_attack_ber": float(np.min(attack_bers)) if attack_bers else 0.0,
                "max_attack_ber": float(np.max(attack_bers)) if attack_bers else 0.0,
            }
        finally:
            self.lambda_strength = old_lambda

    def optimize_lambda_firefly(
        self,
        host_rgb: np.ndarray,
        watermark_binary: np.ndarray,
        attack_suite: list[AttackConfig],
        *,
        n_fireflies: int = 10,
        n_iterations: int = 10,
        alpha_fa: float = 0.01,
        beta0: float = 1.0,
        gamma: float = 1.0,
        robustness_weight: float = 30.0,
        lambda_min: float = 0.05,
        lambda_max: float = 20.0,
        seed: int = 2017,
    ) -> GuoFAResult:
        """Search lambda by Firefly Algorithm using the paper objective.

        The paper minimizes ``[1 - SSIM(X, Xw)] + 30 * mean(BER)``.  This
        implementation treats smaller objective values as brighter fireflies.
        """
        n_fireflies = int(n_fireflies)
        n_iterations = int(n_iterations)
        if n_fireflies < 2:
            raise ValueError("n_fireflies must be at least 2")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        lambda_min = float(lambda_min)
        lambda_max = float(lambda_max)
        if not (lambda_max > lambda_min > 0):
            raise ValueError("Expected 0 < lambda_min < lambda_max")

        rng = np.random.default_rng(int(seed))
        x = rng.uniform(lambda_min, lambda_max, size=n_fireflies).astype(np.float64)

        def eval_all(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
            records = [
                self._objective_for_lambda(
                    float(v),
                    host_rgb,
                    watermark_binary,
                    attack_suite,
                    robustness_weight=robustness_weight,
                )
                for v in values
            ]
            return np.asarray([r["objective"] for r in records], dtype=np.float64), records

        obj, records = eval_all(x)
        history: list[dict[str, Any]] = []
        best_idx = int(np.argmin(obj))
        best_lambda = float(x[best_idx])
        best_obj = float(obj[best_idx])
        best_record = dict(records[best_idx])

        for it in range(n_iterations):
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    # Lower objective is better/brighter.
                    if obj[j] < obj[i]:
                        rij = abs(float(x[i] - x[j])) / max(lambda_max - lambda_min, 1e-12)
                        beta = float(beta0) * np.exp(-float(gamma) * (rij ** 2))
                        random_step = float(alpha_fa) * (rng.random() - 0.5) * (lambda_max - lambda_min)
                        x[i] = x[i] + beta * (x[j] - x[i]) + random_step
                        x[i] = float(np.clip(x[i], lambda_min, lambda_max))
            obj, records = eval_all(x)
            iter_best_idx = int(np.argmin(obj))
            if float(obj[iter_best_idx]) < best_obj:
                best_obj = float(obj[iter_best_idx])
                best_lambda = float(x[iter_best_idx])
                best_record = dict(records[iter_best_idx])
            history.append({
                "iteration": it + 1,
                "best_lambda": best_lambda,
                "best_objective": best_obj,
                "population_min_objective": float(np.min(obj)),
                "population_mean_objective": float(np.mean(obj)),
                "population_best_lambda": float(x[iter_best_idx]),
            })

        # Set the method to the best lambda found so normal use immediately uses it.
        self.lambda_strength = best_lambda
        return GuoFAResult(
            lambda_strength=best_lambda,
            objective=best_obj,
            clean_ssim=float(best_record.get("clean_ssim", np.nan)),
            mean_attack_ber=float(best_record.get("mean_attack_ber", np.nan)),
            history=history,
            fa_params={
                "alpha_fa": float(alpha_fa),
                "beta0": float(beta0),
                "gamma": float(gamma),
                "n_fireflies": n_fireflies,
                "n_iterations": n_iterations,
                "robustness_weight": float(robustness_weight),
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "seed": int(seed),
            },
        )


__all__ = ["Guo2017DWTQRFA", "Guo2017Key", "GuoFAResult"]
