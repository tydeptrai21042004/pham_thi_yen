from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any

import cv2
import numpy as np
try:
    import pywt
except Exception:  # pragma: no cover - tests may run before requirements are installed
    pywt = None
from scipy.linalg import hessenberg

from watermarklab.common.dwt import split_4bands, merge_4bands
from watermarklab.common.chaos import arnold_scramble, arnold_unscramble
from watermarklab.common.attack import default_attack_suite, apply_attack as apply_benchmark_attack, AttackConfig
from watermarklab.common.metrics import psnr as metric_psnr, ssim as metric_ssim, nc as metric_nc, ber as metric_ber


# =========================================================
# Constants: kept aligned with the uploaded Python script
# =========================================================
BLOCKSIZE = 4
WM_SIZE = 64
HOST_SIZE = 512
PRIVATE_KEY = "KB123"
ARNOLD_ITERATIONS = 17
DWT_WAVELET = "haar"
DWT_LEVEL = 1
DWT_BANDS = ("LL", "HL", "HH", "LH")
HOST_EMBED_COLOR_SPACE = "YCrCb"
HOST_CHANNELS = (0,)  # Y only after RGB->BGR->YCrCb conversion
WM_BIN_THRESH = 127

FLAG_Q4 = 0
FLAG_HPOS = 1
FLAG_SKIP = 2
HPOS_NONE = -1

Q4_TAU = 0.50
Q4_MARGIN = 0.08
H01_Q = 7.0
H01_MARGIN = 0.90

HPOS_CANDIDATES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("h21", (1, 0)),
    ("h22", (1, 1)),
)

EPS_CONF = 1e-12

MIN_SURVIVAL_RATE = 0.75
BSS_WEIGHT = 0.75
MSE_WEIGHT = 0.25
MAX_Q_CAND_MSE = 0.18
MAX_H_CAND_MSE = 0.24

Q4_GIVENS_THETA_MAX = 0.42
Q4_GIVENS_COARSE_STEPS = 13
Q4_GIVENS_FINE_THETA_MAX = 0.08
Q4_GIVENS_FINE_STEPS = 0
Q4_GIVENS_PASSES = 1
Q4_GIVENS_MSE_WEIGHT = 0.10
Q4_GIVENS_EXTRA_MARGIN_WEIGHT = 0.04
Q4_GIVENS_PAIRS = ((0, 2), (1, 3), (0, 1), (1, 2))

FAST_CANDIDATE_SCORING = True
USE_STRUCTURED_REPETITION = True

# Firefly optimization parameter space copied from the standalone script.
# Only these four variables are optimized; all other parameters keep their
# script-default values unless explicitly supplied.
PARAM_SPECS = [
    {"name": "q4_tau", "source_name": "Q4_TAU", "min": 0.35, "max": 0.65},
    {"name": "q4_margin", "source_name": "Q4_MARGIN", "min": 0.04, "max": 0.14},
    {"name": "h01_q", "source_name": "H01_Q", "min": 5.0, "max": 10.0},
    {"name": "h01_margin", "source_name": "H01_MARGIN", "min": 0.50, "max": 1.20},
]


@dataclass
class ProposalParams:
    """Parameters for the proposal method.

    The defaults are intentionally the same as the user's standalone Python
    script.  The benchmark package receives RGB arrays, while the standalone
    script receives OpenCV BGR images; the class handles that I/O-format bridge
    internally without changing the algorithmic embedding domain.
    """

    arnold_iterations: int = ARNOLD_ITERATIONS
    dwt_bands: tuple[str, ...] = DWT_BANDS
    # "pywt" is the source-script-faithful mode. It calls pywt.dwt2/idwt2
    # directly instead of the package's lightweight Haar implementation.
    dwt_mode: str = "pywt"
    block_size: int = BLOCKSIZE
    private_key: str = PRIVATE_KEY

    # None = full structured repetition, exactly like USE_STRUCTURED_REPETITION=True
    # in the standalone script.  An integer can still be supplied for quick local
    # experiments, but the faithful/default mode is None.
    repeat: int | None = None

    q4_tau: float = Q4_TAU
    q4_margin: float = Q4_MARGIN
    h01_q: float = H01_Q
    h01_margin: float = H01_MARGIN

    min_survival_rate: float = MIN_SURVIVAL_RATE
    bss_weight: float = BSS_WEIGHT
    mse_weight: float = MSE_WEIGHT
    max_q_cand_mse: float = MAX_Q_CAND_MSE
    max_h_cand_mse: float = MAX_H_CAND_MSE

    q4_givens_theta_max: float = Q4_GIVENS_THETA_MAX
    q4_givens_coarse_steps: int = Q4_GIVENS_COARSE_STEPS
    q4_givens_fine_theta_max: float = Q4_GIVENS_FINE_THETA_MAX
    q4_givens_fine_steps: int = Q4_GIVENS_FINE_STEPS
    q4_givens_passes: int = Q4_GIVENS_PASSES
    q4_givens_mse_weight: float = Q4_GIVENS_MSE_WEIGHT
    q4_givens_extra_margin_weight: float = Q4_GIVENS_EXTRA_MARGIN_WEIGHT
    q4_givens_pairs: tuple[tuple[int, int], ...] = Q4_GIVENS_PAIRS
    fast_candidate_scoring: bool = FAST_CANDIDATE_SCORING

    watermark_dwt_enabled: bool = True
    watermark_payload_band: str = "LL"
    host_embed_color_space: str = HOST_EMBED_COLOR_SPACE
    host_channels: tuple[int, ...] = HOST_CHANNELS
    structured_repetition_enabled: bool = USE_STRUCTURED_REPETITION

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProposalParams":
        """Create params from package-style or source-script-style dictionaries.

        The standalone script exports uppercase names such as ``Q4_TAU`` and
        ``H01_MARGIN``. The cleaned package uses lowercase dataclass fields.
        Supporting both formats is important for the two-phase workflow where
        an optimization phase writes parameters and a normal phase later reads
        them automatically.
        """
        params = cls()
        if not data:
            return params

        aliases = {
            "ARNOLD_ITERATIONS": "arnold_iterations",
            "DWT_BANDS": "dwt_bands",
            "Q4_TAU": "q4_tau",
            "Q4_MARGIN": "q4_margin",
            "H01_Q": "h01_q",
            "H01_MARGIN": "h01_margin",
            "MIN_SURVIVAL_RATE": "min_survival_rate",
            "BSS_WEIGHT": "bss_weight",
            "MSE_WEIGHT": "mse_weight",
            "MAX_Q_CAND_MSE": "max_q_cand_mse",
            "MAX_H_CAND_MSE": "max_h_cand_mse",
            "Q4_GIVENS_THETA_MAX": "q4_givens_theta_max",
            "Q4_GIVENS_MSE_WEIGHT": "q4_givens_mse_weight",
            "Q4_GIVENS_EXTRA_MARGIN_WEIGHT": "q4_givens_extra_margin_weight",
        }

        for key, value in data.items():
            key2 = aliases.get(str(key), str(key))
            if not hasattr(params, key2):
                continue
            if key2 in {"dwt_bands", "host_channels"}:
                value = tuple(value)
            elif key2 == "q4_givens_pairs":
                value = tuple(tuple(x) for x in value)
            setattr(params, key2, value)

        # Keep H-domain margin inside the same safe interval used by the source script.
        params.h01_margin = min(float(params.h01_margin), 0.49 * float(params.h01_q))
        return params

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["dwt_bands"] = list(self.dwt_bands)
        out["host_channels"] = list(self.host_channels)
        out["q4_givens_pairs"] = [list(x) for x in self.q4_givens_pairs]
        return out


@dataclass
class ProposalKey:
    flags: list[int]
    hpos_list: list[int]
    payload_meta: dict[str, Any]
    params: ProposalParams
    repeat_factor: int
    usable_blocks: int
    total_blocks: int
    support_counts: list[int] = field(default_factory=list)
    # Kept for compatibility/debugging.  Extraction regenerates the schedule
    # from the key and image, matching the standalone script behavior.
    schedule: list[tuple[int, int, str, int, int]] = field(default_factory=list)

    @property
    def wm_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.payload_meta.get("wm_shape", (WM_SIZE, WM_SIZE)))


# =========================================================
# JSON / numeric helpers
# =========================================================
def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _flag_stats(flags):
    q4_used = sum(1 for f in flags if int(f) == FLAG_Q4)
    hpos_used = sum(1 for f in flags if int(f) == FLAG_HPOS)
    skip_used = sum(1 for f in flags if int(f) == FLAG_SKIP)
    return q4_used, hpos_used, skip_used


def decode_firefly_position(position: np.ndarray | list[float], base_params: ProposalParams | None = None) -> ProposalParams:
    """Decode a normalized Firefly position into ProposalParams.

    This follows the uploaded script's four-parameter search space exactly:
    Q4_TAU, Q4_MARGIN, H01_Q, H01_MARGIN.
    """
    params = ProposalParams.from_dict(base_params.to_dict() if base_params is not None else None)
    pos = np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0)
    for idx, spec in enumerate(PARAM_SPECS):
        z = float(pos[idx])
        value = float(spec["min"] + z * (spec["max"] - spec["min"]))
        setattr(params, spec["name"], value)
    params.h01_margin = min(float(params.h01_margin), 0.49 * float(params.h01_q))
    return params


def firefly_cache_key(position: np.ndarray | list[float], digits: int = 4) -> tuple[float, ...]:
    arr = np.round(np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0), int(digits))
    return tuple(float(x) for x in arr.tolist())


def optimization_param_snapshot(params: ProposalParams) -> dict[str, Any]:
    """Return lowercase and source-script uppercase optimized parameters."""
    lower = {spec["name"]: float(getattr(params, spec["name"])) for spec in PARAM_SPECS}
    upper = {spec["source_name"]: lower[spec["name"]] for spec in PARAM_SPECS}
    return {**lower, "source_script_names": upper}


# =========================================================
# RGB package I/O bridge to the source script's OpenCV-BGR domain
# =========================================================
def _ensure_uint8_rgb_basic(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must have 3 channels.")
    return np.clip(img, 0, 255).astype(np.uint8)


def _rgb_to_source_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return _ensure_uint8_rgb_basic(img_rgb)[:, :, ::-1].copy()


def _source_bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return _ensure_uint8_rgb_basic(img_bgr)[:, :, ::-1].copy()


def _host_bgr_to_embed_space(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = _ensure_uint8_rgb_basic(img_bgr)  # shape/type guard only
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.uint8)


def _embed_space_to_host_bgr(img_embed: np.ndarray) -> np.ndarray:
    img_embed = _ensure_uint8_rgb_basic(img_embed)
    return cv2.cvtColor(img_embed, cv2.COLOR_YCrCb2BGR).astype(np.uint8)


# =========================================================
# DWT helpers: local Haar orthonormal implementation mirrors pywt Haar
# =========================================================
def _required_multiple_for_dwt(level: int, block_size: int) -> int:
    return (2 ** int(level)) * int(block_size)


def _crop_for_dwt(channel_u8: np.ndarray, level: int, block_size: int):
    req = _required_multiple_for_dwt(level, block_size)
    h, w = channel_u8.shape
    h0 = (h // req) * req
    w0 = (w // req) * req
    if h0 == 0 or w0 == 0:
        raise ValueError("Image too small for DWT + BLOCKSIZE constraint.")
    return channel_u8[:h0, :w0].astype(np.float64), h0, w0


def _dwt_split_4bands(channel_f64: np.ndarray, params: ProposalParams):
    """Split into LL/LH/HL/HH.

    In source-script-faithful mode this calls PyWavelets exactly like:
    ``LL, (LH, HL, HH) = pywt.dwt2(x, "haar")``. The older package-local
    modes remain available only for backward compatibility.
    """
    if str(params.dwt_mode).lower() in {"pywt", "pywt_haar", "script", "source"}:
        if pywt is not None:
            LL, (LH, HL, HH) = pywt.dwt2(np.asarray(channel_f64, dtype=np.float64), DWT_WAVELET)
            return {"LL": LL.copy(), "LH": LH.copy(), "HL": HL.copy(), "HH": HH.copy()}
        # Fallback for environments that have not installed PyWavelets yet.
        # requirements.txt/pyproject.toml include PyWavelets, so normal runs use pywt.
        return split_4bands(channel_f64, mode="orthonormal")
    return split_4bands(channel_f64, mode=params.dwt_mode)


def _dwt_merge_4bands(bands: dict[str, np.ndarray], params: ProposalParams):
    if str(params.dwt_mode).lower() in {"pywt", "pywt_haar", "script", "source"}:
        if pywt is not None:
            return pywt.idwt2((bands["LL"], (bands["LH"], bands["HL"], bands["HH"])), DWT_WAVELET)
        return merge_4bands(bands, mode="orthonormal")
    return merge_4bands(bands, mode=params.dwt_mode)


# =========================================================
# Watermark payload: source-script faithful LL-only watermark DWT
# =========================================================
def _force_binary_watermark_exact(img_u8: np.ndarray, size: int = WM_SIZE, thresh: int = WM_BIN_THRESH) -> np.ndarray:
    if img_u8 is None:
        raise ValueError("Watermark image is None.")
    out = np.asarray(img_u8, dtype=np.uint8)
    if out.ndim == 3:
        out = np.mean(out[:, :, :3], axis=2).astype(np.uint8)
    if out.shape != (size, size):
        raise ValueError(f"Binary watermark must be exactly {size}x{size}. Got {out.shape}.")
    return np.where(out >= int(thresh), 255, 0).astype(np.uint8)


def prepare_binary_watermark_payload_from_array(
    watermark_binary: np.ndarray,
    params: ProposalParams,
):
    wm_binary = _force_binary_watermark_exact(watermark_binary, WM_SIZE, WM_BIN_THRESH)
    wm_bits_2d = (wm_binary >= WM_BIN_THRESH).astype(np.float64)

    # Same as pywt.dwt2(wm_bits_2d, 'haar') -> LL, (LH, HL, HH)
    bands = _dwt_split_4bands(wm_bits_2d, params)
    LL, LH, HL, HH = bands["LL"], bands["LH"], bands["HL"], bands["HH"]

    ll_threshold = 1.0
    ll_bits_2d = (LL >= ll_threshold).astype(np.uint8)

    low_vals = LL[LL < ll_threshold]
    high_vals = LL[LL >= ll_threshold]
    ll_low_value = float(np.mean(low_vals)) if low_vals.size > 0 else 0.0
    ll_high_value = float(np.mean(high_vals)) if high_vals.size > 0 else 2.0

    scrambled_bits_2d = arnold_scramble(ll_bits_2d, int(params.arnold_iterations))
    payload_bits = scrambled_bits_2d.reshape(-1).astype(np.uint8)

    meta = {
        "wm_shape": tuple(int(x) for x in wm_binary.shape),
        "wm_size": int(WM_SIZE),
        "original_wm_bit_len": int(wm_bits_2d.size),
        "wm_bit_len": int(ll_bits_2d.size),
        "payload_len": int(payload_bits.size),
        "payload_shape": tuple(int(x) for x in ll_bits_2d.shape),
        "arnold_enabled": True,
        "arnold_iterations": int(params.arnold_iterations),
        "watermark_dwt_enabled": True,
        "watermark_dwt_wavelet": DWT_WAVELET,
        "watermark_dwt_level": 1,
        "watermark_payload_band": "LL",
        "watermark_ll_threshold": float(ll_threshold),
        "watermark_ll_low_value": float(ll_low_value),
        "watermark_ll_high_value": float(ll_high_value),
        "watermark_detail_subbands": {
            "LH": LH.astype(float).tolist(),
            "HL": HL.astype(float).tolist(),
            "HH": HH.astype(float).tolist(),
        },
    }
    return wm_binary, payload_bits, meta


def reconstruct_binary_watermark_from_payload_bits(payload_bits, meta: dict, params: ProposalParams):
    arnold_iterations = int(meta.get("arnold_iterations", params.arnold_iterations))

    if bool(meta.get("watermark_dwt_enabled", False)):
        wm_size = int(meta.get("wm_size", WM_SIZE))
        payload_len = int(meta["payload_len"])
        payload_shape = tuple(int(x) for x in meta.get("payload_shape", (wm_size // 2, wm_size // 2)))

        bits_scrambled = np.array(payload_bits[:payload_len], dtype=np.uint8).reshape(-1)
        if bits_scrambled.size < payload_len:
            pad = np.zeros((payload_len - bits_scrambled.size,), dtype=np.uint8)
            bits_scrambled = np.concatenate([bits_scrambled, pad], axis=0)
        elif bits_scrambled.size > payload_len:
            bits_scrambled = bits_scrambled[:payload_len]

        scrambled_2d = bits_scrambled.reshape(payload_shape).astype(np.uint8)
        recovered_ll_bits = arnold_unscramble(scrambled_2d, arnold_iterations)

        ll_low = float(meta.get("watermark_ll_low_value", 0.0))
        ll_high = float(meta.get("watermark_ll_high_value", 2.0))
        LL_rec = np.where(recovered_ll_bits > 0, ll_high, ll_low).astype(np.float64)

        details = meta.get("watermark_detail_subbands", {})
        LH = np.asarray(details.get("LH"), dtype=np.float64)
        HL = np.asarray(details.get("HL"), dtype=np.float64)
        HH = np.asarray(details.get("HH"), dtype=np.float64)
        if LH.shape != LL_rec.shape or HL.shape != LL_rec.shape or HH.shape != LL_rec.shape:
            raise ValueError("Saved watermark DWT detail subbands do not match extracted LL shape.")

        wm_float = _dwt_merge_4bands({"LL": LL_rec, "LH": LH, "HL": HL, "HH": HH}, params)
        wm_float = wm_float[:wm_size, :wm_size]
        wm_rec = np.where(wm_float >= 0.5, 255, 0).astype(np.uint8)
        return _force_binary_watermark_exact(wm_rec, wm_size, WM_BIN_THRESH)

    # Backward-compatible fallback; not used by the source-script-faithful path.
    wm_size = int(meta["wm_size"])
    wm_bit_len = int(meta["wm_bit_len"])
    payload_len = int(meta["payload_len"])
    bits_scrambled = np.array(payload_bits[:payload_len], dtype=np.uint8).reshape(-1)
    if bits_scrambled.size < wm_bit_len:
        bits_scrambled = np.concatenate([bits_scrambled, np.zeros((wm_bit_len - bits_scrambled.size,), dtype=np.uint8)])
    elif bits_scrambled.size > wm_bit_len:
        bits_scrambled = bits_scrambled[:wm_bit_len]
    scrambled_2d = bits_scrambled.reshape((wm_size, wm_size)).astype(np.uint8)
    recovered_bits_2d = arnold_unscramble(scrambled_2d, arnold_iterations)
    return _force_binary_watermark_exact((recovered_bits_2d * 255).astype(np.uint8), wm_size, WM_BIN_THRESH)


# =========================================================
# Q and H candidates: same branch logic as the source script
# =========================================================
def _safe_sign(x: float) -> float:
    return 1.0 if float(x) >= 0.0 else -1.0


def _nearest_nonnegative_mod_value(a: float, target: float, q: float) -> float:
    k0 = int(np.round((float(a) - float(target)) / max(float(q), EPS_CONF)))
    candidates = []
    for k in range(k0 - 3, k0 + 4):
        v = float(target) + k * float(q)
        if v >= 0:
            candidates.append(v)
    if not candidates:
        return max(0.0, float(target))
    return float(min(candidates, key=lambda v: abs(v - float(a))))


def _q4_stat(Qm: np.ndarray) -> float:
    return float(Qm[1, 1] ** 2 + Qm[2, 1] ** 2)


def _givens4(a: int, b: int, theta: float) -> np.ndarray:
    G = np.eye(4, dtype=np.float64)
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    G[a, a] = c
    G[a, b] = s
    G[b, a] = -s
    G[b, b] = c
    return G


def _q4_signed_margin_from_Q(Qm: np.ndarray, bit: int, params: ProposalParams) -> float:
    d = _q4_stat(Qm) - float(params.q4_tau)
    return float(d) if int(bit) == 1 else -float(d)


# Compatibility alias used by existing validation code.
def _q4_signed_margin_from_q(Qm: np.ndarray, bit: int, params: ProposalParams) -> float:
    return _q4_signed_margin_from_Q(Qm, bit, params)


def _q4_givens_loss(block0: np.ndarray, Hh: np.ndarray, Qtrial: np.ndarray, bit: int, params: ProposalParams):
    signed_margin = _q4_signed_margin_from_Q(Qtrial, bit, params)
    target = float(params.q4_margin)

    if signed_margin >= target:
        margin_loss = 0.0
        extra_margin = signed_margin - target
    else:
        margin_loss = target - signed_margin
        extra_margin = 0.0

    block_trial = Qtrial @ Hh @ Qtrial.T
    mse = float(np.mean((block0 - block_trial) ** 2))
    loss = (
        margin_loss
        + float(params.q4_givens_mse_weight) * math.sqrt(max(mse, 0.0))
        + float(params.q4_givens_extra_margin_weight) * max(extra_margin, 0.0)
    )
    return float(loss), float(mse), float(signed_margin)


def _build_q4_candidate(block: np.ndarray, bit: int, params: ProposalParams) -> np.ndarray:
    Hh, Qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    best_Q = Qm.copy()
    best_loss, best_mse, best_margin = _q4_givens_loss(block, Hh, best_Q, bit, params)

    if best_margin >= float(params.q4_margin):
        return best_Q @ Hh @ best_Q.T

    grids = [
        np.linspace(-float(params.q4_givens_theta_max), float(params.q4_givens_theta_max), int(params.q4_givens_coarse_steps)),
    ]
    if int(params.q4_givens_fine_steps) > 1:
        grids.append(
            np.linspace(
                -float(params.q4_givens_fine_theta_max),
                float(params.q4_givens_fine_theta_max),
                int(params.q4_givens_fine_steps),
            )
        )

    for pass_idx in range(max(1, int(params.q4_givens_passes))):
        theta_grid = grids[min(pass_idx, len(grids) - 1)]
        improved = False

        for a, b in params.q4_givens_pairs:
            local_best_Q = best_Q
            local_best_key = (best_loss, best_mse, -best_margin)

            for th in theta_grid:
                if abs(float(th)) < 1e-15:
                    continue
                G = _givens4(a, b, float(th))
                Qtrial = G @ best_Q
                loss, mse, signed_margin = _q4_givens_loss(block, Hh, Qtrial, bit, params)
                key = (loss, mse, -signed_margin)
                if key < local_best_key:
                    local_best_key = key
                    local_best_Q = Qtrial

            if local_best_Q is not best_Q:
                best_Q = local_best_Q
                best_loss, best_mse, best_margin = _q4_givens_loss(block, Hh, best_Q, bit, params)
                improved = True

        if not improved:
            break

        if best_margin >= float(params.q4_margin) and best_loss <= float(params.q4_givens_mse_weight) * math.sqrt(max(best_mse, 0.0)) + 1e-12:
            break

    return best_Q @ Hh @ best_Q.T


def _build_hpos_candidate(block: np.ndarray, bit: int, params: ProposalParams, pos: tuple[int, int]) -> np.ndarray:
    Hh, Qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    H2 = Hh.copy()

    q_h01 = max(float(params.h01_q), 1e-6)
    h01_margin = min(max(0.0, float(params.h01_margin)), 0.49 * q_h01)

    rr, cc = pos
    v = float(H2[rr, cc])
    sgn = _safe_sign(v)
    a = abs(v)
    z = a % q_h01

    safe = False
    if int(bit) == 1:
        if z >= (0.5 * q_h01 + h01_margin) and z <= (q_h01 - h01_margin):
            safe = True
    else:
        if z >= h01_margin and z <= (0.5 * q_h01 - h01_margin):
            safe = True

    if not safe:
        target = 0.75 * q_h01 if int(bit) == 1 else 0.25 * q_h01
        a_new = _nearest_nonnegative_mod_value(a, target, q_h01)
        H2[rr, cc] = sgn * a_new

    return Qm @ H2 @ Qm.T


def _extract_bit_from_candidate_block(block: np.ndarray, mode: int, params: ProposalParams, pos: tuple[int, int] | None = None) -> int:
    Hh, Qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)

    if int(mode) == FLAG_Q4:
        return 1 if _q4_stat(Qm) >= float(params.q4_tau) else 0

    if int(mode) == FLAG_HPOS:
        if pos is None:
            raise ValueError("H-domain candidate requires a coefficient position.")
        rr, cc = pos
        q_h01 = max(float(params.h01_q), 1e-6)
        z = abs(float(Hh[rr, cc])) % q_h01
        return 1 if z >= 0.5 * q_h01 else 0

    raise ValueError("Unknown candidate mode.")


def _candidate_attack_score(block0: np.ndarray, block_cand: np.ndarray, mode: int, bit: int, params: ProposalParams, pos=None):
    mse = float(np.mean((block0 - block_cand) ** 2))
    base = block_cand.astype(np.float64)

    stress_tests = [
        ("exact", base),
        ("rounded", np.rint(base)),
        ("positive_drift", base + 0.25),
        ("negative_drift", base - 0.25),
    ]
    if not bool(params.fast_candidate_scoring):
        stress_tests.extend([
            ("scale_up", 1.01 * base),
            ("scale_down", 0.99 * base),
            ("strong_positive_drift", base + 0.50),
            ("strong_negative_drift", base - 0.50),
        ])

    pass_count = 0
    total_count = 0
    for _, test_block in stress_tests:
        total_count += 1
        try:
            extracted_bit = _extract_bit_from_candidate_block(test_block, mode, params, pos=pos)
            if int(extracted_bit) == int(bit):
                pass_count += 1
        except Exception:
            pass

    survival_rate = float(pass_count / max(total_count, 1))
    ok = survival_rate >= float(params.min_survival_rate)
    return survival_rate, mse, survival_rate, survival_rate, survival_rate, ok


def _candidate_mse_limit(cand: dict, params: ProposalParams) -> float:
    if int(cand["flag"]) == FLAG_Q4:
        return max(float(params.max_q_cand_mse), EPS_CONF)
    if int(cand["flag"]) == FLAG_HPOS:
        return max(float(params.max_h_cand_mse), EPS_CONF)
    return 1.0


def _combined_selection_score(cand: dict, params: ProposalParams) -> float:
    bss = float(cand["score"])
    mse = float(cand["mse"])
    normalized_mse = min(mse / _candidate_mse_limit(cand, params), 2.0)
    return float(params.bss_weight) * bss - float(params.mse_weight) * normalized_mse


def _candidate_rank_key(cand: dict, params: ProposalParams):
    return (_combined_selection_score(cand, params), float(cand["score"]), -float(cand["mse"]))


def _candidate_is_strong_enough(cand: dict, params: ProposalParams) -> bool:
    if not bool(cand["ok"]):
        return False
    if float(cand["score"]) < float(params.min_survival_rate):
        return False
    if int(cand["flag"]) == FLAG_Q4 and float(cand["mse"]) > float(params.max_q_cand_mse):
        return False
    if int(cand["flag"]) == FLAG_HPOS and float(cand["mse"]) > float(params.max_h_cand_mse):
        return False
    return True


def _best_hpos_candidate(block0: np.ndarray, bit: int, params: ProposalParams) -> dict:
    all_h: list[dict] = []
    valid_h: list[dict] = []

    for hpos_idx, (hname, pos) in enumerate(HPOS_CANDIDATES):
        block_h = _build_hpos_candidate(block0, bit, params, pos)
        score_h, mse_h, raw_h, avg_h, worst_h, ok_h = _candidate_attack_score(block0, block_h, FLAG_HPOS, bit, params, pos=pos)
        cand = {
            "flag": FLAG_HPOS,
            "block": block_h,
            "score": score_h,
            "mse": mse_h,
            "raw": raw_h,
            "avg": avg_h,
            "worst": worst_h,
            "ok": ok_h,
            "hpos_idx": hpos_idx,
            "hpos_name": hname,
            "pos": pos,
        }
        all_h.append(cand)
        if _candidate_is_strong_enough(cand, params):
            valid_h.append(cand)

    if valid_h:
        return min(valid_h, key=lambda c: float(c["mse"]))
    return min(all_h, key=lambda c: float(c["mse"]))


# =========================================================
# Block selection and structured repetition
# =========================================================
def _all_block_positions(bands_by_ch: dict[int, dict[str, np.ndarray]], params: ProposalParams) -> list[tuple[int, str, int, int]]:
    positions: list[tuple[int, str, int, int]] = []
    bs = int(params.block_size)
    for ch in params.host_channels:
        for band_name in params.dwt_bands:
            band = bands_by_ch[int(ch)][band_name]
            h0 = (band.shape[0] // bs) * bs
            w0 = (band.shape[1] // bs) * bs
            for i in range(0, h0, bs):
                for j in range(0, w0, bs):
                    positions.append((int(ch), band_name, i, j))
    return positions


def _shuffle_positions(positions: list[tuple[int, str, int, int]], private_key: str) -> list[tuple[int, str, int, int]]:
    key_bytes = hashlib.sha256(str(private_key).encode("utf-8")).digest()
    seed_int = int.from_bytes(key_bytes, "big")
    rng = random.Random(seed_int)
    indices = list(range(len(positions)))
    rng.shuffle(indices)
    return [positions[idx] for idx in indices]


def _make_structured_schedule(blocks: list[tuple[int, str, int, int]], payload_len: int, repeat: int | None = None):
    payload_len = int(payload_len)
    if payload_len <= 0:
        raise ValueError("payload_len must be positive.")

    max_repeat = len(blocks) // payload_len
    if max_repeat <= 0:
        raise ValueError(f"Capacity insufficient: payload_len={payload_len}, total_blocks={len(blocks)}")

    # Faithful source-script mode: repeat=None means all usable capacity.
    repeat_factor = max_repeat if repeat is None else max(1, min(int(repeat), max_repeat))
    usable_blocks = repeat_factor * payload_len
    selected_blocks = list(blocks[:usable_blocks])

    schedule: list[tuple[int, int, str, int, int]] = []
    for rep in range(repeat_factor):
        base = rep * payload_len
        for kpos in range(payload_len):
            ch, band_name, i, j = selected_blocks[base + kpos]
            schedule.append((kpos, ch, band_name, i, j))

    return schedule, repeat_factor, usable_blocks


# =========================================================
# Main class
# =========================================================
class ProposalQHDWTHess:
    """Proposal method patched to match the user's standalone Python script.

    Matched algorithmic choices:
      - RGB benchmark input is internally converted to OpenCV BGR, then to YCrCb.
      - Only Y channel is used for DWT embedding/extraction.
      - Watermark is DWT-transformed and only its LL subband is embedded.
      - LH/HL/HH watermark detail subbands are retained in metadata and reused
        during reconstruction, exactly like the uploaded script.
      - Full structured repetition is the default.
      - Q4/H-position candidates, BSS, MSE filtering, and majority voting follow
        the uploaded script.
    """

    name = "Proposal_QH_DWT_Hess"

    def __init__(
        self,
        params: ProposalParams | dict[str, Any] | None = None,
        *,
        use_optimizer: bool = False,
        optimizer_trials: int = 4,
        optimizer_seed: int = 123,
    ):
        self.params = params if isinstance(params, ProposalParams) else ProposalParams.from_dict(params)
        self.use_optimizer = bool(use_optimizer)
        self.optimizer_trials = int(optimizer_trials)
        self.optimizer_seed = int(optimizer_seed)

    def _prepare_embed_space_from_rgb(self, host_rgb: np.ndarray):
        host_bgr = _rgb_to_source_bgr(host_rgb)
        host_embed_orig = _host_bgr_to_embed_space(host_bgr)
        bands_by_ch: dict[int, dict[str, np.ndarray]] = {}
        H0 = W0 = None
        for ch in self.params.host_channels:
            work, h0, w0 = _crop_for_dwt(host_embed_orig[:, :, int(ch)], DWT_LEVEL, self.params.block_size)
            if H0 is None:
                H0, W0 = h0, w0
            bands_by_ch[int(ch)] = _dwt_split_4bands(work, self.params)
        return host_bgr, host_embed_orig, bands_by_ch, int(H0), int(W0)

    def _embed_with_params(self, host_rgb: np.ndarray, watermark_binary: np.ndarray, params: ProposalParams):
        self.params = params
        host_rgb = _ensure_uint8_rgb_basic(host_rgb)
        if host_rgb.shape[0] != HOST_SIZE or host_rgb.shape[1] != HOST_SIZE:
            raise ValueError(f"Host image must be exactly {HOST_SIZE}x{HOST_SIZE}. Got {host_rgb.shape[1]}x{host_rgb.shape[0]}.")

        host_bgr, host_embed_orig, bands_by_ch, H0, W0 = self._prepare_embed_space_from_rgb(host_rgb)
        wm_binary, payload_bits, wm_meta = prepare_binary_watermark_payload_from_array(watermark_binary, params)
        L = int(payload_bits.size)

        positions = _shuffle_positions(_all_block_positions(bands_by_ch, params), params.private_key)
        if len(positions) < L:
            raise ValueError(f"Capacity insufficient: need {L} watermark payload bits but only {len(positions)} host blocks are available.")

        repeat_arg = None if params.structured_repetition_enabled else 1
        if params.repeat is not None:
            # Non-default quick mode remains possible, but default None is source-script faithful.
            repeat_arg = params.repeat
        schedule, repeat_factor, usable_blocks = _make_structured_schedule(positions, L, repeat_arg)

        flags: list[int] = []
        hpos_list: list[int] = []
        support_counts = np.zeros(L, dtype=np.int32)
        support_sel_sum = np.zeros(L, dtype=np.float64)
        bs = int(params.block_size)

        for kpos, ch, band_name, i, j in schedule:
            bit = int(payload_bits[kpos])
            band = bands_by_ch[int(ch)][band_name]
            block0 = band[i : i + bs, j : j + bs].copy()

            try:
                block_q4 = _build_q4_candidate(block0, bit, params)
                score_q4, mse_q4, raw_q4, avg_q4, worst_q4, ok_q4 = _candidate_attack_score(block0, block_q4, FLAG_Q4, bit, params)
                q_cand = {
                    "flag": FLAG_Q4,
                    "block": block_q4,
                    "score": score_q4,
                    "mse": mse_q4,
                    "raw": raw_q4,
                    "avg": avg_q4,
                    "worst": worst_q4,
                    "ok": ok_q4,
                    "hpos_idx": HPOS_NONE,
                    "hpos_name": "none",
                }

                best_h = _best_hpos_candidate(block0, bit, params)
                cands = [q_cand, best_h]
                strong_cands = [c for c in cands if _candidate_is_strong_enough(c, params)]

                if strong_cands:
                    best = sorted(strong_cands, key=lambda c: _candidate_rank_key(c, params), reverse=True)[0]
                    band[i : i + bs, j : j + bs] = best["block"]
                    flags.append(int(best["flag"]))
                    hpos_list.append(int(best.get("hpos_idx", HPOS_NONE)))
                    support_counts[kpos] += 1
                    support_sel_sum[kpos] += float(_combined_selection_score(best, params))
                else:
                    flags.append(FLAG_SKIP)
                    hpos_list.append(HPOS_NONE)
            except Exception:
                flags.append(FLAG_SKIP)
                hpos_list.append(HPOS_NONE)

        watermarked_embed = host_embed_orig.copy()
        for ch in params.host_channels:
            rec_channel = _dwt_merge_4bands(bands_by_ch[int(ch)], params)
            rec_channel = rec_channel[:H0, :W0]
            out_channel = host_embed_orig[:, :, int(ch)].astype(np.float64).copy()
            out_channel[:H0, :W0] = np.clip(np.rint(rec_channel), 0, 255)
            watermarked_embed[:, :, int(ch)] = out_channel.astype(np.uint8)

        watermarked_bgr = _embed_space_to_host_bgr(watermarked_embed)
        watermarked_rgb = _source_bgr_to_rgb(watermarked_bgr)

        q4_used, hpos_used, skip_used = _flag_stats(flags)
        wm_meta.update(
            {
                "structured_repetition_enabled": bool(params.structured_repetition_enabled),
                "structured_repeat_factor": int(repeat_factor),
                "structured_usable_blocks": int(usable_blocks),
                "dwt_bands": list(params.dwt_bands),
                "host_embed_color_space": str(params.host_embed_color_space),
                "host_embed_channels": list(params.host_channels),
                "host_embed_channel_description": "Y channel only after RGB_to_BGR_then_BGR_to_YCrCb",
                "q4_used": int(q4_used),
                "hpos_used": int(hpos_used),
                "skip_used": int(skip_used),
                "total_flags": int(len(flags)),
                "parameter_snapshot": _to_json_safe(params.to_dict()),
                "block_hash": "SHA-256",
                "h_candidates": [name for name, _ in HPOS_CANDIDATES],
                "weighted_voting_removed": True,
                "confidence_scoring_removed": True,
                "bit_survival_score": "BSS = passed_stress_tests / total_stress_tests",
                "logistic_removed": True,
                "rsa_removed": True,
            }
        )

        key = ProposalKey(
            flags=[int(x) for x in flags],
            hpos_list=[int(x) for x in hpos_list],
            payload_meta=_to_json_safe(wm_meta),
            params=ProposalParams.from_dict(params.to_dict()),
            repeat_factor=int(repeat_factor),
            usable_blocks=int(usable_blocks),
            total_blocks=int(len(positions)),
            support_counts=support_counts.astype(int).tolist(),
            schedule=schedule,
        )
        return watermarked_rgb, key

    def _evaluate_params_for_optimization(
        self,
        host_rgb: np.ndarray,
        watermark_binary: np.ndarray,
        params: ProposalParams,
        attack_suite: list[AttackConfig] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one candidate parameter set.

        Objective mirrors the standalone script: if clean PSNR is above the
        threshold, maximize 1 + mean attack NC; otherwise penalize the PSNR gap.
        """
        if attack_suite is None:
            attack_suite = default_attack_suite(include_none=False, preset="lite")

        result = {
            "params": optimization_param_snapshot(params),
            "objective": -1e9,
            "clean_psnr": 0.0,
            "clean_ssim": 0.0,
            "clean_nc": 0.0,
            "clean_ber": 1.0,
            "mean_attack_nc": 0.0,
            "min_attack_nc": 0.0,
            "psnr_threshold": 55.0,
            "psnr_feasible": False,
            "psnr_gap": None,
            "q4_used": None,
            "hpos_used": None,
            "skip_used": None,
            "repeat_factor": None,
            "attack_results": [],
            "error": None,
        }
        try:
            watermarked, key = self._embed_with_params(host_rgb, watermark_binary, params)
            clean_ext = self.extract(watermarked, key, host_rgb=host_rgb)
            clean_psnr = float(metric_psnr(host_rgb, watermarked))
            clean_ssim = float(metric_ssim(host_rgb, watermarked))
            clean_nc = float(metric_nc(watermark_binary, clean_ext))
            clean_ber = float(metric_ber(watermark_binary, clean_ext))
            q4_used, hpos_used, skip_used = _flag_stats(key.flags)

            attack_ncs: list[float] = []
            for attack in attack_suite:
                if attack.name == "no_attack":
                    continue
                rec = {"attack": attack.name, "nc": 0.0, "ber": 1.0, "error": None}
                try:
                    attacked = apply_benchmark_attack(watermarked, attack)
                    ext = self.extract(attacked, key, host_rgb=host_rgb)
                    rec["nc"] = float(metric_nc(watermark_binary, ext))
                    rec["ber"] = float(metric_ber(watermark_binary, ext))
                    attack_ncs.append(float(rec["nc"]))
                except Exception as e:
                    rec["error"] = repr(e)
                    attack_ncs.append(0.0)
                result["attack_results"].append(rec)

            mean_attack_nc = float(np.mean(attack_ncs)) if attack_ncs else clean_nc
            min_attack_nc = float(np.min(attack_ncs)) if attack_ncs else clean_nc
            threshold = float(result["psnr_threshold"])
            feasible = bool(np.isfinite(clean_psnr) and clean_psnr > threshold)
            if feasible:
                objective = 1.0 + mean_attack_nc
                psnr_gap = 0.0
            else:
                psnr_gap = max(0.0, threshold - (clean_psnr if np.isfinite(clean_psnr) else 0.0))
                objective = -(psnr_gap / max(threshold, EPS_CONF)) + 0.001 * mean_attack_nc

            result.update({
                "objective": float(objective),
                "clean_psnr": clean_psnr if np.isfinite(clean_psnr) else 0.0,
                "clean_ssim": clean_ssim if np.isfinite(clean_ssim) else 0.0,
                "clean_nc": clean_nc if np.isfinite(clean_nc) else 0.0,
                "clean_ber": clean_ber if np.isfinite(clean_ber) else 1.0,
                "mean_attack_nc": mean_attack_nc,
                "min_attack_nc": min_attack_nc,
                "psnr_feasible": feasible,
                "psnr_gap": float(psnr_gap),
                "q4_used": int(q4_used),
                "hpos_used": int(hpos_used),
                "skip_used": int(skip_used),
                "repeat_factor": int(key.repeat_factor),
            })
        except Exception as e:
            result["error"] = repr(e)
        return result

    def optimize_params(
        self,
        host_rgb: np.ndarray,
        watermark_binary: np.ndarray,
        *,
        n_fireflies: int | None = None,
        n_generations: int = 2,
        alpha: float = 0.18,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha_decay: float = 0.80,
        seed: int | None = None,
        attack_suite: list[AttackConfig] | None = None,
    ) -> tuple[ProposalParams, dict[str, Any]]:
        """Run the two-phase optimization algorithm for one image.

        This is an array-based Firefly implementation of the same four-parameter
        search space as the standalone script. It returns the best parameters
        and a serializable optimization record.
        """
        n_fireflies = int(n_fireflies if n_fireflies is not None else self.optimizer_trials)
        n_generations = int(n_generations)
        rng = np.random.default_rng(int(seed if seed is not None else self.optimizer_seed))
        dim = len(PARAM_SPECS)
        fireflies = rng.uniform(0.0, 1.0, size=(max(1, n_fireflies), dim))
        scores = np.full((fireflies.shape[0],), -np.inf, dtype=np.float64)
        cache: dict[tuple[float, ...], dict[str, Any]] = {}
        history: list[dict[str, Any]] = []
        best_score = -np.inf
        best_position = fireflies[0].copy()
        best_result: dict[str, Any] | None = None
        current_alpha = float(alpha)

        def evaluate_position(pos, generation: int, firefly_id: int):
            nonlocal best_score, best_position, best_result
            key = firefly_cache_key(pos)
            if key in cache:
                return cache[key]
            params = decode_firefly_position(pos, self.params)
            res = self._evaluate_params_for_optimization(host_rgb, watermark_binary, params, attack_suite=attack_suite)
            res["generation"] = int(generation)
            res["firefly_id"] = int(firefly_id)
            res["position"] = [float(x) for x in np.clip(pos, 0.0, 1.0).tolist()]
            cache[key] = res
            hist = {k: res.get(k) for k in [
                "generation", "firefly_id", "objective", "clean_psnr", "psnr_feasible",
                "psnr_gap", "mean_attack_nc", "min_attack_nc", "clean_nc", "clean_ber",
                "q4_used", "hpos_used", "skip_used", "repeat_factor", "error"
            ]}
            hist["params"] = res.get("params")
            history.append(hist)
            if float(res.get("objective", -1e9)) > best_score:
                best_score = float(res.get("objective", -1e9))
                best_position = np.clip(pos.copy(), 0.0, 1.0)
                best_result = dict(res)
            return res

        for i in range(fireflies.shape[0]):
            res = evaluate_position(fireflies[i], 0, i)
            scores[i] = float(res.get("objective", -1e9))

        for gen in range(1, n_generations + 1):
            for i in range(fireflies.shape[0]):
                for j in range(fireflies.shape[0]):
                    if scores[j] > scores[i]:
                        rij = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = float(beta0) * math.exp(-float(gamma) * (rij ** 2))
                        random_step = current_alpha * (rng.random(dim) - 0.5)
                        new_pos = np.clip(fireflies[i] + beta * (fireflies[j] - fireflies[i]) + random_step, 0.0, 1.0)
                        new_res = evaluate_position(new_pos, gen, i)
                        new_score = float(new_res.get("objective", -1e9))
                        if new_score > scores[i]:
                            fireflies[i] = new_pos
                            scores[i] = new_score
            current_alpha *= float(alpha_decay)

        best_params = decode_firefly_position(best_position, self.params)
        final_result = self._evaluate_params_for_optimization(host_rgb, watermark_binary, best_params, attack_suite=attack_suite)
        record = {
            "best_position": [float(x) for x in best_position.tolist()],
            "best_params": optimization_param_snapshot(best_params),
            "best_result": final_result,
            "history": history,
            "optimizer": {
                "algorithm": "firefly",
                "n_fireflies": int(n_fireflies),
                "n_generations": int(n_generations),
                "alpha": float(alpha),
                "beta0": float(beta0),
                "gamma": float(gamma),
                "alpha_decay": float(alpha_decay),
                "seed": int(seed if seed is not None else self.optimizer_seed),
                "param_specs": PARAM_SPECS,
            },
        }
        if best_result is not None:
            record["search_best_result"] = best_result
        return best_params, record

    def _quick_optimize_params(self, host_rgb: np.ndarray, watermark_binary: np.ndarray) -> ProposalParams:
        params, _record = self.optimize_params(host_rgb, watermark_binary)
        return params

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        params = self._quick_optimize_params(host_rgb, watermark_binary) if self.use_optimizer else self.params
        return self._embed_with_params(host_rgb, watermark_binary, params)

    def _extract_payload_bits_from_rgb(self, possibly_attacked_rgb: np.ndarray, key: ProposalKey):
        params = key.params
        payload_meta = key.payload_meta
        payload_len = int(payload_meta["payload_len"])
        dwt_bands_for_meta = tuple(payload_meta.get("dwt_bands", params.dwt_bands))

        img_bgr = _rgb_to_source_bgr(possibly_attacked_rgb)
        embed_img = _host_bgr_to_embed_space(img_bgr)
        _, H0, W0 = _crop_for_dwt(embed_img[:, :, int(params.host_channels[0])], DWT_LEVEL, params.block_size)
        work_raw_u8 = embed_img[:H0, :W0, :].copy()

        local_params = ProposalParams.from_dict(params.to_dict())
        local_params.dwt_bands = tuple(dwt_bands_for_meta)
        bands_by_ch: dict[int, dict[str, np.ndarray]] = {}
        for ch in local_params.host_channels:
            bands_by_ch[int(ch)] = _dwt_split_4bands(work_raw_u8[:, :, int(ch)].astype(np.float64), local_params)

        positions = _shuffle_positions(_all_block_positions(bands_by_ch, local_params), local_params.private_key)
        if payload_meta.get("structured_repetition_enabled", True):
            schedule, repeat_factor, usable_blocks = _make_structured_schedule(positions, payload_len, None)
        else:
            schedule, repeat_factor, usable_blocks = _make_structured_schedule(positions, payload_len, 1)

        n = min(len(schedule), len(key.flags), len(key.hpos_list))
        votes = [[] for _ in range(payload_len)]
        bs = int(local_params.block_size)

        for idx in range(n):
            kpos, ch, band_name, i, j = schedule[idx]
            flag = int(key.flags[idx])
            if flag == FLAG_SKIP:
                continue
            try:
                block = bands_by_ch[int(ch)][band_name][i : i + bs, j : j + bs]
                if block.shape != (bs, bs):
                    continue
                if flag == FLAG_Q4:
                    extracted_bit = _extract_bit_from_candidate_block(block, FLAG_Q4, local_params)
                elif flag == FLAG_HPOS:
                    hpos_idx = int(key.hpos_list[idx]) if idx < len(key.hpos_list) else HPOS_NONE
                    if not (0 <= hpos_idx < len(HPOS_CANDIDATES)):
                        continue
                    _, pos = HPOS_CANDIDATES[hpos_idx]
                    extracted_bit = _extract_bit_from_candidate_block(block, FLAG_HPOS, local_params, pos=pos)
                else:
                    continue
                votes[int(kpos)].append(int(extracted_bit))
            except Exception:
                continue

        out_bits = np.zeros(payload_len, dtype=np.uint8)
        valid_count = 0
        total_votes = 0
        for kpos, bit_votes in enumerate(votes):
            if not bit_votes:
                out_bits[kpos] = 0
                continue
            valid_count += 1
            total_votes += len(bit_votes)
            ones = int(np.sum(bit_votes))
            zeros = len(bit_votes) - ones
            out_bits[kpos] = 1 if ones >= zeros else 0

        coverage = float(valid_count / payload_len) if payload_len > 0 else 0.0
        return out_bits, coverage

    def extract(self, possibly_attacked_rgb: np.ndarray, key: ProposalKey, host_rgb: np.ndarray | None = None):
        best_bits, _coverage = self._extract_payload_bits_from_rgb(possibly_attacked_rgb, key)
        wm_rec = reconstruct_binary_watermark_from_payload_bits(best_bits, key.payload_meta, key.params)
        return _force_binary_watermark_exact(wm_rec, int(key.payload_meta.get("wm_size", WM_SIZE)), WM_BIN_THRESH)


__all__ = [
    "ProposalQHDWTHess",
    "ProposalParams",
    "ProposalKey",
    "FLAG_Q4",
    "FLAG_HPOS",
    "FLAG_SKIP",
    "HPOS_CANDIDATES",
    "PARAM_SPECS",
    "decode_firefly_position",
    "firefly_cache_key",
    "optimization_param_snapshot",
    "_nearest_nonnegative_mod_value",
    "_q4_stat",
    "_build_q4_candidate",
    "_build_hpos_candidate",
    "_extract_bit_from_candidate_block",
    "_make_structured_schedule",
    "_shuffle_positions",
    "prepare_binary_watermark_payload_from_array",
    "reconstruct_binary_watermark_from_payload_bits",
]
