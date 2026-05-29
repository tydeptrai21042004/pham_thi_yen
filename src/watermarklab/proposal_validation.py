from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.linalg import hessenberg

from watermarklab.common.metrics import psnr, nc, ber
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalQHDWTHess,
    ProposalParams,
    FLAG_Q4,
    FLAG_HPOS,
    FLAG_SKIP,
    HPOS_CANDIDATES,
    _nearest_nonnegative_mod_value,
    _q4_stat,
    _build_q4_candidate,
    _build_hpos_candidate,
    _extract_bit_from_candidate_block,
    _make_structured_schedule,
    _shuffle_positions,
)


def proposal_notebook_contract() -> dict[str, Any]:
    """Constants and algorithmic choices expected from optimal_dwt_hess.ipynb.

    This is an executable contract. Unit tests call it so that accidental edits
    to the proposal implementation are caught immediately.
    """

    return {
        "host_shape": (512, 512, 3),
        "watermark_shape": (64, 64),
        "watermark_payload_shape": (32, 32),
        "watermark_payload_len": 1024,
        "watermark_dwt_enabled": True,
        "watermark_payload_band": "LL",
        "host_embed_color_space": "YCrCb",
        "block_size": 4,
        "private_key": "KB123",
        "arnold_iterations": 17,
        "dwt_wavelet": "haar",
        "dwt_level": 1,
        "dwt_mode": "orthonormal",
        "dwt_bands": ("LL", "HL", "HH", "LH"),
        "host_channels": (0,),
        "flag_q4": 0,
        "flag_hpos": 1,
        "flag_skip": 2,
        "q4_tau": 0.50,
        "q4_margin": 0.08,
        "h01_q": 7.0,
        "h01_margin": 0.90,
        "hpos_candidates": (("h21", (1, 0)), ("h22", (1, 1))),
        "min_survival_rate": 0.75,
        "structured_repetition": True,
    }


def validate_proposal_params(params: ProposalParams | None = None) -> dict[str, Any]:
    """Validate default proposal parameters against the notebook contract."""

    p = params or ProposalParams()
    c = proposal_notebook_contract()
    checks = {
        "block_size": int(p.block_size) == c["block_size"],
        "private_key": str(p.private_key) == c["private_key"],
        "arnold_iterations": int(p.arnold_iterations) == c["arnold_iterations"],
        "dwt_mode": str(p.dwt_mode) == c["dwt_mode"],
        "dwt_bands": tuple(p.dwt_bands) == tuple(c["dwt_bands"]),
        "host_channels": tuple(p.host_channels) == tuple(c["host_channels"]),
        "watermark_dwt_enabled": bool(p.watermark_dwt_enabled) == c["watermark_dwt_enabled"],
        "watermark_payload_band": str(p.watermark_payload_band) == c["watermark_payload_band"],
        "q4_tau": np.isclose(float(p.q4_tau), c["q4_tau"]),
        "q4_margin": np.isclose(float(p.q4_margin), c["q4_margin"]),
        "h01_q": np.isclose(float(p.h01_q), c["h01_q"]),
        "h01_margin": np.isclose(float(p.h01_margin), c["h01_margin"]),
        "min_survival_rate": np.isclose(float(p.min_survival_rate), c["min_survival_rate"]),
        "hpos_candidates": tuple(HPOS_CANDIDATES) == tuple(c["hpos_candidates"]),
        "flags": (FLAG_Q4, FLAG_HPOS, FLAG_SKIP) == (c["flag_q4"], c["flag_hpos"], c["flag_skip"]),
    }
    return {
        "ok": bool(all(bool(v) for v in checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "params": p.to_dict(),
        "contract": c,
    }


def validate_proposal_math_branches(params: ProposalParams | None = None) -> dict[str, Any]:
    """Check Q4 and H-domain candidate construction independently of images."""

    p = params or ProposalParams(repeat=1)
    # A deterministic non-symmetric 4x4 block selected so both Q4 bit states
    # are reachable with the Givens search. Some arbitrary blocks are handled
    # by the H-domain fallback branch instead, so this test validates Q4 math
    # on a reachable case rather than requiring Q4 to solve every block.
    rng = np.random.default_rng(1)
    block = rng.normal(loc=0.0, scale=3.0, size=(4, 4))

    q_results = []
    h_results = []
    qim_results = []
    for bit in (0, 1):
        q_block = _build_q4_candidate(block, bit, p)
        q_bit = _extract_bit_from_candidate_block(q_block, FLAG_Q4, p)
        q_stat = _q4_stat(hessenberg(q_block, calc_q=True)[1])
        q_results.append({"bit": bit, "extracted": int(q_bit), "q4_stat": float(q_stat), "ok": int(q_bit) == bit})

        target = 0.75 * p.h01_q if bit == 1 else 0.25 * p.h01_q
        qim_value = _nearest_nonnegative_mod_value(13.2, target, p.h01_q)
        residue = float(qim_value % p.h01_q)
        qim_results.append({"bit": bit, "target": float(target), "residue": residue, "ok": abs(residue - target) < 1e-9})

        for hpos_idx, (name, pos) in enumerate(HPOS_CANDIDATES):
            h_block = _build_hpos_candidate(block, bit, p, pos)
            h_bit = _extract_bit_from_candidate_block(h_block, FLAG_HPOS, p, pos=pos)
            h_results.append({"bit": bit, "candidate": name, "extracted": int(h_bit), "ok": int(h_bit) == bit})

    checks = {
        "q4_candidate_extracts_requested_bit": all(x["ok"] for x in q_results),
        "h_candidate_extracts_requested_bit": all(x["ok"] for x in h_results),
        "qim_residue_targets_are_exact": all(x["ok"] for x in qim_results),
    }
    return {"ok": all(checks.values()), "checks": checks, "q4": q_results, "hpos": h_results, "qim": qim_results}


def validate_structured_schedule(total_blocks: int = 16384, payload_len: int = 1024, repeat: int | None = None) -> dict[str, Any]:
    """Validate deterministic structured repetition schedule math.

    The source-script-faithful proposal uses 512x512 host images, one 1-level Haar split
    on the Y channel and four subbands. With 4x4 blocks this gives 16,384
    candidate blocks. The watermark-side DWT embeds only the 32x32 LL payload,
    so the maximum repeat factor is 16.
    """

    blocks = [(0, "LL", i, 0) for i in range(int(total_blocks))]
    schedule, repeat_factor, usable_blocks = _make_structured_schedule(blocks, payload_len, repeat)
    counts = np.zeros(payload_len, dtype=np.int32)
    for bit_idx, *_ in schedule:
        counts[int(bit_idx)] += 1
    checks = {
        "schedule_length_equals_usable_blocks": len(schedule) == usable_blocks,
        "usable_blocks_equals_repeat_times_payload": usable_blocks == repeat_factor * payload_len,
        "every_payload_bit_has_equal_repetition": int(counts.min()) == int(counts.max()) == int(repeat_factor),
        "repeat_factor_within_capacity": repeat_factor <= total_blocks // payload_len,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "repeat_factor": int(repeat_factor),
        "usable_blocks": int(usable_blocks),
        "total_blocks": int(total_blocks),
        "payload_len": int(payload_len),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
    }


def validate_proposal_end_to_end(host_rgb: np.ndarray, watermark_binary: np.ndarray, params: ProposalParams | None = None) -> dict[str, Any]:
    """Run a real no-attack embed/extract check and return a readable report."""

    p = params or ProposalParams(repeat=1)
    method = ProposalQHDWTHess(params=p, use_optimizer=False)
    watermarked, key = method.embed(host_rgb, watermark_binary)
    extracted = method.extract(watermarked, key, host_rgb=host_rgb)
    wm_u8 = (np.asarray(watermark_binary) >= 127).astype(np.uint8) * 255
    flags = np.asarray(key.flags, dtype=np.int32)
    q = int(np.sum(flags == FLAG_Q4))
    h = int(np.sum(flags == FLAG_HPOS))
    skip = int(np.sum(flags == FLAG_SKIP))
    support = np.asarray(key.support_counts, dtype=np.int32)
    out = {
        "ok": True,
        "psnr": float(psnr(host_rgb, watermarked)),
        "nc": float(nc(wm_u8, extracted)),
        "ber": float(ber(wm_u8, extracted)),
        "repeat_factor": int(key.repeat_factor),
        "usable_blocks": int(key.usable_blocks),
        "total_blocks": int(key.total_blocks),
        "flags_total": int(len(key.flags)),
        "q4_used": q,
        "h_used": h,
        "skip_used": skip,
        "skip_rate": float(skip / max(len(key.flags), 1)),
        "support_min": int(support.min()) if support.size else 0,
        "support_max": int(support.max()) if support.size else 0,
        "optimizer_used": False,
    }
    out["checks"] = {
        "watermarked_shape_matches_host": tuple(watermarked.shape) == tuple(np.asarray(host_rgb).shape),
        "extracted_shape_matches_watermark": tuple(extracted.shape) == tuple(np.asarray(watermark_binary).shape),
        "clean_psnr_above_40db": out["psnr"] > 40.0,
        "clean_nc_above_0p90": out["nc"] > 0.90,
        "clean_ber_below_0p10": out["ber"] < 0.10,
        "uses_at_least_one_candidate_mode": (q + h) > 0,
        "skip_rate_below_25_percent": out["skip_rate"] < 0.25,
        "support_counts_do_not_exceed_repeat": out["support_max"] <= int(key.repeat_factor),
    }
    out["ok"] = bool(all(out["checks"].values()))
    return out


def run_full_proposal_validation(host_rgb: np.ndarray | None = None, watermark_binary: np.ndarray | None = None) -> dict[str, Any]:
    """Run all non-I/O proposal validation layers."""

    report = {
        "params": validate_proposal_params(),
        "math_branches": validate_proposal_math_branches(),
        "structured_schedule": validate_structured_schedule(),
    }
    if host_rgb is not None and watermark_binary is not None:
        report["end_to_end"] = validate_proposal_end_to_end(host_rgb, watermark_binary)
    report["ok"] = all(part.get("ok", False) for part in report.values() if isinstance(part, dict))
    return report


__all__ = [
    "proposal_notebook_contract",
    "validate_proposal_params",
    "validate_proposal_math_branches",
    "validate_structured_schedule",
    "validate_proposal_end_to_end",
    "run_full_proposal_validation",
]
