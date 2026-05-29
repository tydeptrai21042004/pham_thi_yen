from __future__ import annotations
from typing import Any

from watermarklab.methods.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from watermarklab.methods.guo2017_dwt_qr_fa import Guo2017DWTQRFA
from watermarklab.methods.gaata2022_dwt_hess_fwa import Gaata2022DWTHessFWA
from watermarklab.methods.dwt_hd_svd2025 import DWTHDSVD2025
from watermarklab.methods.proposal_qh_dwt_hess import ProposalQHDWTHess, ProposalParams

BASELINE_METHOD_IDS = [
    "kumar2021",
    "guo2017_dwt_qr_fa",
    "gaata2022_dwt_hess_fwa",
    "dwt_hd_svd_2025",
]
DEFAULT_METHOD_IDS = BASELINE_METHOD_IDS + ["proposal"]


def build_methods(selected: list[str] | None = None, proposal_options: dict[str, Any] | None = None, baseline_modes: dict[str, str] | None = None, guo_options: dict[str, Any] | None = None):
    """Build registered methods.

    selected may be None/['all'], ['baselines'], or an explicit list of method ids.
    """
    proposal_options = dict(proposal_options or {})
    params_data = proposal_options.pop("params", None)
    proposal_params = params_data if isinstance(params_data, ProposalParams) else ProposalParams.from_dict(params_data)
    baseline_modes = dict(baseline_modes or {})
    guo_options = dict(guo_options or {})

    all_methods = {
        "kumar2021": Kumar2021DWTEntropy(mode=baseline_modes.get("kumar2021", "adapt")),
        "guo2017_dwt_qr_fa": Guo2017DWTQRFA(mode=baseline_modes.get("guo2017_dwt_qr_fa", "adapt"), **guo_options),
        "gaata2022_dwt_hess_fwa": Gaata2022DWTHessFWA(mode=baseline_modes.get("gaata2022_dwt_hess_fwa", "adapt")),
        "dwt_hd_svd_2025": DWTHDSVD2025(mode=baseline_modes.get("dwt_hd_svd_2025", "adapt")),
        "proposal": ProposalQHDWTHess(params=proposal_params, **proposal_options),
    }

    if selected is None or selected == ["all"]:
        return {k: all_methods[k] for k in DEFAULT_METHOD_IDS}
    if selected == ["baselines"]:
        return {k: all_methods[k] for k in BASELINE_METHOD_IDS}

    missing = [k for k in selected if k not in all_methods]
    if missing:
        valid = ", ".join(DEFAULT_METHOD_IDS + ["baselines", "all"])
        raise KeyError(f"Unknown method id(s): {missing}. Valid choices: {valid}")
    return {k: all_methods[k] for k in selected}


__all__ = [
    "Kumar2021DWTEntropy",
    "Guo2017DWTQRFA",
    "Gaata2022DWTHessFWA",
    "DWTHDSVD2025",
    "ProposalQHDWTHess",
    "ProposalParams",
    "BASELINE_METHOD_IDS",
    "DEFAULT_METHOD_IDS",
    "build_methods",
]
