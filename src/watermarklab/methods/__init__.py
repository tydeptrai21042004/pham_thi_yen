from __future__ import annotations
from typing import Any

from watermarklab.methods.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from watermarklab.methods.guo2017_dwt_qr_fa import Guo2017DWTQRFA
from watermarklab.methods.gaata2022_dwt_hess_fwa import Gaata2022DWTHessFWA
from watermarklab.methods.dwt_hd_svd2025 import DWTHDSVD2025
from watermarklab.methods.hess_nha2023 import HessNha2023Hessenberg
from watermarklab.methods.roy2018_dwt_svd import Roy2018DWTSVD
from watermarklab.methods.dwt_wht_svd2024 import DWTWHTSVD2024
from watermarklab.methods.qwt_qsvd_zhang2022 import QWTQSVDZhang2022
from watermarklab.methods.zhu2021_iwt_svd import Zhu2021IWTSVD
from watermarklab.methods.proposal_qh_dwt_hess import ProposalQHDWTHess, ProposalParams

BASELINE_METHOD_IDS = [
    "kumar2021",
    "guo2017_dwt_qr_fa",
    "gaata2022_dwt_hess_fwa",
    "dwt_hd_svd_2025",
    "hess_nha2023",
    "roy2018_dwt_svd",
    "dwt_wht_svd_2024",
    "qwt_qsvd_zhang2022_blind",
    "qwt_qsvd_zhang2022_semiblind",
    "zhu2021_iwt_svd_adapted",
]
DEFAULT_METHOD_IDS = BASELINE_METHOD_IDS + ["proposal"]


def build_methods(
    selected: list[str] | None = None,
    proposal_options: dict[str, Any] | None = None,
    baseline_modes: dict[str, str] | None = None,
    guo_options: dict[str, Any] | None = None,
    gaata_options: dict[str, Any] | None = None,
):
    """Build registered methods.

    selected may be None/['all'], ['baselines'], or an explicit list of method ids.
    """
    proposal_options = dict(proposal_options or {})
    params_data = proposal_options.pop("params", None)
    proposal_params = params_data if isinstance(params_data, ProposalParams) else ProposalParams.from_dict(params_data)
    baseline_modes = dict(baseline_modes or {})
    guo_options = dict(guo_options or {})
    gaata_options = dict(gaata_options or {})

    all_methods = {
        "kumar2021": Kumar2021DWTEntropy(mode=baseline_modes.get("kumar2021", "adapt")),
        "guo2017_dwt_qr_fa": Guo2017DWTQRFA(mode=baseline_modes.get("guo2017_dwt_qr_fa", "adapt"), **guo_options),
        "gaata2022_dwt_hess_fwa": Gaata2022DWTHessFWA(mode=baseline_modes.get("gaata2022_dwt_hess_fwa", "adapt"), **gaata_options),
        "dwt_hd_svd_2025": DWTHDSVD2025(mode=baseline_modes.get("dwt_hd_svd_2025", "adapt")),
        "hess_nha2023": HessNha2023Hessenberg(mode=baseline_modes.get("hess_nha2023", "adapt")),
        "roy2018_dwt_svd": Roy2018DWTSVD(mode=baseline_modes.get("roy2018_dwt_svd", "adapt")),
        "dwt_wht_svd_2024": DWTWHTSVD2024(mode=baseline_modes.get("dwt_wht_svd_2024", "adapt")),
        "qwt_qsvd_zhang2022_blind": QWTQSVDZhang2022(mode=baseline_modes.get("qwt_qsvd_zhang2022_blind", "adapt"), extraction_mode="blind"),
        "qwt_qsvd_zhang2022_semiblind": QWTQSVDZhang2022(mode=baseline_modes.get("qwt_qsvd_zhang2022_semiblind", "adapt"), extraction_mode="semi-blind"),
        "zhu2021_iwt_svd_adapted": Zhu2021IWTSVD(mode=baseline_modes.get("zhu2021_iwt_svd_adapted", "adapt")),
        "proposal": ProposalQHDWTHess(params=proposal_params, **proposal_options),
    }

    if selected is None:
        selected_ids = list(DEFAULT_METHOD_IDS)
    else:
        selected_ids: list[str] = []
        for item in selected:
            if item == "all":
                selected_ids.extend(DEFAULT_METHOD_IDS)
            elif item == "baselines":
                selected_ids.extend(BASELINE_METHOD_IDS)
            else:
                selected_ids.append(item)
        # Preserve user order while removing duplicates.
        selected_ids = list(dict.fromkeys(selected_ids))

    missing = [k for k in selected_ids if k not in all_methods]
    if missing:
        valid = ", ".join(DEFAULT_METHOD_IDS + ["baselines", "all"])
        raise KeyError(f"Unknown method id(s): {missing}. Valid choices: {valid}")
    return {k: all_methods[k] for k in selected_ids}


__all__ = [
    "Kumar2021DWTEntropy",
    "Guo2017DWTQRFA",
    "Gaata2022DWTHessFWA",
    "DWTHDSVD2025",
    "HessNha2023Hessenberg",
    "Roy2018DWTSVD",
    "DWTWHTSVD2024",
    "QWTQSVDZhang2022",
    "Zhu2021IWTSVD",
    "ProposalQHDWTHess",
    "ProposalParams",
    "BASELINE_METHOD_IDS",
    "DEFAULT_METHOD_IDS",
    "build_methods",
]
