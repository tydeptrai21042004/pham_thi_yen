from __future__ import annotations
from typing import Any
from pathlib import Path
import pandas as pd

BASELINE_IDS = {"kumar2021", "guo2017_dwt_qr_fa", "gaata2022_dwt_hess_fwa", "dwt_hd_svd_2025", "hess_nha2023", "roy2018_dwt_svd"}

def _r(method_id: str, paper: str, table: str, phase: str, image: str, attack: str, metric: str, value: Any, unit: str = "", note: str = ""):
    return {"method_id": method_id, "paper": paper, "table": table, "mode": "original_reported", "phase": phase, "image": image, "attack": attack, "metric": metric, "value": value, "unit": unit, "note": note}

def _mean(vals):
    return sum(vals) / len(vals)

# Compact exact data copied from the papers. It includes the key before/after-attack
# values needed for manuscript tables. Full per-image tables can be extended here.
def paper_reported_rows(method_ids: list[str] | None = None) -> list[dict[str, Any]]:
    ids = set(method_ids or BASELINE_IDS)
    rows: list[dict[str, Any]] = []
    if "kumar2021" in ids:
        p = "Kumar and Singh 2021"
        rows += [
            _r("kumar2021", p, "Table 5 average", "before_attack", "Average", "no_attack", "PSNR", 51.6145, "dB"),
            _r("kumar2021", p, "Table 5 average", "before_attack", "Average", "no_attack", "SSIM", 0.9992),
            _r("kumar2021", p, "Table 5 average", "before_attack", "Average", "no_attack", "NCC_cover_watermarked", 0.9998),
            _r("kumar2021", p, "Table 7", "before_attack", "Average", "no_attack", "NCC_watermark", 1.0),
            _r("kumar2021", p, "Table 8", "before_attack", "Average", "no_attack", "BER", 0.0422, note="Paper states average BER under no attack is 0.0422"),
            _r("kumar2021", p, "Section 6.3", "metadata", "Average", "runtime", "runtime", 0.453745, "sec"),
            _r("kumar2021", p, "Section 6.4", "metadata", "All", "payload", "payload", 0.04166, "bpp"),
        ]
        # Exact mean after-attack values computed from Tables 7 and 8 in the paper.
        kumar_ncc = {
            "salt_pepper_0.02": [0.9910,0.9926,0.9895,0.9898,0.9861,0.9884,0.9896],
            "salt_pepper_0.05": [0.9790,0.9815,0.9704,0.9782,0.9656,0.9717,0.9818],
            "gaussian_noise": [0.9867,0.9886,0.9878,0.9882,0.9927,0.9875,0.9880],
            "speckle_noise": [0.9826,0.9851,0.9945,0.9809,0.9725,0.9917,0.9835],
            "gaussian_filtering": [0.9992,0.9997,0.9998,0.9999,0.9995,0.9997,0.9971],
            "median_filtering": [0.9950,0.9991,0.9991,0.9993,0.9987,0.9985,0.9869],
            "histogram_equalization": [0.9989,0.9999,0.9995,0.9934,0.9972,0.9997,0.9973],
            "sharpening": [0.9976,0.9992,0.9995,0.9996,0.9987,0.9990,0.9915],
            "rotation_10deg": [0.9929,0.9989,0.9979,0.9983,0.9978,0.9971,0.9830],
            "crop_10pct": [1,1,1,1,1,1,0.9945],
            "jpeg_qf90": [0.9999,0.9998,0.9999,0.9999,0.9999,0.9995,0.9985],
            "jpeg_qf70": [0.9992,0.9992,0.9995,0.9996,0.9994,0.9988,0.9928],
            "jpeg_qf50": [0.9970,0.9991,0.9992,0.9994,0.9986,0.9986,0.9888],
        }
        for attack, vals in kumar_ncc.items():
            rows.append(_r("kumar2021", p, "Table 7 mean", "after_attack", "Average", attack, "NCC_watermark", round(_mean(vals), 6)))
    if "guo2017_dwt_qr_fa" in ids:
        p = "Guo, Li and Goel 2017"
        rows += [
            _r("guo2017_dwt_qr_fa", p, "Table 3 mean", "before_attack", "Lena", "Mean", "SSIM", 0.9352),
            _r("guo2017_dwt_qr_fa", p, "Table 3 mean", "before_attack", "Lena", "Mean", "PSNR", 37.9956, "dB"),
            _r("guo2017_dwt_qr_fa", p, "Table 3 mean", "before_attack", "Lena", "Mean", "NC_cover_watermarked", 0.9997),
            _r("guo2017_dwt_qr_fa", p, "Table 3 mean", "before_attack", "Lena", "Mean", "Corr_cover_watermarked", 0.9979),
            _r("guo2017_dwt_qr_fa", p, "Table 4 mean", "before_attack", "Elaine", "Mean", "SSIM", 0.9413),
            _r("guo2017_dwt_qr_fa", p, "Table 4 mean", "before_attack", "Elaine", "Mean", "PSNR", 37.5005, "dB"),
            _r("guo2017_dwt_qr_fa", p, "Table 4 mean", "before_attack", "Elaine", "Mean", "NC_cover_watermarked", 0.9997),
            _r("guo2017_dwt_qr_fa", p, "Table 4 mean", "before_attack", "Elaine", "Mean", "Corr_cover_watermarked", 0.9974),
        ]
        attacks = {1:"rotation_45deg",2:"rescale",3:"jpeg_qf25",4:"gamma_0.1",5:"gaussian_noise",6:"salt_pepper",7:"speckle",8:"gaussian_lowpass",9:"median",10:"histogram_equalization"}
        lena_ber=[0.0012,0,0,0,0.0142,0.0044,0.0076,0,0,0.0156]
        elaine_ber=[0.0012,0,0,0,0.0010,0.0029,0.0005,0,0,0.0159]
        lena_nc=[0.9993,1,1,1,0.9920,0.9975,0.9957,1,1,0.9912]
        elaine_nc=[0.9993,1,1,1,0.9994,0.9983,0.9997,1,1,0.9910]
        for i in range(10):
            rows.append(_r("guo2017_dwt_qr_fa", p, "Tables 5-6", "after_attack", "Lena", attacks[i+1], "BER", lena_ber[i]))
            rows.append(_r("guo2017_dwt_qr_fa", p, "Tables 5-6", "after_attack", "Lena", attacks[i+1], "NC_watermark", lena_nc[i]))
            rows.append(_r("guo2017_dwt_qr_fa", p, "Tables 5-6", "after_attack", "Elaine", attacks[i+1], "BER", elaine_ber[i]))
            rows.append(_r("guo2017_dwt_qr_fa", p, "Tables 5-6", "after_attack", "Elaine", attacks[i+1], "NC_watermark", elaine_nc[i]))
    if "gaata2022_dwt_hess_fwa" in ids:
        p = "Gaata et al. 2022"
        rows += [
            _r("gaata2022_dwt_hess_fwa", p, "Table 3 average", "before_attack", "Average", "no_attack", "MSE_non_optimization", 0.04083),
            _r("gaata2022_dwt_hess_fwa", p, "Table 3 average", "before_attack", "Average", "no_attack", "MSE_FWA", 0.03062),
            _r("gaata2022_dwt_hess_fwa", p, "Table 3 average", "before_attack", "Average", "no_attack", "PSNR_non_optimization", 38.00254, "dB"),
            _r("gaata2022_dwt_hess_fwa", p, "Table 3 average", "before_attack", "Average", "no_attack", "PSNR_FWA", 39.25193, "dB"),
        ]
        for attack, val in {"jpg_compression":98.12,"scaling":99.005,"rotation":97.62,"gaussian_noise":95.245,"histogram_equalization":94.845,"image_adjustment":93.953}.items():
            rows.append(_r("gaata2022_dwt_hess_fwa", p, "Table 4 FWA mean", "after_attack", "Average", attack, "retrieval_percent_FWA", val, "%"))
    if "hess_nha2023" in ids:
        p = "Nha et al. 2023"
        rows += [
            _r("hess_nha2023", p, "Section 4.3 statement", "before_attack", "Average", "no_attack", "PSNR", 54.0, "dB", note="Paper states average PSNR is higher than 54 dB; exact per-image table can be added if needed."),
            _r("hess_nha2023", p, "Section 4.3 statement", "before_attack", "Average", "no_attack", "SSIM", 0.9991, note="Paper states average SSIM is higher than 0.9991."),
            _r("hess_nha2023", p, "Section 4.3 statement", "before_attack", "Average", "no_attack", "NC_watermark", 1.0, note="Paper states NC is 1 under no attack."),
            _r("hess_nha2023", p, "Abstract statement", "after_attack", "Average", "common_attacks", "NC_watermark", 0.93, note="Paper states average NC is higher than 0.93 under common attacks; this is a lower-bound statement, not an exact mean."),
            _r("hess_nha2023", p, "Section 4.2", "metadata", "All", "payload", "watermark_side", 32, "px", note="Original paper watermarks are 32x32 binary; project implementation adapts to 64x64."),
        ]

    if "roy2018_dwt_svd" in ids:
        p = "Roy and Pal 2018"
        rows += [
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Lena", "no_attack", "PSNR", 51.1464, "dB"),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Lena", "no_attack", "NC_watermark", 0.9992),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Lena", "no_attack", "BER", 0.006),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Peppers", "no_attack", "PSNR", 51.1286, "dB"),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Peppers", "no_attack", "NC_watermark", 0.9989),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Peppers", "no_attack", "BER", 0.007),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Mandril", "no_attack", "PSNR", 51.1269, "dB"),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Mandril", "no_attack", "NC_watermark", 0.9987),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Mandril", "no_attack", "BER", 0.007),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Agni-III missile", "no_attack", "PSNR", 51.1925, "dB"),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Agni-III missile", "no_attack", "NC_watermark", 0.9979),
            _r("roy2018_dwt_svd", p, "Table 1", "before_attack", "Agni-III missile", "no_attack", "BER", 0.008),
            _r("roy2018_dwt_svd", p, "Section 4", "metadata", "All", "watermark", "watermark_size", "64x64"),
            _r("roy2018_dwt_svd", p, "Section 4", "metadata", "All", "embedding_strength", "alpha", 0.02),
            _r("roy2018_dwt_svd", p, "Equations 5-16", "metadata", "All", "side_information", "side_information", "Sp,UWp,VWp", note="Stored from embedding phase for extraction; not a fully blind method."),
        ]
    if "dwt_hd_svd_2025" in ids:
        p = "Dong, Yan and Yin 2024"
        rows += [
            _r("dwt_hd_svd_2025", p, "Table 2 average", "before_attack", "Average", "no_attack", "PSNR", 45.3437, "dB"),
            _r("dwt_hd_svd_2025", p, "Table 2 average", "before_attack", "Average", "no_attack", "SSIM", 0.9987),
            _r("dwt_hd_svd_2025", p, "Table 4", "before_attack", "Average", "no_attack", "NCC_watermark", 1.0),
            _r("dwt_hd_svd_2025", p, "Table 5", "before_attack", "Average", "no_attack", "BER", 0.0),
            _r("dwt_hd_svd_2025", p, "Table 7", "metadata", "Average", "runtime", "total_time", 1.1136, "sec"),
            _r("dwt_hd_svd_2025", p, "Table 8", "metadata", "All", "payload", "payload", 0.6667, "bpp"),
        ]
        ncc = {
            "salt_pepper_v0.01":[0.9920,0.9968,0.9929,0.9947,0.9969,0.9951],
            "salt_pepper_v0.05":[0.9337,0.9740,0.9451,0.9508,0.9600,0.9616],
            "gaussian_noise_v0.01":[0.9772,0.9855,0.9681,0.9812,0.9859,0.9779],
            "gaussian_noise_v0.05":[0.8380,0.8927,0.8545,0.8746,0.8824,0.9086],
            "rotation_10deg":[0.8445,0.8771,0.8463,0.8412,0.8913,0.9763],
            "cropping_20pct":[0.9239,0.9664,0.9831,0.9642,0.9155,0.9859],
            "jpeg_qf90":[0.9988,0.9965,0.9991,0.9999,0.9996,0.9999],
            "gamma_0.8":[0.9800,0.9748,0.9883,0.9839,0.9821,0.9860],
        }
        for attack, vals in ncc.items():
            rows.append(_r("dwt_hd_svd_2025", p, "Table 4 mean", "after_attack", "Average", attack, "NCC_watermark", round(_mean(vals), 6)))
    return rows

def paper_reported_dataframe(method_ids: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(paper_reported_rows(method_ids))

def write_paper_reported(output_dir: str | Path, method_ids: list[str] | None = None) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = paper_reported_dataframe(method_ids)
    df.to_csv(output / "paper_reported_results.csv", index=False)
    if not df.empty:
        summary = df.groupby(["method_id", "phase", "metric"], dropna=False)["value"].agg(["count", "mean", "min", "max"]).reset_index()
        summary.to_csv(output / "paper_reported_summary.csv", index=False)
    (output / "README_ORIGINAL_MODE.md").write_text(
        "# Original reported mode\n\n"
        "This folder contains values copied from the published paper tables. "
        "It is the only mode that can match paper numbers exactly. It is not a bit-for-bit rerun.\n\n"
        "For local experiments on your images/watermark, use `--baseline-mode adapt`.\n",
        encoding="utf-8",
    )
    return df
