"""Independent negative-control and robustness audit (frozen optimized params optional).

Usage: python -m watermarklab.experiments.review_audit --host-dir data/host
       --watermark data/watermark/wm.png --output results/revision_audit
       --max-images 2 --host-size 128 --watermark-size 16 --repeat 1

Results measure extraction and false-positive *risks*, NOT a calibrated detector.
A high NC in an unwatermarked host is a failure signal, not a success.
"""
from __future__ import annotations
import argparse
import csv
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image
from watermarklab.common.metrics import ber, nc, ncc, psnr, ssim
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.benchmark import load_proposal_param_file
from watermarklab.experiments.attack_selection import stratified_attack_subset
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalQHDWTHess, ProposalParams, prepare_binary_watermark_payload_from_array,
    _host_bgr_to_embed_space, _embed_space_to_host_bgr, _rgb_to_source_bgr,
    _source_bgr_to_rgb,
)


def binary_bit_metrics(original: np.ndarray, recovered: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    a = np.asarray(original, dtype=np.uint8).reshape(-1)
    b = np.asarray(recovered, dtype=np.uint8).reshape(-1)
    if a.size != b.size:
        raise ValueError("payload lengths differ")
    if valid_mask is None:
        valid_mask = np.ones(a.size, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid_mask.size != a.size:
        raise ValueError("valid-mask length differs")
    return {"payload_ber_all_positions": float(np.mean(a != b)),
            "payload_ber_valid_only": float(np.mean(a[valid_mask] != b[valid_mask])) if valid_mask.any() else None,
            "coverage": float(np.mean(valid_mask))}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), resample=Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def load_binary(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        if size:
            im = im.resize((size, size), resample=Image.Resampling.NEAREST)
        return np.where(np.asarray(im) >= 127, 255, 0).astype(np.uint8)


def side_information_sizes(key) -> dict:
    """Estimate the JSON-encoded extraction contract, without storing private keys.

    Layout keys and payload_meta are counted separately; no claim of compressed
    optimal storage or cryptographic security is made.
    """
    elements = {"flags": key.flags, "hpos_list": key.hpos_list,
                "payload_meta": key.payload_meta, "params": key.params.to_dict(),
                "repeat_factor": int(key.repeat_factor),
                "usable_blocks": int(key.usable_blocks), "total_blocks": int(key.total_blocks)}
    dump = lambda item: len(json.dumps(item, sort_keys=True, default=str).encode('utf-8'))
    result = {f"{name}_json_bytes": dump(val) for name,val in elements.items()}
    result["full_extraction_contract_json_bytes"] = dump(elements)
    result["original_watermark_source_bits"] = int(np.prod(key.wm_shape))
    return result


def audit(host_dir: Path, watermark_path: Path, output: Path, *, max_images: int = 2,
          host_size: int | None = 128, watermark_size: int | None = 16,
          repeat: int | None = None, direct_binary: bool = False, contrastive: bool = False,
          attack_preset: str = "lite", max_attacks: int = 4,
          param_file: Path | None = None, threshold_nc: float = 0.95) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in host_dir.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})[:max_images]
    if not paths:
        raise ValueError("no host images found")
    watermark = load_binary(watermark_path, watermark_size)
    raw_params = {"repeat": repeat, "watermark_dwt_enabled": not direct_binary,
                  "require_contrastive_embedding": contrastive}
    if param_file:
        loaded = load_proposal_param_file(param_file)
        if not loaded.get("loaded") or not loaded.get("global_params"):
            raise ValueError(f"Missing global frozen optimized parameters: {param_file}")
        for name in ("q4_tau", "q4_margin", "h01_q", "h01_margin"):
            record = loaded["global_params"]
            if name in record: raw_params[name] = record[name]
            elif name.upper() in record: raw_params[name.upper()] = record[name.upper()]
    params = ProposalParams.from_dict(raw_params)
    truth = prepare_binary_watermark_payload_from_array(watermark, params)[1]
    method = ProposalQHDWTHess(params=params, use_optimizer=False)
    hosts = [load_rgb(p, host_size) for p in paths]
    rows = []
    embeds = []
    attacks = stratified_attack_subset(default_attack_suite(include_none=False, preset=attack_preset),max_attacks)
    for i, (path, host) in enumerate(zip(paths, hosts)):
        t0 = time.perf_counter()
        marked, key = method.embed(host, watermark)
        embed_sec = time.perf_counter()-t0
        key_json = json.dumps({"flags": key.flags, "hpos_list": key.hpos_list,
                               "payload_meta": key.payload_meta,
                               "params": key.params.to_dict()}, sort_keys=True)
        side_sizes = side_information_sizes(key)
        conv_bgr = _rgb_to_source_bgr(host)
        converted_only = _source_bgr_to_rgb(_embed_space_to_host_bgr(_host_bgr_to_embed_space(conv_bgr)))
        embeds.append((marked,key))
        variants = [("marked_clean", marked), ("original_unmarked", host)]
        if len(hosts) > 1:
            variants.append(("unrelated_host", hosts[(i+1)%len(hosts)]))
        for att in attacks:
            try:
                variants.append(("attack:"+att.name, apply_attack(marked, att)))
            except Exception as e:
                rows.append({"host":path.name,"case":"attack:"+att.name,"error":str(e)})
        # Wrong-key test: change the actual schedule-generating secret in a
        # copied key. This is not equivalent to simply rotating the watermark.
        wrong_key = copy.deepcopy(key)
        wrong_key.params.private_key = key.params.private_key + ":wrong-key-control"
        variants.append(("wrong_secret_key", marked, wrong_key))
        # Three independent one-character schedule-key changes (not password entropy claims).
        for suffix in ("A", "B", "C"):
            perturbed=copy.deepcopy(key)
            perturbed.params.private_key = key.params.private_key + suffix
            variants.append(("wrong_secret_key_"+suffix,marked,perturbed))
        # Side-info tampering is an adversary action, not a pixel attack.
        bad_meta=copy.deepcopy(key)
        h_positions=[j for j,flag in enumerate(bad_meta.flags) if int(flag)==1]
        if h_positions:
            bad_meta.hpos_list[h_positions[0]]=99
        elif bad_meta.hpos_list:
            bad_meta.hpos_list[0]=99
        variants.append(("tampered_position_metadata",marked,bad_meta))
        # Tampered metadata must be measured as a separate side-information
        # integrity threat; it is not a pixel attack.
        corrupted_key = copy.deepcopy(key)
        corrupted_key.flags = list(corrupted_key.flags)[:-1]
        variants.append(("corrupted_side_information", marked, corrupted_key))
        for record in variants:
            case, candidate = record[:2]
            selected_key = record[2] if len(record)==3 else key
            t1 = time.perf_counter()
            try:
                diag = method.extract_payload_diagnostics(candidate,selected_key)
                wm = method.extract(candidate,selected_key)
                payload = binary_bit_metrics(truth,diag.bits,diag.valid_mask)
                rows.append({"host":path.name,"case":case,**payload,
                             "watermark_ber":ber(watermark,wm),"watermark_nc":nc(watermark,wm),
                             "watermark_centered_ncc":ncc(watermark,wm),
                             "erasures":diag.erasures,"ties":diag.ties,
                             "decoder_errors":len(diag.errors),"extract_seconds":round(time.perf_counter()-t1,6),
                             "embed_seconds":round(embed_sec,6),
                             "side_info_json_bytes":len(key_json.encode("utf-8")),
                             **side_sizes,
                             "embedded_payload_bits":int(truth.size),
                             "psnr_host_to_marked":psnr(host,marked),
                             "psnr_host_to_conversion_only":psnr(host,converted_only),
                             "ssim_host_to_marked":ssim(host,marked),
                             "selected_q":sum(x==0 for x in key.flags),
                             "selected_h":sum(x==1 for x in key.flags),
                             "skipped":sum(x==2 for x in key.flags)})
            except Exception as e:
                rows.append({"host":path.name,"case":case,"error":repr(e)})
    write_csv(output/"per_case_results.csv",rows)
    component_rows = [dict(host=r['host'], **{k: v for k,v in r.items()
                         if k.endswith('_json_bytes') or k in ('original_watermark_source_bits','embedded_payload_bits')})
                      for r in rows if r.get('case') == 'marked_clean' and 'error' not in r]
    write_csv(output/"side_information_overhead.csv", component_rows)
    # These figures are a descriptive threshold audit; the threshold is fixed
    # before reviewing measurements and is NOT calibrated on the test images.
    positive=[r for r in rows if r.get("case")=="marked_clean" and r.get("watermark_nc") is not None]
    negatives=[r for r in rows if r.get("case") in ("original_unmarked","unrelated_host") and r.get("watermark_nc") is not None]
    detector_diagnostic={
        "fixed_nc_threshold":float(threshold_nc),
        "true_positive_rate_at_fixed_threshold":float(np.mean([r["watermark_nc"]>=threshold_nc for r in positive])) if positive else None,
        "false_positive_rate_at_fixed_threshold":float(np.mean([r["watermark_nc"]>=threshold_nc for r in negatives])) if negatives else None,
        "positive_n":len(positive),"negative_n":len(negatives),
        "warning":"NC with watermark detail side information is not an independently authenticated ownership detector. This threshold is descriptive, not validated/calibrated."}
    (output/"fixed_threshold_diagnostics.json").write_text(json.dumps(detector_diagnostic,indent=2))
    manifest={"schema":"watermarklab-review-audit-v2","optimization":"frozen_global" if param_file else "defaults",
              "negative_controls":"original, unrelated host, wrong secret key and corrupted side information",
              "detector_calibrated":False,"hosts":[str(x) for x in paths],"watermark":str(watermark_path),
              "watermark_sha256":hashlib.sha256(watermark.tobytes()).hexdigest(),
              "host_size":host_size,"watermark_size":watermark_size,"direct_binary":direct_binary,
              "repeat":repeat,"contrastive":contrastive,"param_file":str(param_file) if param_file else None,
              "fixed_threshold":threshold_nc,"threshold_calibrated_on_test":False,
              "params":{**params.to_dict(),"private_key":"[REDACTED - supply locally]"},
              "attack_suite":[{"name":x.name,"group":x.group,"params":x.params} for x in attacks],
              "python":platform.python_version(),"numpy":np.__version__,"caveat":"image-to-key conditioning may cause false positives; no presence verification is claimed"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    errors=[r for r in rows if "error" in r]
    report={"output":str(output),"measurements":len(rows),"errors":len(errors),
            "unwatermarked_original_nc":[r.get("watermark_nc") for r in rows if r.get("case")=="original_unmarked"],
            "fixed_threshold_diagnostic":detector_diagnostic,
            "status":"audit_complete_but_detection_not_verified"}
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-dir",type=Path,required=True)
    ap.add_argument("--watermark",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-images",type=int,default=2)
    ap.add_argument("--host-size",type=int,default=128)
    ap.add_argument("--watermark-size",type=int,default=16)
    ap.add_argument("--repeat",type=int,default=None)
    ap.add_argument("--direct-binary",action="store_true")
    ap.add_argument("--contrastive",action="store_true",help="Experimental: exclude naturally matching unmarked blocks")
    ap.add_argument("--attack-preset",default="lite")
    ap.add_argument("--max-attacks",type=int,default=4)
    ap.add_argument("--param-file",type=Path,help="Training-only frozen optimized proposal params")
    ap.add_argument("--threshold-nc",type=float,default=0.95,help="Prespecified diagnostic NC threshold, not a learned detector")
    args=ap.parse_args()
    print(json.dumps(audit(args.host_dir,args.watermark,args.output,max_images=args.max_images,
                           host_size=args.host_size,watermark_size=args.watermark_size,
                           repeat=args.repeat,direct_binary=args.direct_binary,contrastive=args.contrastive,
                           attack_preset=args.attack_preset,max_attacks=args.max_attacks,
                           param_file=args.param_file,threshold_nc=args.threshold_nc),indent=2))

if __name__ == "__main__":
    main()
