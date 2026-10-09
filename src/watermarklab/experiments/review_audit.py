"""Independent negative-control and robustness audit (NO optimized parameters).

Usage: python -m watermarklab.experiments.review_audit --host-dir data/host
       --watermark data/watermark/wm.png --output results/revision_audit
       --max-images 2 --host-size 128 --watermark-size 16 --repeat 1

Results measure extraction and false-positive *risks*, NOT a calibrated detector.
A high NC in an unwatermarked host is a failure signal, not a success.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image
from watermarklab.common.metrics import ber, nc, ncc, psnr, ssim
from watermarklab.common.attack import default_attack_suite, apply_attack
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


def audit(host_dir: Path, watermark_path: Path, output: Path, *, max_images: int = 2,
          host_size: int | None = 128, watermark_size: int | None = 16,
          repeat: int | None = 1, direct_binary: bool = False, contrastive: bool = False,
          attack_preset: str = "lite", max_attacks: int = 4) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in host_dir.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})[:max_images]
    if not paths:
        raise ValueError("no host images found")
    watermark = load_binary(watermark_path, watermark_size)
    params = ProposalParams.from_dict({"repeat": repeat, "watermark_dwt_enabled": not direct_binary,
                                       "require_contrastive_embedding": contrastive})
    truth = prepare_binary_watermark_payload_from_array(watermark, params)[1]
    method = ProposalQHDWTHess(params=params, use_optimizer=False)
    hosts = [load_rgb(p, host_size) for p in paths]
    rows = []
    embeds = []
    attacks = default_attack_suite(include_none=False, preset=attack_preset)[:max_attacks]
    for i, (path, host) in enumerate(zip(paths, hosts)):
        t0 = time.perf_counter()
        marked, key = method.embed(host, watermark)
        embed_sec = time.perf_counter()-t0
        key_json = json.dumps({"flags": key.flags, "hpos_list": key.hpos_list,
                               "payload_meta": key.payload_meta,
                               "params": key.params.to_dict()}, sort_keys=True)
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
        for case, candidate in variants:
            t1 = time.perf_counter()
            try:
                diag = method.extract_payload_diagnostics(candidate,key)
                wm = method.extract(candidate,key)
                payload = binary_bit_metrics(truth,diag.bits,diag.valid_mask)
                rows.append({"host":path.name,"case":case,**payload,
                             "watermark_ber":ber(watermark,wm),"watermark_nc":nc(watermark,wm),
                             "watermark_centered_ncc":ncc(watermark,wm),
                             "erasures":diag.erasures,"ties":diag.ties,
                             "decoder_errors":len(diag.errors),"extract_seconds":round(time.perf_counter()-t1,6),
                             "embed_seconds":round(embed_sec,6),
                             "side_info_json_bytes":len(key_json.encode("utf-8")),
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
    manifest={"schema":"watermarklab-review-audit-v1","optimization":"disabled",
              "negative_controls":"original image and different host under identical embedding-derived side information",
              "detector_calibrated":False,"hosts":[str(x) for x in paths],"watermark":str(watermark_path),
              "watermark_sha256":hashlib.sha256(watermark.tobytes()).hexdigest(),
              "host_size":host_size,"watermark_size":watermark_size,"direct_binary":direct_binary,
              "repeat":repeat,"contrastive":contrastive,
              "params":{**params.to_dict(),"private_key":"[REDACTED - supply locally]"},
              "attack_suite":[{"name":x.name,"group":x.group,"params":x.params} for x in attacks],
              "python":platform.python_version(),"numpy":np.__version__,"caveat":"image-to-key conditioning may cause false positives; no presence verification is claimed"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    errors=[r for r in rows if "error" in r]
    report={"output":str(output),"measurements":len(rows),"errors":len(errors),
            "unwatermarked_original_nc":[r.get("watermark_nc") for r in rows if r.get("case")=="original_unmarked"],
            "status":"audit_only_not_verified"}
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
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--direct-binary",action="store_true")
    ap.add_argument("--contrastive",action="store_true",help="Experimental: exclude naturally matching unmarked blocks")
    ap.add_argument("--attack-preset",default="lite")
    ap.add_argument("--max-attacks",type=int,default=4)
    args=ap.parse_args()
    print(json.dumps(audit(args.host_dir,args.watermark,args.output,max_images=args.max_images,
                           host_size=args.host_size,watermark_size=args.watermark_size,
                           repeat=args.repeat,direct_binary=args.direct_binary,contrastive=args.contrastive,
                           attack_preset=args.attack_preset,max_attacks=args.max_attacks),indent=2))

if __name__ == "__main__":
    main()
