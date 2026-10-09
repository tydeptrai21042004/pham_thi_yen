"""Measure whether local BSS predicts the survival of REAL image attacks.

This is evidence collection, not a theoretical guarantee.  Results with only
one BSS level or too few samples have an undefined rank correlation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.experiments.review_audit import load_binary,load_rgb
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams,ProposalQHDWTHess,FLAG_Q4,FLAG_HPOS,FLAG_SKIP,
    HPOS_CANDIDATES,_rgb_to_source_bgr,_host_bgr_to_embed_space,
    _dwt_split_4bands,_extract_bit_from_candidate_block,
    prepare_binary_watermark_payload_from_array,
)


def run(host:Path,watermark:Path,output:Path,*,host_size=128,watermark_size=16,repeat=1,max_attacks=3):
    params=ProposalParams.from_dict({"repeat":repeat})
    original=load_rgb(host,host_size)
    wm=load_binary(watermark,watermark_size)
    truth=prepare_binary_watermark_payload_from_array(wm,params)[1]
    method=ProposalQHDWTHess(params=params,use_optimizer=False)
    marked,key=method.embed(original,wm)
    attacks=default_attack_suite(include_none=False,preset="reviewer_stress")[:max_attacks]
    rows=[]
    for attack in attacks:
        try:
            attacked=apply_attack(marked,attack)
            space=_host_bgr_to_embed_space(_rgb_to_source_bgr(attacked))
            bands={ch:_dwt_split_4bands(space[:,:,ch].astype(float),params) for ch in params.host_channels}
            for idx,(kpos,ch,band,i,j) in enumerate(key.schedule):
                flag=int(key.flags[idx]);
                if flag==FLAG_SKIP:continue
                block=bands[ch][band][i:i+params.block_size,j:j+params.block_size]
                hpos_idx=int(key.hpos_list[idx])
                pos=HPOS_CANDIDATES[hpos_idx][1] if flag==FLAG_HPOS else None
                try:
                    bit=_extract_bit_from_candidate_block(block,flag,params,pos=pos)
                    survived=int(bit==int(truth[kpos]))
                    error=None
                except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    survived=0;error=repr(exc)
                bss=float(key.selected_bss[idx])
                rows.append({"host":host.name,"attack":attack.name,"scheduled_index":idx,
                             "payload_index":kpos,"branch":"Q" if flag==FLAG_Q4 else "H",
                             "bss":bss,"mse":float(key.selected_mse[idx]),
                             "actual_survival":survived,"error":error})
        except Exception as exc:
            rows.append({"host":host.name,"attack":attack.name,"error":repr(exc)})
    output.mkdir(parents=True,exist_ok=True)
    if rows:
        names=list(dict.fromkeys(k for x in rows for k in x))
        with (output/"bss_per_block.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    summary=[]
    for attack in attacks:
        vals=[x for x in rows if x.get("attack")==attack.name and x.get("bss") is not None]
        ranks=[float(x["bss"]) for x in vals]
        survival=[int(x["actual_survival"]) for x in vals]
        correlation=None
        if len(set(ranks))>1 and len(set(survival))>1:
            corr,_p=spearmanr(ranks,survival)
            if np.isfinite(corr):correlation=float(corr)
        summary.append({"attack":attack.name,"samples":len(vals),"survival_rate":float(np.mean(survival)) if vals else None,
                        "spearman_rho":correlation})
    (output/"bss_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("host","watermark","output"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--host-size",type=int,default=128)
    ap.add_argument("--watermark-size",type=int,default=16)
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--max-attacks",type=int,default=3)
    a=ap.parse_args()
    print(json.dumps(run(a.host,a.watermark,a.output,host_size=a.host_size,watermark_size=a.watermark_size,repeat=a.repeat,max_attacks=a.max_attacks),indent=2))
