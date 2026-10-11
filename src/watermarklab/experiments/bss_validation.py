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
from watermarklab.experiments.block_size_comparison import frozen_params
from watermarklab.experiments.attack_selection import stratified_attack_subset
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.experiments.review_audit import load_binary,load_rgb
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams,ProposalQHDWTHess,FLAG_Q4,FLAG_HPOS,FLAG_SKIP,
    HPOS_CANDIDATES,_rgb_to_source_bgr,_host_bgr_to_embed_space,
    _dwt_split_4bands,_extract_bit_from_candidate_block,
    prepare_binary_watermark_payload_from_array,
)


def run(host:Path,watermark:Path,output:Path,*,host_size=128,watermark_size=16,repeat=None,max_attacks=3,param_file=None):
    params=frozen_params(param_file, repeat=repeat)
    original=load_rgb(host,host_size)
    wm=load_binary(watermark,watermark_size)
    truth=prepare_binary_watermark_payload_from_array(wm,params)[1]
    method=ProposalQHDWTHess(params=params,use_optimizer=False)
    marked,key=method.embed(original,wm)
    attacks=stratified_attack_subset(default_attack_suite(include_none=False,preset="reviewer_stress"),max_attacks)
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
        mse_corr=None
        mse_vals=[-float(x["mse"]) for x in vals]
        if len(set(mse_vals))>1 and len(set(survival))>1:
            coeff,_p=spearmanr(mse_vals,survival)
            if np.isfinite(coeff): mse_corr=float(coeff)
        summary.append({"host":host.name,"attack":attack.name,"attack_group":attack.group,
                        "samples":len(vals),"survival_rate":float(np.mean(survival)) if vals else None,
                        "spearman_rho_bss":correlation,"spearman_rho_neg_mse":mse_corr})
    (output/"bss_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("watermark","output"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--host",type=Path)
    ap.add_argument("--host-dir",type=Path)
    ap.add_argument("--max-hosts",type=int,default=9)
    ap.add_argument("--param-file",type=Path)
    ap.add_argument("--host-size",type=int,default=128)
    ap.add_argument("--watermark-size",type=int,default=16)
    ap.add_argument("--repeat",type=int,default=None)
    ap.add_argument("--max-attacks",type=int,default=3)
    a=ap.parse_args()
    if not a.host and not a.host_dir: ap.error("Provide --host or --host-dir")
    hosts=([a.host] if a.host else sorted(p for p in a.host_dir.iterdir() if p.suffix.lower() in ('.bmp','.png','.jpg'))[:a.max_hosts])
    a.output.mkdir(parents=True,exist_ok=True)
    combined=[]
    for host in hosts:
        result=run(host,a.watermark,a.output/host.stem,host_size=a.host_size,
                   watermark_size=a.watermark_size,repeat=a.repeat,max_attacks=a.max_attacks,
                   param_file=a.param_file)
        combined.extend(result)
    (a.output/'host_level_summary.json').write_text(json.dumps(combined,indent=2))
    if combined:
        import pandas as pd
        df=pd.DataFrame(combined)
        # Attack-level and category-level summaries retain undefined correlations
        # rather than replacing them with apparently successful zeroes.
        df.to_csv(a.output/'bss_per_host_attack.csv',index=False)
        df.groupby('attack_group',dropna=False).agg(
            mean_bss_rho=('spearman_rho_bss','mean'),
            mean_negative_mse_rho=('spearman_rho_neg_mse','mean'),
            host_attack_measurements=('attack','count'),
            total_samples=('samples','sum')).reset_index().to_csv(a.output/'bss_by_attack_group.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'hosts':[p.name for p in hosts],
          'n_hosts':len(hosts),'rows':len(combined),
          'interpretation':'correlations are empirical, not mathematical proof of global robustness'},indent=2))
    print(json.dumps({'hosts':len(hosts),'summary_records':len(combined)},indent=2))
