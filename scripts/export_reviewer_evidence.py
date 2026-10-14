"""Workstream A: export reviewer evidence from EXISTING result files (no model runs).

Use --results-dir on the extracted pham_yen_groupwise_FULL_RESULTS root. This
script does not alter the proposed method, source numbers or experiment files.
It emits attack-wise ablation, classical vs proposal statistics, security and
runtime tables with reviewer IDs; missing evidence is explicitly recorded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def _read(path):
    return pd.read_csv(path) if path.is_file() else None


def _mean_ci_by_host(df, method_col='method_id', metric='nc', seed=3101):
    # Within-host attack repetitions are correlated; bootstrap hosts only.
    rng=np.random.default_rng(seed)
    rows=[]
    if df is None or not {method_col,'image',metric}.issubset(df): return rows
    for method,part in df.groupby(method_col):
        host_means=part.groupby('image')[metric].mean().dropna().to_numpy(dtype=float)
        if len(host_means)<2: continue
        boot=rng.choice(host_means,size=(2000,len(host_means)),replace=True).mean(axis=1)
        rows.append(dict(method_id=method,metric=metric,n_hosts=len(host_means),
                         host_mean=float(host_means.mean()),host_sd=float(host_means.std(ddof=1)),
                         ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                         method='host_bootstrap'))
    return rows


def export(results_dir:Path, output:Path):
    results_dir,output=Path(results_dir),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    required={
        'R1.2_R2.9_security': 'review_audit/per_case_results.csv',
        'R1.4_classical_stats': 'heldout_full/per_image_attack_results.csv',
        'R1.6_R2.4_bss': 'bss_validation/bss_per_host_attack.csv',
        'R2.5_ablation': 'ablation/full/per_image_attack_results.csv',
        'R2.1_block_size': 'block_size_comparison/per_case.csv',
        'R2.10_color_channels': 'channel_ablation/channel_ablation.csv',
    }
    index=[]
    for ref,rel in required.items():
        src=results_dir/rel
        index.append({'reviewer':ref,'source':str(src),'exists':src.is_file()})
    pd.DataFrame(index).to_csv(output/'reviewer_evidence_index.csv',index=False)
    src=_read(results_dir/'heldout_full/per_image_attack_results.csv')
    if src is not None:
        after=src[src.phase=='after_attack'].copy()
        for metric in ('nc','ber','psnr','ssim','embed_time_sec','extract_time_sec'):
            if metric in after:
                pd.DataFrame(_mean_ci_by_host(after,metric=metric)).to_csv(output/f'classical_{metric}_host_ci.csv',index=False)
        if {'method_id','attack','nc','ber'}.issubset(after):
            summary=after.groupby(['method_id','attack']).agg(n=('image','nunique'),
                nc_mean=('nc','mean'),nc_sd=('nc','std'),ber_mean=('ber','mean'),ber_sd=('ber','std')).reset_index()
            summary.to_csv(output/'classical_attackwise.csv',index=False)
        if {'method_id','image','embed_time_sec','extract_time_sec'}.issubset(src):
            clean=src[src.phase=='before_attack']
            clean.groupby('method_id',as_index=False)[['embed_time_sec','extract_time_sec']].agg(['mean','std']).to_csv(output/'runtime_summary.csv')
    ablations=[]
    for folder in sorted((results_dir/'ablation').glob('*')):
        data=_read(folder/'per_image_attack_results.csv')
        if data is None or 'phase' not in data:continue
        subset=data[data.phase=='after_attack']
        for attack, group in subset.groupby('attack'):
            ablations.append(dict(variant=folder.name,attack=attack,hosts=group.image.nunique(),
                nc_mean=float(group.nc.mean()),nc_sd=float(group.nc.std()),
                ber_mean=float(group.ber.mean()),ber_sd=float(group.ber.std())))
    if ablations:pd.DataFrame(ablations).to_csv(output/'attackwise_ablation.csv',index=False)
    for subpath, target in [('block_size_comparison/per_case.csv','block_size_original_results.csv'),
                            ('channel_ablation/channel_ablation.csv','channel_original_results.csv'),
                            ('bss_validation/bss_per_host_attack.csv','bss_original_uncorrected_summary.csv')]:
        previous=_read(results_dir/subpath)
        if previous is not None: previous.to_csv(output/target,index=False)
    audit=_read(results_dir/'review_audit/per_case_results.csv')
    if audit is not None:
        keep=[x for x in ('host','case','watermark_nc','watermark_ber','payload_ber_all_positions',
                         'payload_ber_valid_only','coverage','side_info_json_bytes',
                         'embedded_payload_bits','erasures','decoder_errors') if x in audit]
        audit[keep].to_csv(output/'security_and_sideinfo_summary.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'source':str(results_dir),
       'generated_from_saved_results_only':True,
       'missing':[x for x in index if not x['exists']],
       'note':'NC and BER reflect each original experiment protocol. Do not merge native neural capacities with matched 4096-bit results.'},indent=2))
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(export(a.results_dir,a.output),indent=2))
