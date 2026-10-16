"""Build reproducible reviewer-facing tables from completed classical results.

Does NOT rerun methods or transform results. All aggregates disclose coverage and
never treat unmeasured attacks as valid observations. Requires per-method/host
completeness to mark the reviewer report COMPLETE.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
import pandas as pd
from watermarklab.methods import DEFAULT_METHOD_IDS
from watermarklab.benchmark import PROPOSAL_ABLATION_VARIANTS
from watermarklab.common.attack import default_attack_suite
from watermarklab.experiments.splits import ORIGINAL_TEST_NAMES


TABLES={
 'classical_full':'heldout_full/per_image_attack_results.csv',
 'security':'review_audit/per_case_results.csv',
 'bss':'bss_validation/bss_per_host_attack.csv',
 'channel':'channel_ablation/channel_ablation.csv',
 'block_size':'block_size_comparison/per_case.csv',
 **{f'ablation_{v}':f'ablation/{v}/per_image_attack_results.csv' for v in PROPOSAL_ABLATION_VARIANTS},
}


def _stats(df:pd.DataFrame, keys, metrics, *, n_hosts=None):
    metrics=[col for col in metrics if col in df.columns]
    if not metrics:return pd.DataFrame()
    result=df.groupby(keys,dropna=False).agg(
        n_rows=(metrics[0],'count'),
        **{f'{col}_mean':(col,'mean') for col in metrics},
        **{f'{col}_sd':(col,'std') for col in metrics}).reset_index()
    if n_hosts and n_hosts in df.columns:
        unique=df.groupby(keys,dropna=False)[n_hosts].nunique().reset_index(name='n_hosts')
        result=result.merge(unique,on=keys)
    return result


def build(classical:Path, output:Path, *, watermark='wm.png'):
    classical,output=Path(classical),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    available={key:(classical/path).is_file() for key,path in TABLES.items()}
    result={'watermark':watermark,'sources':available,'complete':False,
            'notes':['Only measured rows are reported; failures/missing results are not imputed.',
                     'Uncertainty is across independent held-out hosts, not attack-row independence.',
                     'Neural native-payload comparisons are deliberately excluded.']}
    if not all(available.values()):
        result['missing']=[key for key,exists in available.items() if not exists]
        (output/'review_coverage.json').write_text(json.dumps(result,indent=2))
        return result

    classical_rows=pd.read_csv(classical/TABLES['classical_full'])
    attack_cfg=default_attack_suite(False,'full')
    attack_map={a.name:a.group for a in attack_cfg}
    hosts={Path(h).stem for h in ORIGINAL_TEST_NAMES}
    exp={(m,h,'before_attack','no_attack') for m in DEFAULT_METHOD_IDS for h in hosts}
    exp|={(m,h,'after_attack',a.name) for m in DEFAULT_METHOD_IDS for h in hosts for a in attack_cfg}
    observed=[(r.method_id,r.image,r.phase,r.attack) for r in classical_rows.itertuples(index=False)]
    result['expected_classical_rows']=len(exp)
    result['actual_classical_rows']=len(observed)
    result['classical_unique_complete']=len(observed)==len(exp) and set(observed)==exp
    metrics=('psnr','ssim','nc','ber','embed_time_sec','extract_time_sec')
    numeric=classical_rows[list(metrics)].apply(pd.to_numeric,errors='coerce')
    result['finite_metrics']=bool(np.isfinite(numeric.to_numpy()).all())
    if not result['classical_unique_complete'] or not result['finite_metrics']:
        result['problem']='Incomplete/nonfinite held-out grid; report withheld to prevent misleading comparisons'
        (output/'review_coverage.json').write_text(json.dumps(result,indent=2))
        return result

    classical_rows['attack_family']=classical_rows.attack.map(attack_map).fillna('clean')
    clean=classical_rows[classical_rows.phase=='before_attack']
    attack=classical_rows[classical_rows.phase=='after_attack']
    _stats(clean,['method_id'],metrics,n_hosts='image').to_csv(output/'clean_per_method.csv',index=False)
    _stats(attack,['method_id','attack','attack_family'],metrics,n_hosts='image').to_csv(output/'all_attacks_methodwise.csv',index=False)
    _stats(attack,['method_id','attack_family'],metrics,n_hosts='image').to_csv(output/'attack_family_methodwise.csv',index=False)
    _stats(attack,['method_id'],metrics,n_hosts='image').to_csv(output/'all_attacks_overall.csv',index=False)

    # Paired, host-level uncertainty: repeat attacks within each host are correlated.
    means=attack.groupby(['method_id','image'],as_index=False)[list(metrics)].mean()
    ref=means[means.method_id=='proposal'].set_index('image')
    boot=np.random.default_rng(3101)
    rows=[]
    for method in DEFAULT_METHOD_IDS:
        if method=='proposal':continue
        sub=means[means.method_id==method].set_index('image')
        if set(sub.index)!=set(ref.index):
            result['problem']=f'Host mismatch for {method}'
            (output/'review_coverage.json').write_text(json.dumps(result,indent=2));return result
        for metric in metrics:
            delta=(ref.loc[sorted(hosts),metric]-sub.loc[sorted(hosts),metric]).to_numpy(dtype=float)
            sampled=boot.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
            rows.append({'baseline':method,'metric':metric,'n_hosts':len(delta),
                         'proposal_minus_baseline':float(delta.mean()),
                         'host_sd':float(delta.std(ddof=1)),
                         'ci95_low':float(np.quantile(sampled,.025)),
                         'ci95_high':float(np.quantile(sampled,.975)),
                         'resampling':'paired-host-bootstrap-2000'})
    pd.DataFrame(rows).to_csv(output/'paired_host_confidence.csv',index=False)

    reviewer_names={a.name for a in default_attack_suite(False,'reviewer_stress')}
    stress=attack[attack.attack.isin(reviewer_names)]
    _stats(stress,['method_id','attack','attack_family'],metrics,n_hosts='image').to_csv(
        output/'reviewer_stress_attacks.csv',index=False)
    geom=attack[attack.attack_family.astype(str).str.contains('rot|geo|crop|resize|trans|scale|shear',case=False,regex=True)]
    if len(geom):
        _stats(geom,['method_id','attack','attack_family'],metrics,n_hosts='image').to_csv(
            output/'geometric_attacks.csv',index=False)
    result['reviewer_stress_attack_count']=len(reviewer_names)

    ablation_attack_names={a.name for a in default_attack_suite(False,'reviewer_stress')}
    expected_ablation={(h,'before_attack','no_attack') for h in hosts}
    expected_ablation|={(h,'after_attack',a) for h in hosts for a in ablation_attack_names}
    ablations=[]
    for variant in PROPOSAL_ABLATION_VARIANTS:
        data=pd.read_csv(classical/TABLES[f'ablation_{variant}'])
        actual=[(str(x.image),str(x.phase),str(x.attack)) for x in data.itertuples(index=False)]
        if len(actual)!=len(expected_ablation) or set(actual)!=expected_ablation:
            result['problem']=f'Ablation {variant} incomplete, observed={len(actual)}, expected={len(expected_ablation)}'
            (output/'review_coverage.json').write_text(json.dumps(result,indent=2));return result
        subset=data[data.phase=='after_attack'].copy()
        subset['variant']=variant
        ablations.append(subset)
    ab=pd.concat(ablations,ignore_index=True)
    _stats(ab,['variant','attack'],('nc','ber','psnr','ssim'),n_hosts='image').to_csv(
        output/'ablation_by_attack.csv',index=False)
    _stats(ab,['variant'],('nc','ber','psnr','ssim'),n_hosts='image').to_csv(
        output/'ablation_overall.csv',index=False)

    ch=pd.read_csv(classical/TABLES['channel'])
    _stats(ch[ch.phase=='after_attack'],['channel','attack'],('nc','ber','psnr_clean','ssim_clean'),n_hosts='host').to_csv(
        output/'channel_by_attack.csv',index=False)
    block=pd.read_csv(classical/TABLES['block_size'])
    _stats(block,['block_size','attack'],('nc','ber','psnr','ssim','coverage','embed_seconds','extract_seconds'),n_hosts='image').to_csv(
        output/'block_size_by_attack.csv',index=False)
    bss=pd.read_csv(classical/TABLES['bss'])
    bss.to_csv(output/'bss_measurements_raw.csv',index=False)
    security=pd.read_csv(classical/TABLES['security'])
    security.to_csv(output/'security_measurements_raw.csv',index=False)
    if 'side_info_json_bytes' in security:
        security.groupby('case',as_index=False).agg(
            n=('host','count'),side_info_json_bytes_mean=('side_info_json_bytes','mean')).to_csv(
            output/'side_information_bytes.csv',index=False)
    result['complete']=True
    result['n_test_hosts']=len(hosts)
    result['methods']=len(DEFAULT_METHOD_IDS)
    result['full_attacks']=len(attack_cfg)
    result['ablation_variants']=len(PROPOSAL_ABLATION_VARIANTS)
    result['limitations']=[
       'Block-size 8/16 configurations are experimental variants; capacity parity should be interpreted using recorded payload and coverage.',
       'Side information is required for semi-blind extraction; the byte counts are empirical measurements, not encryption guarantees.',
       'BSS theoretical validity, key-space assumptions, and geometric mechanism need discussion in the manuscript.',
    ]
    (output/'review_coverage.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--watermark',default='wm.png')
    args=ap.parse_args()
    status=build(args.results_dir,args.out,watermark=args.watermark)
    print(json.dumps(status,indent=2))
    sys.exit(0 if status['complete'] else 2)
