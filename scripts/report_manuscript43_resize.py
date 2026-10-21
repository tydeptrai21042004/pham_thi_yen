"""Summarize only COMPLETE paired 64-vs-128 43-attack results; no filled-in estimates.

 python scripts/report_manuscript43_resize.py --input results/manuscript43/resized64_vs_128

Generates, only after strict completeness checks:
 - paper43_size_summary.csv
 - paper43_by_attack.csv
 - paper43_by_family.csv
 - paper43_paired_128_minus_64.csv
 - paper43_report_status.json
"""
from __future__ import annotations
import argparse,json,sys
import numpy as np
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from watermarklab.common.attack import default_attack_suite,manuscript43_family,reviewer45_family


def build(input_dir:Path, expected_hosts:int=12, expected_watermarks:int=2, attack_preset:str="manuscript43"):
    input_dir=Path(input_dir)
    clean_path=input_dir/'proposal_size_sweep_clean_results.csv'
    attacked_path=input_dir/'proposal_size_sweep_attack_results.csv'
    if not clean_path.is_file() or not attacked_path.is_file():
        raise FileNotFoundError('Complete size-sweep clean and attacked CSVs do not exist; execute resize128 stage first')
    clean=pd.read_csv(clean_path);attacked=pd.read_csv(attacked_path)
    expected_names={a.name for a in default_attack_suite(False,attack_preset)}
    family=manuscript43_family if attack_preset=='manuscript43' else reviewer45_family
    prefix='paper43' if attack_preset=='manuscript43' else 'reviewer45'
    sizes=(64,128)
    wms={Path(x).stem for x in clean['watermark_file'].dropna().unique()}
    hosts=set(clean['host_image'].dropna().unique())
    problems=[]
    if len(wms)!=expected_watermarks:problems.append(f'watermarks={len(wms)} expected={expected_watermarks}')
    if len(hosts)!=expected_hosts:problems.append(f'hosts={len(hosts)} expected={expected_hosts}')
    if set(clean['watermark_size'].unique())!={64,128}:problems.append('clean sizes must be 64,128')
    if set(attacked['watermark_size'].unique())!={64,128}:problems.append('attacked sizes must be 64,128')
    expected_grid={(w,size,h) for w in clean['watermark_file'].unique() for size in sizes for h in hosts}
    clean_grid=set(zip(clean.watermark_file,clean.watermark_size,clean.host_image))
    attack_grid=set(zip(attacked.watermark_file,attacked.watermark_size,attacked.host_image))
    if clean_grid != expected_grid: problems.append('clean grid incomplete or inconsistent across watermark/size/host')
    if attack_grid != expected_grid: problems.append('attack grid incomplete or inconsistent across watermark/size/host')
    expected_clean=len(hosts)*len(wms)*len(sizes)
    expected_attack=expected_clean*len(expected_names)
    if len(clean)!=expected_clean:problems.append(f'clean count {len(clean)} not {expected_clean}')
    if len(attacked)!=expected_attack:problems.append(f'attack count {len(attacked)} not {expected_attack}')
    for df,keys,name in ((clean,['host_image','watermark_file','watermark_size'],'clean'),
                         (attacked,['host_image','watermark_file','watermark_size','attack'],'attack')):
        if df.duplicated(keys).any():problems.append(f'{name} duplicate rows')
    for (w,s,h),grp in attacked.groupby(['watermark_file','watermark_size','host_image'],dropna=False):
        if set(grp.attack)!=expected_names:
            problems.append(f'incomplete set: {Path(w).name}/{s}/{h}: {len(set(grp.attack))}/{len(expected_names)}')
    for key,df in [('clean',clean),('attacked',attacked)]:
        expected_cols=('clean_psnr','clean_ssim','clean_nc','clean_ber') if key=='clean' else ('nc_after_attack','ber_after_attack')
        for col in expected_cols:
            if not np.isfinite(pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)).all():problems.append(f'{key}.{col} has missing/nonfinite values')
    status={'status':'PASS' if not problems else 'FAIL','expected_attack_rows':expected_attack,
            'actual_attack_rows':len(attacked),'expected_clean_rows':expected_clean,
            'actual_clean_rows':len(clean),'problems':problems,'sizes':list(sizes),
            'watermark_names':sorted(wms),'hosts':sorted(hosts)}
    (input_dir/f'{prefix}_report_status.json').write_text(json.dumps(status,indent=2))
    if problems:raise RuntimeError(f'{len(expected_names)}-attack completeness check failed: '+'; '.join(problems[:8]))
    labels={w:('W1' if Path(w).name=='wm.png' else 'W2' if 'logo' in Path(w).name.lower() else Path(w).stem) for w in clean.watermark_file.unique()}
    def common(df):
        d=df.copy();d['watermark_label']=d.watermark_file.map(labels);return d
    clean=common(clean);attacked=common(attacked)
    attacked['attack_family']=attacked['attack'].map(family)
    clean_agg=clean.groupby(['watermark_label','watermark_size']).agg(
        mean_clean_psnr=('clean_psnr','mean'),mean_clean_ssim=('clean_ssim','mean'),
        mean_clean_nc=('clean_nc','mean'),mean_clean_ber=('clean_ber','mean'),
        mean_repetition=('proposal_repeat_factor','mean'),hosts=('host_image','nunique')).reset_index()
    attacked_agg=attacked.groupby(['watermark_label','watermark_size']).agg(
        mean_attack_nc=('nc_after_attack','mean'),mean_attack_ber=('ber_after_attack','mean'),
        attack_rows=('attack','size'),hosts=('host_image','nunique')).reset_index()
    summary=clean_agg.merge(attacked_agg,on=['watermark_label','watermark_size','hosts'],validate='one_to_one')
    summary.to_csv(input_dir/f'{prefix}_size_summary.csv',index=False)
    attacked.groupby(['watermark_label','watermark_size','attack','attack_family']).agg(
        mean_nc=('nc_after_attack','mean'),mean_ber=('ber_after_attack','mean'),
        hosts=('host_image','nunique')).reset_index().to_csv(input_dir/f'{prefix}_by_attack.csv',index=False)
    attacked.groupby(['watermark_label','watermark_size','attack_family']).agg(
        mean_nc=('nc_after_attack','mean'),mean_ber=('ber_after_attack','mean'),
        conditions=('attack','nunique'),observations=('host_image','size')).reset_index().to_csv(input_dir/f'{prefix}_by_family.csv',index=False)
    by_host=attacked.groupby(['watermark_label','host_image','watermark_size']).agg(
        nc=('nc_after_attack','mean'),ber=('ber_after_attack','mean')).reset_index()
    wide=by_host.pivot(index=['watermark_label','host_image'],columns='watermark_size',values=['nc','ber'])
    wide.columns=[f'{metric}_{int(size)}' for metric,size in wide.columns];wide=wide.reset_index()
    wide['delta_nc_128_minus_64']=wide['nc_128']-wide['nc_64']
    wide['delta_ber_128_minus_64']=wide['ber_128']-wide['ber_64']
    wide.to_csv(input_dir/f'{prefix}_paired_128_minus_64.csv',index=False)
    print(f'PASS: complete {len(expected_names)} attacks across',len(hosts),'hosts and',len(wms),'watermarks at 64 and 128')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expected-hosts',type=int,default=12)
    p.add_argument('--expected-watermarks',type=int,default=2)
    p.add_argument('--attack-preset',default='manuscript43',choices=['manuscript43','reviewer45'])
    a=p.parse_args()
    build(a.input,a.expected_hosts,a.expected_watermarks,a.attack_preset)
