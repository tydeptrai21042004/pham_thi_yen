"""Report actual R1.2 security and R2.2 adapted TrustMark 4096 experiment outputs.
Never turn undefined conditional NC into zero or remove failed extractions.
Usage: python scripts/reviewer_r12_r22_tables.py --root /kaggle/working/R12_R22_GPU --output ...
"""
from __future__ import annotations
import argparse,csv,json,statistics,math
from pathlib import Path
from collections import Counter,defaultdict

HOSTS=('Girl','airplane','athens','baboon','butfly','house','manhatan',
       'milkdrop','pepper','safari','sailboat','tiffany')

def readcsv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def number(v):
    try:
        a=float(v);return a if math.isfinite(a) else None
    except (ValueError,TypeError):return None

def avg(rows,field):
    a=[number(r.get(field)) for r in rows]
    a=[x for x in a if x is not None]
    return round(statistics.mean(a),6) if a else None

def savecsv(dest,rows):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run(root,out,expected_attacks=109):
    out.mkdir(parents=True,exist_ok=True)
    all_neural=[]; audit=[]; records=[]
    for cohort in ('A','B'):
        expected_names=None; cohort_rows=[]
        for host in HOSTS:
            path=root/'NEURAL'/cohort/host/'trustmark'/(host+'_results.csv')
            if not path.is_file():continue
            rows=readcsv(path)
            if any(str(r.get('image'))!=host for r in rows):raise ValueError(f'Wrong host in {path}')
            attacked=[r for r in rows if r.get('phase')=='after_attack']
            clean=[r for r in rows if r.get('phase')=='before_attack']
            controls=[r for r in rows if r.get('phase')=='control']
            names=[r['attack'] for r in attacked]
            if len(names)!=len(set(names)):raise ValueError(f'Duplicate attack rows in {path}')
            if len(clean)!=1 or len(controls)!=1:raise ValueError(f'Missing clean/control for {path}')
            if len(attacked)>expected_attacks:raise ValueError(f'Too many attacks in {path}')
            # Prevent reporting a successful complete grid if attack identity differs.
            if len(attacked)==expected_attacks:
                if expected_names is None: expected_names=set(names)
                elif set(names)!=expected_names:raise ValueError(f'Attack suite differs: {path}')
            all_neural.extend([{**r,'cohort':cohort} for r in rows])
            cohort_rows.extend(attacked)
            records.append({'cohort':cohort,'host':host,'completed_attacks':len(attacked),
                            'expected_attacks':expected_attacks,'clean_coverage':number(clean[0].get('bit_coverage')),
                            'clean_correct_fraction':number(clean[0].get('fraction_correct_all_bits')),
                            'attacked_correct_fraction':avg(attacked,'fraction_correct_all_bits'),
                            'attacked_coverage':avg(attacked,'bit_coverage'),
                            'undecoded_attacks':sum(number(r.get('bit_coverage'))==0 for r in attacked),
                            'run_errors':sum(bool(r.get('error')) for r in attacked)})
        cohort_all=[r for r in all_neural if r.get('cohort')==cohort]
        for phase in ('before_attack','after_attack'):
            rows=[r for r in cohort_all if r.get('phase')==phase]
            if not rows:continue
            summary={'cohort':cohort,'phase':phase,'records':len(rows),
                     'expected_records':12 if phase=='before_attack' else 12*expected_attacks,
                     'mean_psnr':avg(rows,'psnr'),'mean_ssim':avg(rows,'ssim'),
                     'mean_conditional_nc_decoded_only':avg(rows,'nc'),
                     'conditional_nc_valid_cases':sum(number(r.get('nc')) is not None for r in rows),
                     'mean_bit_coverage':avg(rows,'bit_coverage'),
                     'mean_erasure_aware_ber':avg(rows,'ber_erasure_as_error'),
                     'mean_correctly_recovered_fraction':avg(rows,'fraction_correct_all_bits'),
                     'exact_full_payload_recoveries':sum(number(r.get('exact_recovery'))==1 for r in rows),
                     'zero_coverage_cases':sum(number(r.get('bit_coverage'))==0 for r in rows),
                     'execution_error_cases':sum(bool(r.get('error')) for r in rows),
                     'protocol':'TrustMark 4096-bit spatial extension, 68 native BCH_5 chunks; not native 4096-bit model capacity'}
            audit.append(summary)
    savecsv(out/'trustmark_4096_summary.csv',audit)
    savecsv(out/'trustmark_per_host.csv',records)
    savecsv(out/'trustmark_all_cases.csv',all_neural)
    security=[]
    for cohort in ('A','B'):
        file=root/'SECURITY'/cohort/'security_all_cases.csv'
        if file.is_file():
            rows=readcsv(file)
        else:
            # Interrupted security batch: retain all finished per-host controls.
            rows=[]
            for host in HOSTS:
                source=root/'SECURITY'/cohort/'per_host'/host/'per_case_results.csv'
                if source.is_file(): rows.extend(readcsv(source))
        security.extend([{**r,'cohort':cohort} for r in rows])
    savecsv(out/'security_all_cases.csv',security)
    summaries=[]
    for cohort in ('A','B'):
        rows=[r for r in security if r.get('cohort')==cohort]
        for label,subset in (('marked_clean',[r for r in rows if r.get('case')=='marked_clean']),
             ('original_unmarked',[r for r in rows if r.get('case')=='original_unmarked']),
             ('unrelated_host',[r for r in rows if r.get('case')=='unrelated_host']),
             ('random_wrong_key',[r for r in rows if r.get('case','').startswith('wrong_secret_random_')]),
             ('tampered_metadata',[r for r in rows if r.get('case') in ('tampered_position_metadata','corrupted_side_information')])):
            if not subset:continue
            summaries.append({'cohort':cohort,'security_condition':label,'n':len(subset),
                'hosts':len({r.get('host') for r in subset}),
                'mean_watermark_nc':avg(subset,'watermark_nc'),
                'mean_payload_ber':avg(subset,'payload_ber_all_positions'),
                'nc_ge_095_count':sum((number(r.get('watermark_nc')) or float('-inf'))>=.95 for r in subset),
                'external_hmac_rejections':sum(str(r.get('metadata_integrity_valid')).lower()=='false' for r in subset),
                'extraction_rejected':sum(str(r.get('extraction_rejected')).lower()=='true' for r in subset),
                'note':'HMAC only an external integrity audit; watermark presence detection is NOT established'})
    savecsv(out/'security_aggregate.csv',summaries)
    state={'schema':'R12_R22_20261019_results','neural':audit,'security':summaries,
           'trustmark_hosts_with_any_rows':len(records),'security_rows':len(security),
           'all_24_neural_hosts_complete':len(records)==24 and all(r['completed_attacks']==expected_attacks for r in records),
           'metric_caveat':'proposal reconstructed NC and TrustMark decoded-only NC are NOT equal-capacity measures',
           'unmarked_fp_caveat':'do not claim security if the unmarked source gives high NC',
           'security_method_unchanged':True}
    (out/'report.json').write_text(json.dumps(state,indent=2,allow_nan=False))
    print('Generated review tables:',out,'neural host runs',len(records),
          'complete=',state['all_24_neural_hosts_complete'],'security rows',len(security),flush=True)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--attacks',type=int,default=109)
    a=p.parse_args();run(a.root,a.output,a.attacks)
