"""Strict results-completeness audit and paired host-level confidence intervals.

No failed experiment is dropped silently. CI resamples nine independent hosts,
not individual attack rows, to avoid pseudoreplication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from watermarklab.common.attack import default_attack_suite
from watermarklab.methods import DEFAULT_METHOD_IDS


def audit(csv_path: Path, output: Path, *, host_names: list[str],
          methods: list[str] | None = None, attack_preset: str = 'full',
          n_boot: int = 1000, seed: int = 3101):
    methods = methods or DEFAULT_METHOD_IDS
    attacks = [a.name for a in default_attack_suite(False, attack_preset)]
    expected = {(m, Path(h).stem, 'no_attack', 'before_attack') for m in methods for h in host_names}
    expected |= {(m, Path(h).stem, a, 'after_attack') for m in methods for h in host_names for a in attacks}
    df = pd.read_csv(csv_path)
    required = {'method_id','image','attack','phase','psnr','ssim','nc','ber','embed_time_sec','extract_time_sec'}
    absent = sorted(required - set(df.columns))
    if absent: raise ValueError(f'Missing required columns: {absent}')
    keys = list(zip(df.method_id,df.image,df.attack,df.phase))
    from collections import Counter
    count = Counter(keys)
    actual = set(keys)
    missing = sorted(expected-actual)
    extra = sorted(actual-expected)
    duplicates = sorted(k for k,v in count.items() if v!=1)
    invalid_metrics = df[df[['psnr','ssim','nc','ber','embed_time_sec','extract_time_sec']].isna().any(axis=1)]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    issues={'expected_rows':len(expected),'actual_rows':len(df),'missing':len(missing),
            'extra':len(extra),'duplicate_keys':len(duplicates),'non_finite_or_missing_metrics':len(invalid_metrics),
            'status':'PASS' if not (missing or extra or duplicates or len(invalid_metrics)) else 'FAIL',
            'missing_examples':[list(x) for x in missing[:20]],
            'extra_examples':[list(x) for x in extra[:20]],
            'duplicate_examples':[list(x) for x in duplicates[:20]]}
    (output/'completeness.json').write_text(json.dumps(issues,indent=2))
    if issues['status']!='PASS':
        raise RuntimeError(f'Incomplete experiment grid. See {output/"completeness.json"}: {issues}')
    # Each attack row is the same unit for both methods. First reduce to per-host means.
    after=df[df.phase=='after_attack']
    means=after.groupby(['method_id','image'],as_index=False)[['nc','ber','psnr','embed_time_sec','extract_time_sec']].mean()
    means.to_csv(output/'per_host_attack_means.csv',index=False)
    methods_present=sorted(means.method_id.unique())
    if 'proposal' not in methods_present: return issues
    rng=np.random.default_rng(seed)
    records=[]
    ref=means[means.method_id=='proposal'].set_index('image')
    for baseline in methods_present:
        if baseline=='proposal': continue
        cmp=means[means.method_id==baseline].set_index('image')
        if set(ref.index)!=set(cmp.index): raise RuntimeError(f'Host mismatch vs {baseline}')
        for metric in ('nc','ber','psnr','embed_time_sec','extract_time_sec'):
            difference=(ref[metric]-cmp[metric]).to_numpy(dtype=float)
            samples=rng.choice(difference,(n_boot,len(difference)),replace=True).mean(axis=1)
            records.append({'baseline':baseline,'metric':metric,'n_hosts':len(difference),
                            'proposal_minus_baseline_mean':float(difference.mean()),
                            'host_sd':float(difference.std(ddof=1)),
                            'ci95_low':float(np.quantile(samples,.025)),
                            'ci95_high':float(np.quantile(samples,.975)),
                            'confidence_method':'paired_host_bootstrap'})
    pd.DataFrame(records).to_csv(output/'paired_host_bootstrap.csv',index=False)
    by_attack=after.groupby(['method_id','attack'],as_index=False)[['nc','ber','psnr']].agg(['mean','std'])
    by_attack.columns=['_'.join(a for a in col if a) for col in by_attack.columns]
    by_attack.to_csv(output/'per_attack_summary.csv',index=False)
    return issues


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--methods',default='all')
    ap.add_argument('--attack-preset',default='full')
    args=ap.parse_args()
    info=json.loads(args.split.read_text())
    chosen=DEFAULT_METHOD_IDS if args.methods=='all' else args.methods.split(',')
    print(json.dumps(audit(args.csv,args.output,host_names=info['test_names'],methods=chosen,
                           attack_preset=args.attack_preset),indent=2))
