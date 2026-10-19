"""R1.2 experiments ONLY. Embedding/extraction implementations are NOT modified.

Per-host restartable wrong-key, key-sensitivity, original/unrelated-host and
metadata-integrity audit at 512x512 host / 64x64 watermark and frozen FA params.

Examples:
 python scripts/reviewer_r12_security_batch.py --hosts data/host --watermark data/watermark/wm.png \
  --params configs/reviewer_frozen_A.json --output results/r12_A --wrong-keys 16

The external HMAC checks protect an experimental metadata copy, NOT the
proposed watermarking method. The script cannot create a secure detector.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, tempfile, shutil, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from watermarklab.experiments.review_audit import audit, write_csv
from watermarklab.benchmark import load_proposal_param_file

TEST_HOSTS = ('Girl','airplane','athens','baboon','butfly','house','manhatan',
              'milkdrop','pepper','safari','sailboat','tiffany')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def readcsv(path):
    with Path(path).open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def _float(value):
    try:
        v=float(value); return v if math.isfinite(v) else None
    except (ValueError,TypeError): return None

def summarize(rows, wrong_keys):
    if not rows: return {}
    cases=sorted({r.get('case','') for r in rows})
    groups=[]
    for case in cases:
        sample=[r for r in rows if r.get('case')==case]
        nc=[_float(r.get('watermark_nc')) for r in sample]
        ncc=[_float(r.get('watermark_centered_ncc')) for r in sample]
        bp=[_float(r.get('payload_ber_all_positions')) for r in sample]
        ok=[r for r in sample if 'error' not in r and not r.get('extraction_rejected')]
        groups.append({'case':case,'n':len(sample),'measured_n':len(ok),
                       'mean_watermark_nc':float(np.mean([v for v in nc if v is not None])) if any(v is not None for v in nc) else None,
                       'mean_centered_ncc':float(np.mean([v for v in ncc if v is not None])) if any(v is not None for v in ncc) else None,
                       'mean_payload_ber':float(np.mean([v for v in bp if v is not None])) if any(v is not None for v in bp) else None,
                       'false_positive_count_nc_095':sum(v is not None and v>=.95 for v in nc),
                       'rejection_count':sum(bool(r.get('extraction_rejected')) for r in sample),
                       'metadata_integrity_failures':sum(str(r.get('metadata_integrity_valid')).lower()=='false' for r in sample)})
    return groups

def run(args):
    hosts=Path(args.hosts); watermark=Path(args.watermark); params=Path(args.params)
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True)
    if args.wrong_keys<1: raise ValueError('Need at least one wrong-key trial')
    if not load_proposal_param_file(params).get('global_params'):
        raise ValueError('Frozen training-only optimized parameter JSON is invalid')
    host_files=[hosts/(x+'.bmp') for x in TEST_HOSTS]
    for p in host_files:
        if not p.is_file(): raise FileNotFoundError(f'Required held-out host missing: {p}')
    if not watermark.is_file(): raise FileNotFoundError(watermark)
    source_hashes={'watermark':sha(watermark),'params':sha(params),
                   'hosts':{x.stem:sha(x) for x in host_files}}
    contract={'schema':'reviewer-security-experiments-v1','source_sha256':source_hashes,
              'wrong_keys':args.wrong_keys,'attack_count':args.max_attacks,
              'resolution':512,'watermark_resolution':64,'core_method_modified':False,
              'note':'No fitted test threshold. External HMAC is audit-only. Bad negative controls are retained.'}
    pcontract=target/'contract.json'
    if pcontract.exists() and json.loads(pcontract.read_text())!=contract:
        raise RuntimeError('Output directory has a DIFFERENT security protocol; choose new output folder')
    pcontract.write_text(json.dumps(contract,indent=2))
    for i,host in enumerate(host_files):
        destination=target/'per_host'/host.stem
        marker=destination/'summary.json'; raw=destination/'per_case_results.csv'
        if marker.is_file() and raw.is_file():
            rows=readcsv(raw)
            expected=2+1+1+args.wrong_keys+3+2+args.max_attacks
            have={r['case'] for r in rows}
            if (len(rows)==expected and len(have)==expected and
                    not any(r.get('error') for r in rows)):
                print('R1.2 validated checkpoint',host.name,flush=True);continue
            print('R1.2 incomplete, rerunning',host.name,flush=True)
        with tempfile.TemporaryDirectory(prefix='r12_one_host_') as scratch:
            stage=Path(scratch); (stage/'hosts').mkdir()
            shutil.copy2(host, stage/'hosts'/host.name)
            other=host_files[(i+1)%len(host_files)]
            # There is no model modification here; review_audit reuses existing
            # proposal.embed and extract exactly as in the source checkout.
            report=audit(stage/'hosts',watermark,stage/'out',max_images=1,
                         host_size=512,watermark_size=64,repeat=None,
                         direct_binary=False,contrastive=False,
                         attack_preset='full',max_attacks=args.max_attacks,
                         param_file=params,threshold_nc=.95,
                         unrelated_host_path=other,wrong_key_trials=args.wrong_keys)
            rows=readcsv(stage/'out'/'per_case_results.csv')
            if report.get('errors') or len(rows)<args.wrong_keys+8:
                raise RuntimeError(f'R1.2 audit failed for {host.name}: {report}')
            destination.mkdir(parents=True,exist_ok=True)
            for f in (stage/'out').iterdir():
                if f.is_file(): shutil.copy2(f,destination/f.name)
            print('R1.2 completed',host.name,'rows=',len(rows),flush=True)
    all_rows=[]
    for host in host_files:
        raw=target/'per_host'/host.stem/'per_case_results.csv'
        if raw.is_file(): all_rows.extend(readcsv(raw))
    write_csv(target/'security_all_cases.csv',all_rows)
    summary=summarize(all_rows,args.wrong_keys)
    write_csv(target/'security_summary_by_case.csv',summary)
    names={r['host'] for r in all_rows}
    completed=len(names)==len(host_files)
    (target/'manifest.json').write_text(json.dumps({**contract,'completed_hosts':len(names),
        'expected_hosts':len(host_files),'complete':completed,
        'rows':len(all_rows),'frozen_params_sha256':sha(params)},indent=2))
    print('R1.2 complete=',completed,'rows=',len(all_rows),flush=True)
    return completed

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts',type=Path,default=ROOT/'data/host')
    parser.add_argument('--watermark',type=Path,required=True)
    parser.add_argument('--params',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wrong-keys',type=int,default=16)
    parser.add_argument('--max-attacks',type=int,default=0,help='Optional attack cases; normal R1.2 only needs 0')
    a=parser.parse_args()
    sys.exit(0 if run(a) else 2)
