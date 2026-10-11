"""Reviewer R2-1: empirical 4/8/16 block-size extension with frozen full-proposal rules.

Only the orthogonal Givens size and wavelet tiling change. All payload,
Hessenberg coefficients, BSS, quantization and secret scheduling remain fixed.
This is an actual embed->attack->extract comparison, NOT a theoretical table.

Example:
 python -m watermarklab.experiments.block_size_comparison \
   --host-dir data/host --watermark data/watermark/wm.png \
   --param-file results/proposal_optimized_params.json --output results/blocks
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from watermarklab.methods.proposal_qh_dwt_hess import (ProposalQHDWTHess, ProposalParams,
    prepare_binary_watermark_payload_from_array)
from watermarklab.experiments.review_audit import load_rgb, load_binary
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.common.metrics import psnr, ssim, nc, ber
from watermarklab.benchmark import load_proposal_param_file
from watermarklab.experiments.attack_selection import stratified_attack_subset

EXTENSIONS = {'.bmp','.png','.jpeg','.jpg'}


def frozen_params(path: Path | None, *, block_size=4, repeat=None) -> ProposalParams:
    # Freeze only the four FA-optimized embedding parameters; no per-test-image
    # optimization or test-image-driven parameter selection is allowed.
    params = ProposalParams()
    record = None
    if path is not None:
        payload = load_proposal_param_file(path)
        if not payload.get('loaded') or not payload.get('global_params'):
            raise ValueError(f'No global frozen proposal parameters in {path}')
        record = payload['global_params']
    d = params.to_dict()
    if record:
        for key in ('q4_tau','q4_margin','h01_q','h01_margin'):
            if key in record:
                d[key] = record[key]
            elif key.upper() in record:
                d[key] = record[key.upper()]
    d['block_size'] = int(block_size)
    d['repeat'] = repeat
    return ProposalParams.from_dict(d)


def run(host_dir: Path, watermark: Path, output: Path, *, param_file: Path | None=None,
        block_sizes=(4, 8, 16), max_hosts=None, max_attacks=5, attack_preset='reviewer_stress',
        host_size=None, watermark_size=None, repeat=None):
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted([q for q in host_dir.iterdir() if q.suffix.lower() in EXTENSIONS])
    if max_hosts is not None: paths = paths[:max_hosts]
    if not paths: raise ValueError('No host images')
    wm = load_binary(watermark, watermark_size)
    attacks = stratified_attack_subset(default_attack_suite(include_none=False,preset=attack_preset),max_attacks)
    data=[]; failures=[]; subbands=[]
    for bsize in block_sizes:
        params = frozen_params(param_file,block_size=bsize,repeat=repeat)
        for path in paths:
            image = load_rgb(path,host_size)
            model = ProposalQHDWTHess(params)
            begin = time.perf_counter()
            try:
                marked,key = model.embed(image,wm)
                embed_seconds = time.perf_counter()-begin
                payload_len = int(prepare_binary_watermark_payload_from_array(wm,params)[1].size)
                for band in params.dwt_bands:
                    flags=[int(flag) for pos,flag in zip(key.schedule,key.flags) if pos[2]==band]
                    # Schedule entries: (payload_index, channel, band, y, x)
                    subbands.append(dict(block_size=bsize,image=path.name,band=band,
                        scheduled_blocks=len(flags),q_branch=sum(x==0 for x in flags),
                        h_branch=sum(x==1 for x in flags),skipped=sum(x==2 for x in flags),
                        embedded_payload_bits=payload_len,original_watermark_pixels=int(wm.size)))
                jobs=[('clean',None,marked)];jobs.extend((a.name,a,apply_attack(marked,a)) for a in attacks)
                for name,attack,attacked in jobs:
                    t = time.perf_counter();wm_dec=model.extract(attacked,key)
                    diag=model.extract_payload_diagnostics(attacked,key)
                    data.append(dict(block_size=bsize,image=path.name,attack=name,
                        attack_group=attack.group if attack else 'none',
                        psnr=psnr(image,marked),ssim=ssim(image,marked),
                        nc=nc(wm,wm_dec),ber=ber(wm,wm_dec),
                        coverage=diag.coverage,erasures=diag.erasures,
                        payload_bits=payload_len,total_blocks=key.total_blocks,
                        used_blocks=key.usable_blocks,repeat_factor=key.repeat_factor,
                        embed_seconds=embed_seconds,extract_seconds=time.perf_counter()-t,
                        q_used=sum(f==0 for f in key.flags),h_used=sum(f==1 for f in key.flags),
                        skipped=sum(f==2 for f in key.flags)))
                print('[BLOCK SIZE]',bsize,path.name,'complete',flush=True)
            except Exception as err:
                failures.append(dict(block_size=bsize,image=path.name,error=repr(err)))
                print('[BLOCK SIZE ERROR]',bsize,path.name,repr(err),flush=True)
            _save(output,data,failures,subbands)
    expected=len(paths)*len(block_sizes)*(len(attacks)+1)
    manifest={'schema':'block-size-reviewer-study-v1','status':'complete' if len(data)==expected and not failures else 'incomplete',
       'expected_rows':expected,'actual_rows':len(data),'failures':len(failures),
       'block_sizes':list(block_sizes),'hosts':[x.name for x in paths],'attack_names':[a.name for a in attacks],
       'parameter_source':str(param_file) if param_file else 'unoptimized_defaults',
       'frozen_params':frozen_params(param_file).to_dict(),
       'change_scope':'Q Givens matrix dimension and DWT block tiling only; full 4x4 remains original',
       'limitation':'Experimentally generalized 8x8/16x16; independent robustness must be assessed.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    if manifest['status']!='complete':raise RuntimeError(f'Block-size comparison incomplete: {manifest}')
    return manifest


def _save(output, rows, failures, subbands):
    if subbands:
        with (output/'host_subband_use.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(subbands[0]));writer.writeheader();writer.writerows(subbands)
    if rows:
        columns=list(rows[0])
        with (output/'per_case.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        import pandas as pd
        df=pd.DataFrame(rows)
        df.groupby(['block_size','attack_group'],dropna=False).agg(
            nc_mean=('nc','mean'),ber_mean=('ber','mean'),psnr_mean=('psnr','mean'),
            coverage_mean=('coverage','mean'),repetition_mean=('repeat_factor','mean'),
            hosts=('image','nunique'),rows=('attack','count')).reset_index().to_csv(output/'summary_by_attack_group.csv',index=False)
    (output/'failures.json').write_text(json.dumps(failures,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host-dir',type=Path,required=True)
    ap.add_argument('--watermark',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--param-file',type=Path)
    ap.add_argument('--block-sizes',default='4,8,16')
    ap.add_argument('--max-hosts',type=int)
    ap.add_argument('--max-attacks',type=int,default=5)
    ap.add_argument('--attack-preset',default='reviewer_stress')
    ap.add_argument('--host-size',type=int)
    ap.add_argument('--watermark-size',type=int)
    ap.add_argument('--repeat',type=int)
    args=ap.parse_args()
    print(json.dumps(run(args.host_dir,args.watermark,args.output,param_file=args.param_file,
      block_sizes=tuple(int(v) for v in args.block_sizes.split(',')),max_hosts=args.max_hosts,
      max_attacks=args.max_attacks,attack_preset=args.attack_preset,host_size=args.host_size,
      watermark_size=args.watermark_size,repeat=args.repeat),indent=2))
