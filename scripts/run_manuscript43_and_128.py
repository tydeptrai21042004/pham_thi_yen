"""Strict, resumable-by-stage 43-attack manuscript experiments.

Runs only requested stages, never invents missing results and never touches the
manuscript. Select the held-out 12-host directory and a TRAIN-only frozen
proposal parameter JSON to produce publication-comparable results.

Usage:
 python scripts/run_manuscript43_and_128.py --hosts /path/to/test12 \
   --train-hosts /path/to/train9 --proposal-params /path/to/proposal_optimized_params.json \
   --output /kaggle/working/manuscript43 --stages all --dry-run

Each stage writes a log and a manifest. Re-run with a subset of --stages to
recover after a Kaggle session stops. A completed stage is not silently assumed
valid because its output directory happens to exist.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from watermarklab.common.attack import default_attack_suite, manuscript43_family, reviewer45_family
from watermarklab.benchmark import load_proposal_param_file
from watermarklab.common.io_utils import load_host_rgb, list_image_files

PRESET='manuscript43'  # default, overridable with --attack-preset
STAGES=('baseline64','ablation64','block64','channel64','bss64','hparam64','resize128','trustmark64')


def preflight(args):
    preset=args.attack_preset
    family=manuscript43_family if preset=="manuscript43" else reviewer45_family
    attacks=default_attack_suite(False,preset)
    assert len(attacks)==(43 if preset=="manuscript43" else 45) and len({a.name for a in attacks})==len(attacks)
    from collections import Counter
    counts=dict(Counter(family(a.name) for a in attacks))
    expected_counts={'compression':8,'noise':8,'filtering':(6 if preset=='manuscript43' else 5),'geometry':(10 if preset=='manuscript43' else 13),'light_colour':5,'combined':6}
    assert counts==expected_counts, counts
    watermarks=[Path(p).resolve() for p in args.watermarks]
    for p in watermarks:
        if not p.is_file():raise FileNotFoundError(p)
    host_dir=Path(args.hosts).resolve()
    if not host_dir.is_dir():raise NotADirectoryError(host_dir)
    hosts=[p for p in list_image_files(host_dir) if p.stem.lower() not in ('lenna','lena')]
    if not hosts:raise ValueError('No test hosts found')
    if not args.dry_run and args.stages!='preflight' and len(hosts)!=12 and not args.allow_non12_hosts:
        raise ValueError(f'Found {len(hosts)} hosts, not 12. Use the manuscript held-out test12 split or --allow-non12-hosts for a non-paper diagnostic.')
    if not args.dry_run and args.stages!='preflight' and len(watermarks)!=2:
        raise ValueError('The manuscript protocol requires exactly two watermark files')
    proposal_file=Path(args.proposal_params).resolve() if args.proposal_params else None
    if not args.dry_run and args.stages not in ('preflight','trustmark64'):
        if proposal_file is None or not proposal_file.is_file():
            raise FileNotFoundError('A training-only frozen --proposal-params JSON is required (not an auto-optimized heldout file)')
        parsed=load_proposal_param_file(proposal_file)
        if not parsed.get('global_params'):raise ValueError('The proposal file has no frozen global_params')
    for p in hosts:
        if load_host_rgb(p).shape[:2]!=(512,512):
            raise ValueError(f'Host {p.name} is not 512x512')
    train_dir=Path(args.train_hosts).resolve() if args.train_hosts else None
    if args.train_hosts and not train_dir.is_dir():raise NotADirectoryError(train_dir)
    return {'schema':f'{preset}-protocol-v1','attack_preset':preset,'attack_count':len(attacks),
      'clean_control':'no_attack (separate)', 'attack_families':counts,'attack_configs':[
       {'row_number':i+2,'name':a.name,'family':family(a.name),
        'operator':a.group,'params':a.params} for i,a in enumerate(attacks)],
      'host_dir':str(host_dir),'hosts':[p.name for p in hosts],
      'watermarks':[{'filename':p.name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in watermarks],
      'proposal_params':str(proposal_file) if proposal_file else None,
      'train_host_dir':str(train_dir) if train_dir else None,
      'host_pixels':'512x512 RGB', 'watermark_source_sizes':[64,128],
      'effective_DWT_LL_bits':{'64':1024,'128':4096},
      'jpeg2000_codec':'Pillow/OpenJPEG required; no fallback to JPEG',
      'geometry':'non-registered bicubic size-preserving attacks, black fill',
      'note':'Pillow and random-seed choices are explicit implementations; the manuscript does not fully specify interpolation/filters.'}


def _job_commands(args, record):
    host=str(Path(args.hosts).resolve()); out=Path(args.output).resolve(); preset=args.attack_preset
    watermarks=[str(Path(p).resolve()) for p in args.watermarks]
    par=str(Path(args.proposal_params).resolve()) if args.proposal_params else ''
    limit=['--max-images',str(args.max_images)] if args.max_images else []
    max_hosts=['--max-hosts',str(args.max_images)] if args.max_images else []
    common=[sys.executable,str(ROOT/'main.py')]
    py=[sys.executable,'-m']
    jobs=[]
    def add(stage, label, cmd):jobs.append((stage,label,cmd))
    for i,w in enumerate(watermarks,1):
        name=f'w{i}'
        add('baseline64',name,common+['--phase','normal','--host-dir',host,'--watermark',w,
            '--attack-preset',preset,'--methods',args.methods,'--baseline-mode','adapt',
            '--proposal-repeat','full','--proposal-param-file',par,'--proposal-param-mode','require',
            '--gaata-param-mode','ignore','--guo-param-mode','ignore',
            '--resume-attack-journal','--no-save-images','--output',str(out/'baseline64'/name)]+limit)
        add('ablation64',name,common+['--phase','proposal-ablation','--host-dir',host,'--watermark',w,
            '--attack-preset',preset,'--proposal-param-file',par,'--proposal-param-mode','require',
            '--proposal-repeat','full','--no-save-images','--output',str(out/'ablation64'/name)]+limit)
        add('block64',name,py+['watermarklab.experiments.block_size_comparison','--host-dir',host,'--watermark',w,
            '--param-file',par,'--output',str(out/'block64'/name),'--block-sizes','4,8,16',
            '--max-attacks','0','--attack-preset',preset]+max_hosts)
        add('channel64',name,py+['watermarklab.experiments.channel_ablation','--host-dir',host,'--watermark',w,
            '--param-file',par,'--output',str(out/'channel64'/name),'--max-attacks','0',
            '--watermark-size','64','--preset',preset]+limit)
        add('bss64',name,py+['watermarklab.experiments.bss_validation','--host-dir',host,'--watermark',w,
            '--param-file',par,'--output',str(out/'bss64'/name),'--max-attacks','0',
            '--watermark-size','64','--host-size','512','--max-hosts',str(args.max_images or 12),
            '--attack-preset',preset])
        if args.train_hosts:
            add('hparam64',name,common+['--phase','proposal-hparam-sweep','--host-dir',str(Path(args.train_hosts).resolve()),
                '--watermark',w,'--attack-preset',preset,'--hparam-sweep-max-combinations',str(args.hparam_combinations),
                '--proposal-repeat','full','--no-save-images','--output',str(out/'hparam64'/name)])
        add('trustmark64',name,py+['watermarklab.experiments.neural_4096','--host-dir',host,'--watermark',w,
            '--output',str(out/'trustmark64'/name),'--methods','trustmark','--attack-preset',preset,
            '--max-attacks','0','--force-full','--binary-channel',('alpha' if 'logo' in w.lower() else 'luma')]+limit)
    add('resize128','w1_w2',common+['--phase','proposal-size-sweep','--host-dir',host,
        '--watermark',watermarks[0],'--size-sweep-watermarks',','.join(watermarks),
        '--size-sweep-host-sizes','512','--size-sweep-watermark-sizes','64,128',
        '--proposal-param-file',par,'--proposal-param-mode','require',
        '--proposal-repeat','full','--attack-preset',preset,'--no-save-images',
        '--output',str(out/'resized64_vs_128')]+limit)
    return jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attack-preset',default='manuscript43',choices=['manuscript43','reviewer45'],help='Separate 43-original or 45-extended output; do not mix outputs')
    ap.add_argument('--hosts',default=str(ROOT/'data'/'host'),help='The 12 held-out 512x512 test host images, not training hosts')
    ap.add_argument('--watermarks',nargs=2,default=[str(ROOT/'data/watermark/wm.png'),str(ROOT/'data/watermark/Cc.logo.circle.png')])
    ap.add_argument('--train-hosts',help='Separate nine training images; required only for hparam64')
    ap.add_argument('--proposal-params',help='Trained on separate training hosts and frozen before evaluation')
    ap.add_argument('--output',default=str(ROOT/'results/manuscript43'))
    ap.add_argument('--methods',default='proposal,guo2017_dwt_qr_fa,zhu2021_iwt_svd_adapted,qwt_qsvd_zhang2022_blind,hess_nha2023,gaata2022_dwt_hess_fwa,su2016_contourlet_hess',help='Classical methods for baseline64; all require matched 43 attacks')
    ap.add_argument('--stages',default='preflight',help='Comma-separated: '+','.join(STAGES)+', all, or preflight')
    ap.add_argument('--hparam-combinations',type=int,default=24)
    ap.add_argument('--max-images',type=int,help='Debug mode only. Manuscript tables require 12 hosts')
    ap.add_argument('--allow-non12-hosts',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    record=preflight(args)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    other='reviewer45' if args.attack_preset=='manuscript43' else 'manuscript43'
    if (out/f'{other}_protocol.json').exists():
        raise ValueError(f'Output already has {other} protocol; choose a distinct --output folder')
    (out/f'{args.attack_preset}_protocol.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    if args.stages=='preflight':
        print('PREFLIGHT',len(record['hosts']),'hosts;',record['attack_count'],'attacked cases plus clean;',record['effective_DWT_LL_bits'])
        print('Saved',out/f'{args.attack_preset}_protocol.json');return
    selected=STAGES if args.stages=='all' else tuple(x.strip() for x in args.stages.split(',') if x.strip())
    unknown=set(selected)-set(STAGES)
    if unknown:ap.error(f'Invalid --stages: {sorted(unknown)}')
    if 'hparam64' in selected and not args.train_hosts:ap.error('hparam64 requires --train-hosts; never tune on held-out images')
    if args.max_images and not args.dry_run:
        print('WARNING: subset/diagnostic run; results cannot fill 12-host manuscript tables',flush=True)
    jobs=[(s,label,cmd) for s,label,cmd in _job_commands(args,record) if s in selected]
    env=os.environ.copy();env['WATERMARKLAB_STRICT_PYWT']='1';env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    statuses=[]
    manifest_path=out/'stage_status.json'
    for stage,label,cmd in jobs:
        name=f'{stage}_{label}';log=out/f'{name}.log'
        print(('DRY' if args.dry_run else 'RUN'),name, ' '.join(cmd),flush=True)
        if args.dry_run:
            statuses.append({'stage':name,'status':'planned','command':cmd});continue
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as fp:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT)
        status='complete' if proc.returncode==0 else 'failed'
        statuses.append({'stage':name,'status':status,'returncode':proc.returncode,
            'seconds':round(time.monotonic()-start,1),'log':str(log),'command':cmd})
        manifest_path.write_text(json.dumps(statuses,indent=2))
        if proc.returncode:
            raise RuntimeError(f'Stage {name} failed (exit {proc.returncode}); see {log}. Prior outputs preserved.')
    manifest_path.write_text(json.dumps(statuses,indent=2))
    print('Stage record:',manifest_path)

if __name__=='__main__': main()
