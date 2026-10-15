"""Resumable Kaggle experiments; does not alter the proposed method.

Examples (run repeatedly with a persistent output directory):
 python scripts/recover_missing_results.py --step optimize --out results/recovery
 python scripts/recover_missing_results.py --step heldout --out results/recovery --max-jobs 24
 python scripts/recover_missing_results.py --step reviewer --out results/recovery --max-jobs 12
 python scripts/recover_missing_results.py --step neural4096 --out results/recovery --max-jobs 1
 python scripts/recover_missing_results.py --step extras --out results/recovery --max-jobs 1

The output ZIP is a checkpoint, not proof of experimental completeness. Inspect
recovery_status.json and completeness.json, which explicitly record blockers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
WATERMARK = ROOT / 'data' / 'watermark' / 'wm.png'


def jwrite(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def snapshot(out: Path):
    dest = out.parent / (out.name + '.zip')
    tmp = Path(str(dest) + '.partial')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for file in sorted(out.rglob('*')):
            if not file.is_file() or file.suffix == '.tmp':
                continue
            rel = file.relative_to(out)
            if any(k in rel.parts for k in ('splits', 'marked_images', 'single_host',
                                            'native_neural', 'neural_4096')):
                continue
            archive.write(file, arcname=str(Path(out.name) / rel))
    tmp.replace(dest)
    print('Saved checkpoint:', dest, flush=True)


def restore(src: Path, out: Path):
    if not src.exists():
        raise FileNotFoundError(src)
    with zipfile.ZipFile(src) as z:
        if z.testzip():
            raise ValueError('Input checkpoint corrupt')
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2 or parts[0] not in (out.name, src.stem) or '..' in parts:
                raise ValueError(f'Unexpected archive path: {member.filename}')
            rel = Path(*parts[1:])
            if any(k in rel.parts for k in ('native_neural', 'neural_4096')):
                continue  # old non-capacity-matched results are never rerun or repackaged
            if rel.is_absolute() or rel.parts[0] == '.':
                raise ValueError(f'Unsafe archive path: {rel}')
            dest = out / rel
            if dest.exists():
                continue  # never overwrite existing results; resume validators decide validity
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as inp, dest.open('wb') as fp:
                shutil.copyfileobj(inp, fp)


def csvrows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_ok(csv_path: Path, method: str, image: str, attack_names: list[str]) -> bool:
    if not csv_path.is_file():
        return False
    rows = csvrows(csv_path)
    expected = {('before_attack', 'no_attack')} | {('after_attack', a) for a in attack_names}
    actual = {(x.get('phase'), x.get('attack')) for x in rows}
    if len(rows) != len(expected) or actual != expected:
        return False
    for row in rows:
        if row.get('method_id') != method or row.get('image') != image:
            return False
        for col in ('psnr', 'ssim', 'nc', 'ber', 'embed_time_sec', 'extract_time_sec'):
            try:
                import math
                if not math.isfinite(float(row.get(col, ''))):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step', choices=('optimize','heldout','reviewer','neural4096','extras','all'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--restore-zip', type=Path)
    ap.add_argument('--max-jobs', type=int, default=16, help='Maximum NEW individual jobs in this run (0 = unlimited)')
    ap.add_argument('--max-seconds', type=int, default=31500, help='Soft budget: stop after a finished job')
    ap.add_argument('--task-timeout', type=int, default=11000, help='Per-job limit; optimize progress survives timeout')
    ap.add_argument('--include-neural', action='store_true', help='Deprecated: all already includes 4096-bit neural; native never runs')
    ap.add_argument('--checkpoint-seconds', type=int, default=600, help='Periodic snapshots during long jobs')
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.restore_zip:
        restore(args.restore_zip, out)

    env = os.environ.copy()
    env.update(PYTHONPATH=str(SRC), PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    status_file = out / 'recovery_status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {'jobs': {}, 'errors': []}
    status.setdefault('jobs', {})
    status.setdefault('errors', [])
    begin = time.monotonic()
    new_jobs = 0
    blockers = []
    stop = False

    def record(name, cmd, *, validate, timeout=None, extra_env=None):
        nonlocal new_jobs, stop
        if validate():
            status['jobs'][name] = 'PASS'
            return True
        if args.max_jobs and new_jobs >= args.max_jobs:
            stop = True
            return False
        if time.monotonic() - begin >= args.max_seconds:
            stop = True
            return False
        new_jobs += 1
        logs = out / 'logs'
        logs.mkdir(exist_ok=True)
        logfile = logs / (name.replace('/', '__') + '.log')
        print('RUN', name, flush=True)
        try:
            with logfile.open('w', encoding='utf-8') as fp:
                proc = subprocess.Popen(list(map(str, cmd)), cwd=ROOT,
                    env=env if extra_env is None else {**env, **extra_env},
                    stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
                deadline=time.monotonic()+(timeout or args.task_timeout)
                next_backup=time.monotonic()+max(60,args.checkpoint_seconds)
                timed_out=False
                while proc.poll() is None:
                    now=time.monotonic()
                    if now >= deadline or now-begin >= args.max_seconds:
                        timed_out=True
                        try: os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            try: os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            proc.wait()
                        break
                    if now >= next_backup:
                        snapshot(out)  # includes exact cached marked host + per-attack CSV
                        next_backup=now+max(60,args.checkpoint_seconds)
                    time.sleep(3)
            success = (not timed_out) and proc.returncode == 0 and validate()
            reason = ('time limit hit; partial rows checkpointed' if timed_out else
                      f'exit={proc.returncode}; validated={validate()}')
            if timed_out and time.monotonic()-begin >= args.max_seconds:
                stop = True
        except Exception as exc:
            success = False
            reason = repr(exc)
        status['jobs'][name] = 'PASS' if success else 'FAILED: ' + reason
        jwrite(status_file, status)
        snapshot(out)
        print(('PASS' if success else 'FAILED'), name, reason if not success else '', flush=True)
        if not success:
            blockers.append(name)
        return success

    from watermarklab.common.attack import default_attack_suite
    from watermarklab.methods import DEFAULT_METHOD_IDS
    from watermarklab.benchmark import load_proposal_param_file, OPT_PARAM_FORMAT, PROPOSAL_ABLATION_VARIANTS
    from watermarklab.experiments.splits import build_split, ORIGINAL_TEST_NAMES
    import pandas as pd

    split = out / 'host_split.json'
    info = build_split(ROOT/'data/host', split, train_dir=ROOT/'data/host_train', create_dirs=True)
    train, test = Path(info['train_dir']), Path(info['test_dir'])
    classical = out/'classical'
    classical.mkdir(exist_ok=True)
    param = classical/'proposal_optimized_params.json'
    guo = classical/'guo_optimized_params.json'
    gaata = classical/'gaata_optimized_params.json'
    attacks = [a.name for a in default_attack_suite(False, 'full')]

    def common_args():
        return ['--proposal-param-file',param,'--proposal-param-mode','require',
                '--guo-param-file',guo,'--guo-param-mode','require',
                '--gaata-param-file',gaata,'--gaata-param-mode','require']

    def prepared():
        if not all(f.is_file() for f in (param,guo,gaata)):
            return False
        try:
            p=json.loads(param.read_text())
            if p.get('format')!=OPT_PARAM_FORMAT or len(p.get('training_selection',[]))!=9:
                return False
            state_file=param.with_name(param.stem+'.progress.json')
            if not state_file.is_file():return False
            state=json.loads(state_file.read_text())
            if p.get('optimization_contract_sha256') != state.get('fingerprint'):return False
            current_files={f.name:sha(f) for f in [*sorted(train.glob('*.bmp')),WATERMARK]}
            if current_files != state.get('contract',{}).get('files'):return False
            if len({e['candidate_source'] for e in p['training_selection']})!=9:
                return False
            return bool(load_proposal_param_file(param)['global_params'])
        except (OSError,ValueError,KeyError,TypeError):
            return False

    def optimize():
        # Existing Guo/Gaata outputs from the user's first run are reused.
        # On a fresh setup, train them rather than silently using defaults.
        from watermarklab.benchmark import load_guo_param_file, load_gaata_param_file
        baselines = [
          ('guo', guo, 'guo2017_dwt_qr_fa', ['--guo-optimizer-fireflies','4',
                 '--guo-optimizer-generations','2','--guo-optimizer-attack-preset','lite',
                 '--guo-param-file',guo]),
          ('gaata',gaata,'gaata2022_dwt_hess_fwa',['--gaata-optimizer-population','4',
                 '--gaata-optimizer-iterations','2','--gaata-optimizer-sparks','2',
                 '--gaata-param-file',gaata]),
        ]
        for name,path,method,opts in baselines:
            loader = load_guo_param_file if name=='guo' else load_gaata_param_file
            if not record('optimization/'+name,[sys.executable,'main.py','--phase','optimize',
                    '--methods',method,'--host-dir',train,'--watermark',WATERMARK,*opts],
                    validate=lambda path=path,loader=loader: bool(loader(path).get('loaded'))):
                return False
        return record('proposal/optimize', [sys.executable,'main.py','--phase','optimize',
            '--methods','proposal','--host-dir',train,'--watermark',WATERMARK,
            '--proposal-repeat','full','--proposal-optimizer-fireflies','4',
            '--proposal-optimizer-generations','2','--proposal-optimizer-attack-preset','script',
            '--proposal-param-file',param], validate=prepared)

    def one_host(host):
        d=out/'single_host'/host.stem
        d.mkdir(parents=True,exist_ok=True)
        dest=d/host.name
        if not dest.exists() or sha(dest)!=sha(host):
            shutil.copy2(host,dest)
        return d

    def heldout():
        if not prepared():
            blockers.append('Optimization incomplete: no validated frozen parameter file')
            return False
        jobs = []
        for method in DEFAULT_METHOD_IDS:
            for host in sorted(test.glob('*.bmp')):
                folder=classical/'heldout_batches'/method/host.stem
                target=folder/'per_image_attack_results.csv'
                signature={'host':sha(host),'watermark':sha(WATERMARK),
                    'proposal':sha(param),'guo':sha(guo),'gaata':sha(gaata),
                    'code':sha(SRC/'watermarklab/benchmark.py'), 'method':method,
                    'attacks':attacks}
                digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
                guard=folder/'run_contract.json'
                def valid(target=target,guard=guard,digest=digest,method=method,host=host):
                    try:
                        return json.loads(guard.read_text()).get('digest')==digest and phase_ok(target,method,host.stem,attacks)
                    except (FileNotFoundError, ValueError):
                        return False
                if not valid():
                    folder.mkdir(parents=True,exist_ok=True)
                    # Do not certify old outputs under a new parameter/code digest.
                    if target.exists():
                        target.replace(folder/'per_image_attack_results.stale.csv')
                    cmd=[sys.executable,'main.py','--phase','normal','--methods',method,
                         '--baseline-mode','adapt','--host-dir',one_host(host),
                         '--watermark',WATERMARK,'--attack-preset','full',
                         '--proposal-repeat','full',*common_args(),
                         '--no-save-images','--output',folder]
                    ok=record(f'heldout/{method}/{host.stem}',cmd,
                        validate=lambda target=target,method=method,host=host: phase_ok(target,method,host.stem,attacks))
                    # A complete row grid is necessary, but the separate contract
                    # also prevents reusing results from other code/inputs.
                    if ok:
                        jwrite(guard, {'digest':digest,'contract':signature})
                        jwrite(status_file,status)
                    elif stop:
                        return False
                jobs.append((target, valid))
                if stop:
                    return False
        # Strict: only publish a merged table when all 12x13 jobs are valid.
        for target,valid in jobs:
            if not valid():
                blockers.append(f'Missing/invalid experiment: {target}')
                return False
        all_data=pd.concat([pd.read_csv(target) for target,_ in jobs],ignore_index=True)
        dest=classical/'heldout_full'
        dest.mkdir(exist_ok=True)
        all_data.to_csv(dest/'per_image_attack_results.csv',index=False)
        from watermarklab.experiments.evidence import audit
        evidence=classical/'statistical_evidence'
        result=audit(dest/'per_image_attack_results.csv',evidence,host_names=info['test_names'],
                     methods=DEFAULT_METHOD_IDS,attack_preset='full')
        return result['status']=='PASS'

    reviewer_jobs = ('ablation','channel','audit','bss','block')
    def reviewer():
        if not prepared():
            blockers.append('Reviewer experiments require finalized proposal optimization')
            return False
        for kind in reviewer_jobs:
            for host in sorted(test.glob('*.bmp')):
                directory=classical/'reviewer_chunks'/kind/host.stem
                sub=one_host(host)
                if kind=='ablation':
                    cmd=[sys.executable,'main.py','--phase','proposal-ablation','--host-dir',sub,
                        '--watermark',WATERMARK,'--proposal-param-file',param,
                        '--proposal-param-mode','require','--attack-preset','reviewer_stress',
                        '--no-save-images','--output',directory]
                    required=directory/'full'/'per_image_attack_results.csv'
                elif kind=='channel':
                    cmd=[sys.executable,'-m','watermarklab.experiments.channel_ablation','--host-dir',sub,
                        '--watermark',WATERMARK,'--param-file',param,'--preset','reviewer_stress',
                        '--output',directory]
                    required=directory/'channel_ablation.csv'
                elif kind=='audit':
                    cmd=[sys.executable,'-m','watermarklab.experiments.review_audit','--host-dir',sub,
                        '--watermark',WATERMARK,'--param-file',param,'--host-size','512',
                        '--watermark-size','64','--max-images','1','--max-attacks','7',
                        '--attack-preset','reviewer_stress','--output',directory]
                    required=directory/'per_case_results.csv'
                elif kind=='bss':
                    cmd=[sys.executable,'-m','watermarklab.experiments.bss_validation','--host-dir',sub,
                        '--watermark',WATERMARK,'--param-file',param,'--host-size','512',
                        '--watermark-size','64','--max-hosts','1','--max-attacks','14',
                        '--output',directory]
                    required=directory/'bss_per_host_attack.csv'
                else:
                    cmd=[sys.executable,'-m','watermarklab.experiments.block_size_comparison','--host-dir',sub,
                        '--watermark',WATERMARK,'--param-file',param,'--block-sizes','4,8,16',
                        '--max-hosts','1','--max-attacks','7','--output',directory]
                    required=directory/'per_case.csv'
                guard=directory/'run_contract.json'
                signature={'kind':kind,'host':sha(host),'watermark':sha(WATERMARK),
                           'proposal':sha(param),'code':sha(ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py')}
                digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
                def output_valid():
                    if not required.is_file():
                        return False
                    try:
                        rows=csvrows(required)
                        if not rows:
                            return False
                        if kind=='ablation':
                            attack_set={a.name for a in default_attack_suite(False,'reviewer_stress')}
                            expected={('before_attack','no_attack')} | {('after_attack',a) for a in attack_set}
                            for variant in PROPOSAL_ABLATION_VARIANTS:
                                path=directory/variant/'per_image_attack_results.csv'
                                if not path.is_file():return False
                                vrows=csvrows(path)
                                if len(vrows)!=len(expected) or {(r.get('phase'),r.get('attack')) for r in vrows}!=expected:
                                    return False
                            return not json.loads((directory/'proposal_ablation_failures.json').read_text())
                        if kind in ('channel','block'):
                            state=json.loads((directory/'manifest.json').read_text())
                            return (state.get('status')==('PASS' if kind=='channel' else 'complete')
                                    and len(rows)==state.get('expected_rows',-1)
                                    and all(r.get('host',r.get('image'))==host.name for r in rows))
                        if kind=='audit':
                            cases={r.get('case') for r in rows}
                            return ({'marked_clean','original_unmarked','wrong_secret_key'}.issubset(cases)
                                    and all(not r.get('error') for r in rows)
                                    and all(r.get('host')==host.name for r in rows))
                        if kind=='bss':
                            state=json.loads((directory/'manifest.json').read_text())
                            return (state.get('n_hosts')==1 and state.get('rows')==len(rows)
                                    and state.get('hosts')==[host.name]
                                    and all(r.get('image',r.get('host',host.name)) in (host.name,host.stem) for r in rows))
                        return False
                    except (OSError,ValueError,KeyError,TypeError):
                        return False
                def valid():
                    try:
                        return output_valid() and json.loads(guard.read_text()).get('digest')==digest
                    except (ValueError,FileNotFoundError):
                        return False
                if not valid():
                    if required.exists():
                        required.replace(required.with_suffix('.stale.csv'))
                    ok=record(f'reviewer/{kind}/{host.stem}',cmd,
                        validate=output_valid)
                    if ok:
                        jwrite(guard,{'digest':digest,'signature':signature})
                    elif stop:
                        return False
                if not valid():
                    blockers.append(f'{kind}: invalid/incomplete host result {host.name}')
                    return False
                if stop:
                    return False
            # Rebuild aggregate across all 12 hosts; no partial aggregate masquerades as complete.
            root=classical/({'ablation':'ablation','channel':'channel_ablation',
                'audit':'review_audit','bss':'bss_validation','block':'block_size_comparison'}[kind])
            hosts=sorted(test.glob('*.bmp'))
            for host in hosts:
                partial=classical/'reviewer_chunks'/kind/host.stem
                if not (partial/'run_contract.json').is_file():
                    blockers.append(f'{kind}: incomplete host {host.name}')
                    return False
            if kind=='ablation':
                variants = list(PROPOSAL_ABLATION_VARIANTS)
                for variant in variants:
                    paths=[classical/'reviewer_chunks'/kind/h.stem/variant/'per_image_attack_results.csv' for h in hosts]
                    if not all(p.exists() for p in paths):
                        blockers.append(f'Ablation {variant} incomplete')
                        return False
                    d=root/variant;d.mkdir(parents=True,exist_ok=True)
                    pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(d/'per_image_attack_results.csv',index=False)
            else:
                name={'channel':'channel_ablation.csv','audit':'per_case_results.csv',
                      'bss':'bss_per_host_attack.csv','block':'per_case.csv'}[kind]
                paths=[classical/'reviewer_chunks'/kind/h.stem/name for h in hosts]
                if not all(p.is_file() for p in paths):
                    blockers.append(f'{kind} missing host results')
                    return False
                root.mkdir(parents=True,exist_ok=True)
                pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(root/name,index=False)
        subprocess.run([sys.executable,'scripts/export_reviewer_evidence.py',
            '--results-dir',classical,'--output',classical/'reviewer_tables'],cwd=ROOT,env=env,check=True)
        return True

    def neural4096():
        # Intentionally ONLY the original 64x64/4096-bit watermark extension.
        # No main.py --phase neural or native/random 61-/32-bit experiments.
        # Decode failures remain failures; --force-full merely prevents the
        # clean-feasibility gate from skipping all attack measurements.
        from watermarklab.experiments.neural_4096 import RESULT_COLUMNS
        for method in ('trustmark','wam'):
            for host in sorted(test.glob('*.bmp')):
                directory=out/'neural_4096_forced'/method/host.stem
                target=directory/'per_image_attack_results.csv'
                manifest=directory/'manifest.json'
                def valid():
                    try:
                        m=json.loads(manifest.read_text())
                        rows=csvrows(target)
                        expected={('before_attack','no_attack'),('control','original_unmarked')}
                        expected.update(('after_attack', a) for a in attacks)
                        actual={(r.get('phase'),r.get('attack')) for r in rows}
                        if not (m.get('complete_grid') is True and
                                m.get('expected_attacked_rows')==len(attacks) and
                                m.get('actual_attacked_rows')==len(attacks) and
                                len(rows)==len(expected) and actual==expected):
                            return False
                        if any(str(r.get('source_payload_bits'))!='4096' for r in rows):
                            return False
                        if any(r.get('phase')=='embedding_failed' for r in rows):
                            return False
                        for row in rows:
                            if row.get('image')!=host.stem or row.get('watermark')!=WATERMARK.name:
                                return False
                            # Missing bits (coverage=0) are scientifically valid;
                            # missing numeric metrics due to a processing exception are not.
                            for field in ('psnr','ssim','bit_coverage','ber_erasure_as_error'):
                                if not math.isfinite(float(row.get(field,''))):
                                    return False
                        return True
                    except (OSError,ValueError,TypeError,KeyError):
                        return False
                if valid():
                    status['jobs'][f'neural4096/{method}/{host.stem}']='PASS'
                    continue
                local_env={}
                wam_dir=Path(env.get('WAM_REPO_DIR',str(ROOT.parent/'watermark-anything')))
                wam_weights=Path(env.get('WAM_CHECKPOINT',str(wam_dir/'checkpoints/wam_mit.pth')))
                if method=='wam':
                    if not (wam_dir/'checkpoints/params.json').is_file() or not wam_weights.is_file():
                        blockers.append('WAM official checkout/config/checkpoint missing; install before the 4096-bit run')
                        return False
                    local_env={'WAM_REPO_DIR':str(wam_dir),'WAM_CHECKPOINT':str(wam_weights)}
                cmd=[sys.executable,'-m','watermarklab.experiments.neural_4096',
                     '--methods',method,'--host-dir',one_host(host),
                     '--watermark',WATERMARK,'--attack-preset','full','--max-attacks','0',
                     '--device','auto','--resize-to','256','--force-full','--output',directory]
                ok=record(f'neural4096/{method}/{host.stem}',cmd,validate=valid,extra_env=local_env)
                if not ok:
                    if stop:
                        return False
                    # Do not burn the session by repeatedly executing a broken model
                    # or a missing runtime prerequisite across all 12 images.
                    blockers.append(f'neural4096/{method}/{host.stem}: read job log and resume')
                    return False
        # Aggregate only after EVERY image has its complete, validated grid.
        files=[out/'neural_4096_forced'/method/h.stem/'per_image_attack_results.csv'
               for method in ('trustmark','wam') for h in sorted(test.glob('*.bmp'))]
        if not all(x.is_file() for x in files):
            blockers.append('4,096-bit neural attack grid incomplete')
            return False
        all_df=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
        total=2*len(info['test_names'])*(2+len(attacks))
        if len(all_df)!=total:
            blockers.append('4,096-bit neural row count mismatch')
            return False
        target=out/'neural_4096_forced'
        all_df.to_csv(target/'per_image_attack_results.csv',index=False)
        values=['psnr','ssim','nc','ncc','ber','nc_zero_fill', 'ncc_zero_fill',
                'ber_erasure_as_error','bit_coverage','exact_recovery']
        for col in values:all_df[col]=pd.to_numeric(all_df[col],errors='coerce')
        summary=all_df[all_df.phase.isin(('before_attack','after_attack'))].groupby(
            ['method_id','phase'],as_index=False).agg(
            n=('image','count'),**{col+'_mean':(col,'mean') for col in values})
        summary.to_csv(target/'comparison_4096.csv',index=False)
        jwrite(target/'aggregate_manifest.json',{
            'mode':'force_full_4096bit_only_not_native_payload',
            'source_watermark_bits':4096,'clean_gate_not_relaxed':True,
            'hosts':len(info['test_names']),'attack_cases_per_host':len(attacks),
            'methods':['trustmark_4096_spatial','wam_4096_spatial'],
            'expected_rows':total,'actual_rows':len(all_df),
            'warning':'Clean gate failures and incorrect/missing bits remain genuine failures.'})
        return True

    def extras():
        if not prepared():
            blockers.append('Sensitivity and figures require optimized parameters')
            return False
        tasks = [
          ('size_sweep',[sys.executable,'main.py','--phase','proposal-size-sweep',
             '--host-dir',test,'--max-images','2','--watermark',WATERMARK,
             '--attack-preset','lite','--no-save-images','--output',classical/'size_sweep']),
          ('hparam_sweep',[sys.executable,'main.py','--phase','proposal-hparam-sweep',
             '--host-dir',train,'--watermark',WATERMARK,'--attack-preset','lite',
             '--max-images','1','--no-save-images','--proposal-repeat','1',
             '--hparam-sweep-max-combinations','24','--output',classical/'hparam_sweep']),
          ('proposal_figures',[sys.executable,'main.py','--phase','proposal-plot',
             '--host-dir',test,'--watermark',WATERMARK,'--proposal-param-file',param,
             '--proposal-param-mode','require','--proposal-repeat','full',
             '--proposal-plot-layout','grid','--output',out/'proposal_figures']),
        ]
        for name,cmd in tasks:
            target=classical/name if name!='proposal_figures' else out/'proposal_figures'
            marker=target/'run_complete.json'
            signature=sha(param)+'/'+sha(WATERMARK)+'/'+name
            def valid():
                try:
                    return marker.is_file() and json.loads(marker.read_text())['signature']==signature
                except (OSError,ValueError,KeyError):
                    return False
            if valid():continue
            ok=record(f'extras/{name}',cmd,validate=lambda: target.is_dir() and any(target.rglob('*.csv')) or (name=='proposal_figures' and any(target.rglob('*.pdf'))))
            if ok:jwrite(marker,{'signature':signature})
            elif stop:return False
        return True

    methods={'optimize':optimize,'heldout':heldout,'reviewer':reviewer,'neural4096':neural4096,'extras':extras}
    steps=['optimize','heldout','reviewer','extras','neural4096'] if args.step=='all' else [args.step]
    try:
        for step in steps:
            ok=methods[step]()
            if stop or not ok:
                break
    finally:
        status['last_requested_steps']=steps
        status['last_finished_jobs']=new_jobs
        status['blockers']=blockers
        status['paused_by_budget']=stop
        status['classical_final_exists']=(classical/'heldout_full/per_image_attack_results.csv').is_file()
        expected_heldout=len(DEFAULT_METHOD_IDS)*len(info['test_names'])*(1+len(attacks))
        heldout_path=classical/'heldout_full/per_image_attack_results.csv'
        heldout_status=classical/'statistical_evidence/completeness.json'
        heldout_ok=False
        try:
            heldout_ok=(heldout_path.is_file() and len(csvrows(heldout_path))==expected_heldout
                       and json.loads(heldout_status.read_text()).get('status')=='PASS')
        except (OSError, ValueError, KeyError):
            pass
        reviewer_rel=('review_audit/per_case_results.csv', 'bss_validation/bss_per_host_attack.csv',
                      'ablation/full/per_image_attack_results.csv', 'block_size_comparison/per_case.csv',
                      'channel_ablation/channel_ablation.csv')
        reviewer_ok=all((classical/name).is_file() for name in reviewer_rel)
        # Do not trust aggregate CSV existence without validated per-host evidence.
        if reviewer_ok:
            for kind in reviewer_jobs:
                for test_host in sorted(test.glob('*.bmp')):
                    guard=classical/'reviewer_chunks'/kind/test_host.stem/'run_contract.json'
                    try:
                        g=json.loads(guard.read_text())
                        signature=g['signature']
                        hashed=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
                        reviewer_ok &= (g['digest']==hashed and signature['host']==sha(test_host)
                                        and signature['watermark']==sha(WATERMARK)
                                        and signature['proposal']==sha(param))
                    except (OSError, ValueError, KeyError):
                        reviewer_ok=False
                    if not reviewer_ok:break
                if not reviewer_ok:break
        if reviewer_ok:
            for rel in reviewer_rel:
                rows=csvrows(classical/rel)
                column='image' if rel.startswith(('ablation/', 'block_size')) else 'host'
                names={h.stem for h in test.glob('*.bmp')} if rel.startswith('ablation/') else {h.name for h in test.glob('*.bmp')}
                reviewer_ok &= {r.get(column) for r in rows}==names
                if not reviewer_ok:break
        try:
            reviewer_ok &= not json.loads((classical/'reviewer_tables/manifest.json').read_text()).get('missing')
        except (OSError,ValueError,KeyError):
            reviewer_ok=False
        extra_names=('size_sweep','hparam_sweep','proposal_figures')
        extras_ok=True
        for name in extra_names:
            target=classical/name if name!='proposal_figures' else out/'proposal_figures'
            try:
                sig=json.loads((target/'run_complete.json').read_text()).get('signature')
                extras_ok &= (sig==sha(param)+'/'+sha(WATERMARK)+'/'+name)
            except (OSError,ValueError,KeyError):
                extras_ok=False
        neural_manifest=out/'neural_4096_forced/aggregate_manifest.json'
        neural_ok=False
        try:
            state=json.loads(neural_manifest.read_text())
            neural_ok=(state.get('actual_rows') == 2*len(info['test_names'])*(2+len(attacks))
                       and state.get('attack_cases_per_host')==len(attacks))
        except (OSError, ValueError, KeyError):
            pass
        status['strict_completeness']={
            'classical_13_methods_12_hosts_109_attacks':heldout_ok,
            'reviewer_all_hosts_validated':reviewer_ok,
            'extras_sensitivity_and_figures':extras_ok,
            'neural_4096_both_models_full_grid':neural_ok,
            'expected_heldout_rows':expected_heldout,
            'expected_neural_rows':2*len(info['test_names'])*(2+len(attacks)),
            'all_core_sections_complete':all((heldout_ok,reviewer_ok,extras_ok,neural_ok))}
        jwrite(out/'completeness.json', status['strict_completeness'])
        jwrite(status_file,status)
        snapshot(out)
    print('Done. Read',status_file,'before claiming full completeness.')
    if blockers and not stop:
        sys.exit(1)

if __name__=='__main__':
    main()
