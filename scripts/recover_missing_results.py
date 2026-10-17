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
import platform
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
            if rel.parts[0] == 'watermark_2_Cc':
                continue  # secondary has its own checkpoint ZIP kept at this level
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
                continue  # the old native and non-forced neural evidence is not reusable
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
    ap.add_argument('--step', choices=('optimize','heldout','reviewer','reports','neural4096','extras','all','secondary','verify'), required=True)
    ap.add_argument('--watermark', type=Path, help='Watermark for this separate cohort, default: wm.png')
    ap.add_argument('--skip-secondary', action='store_true', help='Internal: prevent a recursive second cohort')
    ap.add_argument('--include-neural4096', action='store_true', help='Explicit opt-in: extended 4096-bit neural baselines; NEVER native 61-/32-bit tests')
    ap.add_argument('--extended-sensitivity', action='store_true', help='Use all 12 test images for size sweep and all 9 training images for sensitivity sweep')
    ap.add_argument('--proposal-fireflies', type=int, default=4, help='Proposal optimization population, default 4')
    ap.add_argument('--proposal-generations', type=int, default=2, help='Proposal optimization generations, default 2')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--restore-zip', type=Path)
    ap.add_argument('--max-jobs', type=int, default=16, help='Maximum NEW individual jobs in this run (0 = unlimited)')
    ap.add_argument('--max-seconds', type=int, default=31500, help='Soft budget: stop after a finished job')
    ap.add_argument('--task-timeout', type=int, default=11000, help='Per-job limit; optimize progress survives timeout')
    ap.add_argument('--include-neural', action='store_true', help='Deprecated unsafe/ambiguous name; use --include-neural4096 instead')
    ap.add_argument('--checkpoint-seconds', type=int, default=600, help='Periodic snapshots during long jobs')
    args = ap.parse_args(argv)
    if args.proposal_fireflies < 1 or args.proposal_generations < 1:
        ap.error('--proposal-fireflies and --proposal-generations must be positive integers')
    if args.include_neural:
        ap.error('Ambiguous --include-neural is disabled. Only --include-neural4096 is supported; native 61/32-bit runs are forbidden.')
    global WATERMARK
    if args.watermark is not None:
        WATERMARK=args.watermark.resolve()
    if not WATERMARK.is_file():raise FileNotFoundError(WATERMARK)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.restore_zip:
        restore(args.restore_zip, out)

    # Reproducible measured-runtime metadata for Reviewer 1 (platform only;
    # execution times remain the real measured benchmark column values).
    def hardware_environment():
        cpu_model=platform.processor() or 'not-reported'
        ram_kib=None
        if Path('/proc/cpuinfo').is_file():
            for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
                if line.lower().startswith('model name'):
                    cpu_model=line.partition(':')[2].strip()
                    break
        if Path('/proc/meminfo').is_file():
            for line in Path('/proc/meminfo').read_text(errors='replace').splitlines():
                if line.startswith('MemTotal:'):
                    ram_kib=int(line.split()[1]);break
        versions={}
        from importlib import metadata
        for package in ('numpy','scipy','pandas','scikit-image','opencv-python',
                        'PyWavelets','Pillow'):
            try:versions[package]=metadata.version(package)
            except metadata.PackageNotFoundError:versions[package]='missing'
        return {'platform':platform.platform(),'python':sys.version.split()[0],
                'cpu_model':cpu_model,'logical_cpu_count':os.cpu_count(),
                'reported_memory_kib':ram_kib,
                'versions':versions,'thread_env':{name:os.getenv(name) for name in
                   ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
                'execution_mode':'classical_cpu_benchmark_unless_explicit_neural4096'}
    jwrite(out/'hardware_environment.json',hardware_environment())

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
        if not success and logfile.exists():
            # Show actionable traceback/tail, without flooding Kaggle notebook output.
            tail=logfile.read_text(encoding='utf-8',errors='replace').splitlines()[-18:]
            print('--- failure log tail: ' + str(logfile) + ' ---', flush=True)
            print('\n'.join(tail), flush=True)
        if not success:
            blockers.append(name)
        return success

    from watermarklab.common.attack import default_attack_suite
    from watermarklab.methods import DEFAULT_METHOD_IDS
    from watermarklab.benchmark import load_proposal_param_file, OPT_PARAM_FORMAT, PROPOSAL_ABLATION_VARIANTS
    from watermarklab.experiments.splits import build_split, ORIGINAL_TEST_NAMES
    from watermarklab.experiments.reviewer_checks import (validate_channel,
        validate_bss, validate_security, validate_gallery)
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
            contract=state.get('contract', {})
            if current_files != contract.get('files'):return False
            if (contract.get('n_fireflies')!=args.proposal_fireflies
                    or contract.get('n_generations')!=args.proposal_generations
                    or contract.get('attack_preset')!='script'
                    or contract.get('repeat')!='full'):
                return False
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
            '--proposal-repeat','full','--proposal-optimizer-fireflies',str(args.proposal_fireflies),
            '--proposal-optimizer-generations',str(args.proposal_generations),'--proposal-optimizer-attack-preset','script',
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
                # Bind a separate validator for each method/host. A closure over
                # the loop variable would otherwise validate only the LAST job
                # during the aggregate phase, even if earlier batches were bad.
                def make_validator(target, guard, digest, method, host_stem, attack_names):
                    def is_valid():
                        try:
                            return (json.loads(guard.read_text()).get('digest') == digest
                                    and phase_ok(target, method, host_stem, attack_names))
                        except (FileNotFoundError, ValueError, OSError):
                            return False
                    return is_valid
                valid=make_validator(target,guard,digest,method,host.stem,attacks)
                if not valid():
                    folder.mkdir(parents=True,exist_ok=True)
                    # Do not certify old outputs under a new parameter/code digest.
                    if target.exists():
                        target.replace(folder/'per_image_attack_results.stale.csv')
                    cmd=[sys.executable,'main.py','--phase','normal','--methods',method,
                         '--baseline-mode','adapt','--host-dir',one_host(host),
                         '--watermark',WATERMARK,'--attack-preset','full',
                         '--proposal-repeat','full',*common_args(),
                         '--resume-attack-journal',
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
                        '--attack-preset','reviewer_stress','--wrong-key-trials','8',
                        '--unrelated-host',next(p for p in sorted(test.glob('*.bmp')) if p.name != host.name),
                        '--output',directory]
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
                           'proposal':sha(param),'code':sha(ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py'),
                           'audit_code':sha(ROOT/'src/watermarklab/experiments/review_audit.py') if kind=='audit' else None,
                           'bss_code':sha(ROOT/'src/watermarklab/experiments/bss_validation.py') if kind=='bss' else None,
                           'channel_code':sha(ROOT/'src/watermarklab/experiments/channel_ablation.py') if kind=='channel' else None,
                           'reviewer_checks':sha(ROOT/'src/watermarklab/experiments/reviewer_checks.py')}
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
                        if kind=='channel':
                            return validate_channel(directory,host.name)
                        if kind=='audit':
                            return validate_security(directory,host.name,wrong_key_trials=8)
                        if kind=='bss':
                            return validate_bss(directory,host.name,max_attacks=14)
                        if kind=='block':
                            state=json.loads((directory/'manifest.json').read_text())
                            return (state.get('status')=='complete'
                                    and len(rows)==state.get('expected_rows',-1)
                                    and all(r.get('image')==host.name for r in rows))
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

    def reports():
        # Numerical reviewer tables are derived ONLY from validated raw experiments;
        # no baseline rerun and no changed proposal equations are involved.
        destination=classical/'reviewer_analysis'
        state=destination/'review_coverage.json'
        fingerprint={'watermark':sha(WATERMARK),
                     'script':sha(ROOT/'scripts/build_review_tables.py'),
                     'heldout':sha(classical/'heldout_full/per_image_attack_results.csv') if
                         (classical/'heldout_full/per_image_attack_results.csv').is_file() else None,
                     'reviewer_inputs':{name:sha(classical/rel) if (classical/rel).is_file() else None
                         for name,rel in (('security','review_audit/per_case_results.csv'),
                                          ('bss','bss_validation/bss_per_host_attack.csv'),
                                          ('channel','channel_ablation/channel_ablation.csv'),
                                          ('blocks','block_size_comparison/per_case.csv'))},
                     'ablation':{v:sha(classical/'ablation'/v/'per_image_attack_results.csv')
                          for v in PROPOSAL_ABLATION_VARIANTS if
                          (classical/'ablation'/v/'per_image_attack_results.csv').is_file()}}
        digest=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
        guard=destination/'run_contract.json'
        def valid():
            try:
                return (json.loads(state.read_text()).get('complete') is True
                        and json.loads(guard.read_text()).get('digest')==digest)
            except (OSError,ValueError,TypeError):
                return False
        if valid():return True
        cmd=[sys.executable,'scripts/build_review_tables.py',
             '--results-dir',classical,'--out',destination,
             '--watermark',WATERMARK.name]
        ok=record('reports/reviewer_tables',cmd,
                  validate=lambda:state.is_file() and json.loads(state.read_text()).get('complete') is True)
        if ok:jwrite(guard,{'digest':digest,'inputs':fingerprint})
        return valid()

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
                                m.get('watermark_sha256')==sha(WATERMARK) and
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

    def extras_output_ok(name, target):
        """Only certify complete output grids; file existence is insufficient."""
        try:
            lite = [a.name for a in default_attack_suite(False, 'lite')]
            expected_hosts = 12 if args.extended_sensitivity else 2
            if name == 'size_sweep':
                clean = csvrows(target/'proposal_size_sweep_clean_results.csv')
                attacked = csvrows(target/'proposal_size_sweep_attack_results.csv')
                failures = json.loads((target/'proposal_size_sweep_failures.json').read_text())
                # Default sweep: 3 host dimensions x 3 payload dimensions.
                count = expected_hosts * 3 * 3
                expected_keys = {(h.name, hs, ws) for h in sorted(test.glob('*.bmp'))[:expected_hosts]
                                 for hs in ('256','512','1024') for ws in ('32','64','128')}
                actual_keys = {(r['host_image'],r['host_size'],r['watermark_size']) for r in clean}
                attacked_keys = {(r['host_image'],r['host_size'],r['watermark_size'],r['attack'])
                                 for r in attacked}
                return (not failures and len(clean)==count and actual_keys==expected_keys
                        and len(attacked)==count*len(lite)
                        and attacked_keys=={(*key,attack) for key in expected_keys for attack in lite}
                        and all(math.isfinite(float(r[k])) for r in clean
                                for k in ('clean_psnr','clean_ssim','clean_nc','clean_ber'))
                        and all(math.isfinite(float(r[k])) for r in attacked
                                for k in ('nc_after_attack','ber_after_attack','psnr_after_attack')))
            if name == 'hparam_sweep':
                rows=csvrows(target/'proposal_hparam_sweep_per_image_attack_results.csv')
                failures=json.loads((target/'proposal_hparam_sweep_failures.json').read_text())
                n_hosts=9 if args.extended_sensitivity else 1
                if failures or len(rows)!=24*n_hosts*(1+len(lite)):
                    return False
                keys={(r['sweep_combo_id'],r['image'],r['phase'],r['attack']) for r in rows}
                combos={r['sweep_combo_id'] for r in rows}
                hostnames={p.stem for p in sorted(train.glob('*.bmp'))[:n_hosts]}
                expected={(combo,host,'before_attack','no_attack') for combo in combos for host in hostnames}
                expected|={(combo,host,'after_attack',a) for combo in combos
                           for host in hostnames for a in lite}
                return (len(combos)==24 and len(keys)==len(rows) and keys==expected
                        and all(math.isfinite(float(r[k])) for r in rows
                                for k in ('psnr','ssim','nc','ber')))
            if name == 'proposal_figures':
                return (any(target.glob('proposal_attack_sensitivity_*.pdf'))
                        and (target/'proposal_plot_detail_all_watermarks.csv').is_file())
        except (OSError,ValueError,KeyError,TypeError,OverflowError):
            return False
        return False

    def extras():
        if not prepared():
            blockers.append('Sensitivity and figures require optimized parameters')
            return False
        tasks = [
          ('size_sweep',[sys.executable,'main.py','--phase','proposal-size-sweep',
             '--host-dir',test,'--max-images',str(12 if args.extended_sensitivity else 2),'--watermark',WATERMARK,
             '--attack-preset','lite','--proposal-param-file',param,
             '--proposal-param-mode','require','--proposal-repeat','full',
             '--no-save-images','--output',classical/'size_sweep']),
          ('hparam_sweep',[sys.executable,'main.py','--phase','proposal-hparam-sweep',
             '--host-dir',train,'--watermark',WATERMARK,'--attack-preset','lite',
             '--max-images',str(9 if args.extended_sensitivity else 1),'--no-save-images','--proposal-repeat','1',
             '--hparam-sweep-max-combinations','24','--output',classical/'hparam_sweep']),
          ('proposal_figures',[sys.executable,'main.py','--phase','proposal-plot',
             '--host-dir',test,'--watermark',WATERMARK,'--proposal-param-file',param,
             '--proposal-param-mode','require','--proposal-repeat','full',
             '--proposal-plot-layout','grid','--output',out/'proposal_figures']),
        ]
        for name,cmd in tasks:
            target=classical/name if name!='proposal_figures' else out/'proposal_figures'
            marker=target/'run_complete.json'
            signature=(sha(param)+'/'+sha(WATERMARK)+'/'+sha(SRC/'watermarklab/benchmark.py')+'/'+name+'/'
                       + ('extended' if args.extended_sensitivity else 'standard'))
            def valid():
                try:
                    return (marker.is_file() and json.loads(marker.read_text())['signature']==signature
                            and extras_output_ok(name,target))
                except (OSError,ValueError,KeyError):
                    return False
            if valid():continue
            ok=record(f'extras/{name}',cmd,validate=lambda n=name,t=target: extras_output_ok(n,t))
            if ok:jwrite(marker,{'signature':signature})
            elif stop:return False
        # Fixed test host and attack list specified BEFORE results are observed.
        gallery_host=sorted(test.glob('*.bmp'))[0]
        gallery=out/'reviewer_visual_gallery'
        signature={'host':sha(gallery_host),'watermark':sha(WATERMARK),
                   'proposal':sha(param),'guo':sha(guo),
                   'code':sha(ROOT/'src/watermarklab/experiments/extracted_gallery.py')}
        digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        guard=gallery/'run_complete.json'
        def gallery_ok():
            try:
                return (validate_gallery(gallery,gallery_host.name,WATERMARK.name)
                        and json.loads(guard.read_text()).get('digest')==digest)
            except (OSError, ValueError):return False
        if not gallery_ok():
            cmd=[sys.executable,'-m','watermarklab.experiments.extracted_gallery',
                 '--host',gallery_host,'--watermark',WATERMARK,'--output',gallery,
                 '--proposal-params',param,'--guo-params',guo]
            ok=record('extras/extracted_gallery',cmd,
                validate=lambda:validate_gallery(gallery,gallery_host.name,WATERMARK.name))
            if ok:jwrite(guard,{'digest':digest,'signature':signature})
            else:return False
        return gallery_ok()

    methods={'optimize':optimize,'heldout':heldout,'reviewer':reviewer,'reports':reports,'neural4096':neural4096,'extras':extras}
    # Classical reviewer experiments are the default. Neural comparisons only
    # execute with an explicit neural4096 step or --include-neural4096.
    steps=(['optimize','heldout','reviewer','reports','extras'] +
           (['neural4096'] if args.include_neural4096 else [])) if args.step=='all' else (
        [] if args.step in ('secondary','verify') else [args.step])
    execution_ok=True
    secondary_path=out/'watermark_2_Cc'
    secondary_archive=out/'watermark_2_Cc.zip'
    # A fresh `verify` session must inspect the saved secondary ZIP, rather
    # than incorrectly interpreting an unexpanded independent cohort as absent.
    if args.step=='verify' and secondary_archive.is_file() and not (secondary_path/'completeness.json').is_file():
        restore(secondary_archive,secondary_path)
    def secondary_ok():
        try:
            state=json.loads((secondary_path/'completeness.json').read_text())
            prep=json.loads((out/'inputs/secondary_watermark_preprocessing.json').read_text())
            original=ROOT/'data/watermark/Cc.logo.circle.png'
            converted=out/'inputs/Cc.logo.circle_alpha_binary.png'
            return (state.get('all_core_sections_complete') is True and
                    prep.get('original_sha256')==sha(original) and
                    prep.get('preprocessed_sha256')==sha(converted))
        except (OSError,ValueError,KeyError):return False
    try:
        for step in steps:
            ok=methods[step]()
            if stop or not ok:
                execution_ok=False
                break
        if (args.step in ('all','secondary') and not args.skip_secondary
                and execution_ok and not stop):
            from watermarklab.experiments.second_watermark import prepare
            second=prepare(ROOT/'data/watermark/Cc.logo.circle.png',
                           out/'inputs/Cc.logo.circle_alpha_binary.png')
            # The separate cohort has its own train optimization, 13-method
            # held-out grid, reviewer experiments and gallery; neural 4096 is opt-in.
            # It never reuses wm.png's optimized parameters or mixes CSV rows.
            if secondary_archive.is_file() and not (secondary_path/'recovery_status.json').is_file():
                restore(secondary_archive,secondary_path)
            if not secondary_ok():
                remaining=0 if not args.max_jobs else args.max_jobs-new_jobs
                if args.max_jobs and remaining<=0:
                    stop=True
                    execution_ok=False
                    blockers.append('Second watermark awaiting next session: job budget exhausted')
                else:
                    remaining_seconds=max(0,int(args.max_seconds-(time.monotonic()-begin)))
                    command=[sys.executable,__file__,'--step','all','--skip-secondary',
                        '--proposal-fireflies',str(args.proposal_fireflies),
                        '--proposal-generations',str(args.proposal_generations),
                        '--watermark',second,'--out',secondary_path,
                        '--max-jobs',str(remaining),'--max-seconds',str(remaining_seconds),
                        '--task-timeout',str(args.task_timeout),
                        '--checkpoint-seconds',str(args.checkpoint_seconds)]
                    if args.include_neural4096:command.append('--include-neural4096')
                    if args.extended_sensitivity:command.append('--extended-sensitivity')
                    if not record('secondary/full_independent_evaluation',command,
                                  validate=secondary_ok,timeout=max(60,remaining_seconds)):
                        execution_ok=False
                        blockers.append('Second watermark independent cohort not complete')
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
        # Aggregate row counts alone are not proof of original host/params.
        if heldout_ok:
            for method in DEFAULT_METHOD_IDS:
                for host in sorted(test.glob('*.bmp')):
                    folder=classical/'heldout_batches'/method/host.stem
                    try:
                        guard=json.loads((folder/'run_contract.json').read_text())
                        contract=guard['contract']
                        digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
                        heldout_ok &= (guard['digest']==digest and
                          contract.get('host')==sha(host) and
                          contract.get('watermark')==sha(WATERMARK) and
                          contract.get('proposal')==sha(param) and
                          contract.get('guo')==sha(guo) and
                          contract.get('gaata')==sha(gaata) and
                          contract.get('code')==sha(SRC/'watermarklab/benchmark.py') and
                          contract.get('method')==method and contract.get('attacks')==attacks and
                          phase_ok(folder/'per_image_attack_results.csv',method,host.stem,attacks))
                    except (OSError,ValueError,KeyError):heldout_ok=False
                    if not heldout_ok:break
                if not heldout_ok:break
        reviewer_rel=('review_audit/per_case_results.csv', 'bss_validation/bss_per_host_attack.csv',
                      'ablation/full/per_image_attack_results.csv', 'block_size_comparison/per_case.csv',
                      'channel_ablation/channel_ablation.csv')
        reviewer_ok=all((classical/name).is_file() for name in reviewer_rel)
        # Every ablation (not just `full`) must be aggregated and complete.
        reviewer_ok &= all((classical/'ablation'/variant/'per_image_attack_results.csv').is_file()
                           for variant in PROPOSAL_ABLATION_VARIANTS)
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
                        chunk=classical/'reviewer_chunks'/kind/test_host.stem
                        reviewer_ok &= (signature.get('kind')==kind and
                            signature.get('code')==sha(ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py') and
                            signature.get('reviewer_checks')==sha(ROOT/'src/watermarklab/experiments/reviewer_checks.py'))
                        for specialized,location in (('audit','review_audit'),('bss','bss_validation'),('channel','channel_ablation')):
                            if kind==specialized:
                                reviewer_ok &= signature.get(specialized+'_code')==sha(ROOT/f'src/watermarklab/experiments/{location}.py')
                        if kind=='channel':reviewer_ok &= validate_channel(chunk,test_host.name)
                        if kind=='audit':reviewer_ok &= validate_security(chunk,test_host.name)
                        if kind=='bss':reviewer_ok &= validate_bss(chunk,test_host.name)
                    except (OSError, ValueError, KeyError):
                        reviewer_ok=False
                    if not reviewer_ok:break
                if not reviewer_ok:break
        if reviewer_ok:
            # Per-host ablation grids must match the exact reviewer attack suite.
            expected_reviewer_attacks={a.name for a in default_attack_suite(False,'reviewer_stress')}
            for variant in PROPOSAL_ABLATION_VARIANTS:
                ablation_path=classical/'ablation'/variant/'per_image_attack_results.csv'
                variant_rows=csvrows(ablation_path)
                exp={(host.stem,'before_attack','no_attack') for host in test.glob('*.bmp')}
                exp|={(host.stem,'after_attack',name) for host in test.glob('*.bmp')
                      for name in expected_reviewer_attacks}
                observed=[(r.get('image'),r.get('phase'),r.get('attack')) for r in variant_rows]
                reviewer_ok &= len(observed)==len(exp) and set(observed)==exp
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
                expected_sig=(sha(param)+'/'+sha(WATERMARK)+'/'+sha(SRC/'watermarklab/benchmark.py')+'/'+name+'/'
                              + ('extended' if args.extended_sensitivity else 'standard'))
                extras_ok &= (sig==expected_sig and extras_output_ok(name,target))
            except (OSError,ValueError,KeyError):
                extras_ok=False
        gallery_host=sorted(test.glob('*.bmp'))[0]
        gallery_dir=out/'reviewer_visual_gallery'
        gallery_ok_final=False
        try:
            g=json.loads((gallery_dir/'run_complete.json').read_text())
            gallery_ok_final=(validate_gallery(gallery_dir,gallery_host.name,WATERMARK.name)
                and g['signature']['host']==sha(gallery_host)
                and g['signature']['watermark']==sha(WATERMARK)
                and g['signature']['proposal']==sha(param)
                and g['signature']['guo']==sha(guo)
                and g['signature']['code']==sha(ROOT/'src/watermarklab/experiments/extracted_gallery.py'))
        except (OSError,ValueError,KeyError):pass
        neural_manifest=out/'neural_4096_forced/aggregate_manifest.json'
        neural_ok=False
        try:
            state=json.loads(neural_manifest.read_text())
            neural_ok=(state.get('actual_rows') == 2*len(info['test_names'])*(2+len(attacks))
                       and state.get('attack_cases_per_host')==len(attacks))
        except (OSError, ValueError, KeyError):
            pass
        if neural_ok:
            for method in ('trustmark','wam'):
                for host in sorted(test.glob('*.bmp')):
                    folder=out/'neural_4096_forced'/method/host.stem
                    try:
                        manifest=json.loads((folder/'manifest.json').read_text())
                        rows=csvrows(folder/'per_image_attack_results.csv')
                        expected={('before_attack','no_attack'),('control','original_unmarked')}
                        expected.update(('after_attack',a) for a in attacks)
                        neural_ok &= (manifest.get('complete_grid') is True
                            and manifest.get('watermark_sha256')==sha(WATERMARK)
                            and len(rows)==len(expected)
                            and {(r.get('phase'),r.get('attack')) for r in rows}==expected
                            and all(r.get('image')==host.stem and r.get('watermark')==WATERMARK.name
                                 and r.get('source_payload_bits')=='4096' and
                                 all(math.isfinite(float(r.get(field,''))) for field in
                                     ('psnr','ssim','bit_coverage','ber_erasure_as_error'))
                                 for r in rows))
                    except (OSError,ValueError,KeyError,TypeError):neural_ok=False
                    if not neural_ok:break
                if not neural_ok:break
        # Neural evidence is optional for this classical-only review protocol.
        # A second watermark is an independent cohort, not rows mixed into wm.png.
        second_requested=args.step in ('all','secondary','verify') and not args.skip_secondary
        neural_requested=args.include_neural4096 or args.step=='neural4096'
        report_ok=False
        try:
            report_path=classical/'reviewer_analysis/review_coverage.json'
            report_ok=(json.loads(report_path.read_text()).get('complete') is True
                       and (classical/'reviewer_analysis/run_contract.json').is_file())
        except (OSError,ValueError,TypeError):pass
        classical_core_complete=all((heldout_ok,reviewer_ok,report_ok,extras_ok,gallery_ok_final))
        status['strict_completeness']={
            'classical_13_methods_12_hosts_109_attacks':heldout_ok,
            'reviewer_all_hosts_validated':reviewer_ok,
            'reviewer_analysis_tables_complete':report_ok,
            'extras_sensitivity_and_figures':extras_ok,
            'neural_4096_both_models_full_grid':neural_ok,
            'neural_4096_required':neural_requested,
            'native_neural_61_32bit_executed':False,
            'visual_extracted_gallery_valid':gallery_ok_final,
            'security_and_bss_strictly_validated':reviewer_ok,
            'secondary_watermark_separate_full_grid':secondary_ok() if second_requested else None,
            'secondary_watermark_required':second_requested,
            'expected_heldout_rows':expected_heldout,
            'expected_neural_rows':2*len(info['test_names'])*(2+len(attacks)),
            'primary_classical_complete':classical_core_complete,
            'all_core_sections_complete':classical_core_complete
               and (neural_ok if neural_requested else True)
               and (secondary_ok() if second_requested else True)}
        jwrite(out/'completeness.json', status['strict_completeness'])
        jwrite(status_file,status)
        snapshot(out)
    print('Done. Read',status_file,'before claiming full completeness.')
    if args.step=='verify' and not status['strict_completeness']['all_core_sections_complete']:
        sys.exit(2)
    if blockers and not stop:
        sys.exit(1)

if __name__=='__main__':
    main()
