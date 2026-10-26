"""Standalone, NO-PRIOR-ARCHIVE classical follow-up for ONE watermark.

Runs independently on Kaggle after a fresh GitHub clone. It does NOT claim to
reuse earlier successful jobs. The scope is intentionally ONLY previously
missing reviewer studies and failed baseline subsets; a full 13-method aggregate
must be made later from matching-provenance raw rows, not fabricated here.

Never invokes TrustMark/WAM. Never changes proposed-method fitness/equations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from watermarklab.experiments.splits import build_split
from watermarklab.common.io_utils import load_watermark_binary
from watermarklab.common.attack import reviewer45_attack_suite


def atomic_json(path: Path, object_):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(object_, indent=2, allow_nan=False, default=str), encoding='utf8')
    tmp.replace(path)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(out: Path):
    dst = out.parent / (out.name + '.zip')
    tmp = dst.with_suffix('.zip.partial')
    excludes = {'splits', 'single_host', 'marked_images', 'native_neural', 'neural_4096', 'neural_4096_forced'}
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for file in sorted(out.rglob('*')):
            if (file.is_file() and not (set(file.relative_to(out).parts) & excludes)
                    and not file.name.endswith(('.partial', '.tmp'))):
                z.write(file, arcname=str(Path(out.name) / file.relative_to(out)))
    tmp.replace(dst)
    print('CHECKPOINT OUTPUT', dst, 'bytes', dst.stat().st_size, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', choices=('W1', 'W2'), required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-seconds', type=int, default=30300)
    p.add_argument('--fireflies', type=int, default=2)
    p.add_argument('--generations', type=int, default=2)
    p.add_argument('--task-timeout', type=int, default=10800)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Leave sufficient time for a final Kaggle output ZIP.
    deadline = started + args.max_seconds
    from watermarklab.experiments.second_watermark import prepare
    source = ROOT / 'data' / 'watermark' / ('wm.png' if args.label == 'W1' else 'Cc.logo.circle.png')
    wm = source if args.label == 'W1' else prepare(source, out/'inputs/Cc.logo.circle_alpha_binary.png')
    wm_array = load_watermark_binary(wm)
    if wm_array.shape != (64, 64) or set(__import__('numpy').unique(wm_array)) != {0, 255}:
        raise RuntimeError(f'Expected nonconstant 64x64 binary watermark for {args.label}')

    split = build_split(ROOT/'data/host', out/'frozen_hosts.json',
                        train_dir=ROOT/'data/host_train', create_dirs=True)
    assert len(split['train_names']) == 9 and len(split['test_names']) == 12
    assert not any('lenna' in name.lower() or 'lena.' in name.lower()
                   for name in split['train_names'] + split['test_names'])
    assert len(reviewer45_attack_suite(False)) == 45
    testdir = out/'splits/test'
    assert len(list(testdir.glob('*.bmp'))) == 12

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    identity_file = out/'run_identity.json'
    identity = {'git_commit': commit, 'watermark_label': args.label,
                'watermark_source_sha256': sha256(source), 'watermark_binary_sha256': sha256(wm),
                'test_hosts': split['test_names'], 'train_hosts': split['train_names'],
                'source_payload_pixels': 4096, 'host_count': 12,
                'restored_previous_results': False,
                'explicitly_not_a_full_13_method_grid': True,
                'neural_methods_enabled': False,
                'fitness_objective': 'unchanged from GitHub checkout',
                'optimizer_fireflies': args.fireflies, 'optimizer_generations': args.generations}
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise RuntimeError('Run identity changed; refuse to mix code/watermark results')
    atomic_json(identity_file, identity)
    state_path=out/'fresh_remaining_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}, 'errors':[]}
    state.setdefault('jobs', {}); state.setdefault('errors', [])

    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    recovery = ROOT/'scripts/recover_missing_results.py'
    common = ['--out', str(out), '--watermark', str(wm), '--skip-secondary',
              '--proposal-fireflies',str(args.fireflies),
              '--proposal-generations',str(args.generations),
              '--max-jobs','0', '--task-timeout',str(args.task_timeout),
              '--checkpoint-seconds','600']

    def left(): return max(0, int(deadline - time.monotonic()))

    def execute(tag, command, *, minimum=240):
        if left()<minimum:
            state['jobs'][tag]='PENDING: time budget'; atomic_json(state_path,state)
            return False
        logfile=out/'fresh_logs'/(tag.replace('/','__')+'.log')
        logfile.parent.mkdir(parents=True,exist_ok=True)
        print('\nRUN',tag,'time remaining',left(),'sec',flush=True)
        # Keep process output on disk and include it in downloadable ZIP.
        with logfile.open('w',encoding='utf8') as log:
            try:
                p=subprocess.Popen([str(c) for c in command],cwd=ROOT,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=max(90,left()-90))
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        p.wait()
                    code=124
            except Exception as error:
                state['errors'].append({'step':tag,'exception':repr(error)})
                code=125
        output=logfile.read_text(encoding='utf8',errors='replace').splitlines()
        print('\n'.join(output[-8:]),flush=True)
        state['jobs'][tag]='PASS' if code==0 else f'INCOMPLETE: exit={code}'
        state['last_git_commit']=commit
        atomic_json(state_path,state)
        pack(out)
        return code==0

    def recovery_step(name, more=None, minimum=240):
        return execute('recovery/'+name+('/'+str(more[1]) if more else ''),
          [sys.executable,str(recovery),'--step',name,*common,
           '--max-seconds',str(max(150,left()-120)),*(more or [])],minimum=minimum)

    try:
        if not recovery_step('optimize'):
            print('Optimization did not complete; no publication results are claimed.',flush=True)
            return 2
        par = out/'classical'
        params = [par/'proposal_optimized_params.json',par/'guo_optimized_params.json',
                  par/'gaata_optimized_params.json']
        if not all(p.exists() for p in params):
            raise RuntimeError('Missing frozen parameters after optimizer; stop instead of using defaults')

        # Independent reviewer studies: one failure must not suppress other groups.
        # Priority: the manuscript's suspicious W2 block-size, then channels/BSS/security/ablation.
        for kind in ('block','channel','bss','audit','ablation'):
            if left()<300:break
            recovery_step('reviewer',['--reviewer-kinds',kind,
                      '--reviewer-attack-preset','reviewer45',
                      '--reviewer-max-attacks','45'])

        # Produce reviewer figures immediately, even if later baseline runs exhaust CPU.
        execute('figures/reviewer_measured',
                [sys.executable,'scripts/plot_remaining_fresh.py','--out',out],minimum=45)

        # Previously failed baseline subgroups, NOT the completed Guo / Proposed grids.
        # Without old result archives this is a NEW subset, not a resumed 13-method grid.
        failed45 = ('gaata2022_dwt_hess_fwa','dwt_wht_svd_2024')
        for method in failed45:
            if left()<300:break
            dest = out/'failed45_from_scratch'/method
            ok=execute('failed45/'+method,
                    [sys.executable,'main.py','--phase','normal','--methods',method,
                     '--baseline-mode','adapt','--host-dir',testdir,'--watermark',wm,
                     '--attack-preset','reviewer45','--proposal-repeat','full',
                     '--proposal-param-file',params[0],'--proposal-param-mode','require',
                     '--guo-param-file',params[1],'--guo-param-mode','require',
                     '--gaata-param-file',params[2],'--gaata-param-mode','require',
                     '--resume-attack-journal','--no-save-images','--output',dest])
            if ok:
                # CLI process exit alone is NOT scientific completeness.
                import csv
                csv_path=dest/'per_image_attack_results.csv'
                failures=dest/'failures.json'
                try:
                    with csv_path.open(newline='') as f:
                        rows=list(csv.DictReader(f))
                    attacks={a.name for a in reviewer45_attack_suite(False)}
                    expected={(h.stem,phase,attack) for h in sorted(testdir.glob('*.bmp'))
                             for phase,attack in ([('before_attack','no_attack')] +
                               [('after_attack', a) for a in attacks])}
                    got={(r['image'],r['phase'],r['attack']) for r in rows}
                    measured=(len(rows)==len(expected) and got==expected and
                              all(r['method_id']==method for r in rows) and
                              all(__import__('math').isfinite(float(r['ber'])) for r in rows))
                    no_failures=(not failures.exists() or not json.loads(failures.read_text()))
                except (FileNotFoundError,ValueError,KeyError,OSError):
                    measured=no_failures=False
                if not (measured and no_failures):
                    state['jobs']['failed45/'+method]='INCOMPLETE: missing/failed attack rows'
                    atomic_json(state_path,state)
                    print('WARNING: incomplete 45-attack baseline',method,flush=True)

        # Previously rejected 109-attack held-out method/host combinations only.
        if left()>=300:
            recovery_step('heldout',['--heldout-methods',','.join((
                'abodena2022_lwt_dct_hess','su2016_contourlet_hess',
                'gaata2022_dwt_hess_fwa','hess_nha2023'))])
        if left()>=300:
            recovery_step('heldout',['--heldout-methods','dwt_wht_svd_2024',
                                    '--heldout-hosts','Girl,milkdrop'])

        # Full-size/sensitivity/figures only if CPU time remains. This does not
        # require a completed baseline grid, and does not invoke neural methods.
        if left()>=300:
            recovery_step('extras',['--extended-sensitivity'])
        if left()>=300:
            recovery_step('reports')

        # Real plots only: derive from completed measured CSVs and record missing ones.
        execute('figures/from_measured_results',
                [sys.executable,'scripts/plot_remaining_fresh.py','--out',out], minimum=45)

        # Auditable progress: incomplete jobs stay incomplete, never fill with 0.
        return 0
    finally:
        state['finished_at_monotonic_elapsed_seconds']=round(time.monotonic()-started,2)
        state['all_scheduled_steps_passed']=bool(state.get('jobs')) and all(
            status == 'PASS' for status in state['jobs'].values())
        atomic_json(state_path,state)
        pack(out)


if __name__ == '__main__':
    raise SystemExit(main())
