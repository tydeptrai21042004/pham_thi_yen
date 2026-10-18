"""Resume ONLY unfinished experiments for BOTH independently checkpointed watermarks.

This is an orchestrator, not a watermarking algorithm change.  Run through
KAGGLE_RESUME_BOTH_ONE_CELL.py.  Both old checkpoint ZIPs must be Kaggle inputs;
missing checkpoints are fatal, never silently replaced by fresh 20,000-row runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
INPUT = Path('/kaggle/input')
DATASETS = {
    'primary': ('watermark_all_results_9train_12test', ROOT/'data/watermark/wm.png'),
    'secondary': ('watermark_2_Cc', WORK/'separate_secondary_inputs/Cc.logo.circle_alpha_binary.png'),
}
# Allow one notebook to make forward progress on both cohorts with checkpoint exports
# before a typical Kaggle session ends. This is a soft budget, not a completion claim.
SESSION_SECONDS = 40_000
RESERVE_SECONDS = 1_200
EXTRAS_SLICE_SECONDS = 11_500
NEURAL_SLICE_SECONDS = 6_900
CHECKPOINT_SECONDS = 600
# Must match the two checkpoint archives; changing them invalidates optimizers.
FIREFLIES = 1
GENERATIONS = 2
EXTENDED_SENSITIVITY = True


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def run(*args, check=True, cwd=ROOT):
    command = list(map(str,args))
    print('>>>', ' '.join(command), flush=True)
    code = subprocess.run(command, cwd=cwd, check=False).returncode
    if check and code:
        raise RuntimeError(f'Exit {code}: {command}')
    return code


def checkpoint_for(folder: str) -> Path:
    if not INPUT.is_dir():
        raise RuntimeError('Kaggle Input is missing. Attach BOTH original result ZIPs.')
    matches = sorted(INPUT.rglob(folder + '.zip'))
    if len(matches) != 1:
        raise RuntimeError(f'Need EXACTLY ONE {folder}.zip attached as a Kaggle Dataset/Input; found {len(matches)}: {matches}')
    return matches[0]


def check_marker(z: zipfile.ZipFile, folder: str, cohort: str) -> dict:
    marker = folder + '/kaggle_evidence_session.json'
    if marker not in z.namelist():
        raise ValueError(f'Not a complete previous-run checkpoint: missing {marker}')
    meta = json.loads(z.read(marker))
    expected = {'cohort': cohort, 'fireflies': FIREFLIES,
                'generations': GENERATIONS, 'extended_sensitivity': EXTENDED_SENSITIVITY}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'{folder}: checkpoint {key}={meta.get(key)!r}, expected {value!r}')
    source = ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
    if meta.get('proposal_sha256') != sha(source):
        raise ValueError(f'{folder}: proposed algorithm differs from checkpoint; refusing to invalidate results')
    return meta


def validate_old_benchmark_contract(out: Path):
    """No accidental regeneration of all heldout jobs after a benchmark.py edit."""
    contracts = sorted((out/'classical/heldout_batches').glob('*/*/run_contract.json'))
    if len(contracts) != 156:
        raise ValueError(f'{out.name}: expected 156 heldout host contracts; found {len(contracts)}. Wrong/incomplete ZIP?')
    actual = sha(ROOT/'src/watermarklab/benchmark.py')
    for path in contracts:
        contract = json.loads(path.read_text())['contract']
        if contract.get('code') != actual:
            raise ValueError(f'{out.name}: benchmark.py differs from old result contract. STOP, not rerunning 156 jobs: {path}')


def status(out: Path) -> dict:
    path = out/'completeness.json'
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def stage_complete(cohort: str, step: str) -> bool:
    out = WORK/DATASETS[cohort][0]
    state = status(out)
    if step == 'extras':
        return (state.get('extras_sensitivity_and_figures') is True
                and state.get('visual_extracted_gallery_valid') is True)
    if step == 'neural4096':
        return state.get('neural_4096_both_models_full_grid') is True
    raise ValueError(step)


def install_dependencies():
    os.environ.update(PYTHONPATH=str(ROOT/'src'), WATERMARKLAB_STRICT_PYWT='1',
                      PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    run(sys.executable,'-m','pip','install','-q','-r',ROOT/'requirements.txt')
    run(sys.executable,'-m','pip','install','-q','trustmark','omegaconf==2.3.0',
        'einops==0.8.0','timm==1.0.11','lpips==0.1.4','pycocotools>=2.0.8')
    run(sys.executable,'-m','pytest','-q',
        'tests/test_finish_both_recovery_2026.py',
        'tests/test_incremental_evidence_2026.py',
        'tests/test_recovery_resume_2026.py', cwd=ROOT)
    wam=WORK/'watermark-anything'
    if not (wam/'checkpoints/params.json').is_file():
        if wam.exists(): shutil.rmtree(wam)
        run('git','clone','--depth','1','https://github.com/facebookresearch/watermark-anything.git',wam)
    weights=wam/'checkpoints/wam_mit.pth'
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        weights.parent.mkdir(parents=True,exist_ok=True)
        temp=weights.with_suffix('.pth.partial')
        with urllib.request.urlopen('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth', timeout=240) as f, temp.open('wb') as target:
            shutil.copyfileobj(f,target)
        if temp.stat().st_size < 1_000_000:
            raise RuntimeError('WAM checkpoint is truncated; do not substitute an alternative model')
        temp.replace(weights)
    if not (wam/'checkpoints/params.json').is_file():
        raise RuntimeError('WAM official params.json missing')
    os.environ.update(WAM_REPO_DIR=str(wam), WAM_CHECKPOINT=str(weights))
    if shutil.which('nvidia-smi'):
        run('nvidia-smi','-L',check=False)
    else:
        print('WARNING: No GPU detected; 4096-bit neural region decoding can take many hours.',flush=True)


def restore_both():
    sys.path.insert(0,str(ROOT/'scripts'))
    from recover_missing_results import restore
    from watermarklab.experiments.second_watermark import prepare
    second_input = DATASETS['secondary'][1]
    prepare(ROOT/'data/watermark/Cc.logo.circle.png',second_input)
    for cohort,(folder,watermark) in DATASETS.items():
        archive = checkpoint_for(folder)
        # Read metadata without extracting gigabytes of evidence unnecessarily.
        with zipfile.ZipFile(archive) as z:
            meta = check_marker(z,folder,cohort)
        out=WORK/folder
        if out.exists():
            # This dedicated cell never silently overlays one cohort's old run
            # on another or continues a dirty notebook working directory.
            raise RuntimeError(f'{out} already exists; use a fresh Kaggle working directory before restoring ZIPs')
        out.mkdir(parents=True)
        print(f'RESTORE {cohort}: {archive} (source {meta["git_commit"]})',flush=True)
        restore(archive,out)
        validate_old_benchmark_contract(out)
        progress = out/'classical/proposal_optimized_params.progress.json'
        try:
            contract = json.loads(progress.read_text())['contract']
            actual_mark = sha(watermark)
            saved_mark = contract['files'][watermark.name]
            if saved_mark != actual_mark:
                raise ValueError(f'{cohort}: watermark checksum changed during preprocessing: '
                                 f'{saved_mark} != {actual_mark}. Do not rerun mismatched evidence.')
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'{cohort}: missing optimizer input-provenance in checkpoint') from exc
        # Checkpoint metadata is from the original run, not falsely re-labeled
        # with the newly cloned patch commit.
        assert (out/'kaggle_evidence_session.json').is_file()
        if not (out/'classical/proposal_optimized_params.json').is_file():
            raise RuntimeError(f'{cohort}: optimizer parameters missing; refuse a fresh retrain')
    print('Both checkpoint archives restored. Existing classical jobs remain untouched.',flush=True)


def run_stage(cohort: str, step: str, *, seconds: int, jobs: int = 0) -> int:
    out=WORK/DATASETS[cohort][0]
    args=[sys.executable,'-u',ROOT/'scripts/recover_missing_results.py',
          '--step',step,'--skip-secondary',
          '--watermark',DATASETS[cohort][1], '--out',out,
          '--proposal-fireflies',str(FIREFLIES),
          '--proposal-generations',str(GENERATIONS),
          '--max-jobs',str(jobs), '--max-seconds',str(seconds),
          '--task-timeout',str(max(900, min(seconds, 14_400))),
          '--checkpoint-seconds',str(CHECKPOINT_SECONDS),
          '--extended-sensitivity','--include-neural4096']
    rc=run(*args,check=False)
    rec=out/'recovery_status.json'
    try:
        state=json.loads(rec.read_text())
    except (OSError,ValueError):
        state={}
    print(f'[{cohort} {step}] rc={rc}; stopped_by_budget={state.get("paused_by_budget")}; '
          f'blockers={state.get("blockers")}; newly_attempted_jobs={state.get("last_finished_jobs")}',flush=True)
    return rc


def main():
    started=time.monotonic()
    # Explicit fail-fast before downloads or any experiments.
    for cohort,(folder,_) in DATASETS.items():
        archive=checkpoint_for(folder)
        with zipfile.ZipFile(archive) as z:
            check_marker(z,folder,cohort)
    install_dependencies()
    restore_both()
    broken=set()

    def remaining():
        return max(0, SESSION_SECONDS - RESERVE_SECONDS - int(time.monotonic()-started))

    # First finish primary's long-running size sweep and remaining extras.
    # Secondary's extras were complete according to the original log.
    for cohort in ('primary','secondary'):
        if stage_complete(cohort,'extras'):
            print(f'SKIP {cohort} extras: checkpoint already complete',flush=True)
            continue
        if remaining()<1800:
            break
        seconds=min(remaining(),EXTRAS_SLICE_SECONDS)
        rc=run_stage(cohort,'extras',seconds=seconds)
        if rc and not status(WORK/DATASETS[cohort][0]).get('extras_sensitivity_and_figures'):
            broken.add((cohort,'extras'))
            print(f'WARNING: {cohort} extras requires investigation; skipping retries within this notebook.',flush=True)

    # Round-robin: BOTH cohorts make progress, with neither running the other's
    # watermark. Each invocation revalidates completed per-host CSVs and SKIPS them.
    while remaining()>=1800:
        pending=[cohort for cohort in ('secondary','primary')
                 if not stage_complete(cohort,'neural4096')
                 and (cohort,'neural4096') not in broken]
        if not pending: break
        for cohort in pending:
            if remaining()<1800: break
            seconds=min(remaining(),NEURAL_SLICE_SECONDS)
            rc=run_stage(cohort,'neural4096',seconds=seconds,jobs=3)
            if rc:
                rec=WORK/DATASETS[cohort][0]/'recovery_status.json'
                try: paused=json.loads(rec.read_text()).get('paused_by_budget')
                except (OSError,ValueError): paused=False
                if not paused:
                    broken.add((cohort,'neural4096'))
                    print(f'ERROR: {cohort} neural stage failed rather than timed out; do not blindly repeat.',flush=True)
            if stage_complete(cohort,'neural4096'):
                print(f'SUCCESS: {cohort} neural 4096-bit grid complete.',flush=True)

    print('\n========== FINAL EVIDENCE STATUS ==========')
    for cohort,(folder,_) in DATASETS.items():
        out=WORK/folder
        # Verify only; this does not execute optimization/classical/neural jobs.
        rc=run_stage(cohort,'verify',seconds=max(120, min(600,remaining()+120)))
        data=status(out)
        print(f'{cohort}: all_core_sections_complete={data.get("all_core_sections_complete")} '
              f'extras={data.get("extras_sensitivity_and_figures")} '
              f'neural={data.get("neural_4096_both_models_full_grid")} verify_exit={rc}',flush=True)
        target=WORK/(folder+'.zip')
        print('  NEXT SESSION CHECKPOINT:',target,'exists:',target.is_file(),flush=True)
    print('If incomplete, save BOTH ZIPs from /kaggle/working, attach them as Input in a NEW notebook run, and run this same cell again.',flush=True)
    # Only report success when both cohorts have independently been certified.
    if all(status(WORK[f]) .get('all_core_sections_complete') is True for f,_ in DATASETS.values()):
        print('SUCCESS: both independent full experimental suites are complete.',flush=True)
    else:
        print('INCOMPLETE: this session saved resumable evidence; no completed stages were intentionally recomputed.',flush=True)


if __name__=='__main__':
    main()
