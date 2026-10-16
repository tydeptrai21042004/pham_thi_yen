# Paste this complete file into ONE Kaggle Python notebook cell.
# Settings: Internet ON. CPU is sufficient for classical; optional neural 4096 prefers GPU.
# IMPORTANT: push the companion patch to the GitHub repository BEFORE running.
# Does not run native TrustMark 61-bit or native WAM 32-bit under any setting.
import os, sys, json, hashlib, shutil, subprocess, zipfile, urllib.request
from pathlib import Path

GITHUB = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'
BRANCH = 'main'
RUN_4096_EXTENSIONS = False  # optional; True runs BOTH forced-grid 4096-bit neural adaptations
RUN_ALL_SENSITIVITY_HOSTS = True  # run full 12-test and 9-train sweep coverage
FIREFLIES = 4
GENERATIONS = 2
SESSION_SECONDS = 31500  # 8h 45m per session; checkpoints are used for long multi-session runs
JOB_TIMEOUT_SECONDS = 30800
CHECKPOINT_SECONDS = 600

WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
INPUT = Path('/kaggle/input')
REPO = WORK / 'pham_thi_yen'
OUT = WORK / 'watermark_all_results_9train_12test'
EXPECTED_PROPOSAL_SHA256 = '10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'
ZIPNAME = OUT.name + '.zip'


def run(argv, *, cwd=None, required=True):
    print('>>>', ' '.join(map(str, argv)), flush=True)
    result = subprocess.run([str(a) for a in argv], cwd=cwd or WORK)
    if required and result.returncode:
        raise RuntimeError(f'Command exit={result.returncode}: {argv}')
    return result.returncode


# Always fetch the CURRENT GitHub main, never overlay an obsolete project ZIP.
if REPO.exists():
    shutil.rmtree(REPO)
run(['git', 'clone', '--depth', '1', '--branch', BRANCH, GITHUB, REPO])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
print('GitHub commit:', commit, flush=True)

proposal = REPO/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
assert proposal.exists(), 'Missing proposed method'
assert hashlib.sha256(proposal.read_bytes()).hexdigest() == EXPECTED_PROPOSAL_SHA256, (
    'Proposal source changed relative to the approved original. Do not run an incompatible revision.')
runner = REPO/'scripts/recover_missing_results.py'
assert runner.is_file() and (REPO/'scripts/build_review_tables.py').is_file(), (
    'Updated patch is not committed to GitHub. Upload and commit the patch first.')
runner_text = runner.read_text()
assert "--proposal-fireflies" in runner_text and "--proposal-generations" in runner_text, (
    'Old recovery runner detected: push the NEW corrected patch before running.')
assert (REPO/'tests/test_full_reviewer_kaggle_contract_2026.py').is_file(), 'New regression tests not committed'
assert FIREFLIES == 4 and GENERATIONS == 2

# Dependency check and preflight regression tests BEFORE time-consuming optimization.
os.environ.update(PYTHONUNBUFFERED='1', WATERMARKLAB_STRICT_PYWT='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONPATH=str(REPO/'src'))
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', REPO/'requirements.txt'])
run([sys.executable, '-c', 'import pywt, cv2, scipy, skimage; print("PyWavelets:",pywt.__version__)'])
run([sys.executable, '-m', 'pytest', '-q',
     'tests/test_full_reviewer_kaggle_contract_2026.py',
     'tests/test_reviewer_experiment_orchestration_2026.py',
     'tests/test_operational_smoke_2026.py',
     'tests/test_dataset_9_train_12_test.py',
     'tests/test_reviewer_neural_4096.py'], cwd=REPO)

# Optional 4096-bit neural extensions. NEVER call main.py --phase neural.
# FULL attack grids are forced for both models, keeping genuine decoding failures.
if RUN_4096_EXTENSIONS:
    run([sys.executable, '-m', 'pip', 'install', '-q', 'trustmark',
         'omegaconf==2.3.0', 'einops==0.8.0', 'timm==1.0.11',
         'lpips==0.1.4', 'pycocotools>=2.0.8'])
    wam = WORK/'watermark-anything'
    if not (wam/'.git').is_dir():
        if wam.exists(): shutil.rmtree(wam)
        run(['git', 'clone', '--depth', '1',
             'https://github.com/facebookresearch/watermark-anything.git', wam])
    weights = wam/'checkpoints/wam_mit.pth'
    if not weights.is_file() or weights.stat().st_size < 1000000:
        weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights.with_suffix('.pth.partial')
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth', str(tmp))
        if tmp.stat().st_size < 1000000: raise RuntimeError('Incomplete WAM checkpoint')
        tmp.replace(weights)
    assert (wam/'checkpoints/params.json').is_file(), 'WAM official params.json missing'
    os.environ['WAM_REPO_DIR'] = str(wam)
    os.environ['WAM_CHECKPOINT'] = str(weights)

OUT.mkdir(parents=True, exist_ok=True)
# On a new Kaggle session add the previous checkpoint ZIP as an Input dataset.
# No old checkpoint -> all work starts clean, including all 11 experiment groups.
# Current notebook's OUT is retained, so rerunning within the SAME session also resumes.
candidates = list(INPUT.rglob(ZIPNAME)) if INPUT.exists() else []
resume = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
args = [sys.executable, 'scripts/recover_missing_results.py',
        '--step', 'all', '--out', OUT,
        '--proposal-fireflies', str(FIREFLIES),
        '--proposal-generations', str(GENERATIONS),
        '--max-jobs', '0', '--max-seconds', str(SESSION_SECONDS),
        '--task-timeout', str(JOB_TIMEOUT_SECONDS),
        '--checkpoint-seconds', str(CHECKPOINT_SECONDS)]
if RUN_ALL_SENSITIVITY_HOSTS: args.append('--extended-sensitivity')
if RUN_4096_EXTENSIONS: args.append('--include-neural4096')
if resume:
    print('RESUMING from previous checkpoint:', resume, flush=True)
    args += ['--restore-zip', resume]
else:
    print('FRESH evaluation: no historical result archive will be reused.', flush=True)

(OUT/'kaggle_git_revision.json').write_text(json.dumps({
    'commit': commit, 'source': GITHUB, 'fireflies': FIREFLIES,
    'generations': GENERATIONS, 'native_neural_enabled': False,
    'extended_4096_enabled': RUN_4096_EXTENSIONS,
    'all_sensitivity_hosts': RUN_ALL_SENSITIVITY_HOSTS}, indent=2))
code = run(args, cwd=REPO, required=False)

for name in ('recovery_status.json','completeness.json'):
    file = OUT/name
    if file.exists():
        state=json.loads(file.read_text())
        print('\n'+name+':', json.dumps(state if name=='completeness.json' else {
            'paused_by_budget':state.get('paused_by_budget'),
            'blockers':state.get('blockers'),
            'finished_jobs':state.get('last_finished_jobs'),
            'failed_jobs':{k:v for k,v in state.get('jobs',{}).items() if v!='PASS'}
        }, indent=2),flush=True)
print('Runner exit code:',code,flush=True)
print('RESULT ZIP:',WORK/ZIPNAME,flush=True)
print('SECONDARY ZIP (inside the primary archive):',OUT/'watermark_2_Cc.zip',flush=True)
print('When incomplete, attach the RESULT ZIP as a Kaggle Input dataset and run THIS SAME CELL again.',flush=True)
if (OUT/'completeness.json').exists():
    complete=json.loads((OUT/'completeness.json').read_text()).get('all_core_sections_complete')
    print('STRICT FULL COMPLETENESS:',complete,flush=True)
