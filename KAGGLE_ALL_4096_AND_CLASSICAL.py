# ONE KAGGLE CELL -- complete classical + reviewer + 4,096-bit neural (NO native neural payload runs).
# Upload these files as Kaggle datasets (Notebook: Internet ON, GPU ON):
#   1) pham_thi_yen-main(20261015-090635).zip (the exact project base)
#   2) WatermarkLab_ALL_EXPERIMENTS_FIXED_PATCH.zip (the patch from ChatGPT)
#   3) watermark_all_results_9train_12test.zip (previous results or latest checkpoint)
# Repeat this cell across sessions when required. Upload each latest results checkpoint to the NEXT
# Kaggle session, so previous attacks are NOT lost. This is CPU/GPU-intensive.
# The real 4096-bit bitmap is split into native network-sized chunks spatially;
# this is an experimental adaptation, not 4096-bit native model support.
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, urllib.request, zipfile

STEP = 'all'         # neural4096 | optimize | heldout | reviewer | extras | all
MAX_JOBS_PER_RUN = 0        # jobs until session budget; 0 = unlimited
MAX_SECONDS_PER_RUN = 32000
TASK_TIMEOUT_SECONDS = 30000
CHECKPOINT_SECONDS = 600    # automatic ZIP snapshots during long jobs
RESUME_ZIP = None           # path to manually select a results checkpoint, or None

WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
INPUT = Path('/kaggle/input')
REPO = WORK / 'pham_thi_yen'
OUTPUT = WORK / 'watermark_all_results_9train_12test'
SOURCE_NAME = 'pham_thi_yen-main(20261015-090635).zip'
PATCH_NAME = 'WatermarkLab_ALL_EXPERIMENTS_FIXED_PATCH.zip'
RESULT_NAME = 'watermark_all_results_9train_12test.zip'
# SHA-256 of the proposal method in this user's supplied base ZIP.
EXPECTED_PROPOSAL_SHA256 = '10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'


def run(cmd, *, cwd=None, required=True):
    print('>>>', ' '.join(map(str, cmd)), flush=True)
    result = subprocess.run(list(map(str, cmd)), cwd=cwd or WORK, check=False)
    if required and result.returncode != 0:
        raise RuntimeError(f'Command failed with {result.returncode}: {cmd}')
    return result.returncode


def overlay(archive: Path):
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError(f'Corrupted input ZIP: {archive}')
        n = 0
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if (len(parts) < 2 or parts[0] != 'pham_thi_yen-main'
                    or '..' in parts or member.filename.startswith('/')):
                raise ValueError(f'Unsafe/unexpected project archive member: {member.filename}')
            dest = REPO.joinpath(*parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            n += 1
    print(f'Overlay installed: {archive.name} ({n} files)', flush=True)


def locate(name):
    hits = list(INPUT.rglob(name)) if INPUT.is_dir() else []
    return sorted(hits, key=lambda x: x.stat().st_mtime)[-1] if hits else None


base = locate(SOURCE_NAME)
patch = locate(PATCH_NAME)
if base:
    # Always start from the exact uploaded base. An old checkout can contain
    # scripts that the new ZIP does not, otherwise contaminating the run.
    if REPO.exists():
        shutil.rmtree(REPO)
    overlay(base)
else:
    if not REPO.is_dir():
        run(['git','clone','--depth','1','https://github.com/tydeptrai21042004/pham_thi_yen.git', REPO])
    print('Using GitHub base. Exact project compatibility will be checked.',flush=True)
method = REPO/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
if not method.is_file() or hashlib.sha256(method.read_bytes()).hexdigest() != EXPECTED_PROPOSAL_SHA256:
    raise RuntimeError('Base proposal differs from the validated user source. Upload the provided SOURCE ZIP instead.')
if patch:
    overlay(patch)
else:
    raise FileNotFoundError(f'Upload {PATCH_NAME} as Kaggle Input. The exact patch is required.')
if not (REPO/'src/watermarklab/experiments/neural_4096.py').is_file():
    raise FileNotFoundError('Missing experimental 4,096-bit comparison source code.')

# Fresh Kaggle sessions need Python deps. No inference via `main.py --phase neural`.
run([sys.executable,'-m','pip','install','-q','-r',REPO/'requirements.txt'])
run([sys.executable,'-c','import pywt,cv2,scipy,skimage;print("PyWavelets",pywt.__version__)'])

if STEP in ('neural4096','all'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for practical WAM 4096-bit stress testing; enable GPU in Notebook Settings.')
    run([sys.executable,'-m','pip','install','-q','trustmark'])
    wam = WORK / 'watermark-anything'
    if not wam.is_dir():
        run(['git','clone','--depth','1','https://github.com/facebookresearch/watermark-anything.git',wam])
    run([sys.executable,'-m','pip','install','-q','--prefer-binary',
        'omegaconf==2.3.0','einops==0.8.0','timm==1.0.11',
        'lpips==0.1.4','pycocotools>=2.0.8'])
    checkpoint = wam/'checkpoints'/'wam_mit.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temp=checkpoint.with_suffix('.pth.partial')
        if temp.exists(): temp.unlink()
        try:
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',temp)
            if temp.stat().st_size < 1_000_000:
                raise RuntimeError('Downloaded WAM checkpoint is suspiciously small')
            temp.replace(checkpoint)
        finally:
            if temp.exists(): temp.unlink()
    if not (wam/'checkpoints'/'params.json').is_file() or checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError('Official WAM config/checkpoint missing or empty.')
    os.environ['WAM_REPO_DIR'] = str(wam)
    os.environ['WAM_CHECKPOINT'] = str(checkpoint)

sources = list(INPUT.rglob(RESULT_NAME)) if INPUT.is_dir() else []
source = Path(RESUME_ZIP) if RESUME_ZIP else (
    max(sources, key=lambda x: x.stat().st_mtime) if sources else None)
if source:
    print('Restoring previous results and neural checkpoints:',source,flush=True)
else:
    print('No existing results ZIP found: starting from scratch.',flush=True)
cmd = [sys.executable,'scripts/recover_missing_results.py',
       '--step',STEP,'--out',OUTPUT,'--max-jobs',str(MAX_JOBS_PER_RUN),
       '--max-seconds',str(MAX_SECONDS_PER_RUN),
       '--task-timeout',str(TASK_TIMEOUT_SECONDS),
       '--checkpoint-seconds',str(CHECKPOINT_SECONDS)]
if source:
    cmd += ['--restore-zip',source]
code=run(cmd,cwd=REPO,required=False)
status=OUTPUT/'recovery_status.json'
if status.is_file():
    obj=json.loads(status.read_text())
    print('STATUS:', json.dumps({k:obj.get(k) for k in
       ('last_requested_steps','last_finished_jobs','blockers','paused_by_budget','classical_final_exists')},indent=2))
print('Runner exit code:',code)
print('RESULTS ZIP:',WORK/RESULT_NAME)
if code:
    print('Some jobs remain incomplete or failed. See recovery_status.json and per-host logs.',flush=True)
print('Check completeness.json: complete experiments require every stage, not only an exit code.')
