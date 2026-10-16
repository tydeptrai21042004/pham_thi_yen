# Kaggle single-cell driver for the 11 classical/reviewer experiments ONLY.
# Attach (1) the user's exact pham_thi_yen-main ZIP, (2) the changes-only ZIP
# supplied alongside this file, (3) optional most recent result checkpoint ZIP.
# Internet ON for Python packages. GPU NOT REQUIRED. Native TrustMark/WAM
# and 4,096-bit neural experiments are NEVER invoked by this driver.
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

STEP = 'all'  # all | optimize | heldout | reviewer | reports | extras | secondary | verify
MAX_SECONDS = 28000    # stop and checkpoint below Kaggle session limit
TASK_TIMEOUT = 26000  # one long optimizer may need nearly a full session
MAX_JOBS = 0          # 0 = as many completed jobs as budget allows
CHECKPOINT_SECONDS = 600
EXTENDED_SENSITIVITY = False  # True = 12 test / 9 train hosts in sensitivity sweeps
RESULT_ZIP = 'watermark_all_results_9train_12test.zip'
PATCH_NAME = 'WatermarkLab_REVIEW_CLASSICAL_ONLY_PATCH_20261016.zip'
EXPECTED_PROPOSAL_SHA256 = '10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'
REPO_GIT = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'

WORK=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
INPUT=Path('/kaggle/input')
REPO=WORK/'pham_thi_yen'
OUTPUT=WORK/'watermark_all_results_9train_12test'


def run(*args, cwd=None, check=True):
    print('>>>', ' '.join(map(str,args)), flush=True)
    p=subprocess.run([str(a) for a in args], cwd=cwd or WORK, check=False)
    if check and p.returncode:raise RuntimeError(f'Exit {p.returncode}: {args}')
    return p.returncode


def find_input(name):
    candidates=list(INPUT.rglob(name)) if INPUT.exists() else []
    return sorted(candidates,key=lambda f:f.stat().st_mtime)[-1] if candidates else None


def overlay(archive, root):
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP corrupted: '+str(archive))
        members=z.infolist()
        for m in members:
            if m.is_dir():continue
            parts=Path(m.filename).parts
            if len(parts)<2 or parts[0]!='pham_thi_yen-main' or '..' in parts or m.filename.startswith('/'):
                raise RuntimeError('Unexpected/unsafe ZIP member: '+m.filename)
            path=root.joinpath(*parts[1:]);path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(m) as source,path.open('wb') as dest:shutil.copyfileobj(source,dest)
    print('Overlay installed:',archive.name,flush=True)


# Prefer exact uploaded archive over a potentially different GitHub revision.
bases=[]
if INPUT.exists():
    bases=[x for x in INPUT.rglob('*.zip') if x.name.startswith('pham_thi_yen-main')]
base=sorted(bases,key=lambda x:x.stat().st_mtime)[-1] if bases else None
if REPO.exists():shutil.rmtree(REPO)
if base:
    overlay(base,REPO)
else:
    run('git','clone','--depth','1',REPO_GIT,REPO)
method=REPO/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
if not method.is_file() or hashlib.sha256(method.read_bytes()).hexdigest()!=EXPECTED_PROPOSAL_SHA256:
    raise RuntimeError('Proposal method mismatch. Upload the exact source ZIP for this changes-only patch.')
patch=find_input(PATCH_NAME)
if patch is None:raise FileNotFoundError('Attach '+PATCH_NAME+' as a Kaggle Input.')
overlay(patch,REPO)
if hashlib.sha256(method.read_bytes()).hexdigest()!=EXPECTED_PROPOSAL_SHA256:
    raise RuntimeError('The patch unexpectedly changed the proposal method.')

run(sys.executable,'-m','pip','install','-q','-r',REPO/'requirements.txt')
run(sys.executable,'-c','import pywt,cv2,scipy,skimage;print("PyWavelets",pywt.__version__)')
os.environ.update(PYTHONUNBUFFERED='1',WATERMARKLAB_STRICT_PYWT='1',
                  OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

resume=find_input(RESULT_ZIP)
args=[sys.executable,'scripts/recover_missing_results.py','--step',STEP,'--out',OUTPUT,
      '--max-jobs',str(MAX_JOBS),'--max-seconds',str(MAX_SECONDS),
      '--task-timeout',str(TASK_TIMEOUT),'--checkpoint-seconds',str(CHECKPOINT_SECONDS)]
if resume:
    args.extend(['--restore-zip',str(resume)])
    print('Restoring previous checkpoint:',resume,flush=True)
if EXTENDED_SENSITIVITY:args.append('--extended-sensitivity')
code=run(*args,cwd=REPO,check=False)
for name in ('recovery_status.json','completeness.json'):
    location=OUTPUT/name
    if location.is_file():
        print(name,location.read_text()[-8000:],flush=True)
print('Exit:',code,'Checkpoint:',OUTPUT.with_suffix('.zip'),flush=True)
print('Use the latest checkpoint ZIP as input in your next Kaggle session.',flush=True)
print('PASS means strict grids validated; do not infer completion from ZIP existence.',flush=True)
