# Kaggle: Accelerator = None for classical; GPU recommended for neural.
# Internet = ON. Paste this file into ONE cell after committing the patch to GitHub.
# Runs the original method unchanged, writes reusable checkpoints, and NEVER
# treats an incomplete experiment subset as a full manuscript reproduction.
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

GITHUB = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'
COHORT = 'primary'        # primary = wm.png; secondary = separate Cc.logo.circle watermark
STEP = 'all'              # all, optimize, heldout, reviewer, reports, extras, neural4096, verify
INCLUDE_NEURAL_4096 = False  # set True only when ready to evaluate expensive adapted TrustMark/WAM
EXTENDED_SENSITIVITY = True
FIREFLIES = 1             # QUICK PILOT ONLY, NOT the manuscript's 10-firefly setting
GENERATIONS = 2          # QUICK PILOT ONLY, NOT the manuscript's 10-iteration setting
MAX_SECONDS = 36000       # up to 10 hours of experiments, allows Kaggle export margin
TASK_TIMEOUT = 30000     # per-job cap; checkpoints preserve successful attack rows
CHECKPOINT_SECONDS = 600

assert COHORT in ('primary', 'secondary')
assert STEP in ('all','optimize','heldout','reviewer','reports','extras','neural4096','verify')
assert FIREFLIES > 0 and GENERATIONS > 0
work = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
inp = Path('/kaggle/input')
repo = work/'pham_thi_yen'
folder_name = 'watermark_all_results_9train_12test' if COHORT=='primary' else 'watermark_2_Cc'
out = work/folder_name
archive = work/(folder_name+'.zip')

def command(*args, cwd=None, check=True):
    args=list(map(str,args));print('>>>',' '.join(args),flush=True)
    result=subprocess.run(args,cwd=cwd or work)
    if check and result.returncode:
        raise RuntimeError(f'Exit {result.returncode}: {args}')
    return result.returncode

# Detect checkpoint before clone; source commits must be identical. An old
# checkpoint without an explicit marker cannot be automatically trusted.
found = sorted(inp.rglob(folder_name+'.zip')) if inp.is_dir() else []
if len(found)>1:
    raise RuntimeError('More than one matching ZIP. Attach only one for this cohort.')
resume=found[0] if found else None
prior=None
if resume:
    with zipfile.ZipFile(resume) as z:
        marker=f'{folder_name}/kaggle_evidence_session.json'
        if marker not in z.namelist():
            raise RuntimeError('Checkpoint lacks provenance marker; start a fresh patched run.')
        prior=json.loads(z.read(marker))

if repo.exists():shutil.rmtree(repo)
command('git','clone','--depth','1',GITHUB,repo)
sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
source=repo/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
proposal_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if prior:
    for key,value in [('git_commit',sha),('proposal_sha256',proposal_sha),
                      ('cohort',COHORT),('fireflies',FIREFLIES),('generations',GENERATIONS),
                      ('extended_sensitivity',EXTENDED_SENSITIVITY)]:
        if prior.get(key)!=value:
            raise RuntimeError(f'Checkpoint incompatible: {key}: {prior.get(key)} != {value}')
    print('RESUME:',resume,flush=True)
else:
    print('FRESH RUN: no existing patch-compatible checkpoint',flush=True)

os.environ.update(PYTHONPATH=str(repo/'src'),WATERMARKLAB_STRICT_PYWT='1',
    PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
command(sys.executable,'-m','pip','install','-q','-r',repo/'requirements.txt')
command(sys.executable,'-c','import pywt,cv2,scipy,skimage; print("PyWavelets",pywt.__version__)')
command(sys.executable,'-m','pytest','-q',
        'tests/test_incremental_evidence_2026.py',
        'tests/test_recovery_resume_2026.py',
        'tests/test_dataset_9_train_12_test.py',
        'tests/test_reviewer_experiment_orchestration_2026.py',cwd=repo)

sys.path.insert(0,str(repo/'src'))
from PIL import Image
wm1=repo/'data/watermark/wm.png'
wm2=repo/'data/watermark/Cc.logo.circle.png'
print('SOURCE WATERMARK SIZES:', Image.open(wm1).size,Image.open(wm2).size,flush=True)
if COHORT=='secondary':
    from watermarklab.experiments.second_watermark import prepare
    watermark=prepare(wm2,work/'separate_secondary_inputs/Cc.logo.circle_alpha_binary.png')
else:
    watermark=wm1

if INCLUDE_NEURAL_4096 or STEP=='neural4096':
    command(sys.executable,'-m','pip','install','-q','trustmark','omegaconf==2.3.0',
            'einops==0.8.0','timm==1.0.11','lpips==0.1.4','pycocotools>=2.0.8')
    wam=work/'watermark-anything'
    if not (wam/'.git').is_dir():
        if wam.exists():shutil.rmtree(wam)
        command('git','clone','--depth','1','https://github.com/facebookresearch/watermark-anything.git',wam)
    weights=wam/'checkpoints/wam_mit.pth'
    if not weights.is_file() or weights.stat().st_size<1_000_000:
        weights.parent.mkdir(parents=True,exist_ok=True)
        tmp=weights.with_suffix('.pth.partial')
        with urllib.request.urlopen('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',timeout=240) as f, tmp.open('wb') as d:
            shutil.copyfileobj(f,d)
        if tmp.stat().st_size<1_000_000:raise RuntimeError('WAM weights download truncated')
        tmp.replace(weights)
    if not (wam/'checkpoints/params.json').exists():raise RuntimeError('Missing WAM params.json')
    os.environ['WAM_REPO_DIR']=str(wam)
    os.environ['WAM_CHECKPOINT']=str(weights)

out.mkdir(parents=True,exist_ok=True)
meta=dict(git_commit=sha,proposal_sha256=proposal_sha,cohort=COHORT,
    fireflies=FIREFLIES,generations=GENERATIONS,
    extended_sensitivity=EXTENDED_SENSITIVITY,
    scientific_warning='1 firefly x 2 generations is a pilot, not manuscript 10 x 10',
    host_train_count=9,host_test_count=12,
    native_neural_payload_evaluation=False)
# If recovering from ZIP, source runner will restore matching provenance metadata.
(out/'kaggle_evidence_session.json').write_text(json.dumps(meta,indent=2))
args=[sys.executable,'scripts/recover_missing_results.py',
      '--step',STEP,'--skip-secondary','--watermark',watermark,'--out',out,
      '--proposal-fireflies',FIREFLIES,'--proposal-generations',GENERATIONS,
      '--max-jobs',0,'--max-seconds',MAX_SECONDS,
      '--task-timeout',TASK_TIMEOUT,'--checkpoint-seconds',CHECKPOINT_SECONDS]
if EXTENDED_SENSITIVITY:args+=['--extended-sensitivity']
if INCLUDE_NEURAL_4096:args+=['--include-neural4096']
if resume:args+=['--restore-zip',resume]
code=command(*args,cwd=repo,check=False)
for name in ('recovery_status.json','completeness.json'):
    f=out/name
    if f.exists():
        data=json.loads(f.read_text())
        if name=='recovery_status.json':
            data={k:data.get(k) for k in ('paused_by_budget','blockers','last_finished_jobs','strict_completeness')}
        print(name,json.dumps(data,indent=2),flush=True)
print('SOURCE COMMIT:',sha,'runner exit:',code,flush=True)
print('CHECKPOINT ZIP:',archive,'exists:',archive.is_file(),flush=True)
print('Reattach that ZIP as a Kaggle Input Dataset on the next session, keeping settings and commit unchanged.',flush=True)
print('W2 in code is 64x64; the manuscript also describes a 128x128 watermark experiment.',flush=True)
