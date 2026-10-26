# Standalone Kaggle CPU cell — W1 — NO previous-result ZIPs required.
# Kaggle settings: Internet = ON; Accelerator = None (CPU).
# BEFORE running: upload the supplied GitHub code patch and commit to main.
import os, sys, subprocess, json, shutil
from pathlib import Path

LABEL = 'W1'
REPO_URL = "https://github.com/tydeptrai21042004/pham_thi_yen.git"
WORK = Path('/kaggle/working')
assert WORK.is_dir(), 'Run this cell inside a Kaggle notebook'
REPO = WORK / ('yen_fresh_' + LABEL.lower() + '_repo')
OUTPUT = WORK / ('yen_' + LABEL.lower() + '_remaining_fresh')
OUT_ZIP = WORK / (OUTPUT.name + '.zip')
os.environ.update(PYTHONUNBUFFERED='1', WATERMARKLAB_STRICT_PYWT='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')

# Completely self-contained: no attachments and no reading of /kaggle/input.
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(['git','clone','--depth','1','--branch','main',REPO_URL,str(REPO)],check=True)
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
print('FRESH GITHUB COMMIT',commit,flush=True)

required = [
    'src/watermarklab/experiments/partial_results.py',
    'scripts/recover_missing_results.py',
    'scripts/run_remaining_fresh.py',
    'scripts/plot_remaining_fresh.py',
    'tests/test_undefined_nc_checkpoint_fix.py',
]
missing=[x for x in required if not (REPO/x).is_file()]
if missing:
    raise RuntimeError('Commit the code patch to GitHub first. Missing: '+str(missing))
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements.txt'],cwd=REPO,check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','pytest'],cwd=REPO,check=True)
env=os.environ.copy();env['PYTHONPATH']=str(REPO/'src')
subprocess.run([sys.executable,'-m','pytest','-q',
                'tests/test_undefined_nc_checkpoint_fix.py',
                'tests/test_recovery_resume_2026.py',
                'tests/test_full_reviewer_kaggle_contract_2026.py'],
               cwd=REPO,env=env,check=True)

print('Starting independent fresh',LABEL,'run: 9 training, 12 test, no Lena, NO neural',flush=True)
cmd=[sys.executable,'scripts/run_remaining_fresh.py',
     '--label',LABEL,'--out',str(OUTPUT),
     '--max-seconds','40500',
     '--fireflies','2','--generations','2',
     '--task-timeout','10500']
try:
    run=subprocess.run(cmd,cwd=REPO,env=env,check=False)
    print('RUN EXIT',run.returncode,flush=True)
finally:
    print('KAGGLE OUTPUT ZIP:',OUT_ZIP,'EXISTS:',OUT_ZIP.exists(),flush=True)
    status=OUTPUT/'fresh_remaining_status.json'
    if status.is_file():
        obj=json.loads(status.read_text())
        print('STATUS:',json.dumps(obj,indent=2)[:12000],flush=True)
    if OUT_ZIP.exists():
        print('OUTPUT BYTES',OUT_ZIP.stat().st_size,flush=True)
# Exit 0 here lets Kaggle publish partial checkpoint evidence; inspect status
# rather than treating an incomplete run as completed experiment.
