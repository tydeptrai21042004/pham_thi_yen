# Paste this entire file into ONE Kaggle Python cell.
# Settings: Internet ON, GPU recommended (but CPU works, much more slowly).
# BEFORE RUNNING: attach BOTH previous result ZIPs as Kaggle INPUT datasets:
#    watermark_all_results_9train_12test.zip  (session A)
#    watermark_2_Cc.zip                       (session B)
# No ZIP = ERROR; this script NEVER starts all experiments from scratch.
# Run in a FRESH notebook session. After each interrupted run, save BOTH
# updated ZIPs from /kaggle/working and attach them to the next session.
import subprocess, sys, shutil
from pathlib import Path

GITHUB = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'
WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
NAMES = ('watermark_all_results_9train_12test.zip','watermark_2_Cc.zip')
for name in NAMES:
    matches = sorted(INPUT.rglob(name)) if INPUT.exists() else []
    if len(matches) != 1:
        raise RuntimeError(f'Attach exactly ONE old {name} to Kaggle Input. Found: {matches}. '
                           'Without both checkpoints, cannot resume ONLY missing jobs.')
    print('FOUND SAVED EVIDENCE:',matches[0],flush=True)
repo=WORK/'pham_thi_yen'
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(['git','clone','--depth','1',GITHUB,str(repo)],check=True)
print('CODE:',subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),flush=True)
script=repo/'scripts/kaggle_resume_both_2026.py'
if not script.is_file():
    raise RuntimeError('Commit the supplied ZIP PATCH to GitHub before running this cell')
result=subprocess.run([sys.executable,'-u',str(script)],cwd=repo)
print('ORCHESTRATOR EXIT:',result.returncode,flush=True)
if result.returncode:
    raise RuntimeError('Recovery orchestrator stopped; examine the first failure above. '
                       'Do not rerun the full original notebook.')
