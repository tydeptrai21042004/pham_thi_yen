# Paste this ENTIRE file into ONE Kaggle Python cell.
# Kaggle: Settings -> Accelerator -> GPU (T4/P100); Internet ON.
# Upload updated patch files to GitHub first. This uses the original 4096-bit
# TrustMark adaptation (68 BCH-5 chunks) and DOES NOT change the proposal.
# One GPU session might NOT finish 24 hosts x 109 attacks x 68 decoder calls.
# Resume with the output CHECKPOINT zip attached to the next Kaggle session.
import os, sys, time, json, shutil, zipfile, hashlib, subprocess
from pathlib import Path

GITHUB_URL = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'
GIT_BRANCH = 'main'
COHORTS = ('A','B')       # For independent sessions use ('A',) and ('B',).
RUN_SECURITY = True     # R1.2, CPU; does NOT alter the proposal.
RUN_TRUSTMARK = True    # R2.2, genuine TrustMark GPU inference, 4096-bit extension.
WRONG_KEY_TRIALS = 8
MAX_SESSION_SECONDS = 11*3600+15*60
CHECKPOINT_EVERY_SECONDS = 15*60
ORIGINAL_PROPOSAL_SHA256 = '10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'

WORK=Path('/kaggle/working'); INPUT=Path('/kaggle/input')
REPO=WORK/'pham_thi_yen'; ROOT=WORK/'R12_R22_GPU'
RESULTS=ROOT/'REVIEW_TABLES'; CHECKPOINT=WORK/'R12_R22_GPU_CHECKPOINT.zip'
HOSTS=('Girl','airplane','athens','baboon','butfly','house','manhatan',
       'milkdrop','pepper','safari','sailboat','tiffany')
START=time.monotonic(); previous_checkpoint_time=START


def command(*args,cwd=None,env=None):
    print('>',' '.join(str(a) for a in args),flush=True)
    subprocess.run([str(a) for a in args],check=True,cwd=cwd or REPO,env=env)


def package_checkpoint():
    ROOT.mkdir(parents=True,exist_ok=True)
    temporary=CHECKPOINT.with_suffix('.zip.partial')
    if temporary.exists():temporary.unlink()
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for item in sorted(ROOT.rglob('*')):
            if item.is_file() and not item.name.endswith(('.tmp','.partial')):
                z.write(item,item.relative_to(ROOT).as_posix())
    temporary.replace(CHECKPOINT)
    print('CHECKPOINT:',CHECKPOINT,'size=',CHECKPOINT.stat().st_size,flush=True)


def restore_checkpoint():
    matches=sorted(INPUT.rglob('R12_R22_GPU_CHECKPOINT.zip')) if INPUT.exists() else []
    if not matches:return
    if len(matches)>1:raise ValueError('Several checkpoint ZIPs attached; retain only the latest')
    print('RESTORING CHECKPOINT:',matches[0],flush=True)
    with zipfile.ZipFile(matches[0]) as z:
        for info in z.infolist():
            path=Path(info.filename)
            if path.is_absolute() or '..' in path.parts or not info.filename:
                raise ValueError('Unsafe checkpoint path: '+info.filename)
        z.extractall(ROOT)


def hours_left():return (MAX_SESSION_SECONDS-(time.monotonic()-START))/3600


def execute_with_checkpoints(args,label,env):
    global previous_checkpoint_time
    print('START',label,'remaining_hours',round(hours_left(),2),flush=True)
    process=subprocess.Popen([str(x) for x in args],cwd=REPO,env=env)
    while process.poll() is None:
        time.sleep(20)
        if time.monotonic()-previous_checkpoint_time>=CHECKPOINT_EVERY_SECONDS:
            package_checkpoint();previous_checkpoint_time=time.monotonic()
        if hours_left() <= .18:
            print('SESSION BUDGET: terminating process to preserve checkpoint:',label,flush=True)
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            package_checkpoint()
            return False
    if process.returncode:
        package_checkpoint()
        raise RuntimeError(f'{label} failed with exit {process.returncode}; inspect log above')
    package_checkpoint();previous_checkpoint_time=time.monotonic()
    print('FINISH',label,flush=True)
    return True


if REPO.exists():shutil.rmtree(REPO)
command('git','clone','--depth','1','--branch',GIT_BRANCH,GITHUB_URL,REPO,cwd=WORK)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
print('GITHUB COMMIT:',commit,flush=True)
proposal=REPO/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
assert hashlib.sha256(proposal.read_bytes()).hexdigest()==ORIGINAL_PROPOSAL_SHA256,\
       'PROPOSAL SOURCE DIFFERS FROM USER-UPLOADED ORIGINAL: refusing to mislabel unchanged protocol'
for f in ('scripts/reviewer_r12_security_batch.py','scripts/reviewer_r12_r22_tables.py',
          'configs/reviewer_r12_frozen_A.json','configs/reviewer_r12_frozen_B.json',
          'src/watermarklab/experiments/neural_4096.py'):
    assert (REPO/f).is_file(),f'Please push the supplied patch to GitHub: {f}'
command(sys.executable,'-m','pip','install','-q','-e',REPO,'trustmark==0.9.2','pytest','PyWavelets')
import torch
assert torch.cuda.is_available(), 'GPU not available: Kaggle Settings > Accelerator > GPU, then restart notebook'
print('CUDA:',torch.cuda.get_device_name(0),'torch:',torch.__version__,flush=True)
from importlib import metadata
print('TrustMark:',metadata.version('trustmark'),flush=True)
run_env=os.environ.copy()
run_env.update(PYTHONPATH=str(REPO/'src'),PYTHONUNBUFFERED='1',
               WATERMARKLAB_STRICT_PYWT='1',OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
command(sys.executable,'-m','pytest','-q','tests/test_reviewer_4096_only.py',
        'tests/test_reviewer_neural_4096.py',
        'tests/test_neural4096_only_cache_and_retry.py',cwd=REPO,env=run_env)
for host in HOSTS:
    assert (REPO/'data/host'/f'{host}.bmp').is_file(),f'Missing test image {host}'
assert COHORTS and set(COHORTS)<=set(('A','B'))
ROOT.mkdir(exist_ok=True);restore_checkpoint()
# OPTIONAL: If all six earlier evidence ZIPs are attached as Kaggle Input,
# produce instant, verified existing comparisons before any expensive reruns.
ARCHIVES=('watermark_all_results_9train_12test.zip',
 'watermark_all_results_9train_12test_missing.zip','watermark_2_Cc.zip',
 'watermark_2_Ccmisssing.zip','trustmark_only_remaining_A.zip',
 'trustmark_only_remaining_B.zip')
prior=[]
for name in ARCHIVES:
    located=sorted(INPUT.rglob(name)) if INPUT.exists() else []
    if located:prior.append(located[0])
if len(prior)==len(ARCHIVES) and (REPO/'scripts/reviewer_4096_report.py').is_file():
    command(sys.executable,'scripts/reviewer_4096_report.py','--archives',
            *prior,'--output',ROOT/'OLD_DATA_REPORT',env=run_env)
    package_checkpoint()
else:
    print('Optional old six evidence archives not all attached; new GPU experiments continue.',flush=True)
ROOT.joinpath('experiment_environment.json').write_text(json.dumps({
 'git_sha':commit,'gpu':torch.cuda.get_device_name(0),
 'torch':torch.__version__,'trustmark':metadata.version('trustmark'),
 'protocol':'4096-bit spatial extension; not native 4096-bit TrustMark',
 'watermark_cohorts':COHORTS,'full_attack_count':109,
 'proposal_sha256':ORIGINAL_PROPOSAL_SHA256},indent=2))
sys.path.insert(0, str(REPO/'src'))
from watermarklab.experiments.second_watermark import prepare
prepared_B=ROOT/'prepared_logo_B.png'
prepare(REPO/'data/watermark/Cc.logo.circle.png',prepared_B)
WM={
 'A':(REPO/'data/watermark/wm.png','luma'),
 'B':(REPO/'data/watermark/Cc.logo.circle.png','alpha')}

try:
    if RUN_SECURITY:
        for cohort in COHORTS:
            if hours_left() <= .18:break
            # Use the PREPARED B input to match the original classical experiment.
            security_watermark=prepared_B if cohort=='B' else WM[cohort][0]
            cmd=[sys.executable,'-u','scripts/reviewer_r12_security_batch.py',
                 '--hosts',REPO/'data/host','--watermark',security_watermark,
                 '--params',REPO/'configs'/f'reviewer_r12_frozen_{cohort}.json',
                 '--output',ROOT/'SECURITY'/cohort,'--wrong-keys',WRONG_KEY_TRIALS,
                 '--max-attacks','0']
            if not execute_with_checkpoints(cmd,f'R1.2 security cohort {cohort}',run_env):break
    if RUN_TRUSTMARK:
        for cohort in COHORTS:
            if hours_left() <= .18:break
            watermark,channel=WM[cohort]
            for host in HOSTS:
                if hours_left() <= .18:break
                out=ROOT/'NEURAL'/cohort/host
                cmd=[sys.executable,'-u','-m','watermarklab.experiments.neural_4096',
                     '--methods','trustmark','--device','cuda','--host-dir',REPO/'data/host'/f'{host}.bmp',
                     '--watermark',watermark,'--binary-channel',channel,
                     '--output',out,'--attack-preset','full','--force-full']
                if not execute_with_checkpoints(cmd,f'R2.2 TrustMark {cohort}/{host}',run_env):break
finally:
    command(sys.executable,'scripts/reviewer_r12_r22_tables.py','--root',ROOT,
            '--output',RESULTS,cwd=REPO,env=run_env)
    package_checkpoint()
    final=WORK/'R12_R22_RESULTS.zip'; temp=final.with_suffix('.zip.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for file in RESULTS.rglob('*'):
            if file.is_file():z.write(file,file.relative_to(RESULTS))
        z.write(ROOT/'experiment_environment.json','experiment_environment.json')
    temp.replace(final)
    print('OUTPUT:',final,'CHECKPOINT:',CHECKPOINT,flush=True)
    assert hashlib.sha256(proposal.read_bytes()).hexdigest()==ORIGINAL_PROPOSAL_SHA256
print('If report.json says complete=false, attach checkpoint ZIP in next GPU session and rerun the cell.')
