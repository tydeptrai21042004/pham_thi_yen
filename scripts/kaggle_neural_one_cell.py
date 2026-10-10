# Copy the WHOLE script into ONE Kaggle Python cell.
# Notebook Settings -> Internet ON; GPU recommended for ChunkySeal.
# IMPORTANT: FIRST commit/push the neural patch files to the GitHub repository.
# This script clones your own updated code, installs optional pretrained models,
# runs held-out hosts, and keeps ZIP checkpoints; no uploaded source ZIP needed.
from pathlib import Path
import csv, json, os, random, shutil, subprocess, sys, time, zipfile

REPO_URL = 'https://github.com/tydeptrai21042004/pham_thi_yen.git'
BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO = BASE / 'pham_thi_yen'
OUT = BASE / 'pham_yen_neural_results'
OUT.mkdir(parents=True, exist_ok=True)
ZIP_PATH = BASE / 'pham_yen_neural_results.zip'
HOST_SEED = 3101
METHODS_TO_RUN = ('trustmark', 'chunkyseal')  # choose ('trustmark',) if required
ATTACK_PRESET = 'full'
MAX_ATTACKS = 0  # 0 = all; choose 2 for fast smoke testing
SAFE_STOP_SECONDS = int(10.25 * 3600)
START = time.monotonic()

# Existing torch/torchvision provided by Kaggle should remain the preferred CUDA build.
ENV = os.environ.copy()
ENV.update(PYTHONPATH=str(REPO / 'src'), PYTHONUNBUFFERED='1',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


def save_zip(note):
    (OUT/'kaggle_status.json').write_text(json.dumps({'note': note, 'elapsed_h': round((time.monotonic()-START)/3600,3)}, indent=2))
    tmp = BASE/'pham_yen_neural_results.incomplete.zip'
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for file in OUT.rglob('*'):
            if file.is_file() and file.suffix not in ('.tmp', '.lock') and 'splits' not in file.parts:
                archive.write(file, str(file.relative_to(BASE)))
    tmp.replace(ZIP_PATH)
    print(f'[SAVE] {note}: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1048576:.1f} MiB)', flush=True)


def run(label, cmd, *, cwd=None, timeout_seconds=7200):
    print(f'\n[{label}] {" ".join(map(str,cmd))}', flush=True)
    log = OUT / (label+'.log')
    started, last_checkpoint = time.monotonic(), time.monotonic()
    with log.open('w', encoding='utf-8') as writer:
        proc = subprocess.Popen(list(map(str,cmd)), cwd=cwd or BASE, env=ENV,
                                stdout=writer, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-START > SAFE_STOP_SECONDS or time.monotonic()-started > timeout_seconds:
                import signal
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                print(f'[LIMIT] {label} interrupted; partial outputs retained', flush=True)
                break
            if time.monotonic()-last_checkpoint >= 600:
                save_zip(f'during {label}')
                last_checkpoint=time.monotonic()
            time.sleep(5)
    result = proc.wait()
    save_zip(f'{label}: exit={result}')
    print(f'[{label}] exit={result}. Log: {log}', flush=True)
    return result == 0


try:
    if not (REPO/'.git').exists():
        if not run('clone', ['git','clone','--depth','1',REPO_URL,REPO], timeout_seconds=360):
            raise RuntimeError('Git clone failed: check Internet.')
    else:
        run('pull', ['git','-C',REPO,'pull','--ff-only'])
    if not run('base_requirements', [sys.executable,'-m','pip','install','-q','-r',REPO/'requirements.txt'], timeout_seconds=1200):
        raise RuntimeError('Basic project dependency installation failed.')
    help_result=subprocess.run([sys.executable,'main.py','--help'],cwd=REPO,env=ENV,capture_output=True,text=True)
    if help_result.returncode or 'neural-max-attacks' not in help_result.stdout:
        raise RuntimeError('This GitHub checkout lacks the patch. Commit/push the attached patch files before cloning.')

    # Reproduce the same 3/10 host split from the original Kaggle notebook.
    images = sorted(p for p in (REPO/'data/host').iterdir() if p.suffix.lower() in {'.bmp','.png','.jpg','.jpeg','.webp','.tif','.tiff'})
    rng=random.Random(HOST_SEED)
    rng.shuffle(images)
    train_n=max(2,round(0.25*len(images)))
    train, test = images[:train_n], images[train_n:]
    test_dir=OUT/'splits'/'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    for item in test: shutil.copy2(item, test_dir/item.name)
    (OUT/'host_split.json').write_text(json.dumps({'seed':HOST_SEED, 'train':[p.name for p in train],
                                              'heldout':[p.name for p in test]}, indent=2))
    print('Held-out:', [p.name for p in test])
    save_zip('host split prepared')

    if 'trustmark' in METHODS_TO_RUN and run('install_trustmark', [sys.executable,'-m','pip','install','-q','trustmark'], timeout_seconds=1200):
        run('trustmark', [sys.executable,'main.py','--phase','neural','--methods','trustmark',
                          '--host-dir',test_dir,'--attack-preset',ATTACK_PRESET,
                          '--neural-max-attacks',str(MAX_ATTACKS),'--no-save-images',
                          '--output',OUT/'heldout_trustmark'], cwd=REPO, timeout_seconds=14400)
    if 'chunkyseal' in METHODS_TO_RUN:
        # The upstream VideoSeal package currently depends on extra libraries, including decord.
        ok=run('install_videoseal', [sys.executable,'-m','pip','install','-q',
                                    'git+https://github.com/facebookresearch/videoseal.git'], timeout_seconds=1800)
        if ok:
            run('chunkyseal', [sys.executable,'main.py','--phase','neural','--methods','chunkyseal',
                               '--host-dir',test_dir,'--attack-preset',ATTACK_PRESET,
                               '--neural-max-attacks',str(MAX_ATTACKS),'--neural-device','auto',
                               '--no-save-images','--output',OUT/'heldout_chunkyseal'],
                cwd=REPO, timeout_seconds=14400)
    print('Neural benchmark outputs saved separately; DO NOT merge with the 4096-bit matched classical table.')
finally:
    save_zip('FINAL')
    print('ARCHIVE:',ZIP_PATH)
