"""Compact checkpointed Kaggle revision run (from an updated repository).

Run in Kaggle with Internet ON and a suitable GPU for --run-neural:
  python scripts/kaggle_review_run.py --out /kaggle/working/reviewer_2026 --run-neural

The archive deliberately excludes images from all benchmarks except the fixed
three-method extracted-watermark gallery. No claim of 100% completion is made
until the stage manifest and strict results checks pass.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
MAX_SECONDS=int(10.5*3600)


def run_all(out:Path, *, run_neural:bool=False, verbose:bool=True, allow_expensive:bool=True):
    if run_neural:
        raise ValueError('Native 61/32-bit neural benchmarks are disabled in this patch. '
                         'Use scripts/recover_missing_results.py --step neural4096 '
                         'for the 4096-bit-only experiments.')
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    scratch=out.parent/(out.name+'_scratch')
    scratch.mkdir(exist_ok=True)
    env=os.environ.copy(); env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    start=time.monotonic(); stages=[]
    archive=out.parent/(out.name+'.zip')
    def pack():
        tmp=archive.with_suffix('.zip.partial')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for f in sorted(out.rglob('*')):
                if not f.is_file() or f.suffix=='.tmp': continue
                relative=f.relative_to(out)
                if ('images' in relative.parts or 'marked_images' in relative.parts or
                    'splits' in relative.parts):continue
                z.write(f,arcname=str(Path(out.name)/relative))
        os.replace(tmp,archive)
    def status():
        state={'utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
               'python':sys.version.split()[0], 'stages':stages,
               'complete':all(r['ok'] for r in stages),
               'elapsed_hours':round((time.monotonic()-start)/3600,3)}
        (out/'run_status.json').write_text(json.dumps(state,indent=2))
        pack()
    def stage(name, cmd, *, timeout=4*3600):
        remaining=MAX_SECONDS-(time.monotonic()-start)
        if remaining<=60:
            stages.append({'name':name,'ok':False,'reason':'session safety stop'})
            status();return False
        log_path=out/(name+'.log')
        print('START',name,flush=True)
        t=time.monotonic()
        try:
            with log_path.open('w') as fp:
                proc=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,env=env,
                    stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+min(timeout,max(1,int(remaining-60)))
                next_checkpoint=time.monotonic()+20*60
                timed_out=False
                while proc.poll() is None:
                    if time.monotonic()>=deadline:
                        timed_out=True
                        try:os.killpg(proc.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        try:proc.wait(timeout=12)
                        except subprocess.TimeoutExpired:
                            try:os.killpg(proc.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            proc.wait()
                        break
                    if time.monotonic()>=next_checkpoint:
                        pack()  # periodic atomic checkpoint for long stages
                        next_checkpoint=time.monotonic()+20*60
                    time.sleep(3)
            code=proc.wait()
            ok=(code==0 and not timed_out)
            error='stage timeout / safety cutoff' if timed_out else ('' if ok else f'exit status {code}')
        except Exception as ex:
            ok=False;error=repr(ex)
        stages.append({'name':name,'ok':ok,'elapsed_seconds':round(time.monotonic()-t,2),
                       'error':error,'log':log_path.name})
        print('FINISHED',name,'PASS' if ok else error,flush=True)
        status()
        return ok
    def cli(*args):return [sys.executable,'main.py',*args]
    water=ROOT/'data/watermark/wm.png'
    split_file=scratch/'split.json'
    if not stage('00_split',[sys.executable,'-m','watermarklab.experiments.splits',
       '--host-dir',ROOT/'data/host','--output',split_file,'--create-dirs']):
        raise RuntimeError('The fixed 9-new-training / 12-original-testing split could not be constructed')
    manifest=json.loads(split_file.read_text());(out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    train=Path(manifest['train_dir']);test=Path(manifest['test_dir'])
    assert len(list(train.glob('*.bmp')))==9 and len(list(test.glob('*.bmp')))==12
    param=out/'proposal_optimized_params.json'
    guo=out/'guo_optimized_params.json'
    gaata=out/'gaata_optimized_params.json'
    stage('01_proposal_optimization',cli('--phase','optimize','--methods','proposal',
         '--host-dir',train,'--watermark',water,'--proposal-repeat','full',
         '--proposal-optimizer-fireflies','4','--proposal-optimizer-generations','2',
         '--proposal-optimizer-attack-preset','script','--proposal-param-file',param),timeout=3*3600)
    stage('02_guo_optimization',cli('--phase','optimize','--methods','guo2017_dwt_qr_fa',
         '--host-dir',train,'--watermark',water,'--guo-optimizer-fireflies','4',
         '--guo-optimizer-generations','2','--guo-optimizer-attack-preset','lite',
         '--guo-param-file',guo),timeout=2*3600)
    stage('03_gaata_optimization',cli('--phase','optimize','--methods','gaata2022_dwt_hess_fwa',
         '--host-dir',train,'--watermark',water,'--gaata-optimizer-population','4',
         '--gaata-optimizer-iterations','2','--gaata-optimizer-sparks','2',
         '--gaata-param-file',gaata),timeout=2*3600)
    # Never replace an absent trained file with defaults in final protocol.
    if run_neural:
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') and not _cuda_available():
            print('WARNING: Kaggle GPU not detected; neural inference may be very slow',flush=True)
        stage('13_install_trustmark',[sys.executable,'-m','pip','install','-q','trustmark'],timeout=1500)
        wam_source=out.parent/'watermark-anything'
        if not wam_source.exists():
            stage('14_clone_wam',['git','clone','--depth','1',
                'https://github.com/facebookresearch/watermark-anything.git',wam_source],timeout=1200)
        if wam_source.exists():
            # WAM's full upstream requirements pin old scientific stack versions.
            # Kaggle Python 3.13 may attempt unsupported source builds while the
            # official *inference* model already works with the runtime stack.
            # Install its inference-only dependencies, retaining Kaggle's usable
            # OpenCV / scipy / scikit-image / scikit-learn installations.
            stage('15_install_wam_dependencies',[
                sys.executable,'-m','pip','install','--prefer-binary',
                'omegaconf==2.3.0','einops==0.8.0','timm==1.0.11',
                'lpips==0.1.4','pycocotools>=2.0.8'],timeout=1800)
            checkpoint=wam_source/'checkpoints'/'wam_mit.pth'
            if not checkpoint.exists():
                stage('16_fetch_wam_weights',[sys.executable,'-c',
                    'import urllib.request,sys;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])',
                    'https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',checkpoint],timeout=1500)
            env['WAM_REPO_DIR']=str(wam_source)
            env['WAM_CHECKPOINT']=str(checkpoint)
        stage('17_neural_models',cli('--phase','neural','--methods','trustmark,wam',
                '--host-dir',test,'--watermark',water,'--attack-preset','full',
                '--no-save-images','--output',out/'neural'),timeout=3*3600)

    if all(f.exists() for f in (param,guo,gaata)):
        stage('04_heldout_full',cli('--phase','normal','--methods','all','--baseline-mode','adapt',
             '--host-dir',test,'--watermark',water,'--attack-preset','full',
             '--proposal-repeat','full','--proposal-param-file',param,'--proposal-param-mode','require',
             '--guo-param-file',guo,'--guo-param-mode','require','--gaata-param-file',gaata,
             '--gaata-param-mode','require','--no-save-images','--output',out/'heldout_full'),timeout=4*3600)
        csv=out/'heldout_full/per_image_attack_results.csv'
        if csv.exists():
            stage('05_statistical_completeness',[sys.executable,'-m','watermarklab.experiments.evidence',
                '--csv',csv,'--split',split_file,'--output',out/'statistical_evidence'],timeout=600)
        # Fixed gallery chosen independently of measured performance.
        stage('06_extracted_gallery',[sys.executable,'-m','watermarklab.experiments.extracted_gallery',
            '--host',test/'airplane.bmp','--watermark',water,'--output',out/'gallery',
            '--proposal-params',param,'--guo-params',guo],timeout=1500)
        if allow_expensive:
            stage('07_proposal_ablation',cli('--phase','proposal-ablation','--host-dir',test,
                '--watermark',water,'--proposal-param-file',param,'--proposal-param-mode','require',
                '--attack-preset','reviewer_stress','--no-save-images','--output',out/'ablation'),timeout=2*3600)
            stage('08_color_channel',[sys.executable,'-m','watermarklab.experiments.channel_ablation',
                '--host-dir',test,'--watermark',water,'--param-file',param,
                '--preset','reviewer_stress','--output',out/'channel_ablation'],timeout=2*3600)
            stage('09_review_audit',[sys.executable,'-m','watermarklab.experiments.review_audit',
                '--host-dir',test,'--watermark',water,'--output',out/'review_audit',
                '--max-images','12','--host-size','512','--watermark-size','64',
                '--param-file',param,'--attack-preset','reviewer_stress','--max-attacks','7'],timeout=2*3600)
            stage('10_bss_validation',[sys.executable,'-m','watermarklab.experiments.bss_validation',
                '--host-dir',test,'--watermark',water,'--output',out/'bss_validation',
                '--param-file',param,'--host-size','512','--watermark-size','64','--max-attacks','14'],timeout=2*3600)
            stage('11_size_sweep',cli('--phase','proposal-size-sweep','--host-dir',test,
                '--max-images','2','--watermark',water,'--attack-preset','lite',
                '--no-save-images','--output',out/'size_sweep'),timeout=2*3600)
            stage('11b_block_size_comparison',[sys.executable,'-m','watermarklab.experiments.block_size_comparison',
                '--host-dir',test,'--watermark',water,'--param-file',param,
                '--output',out/'block_size_comparison','--block-sizes','4,8,16',
                '--max-attacks','7'],timeout=2*3600)
            stage('12_hparam_sweep',cli('--phase','proposal-hparam-sweep','--host-dir',train,
                '--watermark',water,'--attack-preset','lite','--max-images','1',
                '--no-save-images','--proposal-repeat','1','--hparam-sweep-max-combinations','24',
                '--output',out/'hparam_sweep'),timeout=2*3600)
    else:
        stages.append({'name':'final_classical','ok':False,'reason':'missing one or more trained parameter files'})
        status()
    status()
    print('FINAL ZIP',archive,flush=True)
    return stages


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:return False

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/kaggle/working/reviewer_2026'))
    parser.add_argument('--run-neural',action='store_true')
    parser.add_argument('--compact',action='store_true',help='Skip expensive optional reviewer sweeps')
    opts=parser.parse_args()
    results=run_all(opts.out,run_neural=opts.run_neural,allow_expensive=not opts.compact)
    if not all(r['ok'] for r in results):
        print('One or more stages failed. See run_status.json and logs; archive is partial.',flush=True)
        sys.exit(1)
