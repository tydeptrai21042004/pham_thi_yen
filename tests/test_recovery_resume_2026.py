"""Test recovery checkpointing with synthetic candidates; no expensive benchmarks."""
import json
import csv
from pathlib import Path
import numpy as np
import pytest

from watermarklab import benchmark
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams
import importlib.util
_spec = importlib.util.spec_from_file_location("recover_missing_results", Path(__file__).resolve().parents[1] / "scripts" / "recover_missing_results.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
phase_ok = _mod.phase_ok


def test_optimization_checkpoints_and_resumes_without_reoptimizing_completed_image(tmp_path, monkeypatch):
    hostdir=tmp_path/'train';hostdir.mkdir()
    paths=[hostdir/'one.bmp',hostdir/'two.bmp']
    for p in paths:p.write_bytes(p.stem.encode())
    wm=tmp_path/'wm.png';wm.write_bytes(b'watermark-fixture')
    monkeypatch.setattr(benchmark,'list_image_files',lambda _: paths)
    monkeypatch.setattr(benchmark,'load_host_rgb',lambda p:np.zeros((4,4,3),dtype=np.uint8))
    monkeypatch.setattr(benchmark,'load_watermark_binary',lambda *a,**kw:np.ones((4,4),dtype=np.uint8))
    monkeypatch.setattr(benchmark,'default_attack_suite',lambda **kwargs:[])
    counts={'one':0,'two':0,'evaluations':0,'crash':True}
    class FakeModel:
        def __init__(self,**kwargs):pass
        def optimize_params(self,host,watermark,**kwargs):
            index = kwargs['seed']
            image='one' if index==123 else 'two'
            counts[image]+=1
            if image=='two' and counts['crash']:
                raise RuntimeError('simulated interrupted session')
            params=ProposalParams()
            return params,{'best_result':{'objective':float(index)},'history':[], 'optimizer':{}}
        def _evaluate_params_for_optimization(self,host,wm,params,attack_suite=None):
            counts['evaluations']+=1
            return {'objective':1.0}
    monkeypatch.setattr(benchmark,'ProposalQHDWTHess',FakeModel)
    out=tmp_path/'optimized.json'
    with pytest.raises(RuntimeError,match='interrupted'):
        benchmark.run_proposal_optimization_phase(hostdir,wm,out,seed=123)
    progress=json.loads((tmp_path/'optimized.progress.json').read_text())
    assert len(progress['rows'])==1
    counts['crash']=False
    result=benchmark.run_proposal_optimization_phase(hostdir,wm,out,seed=123)
    assert counts['one']==1 and counts['two']==2
    assert counts['evaluations']==4
    assert len(result['training_selection'])==2
    assert len(result['optimization_contract_sha256'])==64
    assert out.is_file()
    progress2=json.loads((tmp_path/'optimized.progress.json').read_text())
    assert len(progress2['selection'])==2
    # Completed work must survive another interrupted/repeated run.
    benchmark.run_proposal_optimization_phase(hostdir,wm,out,seed=123)
    assert counts['one']==1 and counts['two']==2 and counts['evaluations']==4
    with pytest.raises(ValueError,match='different input/configuration'):
        benchmark.run_proposal_optimization_phase(hostdir,wm,out,seed=124)


def test_heldout_strict_grid_rejects_missing_rows(tmp_path):
    result=tmp_path/'result.csv'
    result.write_text('method_id,image,phase,attack,psnr,ssim,nc,ber,embed_time_sec,extract_time_sec\\n'
                      'proposal,Girl,before_attack,no_attack,50,0.9,1,0,2,1\\n')
    assert not phase_ok(result,'proposal','Girl',['jpeg'])


def test_neural_force_full_resumes_each_completed_attack(tmp_path, monkeypatch):
    import watermarklab.experiments.neural_4096 as ext
    from watermarklab.common.attack import AttackConfig
    from PIL import Image
    folder=tmp_path/'hosts'; folder.mkdir()
    Image.fromarray(np.full((512,512,3),80,dtype=np.uint8)).save(folder/'test.png')
    wm=tmp_path/'wm.png'
    pattern=np.fromfunction(lambda i,j:(i+j)%2,(64,64),dtype=int)
    Image.fromarray(np.uint8(pattern*255)).save(wm)
    cases=[AttackConfig('first','fake',{}),AttackConfig('second','fake',{})]
    monkeypatch.setattr(ext,'default_attack_suite',lambda **kw:cases)
    counter={'first':0,'second':0,'interrupt':True}
    def attack(image,case):
        counter[case.name]+=1
        if case.name=='second' and counter['interrupt']:
            raise KeyboardInterrupt('preempted Kaggle session')
        return image
    monkeypatch.setattr(ext,'apply_attack',attack)
    class NotCapableModel:
        capacity=61
        def embed(self,image,bits): return image.copy()
        def detect(self,image): return (np.zeros(61,dtype=np.uint8),False)
    factory=lambda name:ext.SpatialChunk4096(NotCapableModel(),name=name,cols=8,resize_to=0)
    out=tmp_path/'results'
    with pytest.raises(KeyboardInterrupt):
        ext.run(host_dir=folder,watermark=wm,output=out,methods=('trustmark',),
                wrapper_factory=factory,force_full=True)
    partial=out/'trustmark'/'test_results.csv'
    assert len(list(csv.DictReader(partial.open())))==3
    counter['interrupt']=False
    result=ext.run(host_dir=folder,watermark=wm,output=out,methods=('trustmark',),
                   wrapper_factory=factory,force_full=True)
    assert len(result)==4
    assert counter['first']==1 and counter['second']==2
    assert json.loads((out/'manifest.json').read_text())['actual_attacked_rows']==2
    assert all(str(row.get('clean_gate_passed')) in ('False','false','0') for row in result)
