"""Cheap synthetic interruption tests: no training, no network, no GPU required."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from watermarklab import benchmark as b
from watermarklab.experiments.partial_results import AttackCheckpoint, fingerprint


def test_checkpoint_rejects_different_embedded_image_and_signature(tmp_path):
    p=tmp_path/'progress.json'
    c=AttackCheckpoint(p,fingerprint('run_a'),numeric_columns=('nc',))
    c.prepare(np.zeros((2,2),dtype='uint8'))
    c.set_attack('a',{'nc':0.95})
    same=AttackCheckpoint(p,fingerprint('run_a'),numeric_columns=('nc',))
    same.prepare(np.zeros((2,2),dtype='uint8'))
    assert same.attack('a')['nc']==0.95
    same.prepare(np.ones((2,2),dtype='uint8'))
    assert same.attack('a') is None
    different=AttackCheckpoint(p,fingerprint('run_b'),numeric_columns=('nc',))
    assert different.attack('a') is None
    different.prepare(np.zeros((2,2),dtype='uint8'))
    with pytest.raises(ValueError):
        different.set_attack('bad',{'nc':float('nan')})
    # Non-finite evidence must not be checkpointed as a completed measurement.


def _stub_metrics(monkeypatch):
    for name in ('psnr','ssim','nc','ncc','ber'):
        monkeypatch.setattr(b,name,lambda *a,**k:0.98)


def test_normal_attack_grid_resumes_without_repeating_success(tmp_path, monkeypatch):
    hosts=tmp_path/'hosts';hosts.mkdir(); host=hosts/'x.bmp';host.write_bytes(b'host')
    wm=tmp_path/'wm.png';wm.write_bytes(b'wm')
    monkeypatch.setattr(b,'list_image_files',lambda _: [host])
    monkeypatch.setattr(b,'load_host_rgb',lambda _:np.zeros((8,8,3),dtype=np.uint8))
    monkeypatch.setattr(b,'load_watermark_binary',lambda *a,**k:np.zeros((4,4),dtype=np.uint8))
    attacks=[SimpleNamespace(name=n,group='g',params={}) for n in ('no_attack','a','b')]
    monkeypatch.setattr(b,'default_attack_suite',lambda **k:attacks)
    monkeypatch.setattr(b,'apply_attack',lambda image,attack: np.full_like(image,1 if attack.name=='a' else 2))
    _stub_metrics(monkeypatch)
    counts={'a':0,'b':0,'fail_b':True}
    class Fake:
        name='fake'
        def embed(self,host,watermark):return np.zeros_like(host),SimpleNamespace()
        def extract(self,image,key,host_rgb=None):
            v=int(image[0,0,0])
            if v:
                k='a' if v==1 else 'b';counts[k]+=1
                if k=='b' and counts['fail_b']:raise RuntimeError('simulated failure')
            return np.zeros((4,4),dtype=np.uint8)
    monkeypatch.setattr(b,'build_methods',lambda *a,**k:{'fake':Fake()})
    output=tmp_path/'output'
    def run():
        return b.run_benchmark(hosts,wm,output,selected_methods=['fake'],save_outputs=False,
                               attack_preset='lite',resume_attacks=True)
    first=run()
    assert len(first['results'])==2 and len(first['failures'])==1
    saved=output/'attack_progress/fake/x.json'
    assert json.loads(saved.read_text())['attacks'].keys()=={'a'}
    counts['fail_b']=False
    second=run()
    assert len(second['results'])==3 and not second['failures']
    assert counts['a']==1 and counts['b']==2
    third=run()
    assert len(third['results'])==3 and counts['a']==1 and counts['b']==2
    wm.write_bytes(b'changed watermark')
    fourth=run()
    assert len(fourth['results'])==3 and counts['a']==2 and counts['b']==3


def test_size_sweep_resumes_completed_attack_and_rejects_partial(tmp_path,monkeypatch):
    host=tmp_path/'x.bmp';host.write_bytes(b'host')
    wm=tmp_path/'wm.png';wm.write_bytes(b'wm')
    monkeypatch.setattr(b,'list_image_files',lambda _: [host])
    monkeypatch.setattr(b,'_load_resized_rgb_image',lambda *a,**k:np.zeros((64,64,3),dtype=np.uint8))
    monkeypatch.setattr(b,'_load_resized_binary_watermark',lambda *a,**k:np.zeros((16,16),dtype=np.uint8))
    monkeypatch.setattr(b,'default_attack_suite',lambda **k:[SimpleNamespace(name=n,group='g',params={}) for n in ('a','b')])
    monkeypatch.setattr(b,'apply_attack',lambda image,attack:np.full_like(image,1 if attack.name=='a' else 2))
    for name in ('_plot_size_sweep_bar','_plot_size_sweep_heatmap'):
        monkeypatch.setattr(b,name,lambda *a,**k:None)
    _stub_metrics(monkeypatch)
    calls={'a':0,'b':0,'crash':True}
    class Fake:
        def __init__(self,params,**kwargs):self.params=params
        def embed(self,host,watermark):return np.zeros_like(host),SimpleNamespace(flags=[],repeat_factor=1,usable_blocks=10,total_blocks=10)
        def extract(self,image,key,host_rgb=None):
            v=int(image[0,0,0])
            if v:
                kind='a' if v==1 else 'b';calls[kind]+=1
                if kind=='b' and calls['crash']:raise RuntimeError('interrupted')
            return np.zeros((16,16),dtype=np.uint8)
    monkeypatch.setattr(b,'ProposalQHDWTHess',Fake)
    out=tmp_path/'results'
    def run():
        return b.run_proposal_size_sweep_phase(host_dir=tmp_path,watermark_paths=[wm],
            output_dir=out,host_sizes=[64],watermark_sizes=[16],max_images=1,
            attack_preset='lite',proposal_options={'params':{'repeat':1}})
    with pytest.raises(RuntimeError,match='Incomplete size sweep'):
        run()
    assert calls['a']==1 and calls['b']==1
    calls['crash']=False
    complete=run()
    assert len(complete['clean_results'])==1
    assert len(complete['attack_results'])==2
    assert calls['a']==1 and calls['b']==2
    run()
    assert calls['a']==1 and calls['b']==2
