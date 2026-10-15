"""Offline 4096-bit-only integration checks; do not require downloaded weights."""
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import watermarklab.experiments.neural_4096 as neural


def make_inputs(tmp_path):
    hosts = tmp_path / 'hosts'
    hosts.mkdir()
    Image.fromarray(np.full((512, 512, 3), 96, np.uint8)).save(hosts/'image.png')
    wm = tmp_path / 'wm.png'
    Image.fromarray(np.indices((64,64)).sum(axis=0).astype(np.uint8).__mod__(2)*255).save(wm)
    return hosts,wm


class TrackedCodec:
    def __init__(self):
        self.capacity = 61
        self.embeds = 0
    def embed(self, patch, message):
        self.embeds += 1
        return patch.copy()
    def detect(self, patch):
        return None, False


def test_force_all_attack_metrics_even_with_zero_clean_coverage(tmp_path,monkeypatch):
    hosts,wm=make_inputs(tmp_path)
    cases=[SimpleNamespace(name='attack_1',group='synthetic'),SimpleNamespace(name='attack_2',group='synthetic')]
    monkeypatch.setattr(neural,'default_attack_suite',lambda **kw:cases)
    monkeypatch.setattr(neural,'apply_attack',lambda image,case:image)
    codec=TrackedCodec()
    factory=lambda method:neural.SpatialChunk4096(codec,name=method+'_4096_spatial',cols=8,resize_to=0)
    output=tmp_path/'results'
    rows=neural.run(host_dir=hosts,watermark=wm,output=output,methods=('trustmark',),
                    wrapper_factory=factory,force_full=True)
    assert [r['phase'] for r in rows]==['before_attack','control','after_attack','after_attack']
    assert all(float(r['bit_coverage'])==0 for r in rows)
    assert len(list(csv.DictReader((output/'trustmark'/'image_results.csv').open())))==4
    assert (output/'trustmark'/'image_marked_cache.png').is_file()
    assert codec.embeds==68
    # No second embedding if all attacks are complete and cached image is valid.
    rows=neural.run(host_dir=hosts,watermark=wm,output=output,methods=('trustmark',),
                    wrapper_factory=factory,force_full=True)
    assert len(rows)==4 and codec.embeds==68


def test_failed_attack_retry_uses_same_marked_image(tmp_path,monkeypatch):
    hosts,wm=make_inputs(tmp_path)
    cases=[SimpleNamespace(name='fragile',group='synthetic'),SimpleNamespace(name='stable',group='synthetic')]
    monkeypatch.setattr(neural,'default_attack_suite',lambda **kw:cases)
    attempts={'fragile':0,'stable':0}
    def apply(image,case):
        attempts[case.name]+=1
        if case.name=='fragile' and attempts['fragile']==1:
            raise RuntimeError('transient failure')
        return image
    monkeypatch.setattr(neural,'apply_attack',apply)
    codec=TrackedCodec()
    factory=lambda method:neural.SpatialChunk4096(codec,name=method+'_4096_spatial',cols=8,resize_to=0)
    output=tmp_path/'out'
    first=neural.run(host_dir=hosts,watermark=wm,output=output,methods=('trustmark',),
                     wrapper_factory=factory,force_full=True)
    assert len(first)==4
    assert not first[2].get('bit_coverage')
    assert codec.embeds==68
    second=neural.run(host_dir=hosts,watermark=wm,output=output,methods=('trustmark',),
                      wrapper_factory=factory,force_full=True)
    assert codec.embeds==68
    assert attempts=={'fragile':2,'stable':1}
    assert len(second)==4 and all('bit_coverage' in r for r in second)
