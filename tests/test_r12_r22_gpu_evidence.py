"""Offline contract tests; these do NOT pretend that pretrained GPU TrustMark ran."""
import importlib.util
import json, sys, types, csv
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
    path=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location(path.stem,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_frozen_params_are_heldout_training_vectors():
    from watermarklab.benchmark import load_proposal_param_file
    a=load_proposal_param_file(ROOT/'configs/reviewer_r12_frozen_A.json')['global_params']
    b=load_proposal_param_file(ROOT/'configs/reviewer_r12_frozen_B.json')['global_params']
    assert a['q4_tau']==pytest.approx(.448479317241545)
    assert b['h01_q']==pytest.approx(9.75651853405941)
    assert len(a['private_key'])>0


def test_security_summary_preserves_bad_unmarked_control():
    m=load_script('reviewer_r12_security_batch.py')
    rows=[{'case':'original_unmarked','host':'Girl.bmp','watermark_nc':'1.0','payload_ber_all_positions':'0'},
          {'case':'marked_clean','host':'Girl.bmp','watermark_nc':'1.0','payload_ber_all_positions':'0'}]
    summary=m.summarize(rows,1)
    original=next(r for r in summary if r['case']=='original_unmarked')
    assert original['false_positive_count_nc_095']==1
    assert original['mean_watermark_nc']==1.0


def test_report_partial_and_complete_grids(tmp_path):
    m=load_script('reviewer_r12_r22_tables.py')
    dest=tmp_path/'NEURAL/A/Girl/trustmark';dest.mkdir(parents=True)
    rows=[{'image':'Girl','phase':'before_attack','attack':'no_attack','bit_coverage':'1',
           'fraction_correct_all_bits':'1','nc':'1','psnr':'42','ber_erasure_as_error':'0'},
          {'image':'Girl','phase':'control','attack':'original_unmarked','bit_coverage':'0'},
          {'image':'Girl','phase':'after_attack','attack':'jpeg_30','bit_coverage':'0',
           'fraction_correct_all_bits':'0','nc':'','ber_erasure_as_error':'1'},
          {'image':'Girl','phase':'after_attack','attack':'gaussian_05','bit_coverage':'0.4',
           'fraction_correct_all_bits':'0.35','nc':'0.99','ber_erasure_as_error':'.65'}]
    with (dest/'Girl_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    r=m.run(tmp_path,tmp_path/'TABLES',expected_attacks=2)
    assert not r['all_24_neural_hosts_complete']
    assert len(r['neural'])==2
    attack=next(x for x in r['neural'] if x['phase']=='after_attack')
    assert attack['mean_correctly_recovered_fraction']==pytest.approx(.175)
    assert attack['mean_conditional_nc_decoded_only']==pytest.approx(.99)
    assert attack['conditional_nc_valid_cases']==1
    assert attack['zero_coverage_cases']==1


def test_cuda_explicit_mode_and_rejection(monkeypatch):
    fake=types.ModuleType('trustmark')
    class Model:
        class Encoding:
            BCH_5=5
        def __init__(self,**kwargs):self.kwargs=kwargs
        def schemaCapacity(self):return 61
    fake.TrustMark=Model
    monkeypatch.setitem(sys.modules,'trustmark',fake)
    import torch
    from watermarklab.experiments.neural_baselines import TrustMarkAdapter
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    adapter=TrustMarkAdapter(device='cuda')
    assert adapter.device=='cuda'
    assert adapter.model.kwargs['device']=='cuda'
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU requested'):
        TrustMarkAdapter(device='cuda')
