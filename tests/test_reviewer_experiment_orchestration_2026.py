"""Cheap runner/report contract tests; no costly embed/attack benchmarks."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_review_tables', ROOT/'scripts/build_review_tables.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def minimal_grid(tmp_path,monkeypatch):
    attacks=[SimpleNamespace(name='jpeg_q30',group='compression'),
             SimpleNamespace(name='rotation_15deg',group='geometric')]
    monkeypatch.setattr(mod,'DEFAULT_METHOD_IDS',['proposal','baseline'])
    monkeypatch.setattr(mod,'ORIGINAL_TEST_NAMES',['a.bmp','b.bmp'])
    monkeypatch.setattr(mod,'PROPOSAL_ABLATION_VARIANTS',{'full':{},'no_arnold':{}})
    monkeypatch.setattr(mod,'default_attack_suite',lambda *a,**k: attacks)
    monkeypatch.setattr(mod,'TABLES',{
        'classical_full':'heldout_full/per_image_attack_results.csv',
        'security':'review_audit/per_case_results.csv',
        'bss':'bss_validation/bss_per_host_attack.csv',
        'channel':'channel_ablation/channel_ablation.csv',
        'block_size':'block_size_comparison/per_case.csv',
        'ablation_full':'ablation/full/per_image_attack_results.csv',
        'ablation_no_arnold':'ablation/no_arnold/per_image_attack_results.csv'})
    base=tmp_path/'classical'
    classical=[]
    for method in ('proposal','baseline'):
        for host in ('a','b'):
            for phase,name in [('before_attack','no_attack'),
                               ('after_attack','jpeg_q30'),('after_attack','rotation_15deg')]:
                classical.append(dict(method_id=method,image=host,phase=phase,attack=name,
                    psnr=53.,ssim=0.99,nc=0.94 if method=='proposal' else 0.8,
                    ber=.06,embed_time_sec=0.2,extract_time_sec=.1))
    write_csv(base/mod.TABLES['classical_full'],classical)
    for variant in ('full','no_arnold'):
        rows=[{**r} for r in classical if r['method_id']=='proposal']
        write_csv(base/mod.TABLES['ablation_'+variant],rows)
    write_csv(base/mod.TABLES['security'],[dict(host='a.bmp',case='marked_clean',side_info_json_bytes=232)])
    write_csv(base/mod.TABLES['bss'],[dict(host='a.bmp',attack='jpeg_q30',survival_rate=.88)])
    write_csv(base/mod.TABLES['channel'],[dict(host='a.bmp',channel='Y',phase='after_attack',attack='jpeg_q30',nc=.9,ber=.1,psnr_clean=52.,ssim_clean=.99)])
    write_csv(base/mod.TABLES['block_size'],[dict(image='a.bmp',block_size=4,attack='jpeg_q30',nc=.9,ber=.1,psnr=52.,ssim=.99,coverage=1.,embed_seconds=.1,extract_seconds=.1)])
    return base


def test_complete_grid_generates_paired_statistics(tmp_path,monkeypatch):
    base=minimal_grid(tmp_path,monkeypatch)
    output=tmp_path/'tables'
    state=mod.build(base,output)
    assert state['complete'] is True
    assert state['expected_classical_rows']==12
    assert (output/'paired_host_confidence.csv').exists()
    assert (output/'ablation_by_attack.csv').exists()
    assert (output/'geometric_attacks.csv').exists()
    assert (output/'side_information_bytes.csv').exists()


def test_duplicate_row_is_not_certified(tmp_path,monkeypatch):
    base=minimal_grid(tmp_path,monkeypatch)
    file=base/mod.TABLES['classical_full']
    df=pd.read_csv(file)
    pd.concat([df,df.iloc[[0]]],ignore_index=True).to_csv(file,index=False)
    state=mod.build(base,tmp_path/'tables')
    assert state['complete'] is False
    assert state['classical_unique_complete'] is False


def test_missing_reviewer_variant_is_not_certified(tmp_path,monkeypatch):
    base=minimal_grid(tmp_path,monkeypatch)
    (base/'ablation/no_arnold/per_image_attack_results.csv').unlink()
    state=mod.build(base,tmp_path/'tables')
    assert not state['complete']
    assert 'ablation_no_arnold' in state['missing']


def test_patch_cannot_modify_proposal_source():
    import hashlib
    proposal=ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
    assert hashlib.sha256(proposal.read_bytes()).hexdigest() == '10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'
