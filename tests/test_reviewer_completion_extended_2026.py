"""Focused regression tests for reviewer comments #1, #4, #5 and #6.

Tests do not falsely claim the expensive 12-host multi-model Kaggle grid ran.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from watermarklab.experiments.second_watermark import prepare
from watermarklab.experiments.reviewer_checks import (
    validate_security,validate_bss,validate_channel,validate_gallery)
from watermarklab.experiments.bss_validation import validate_summary_rows,normalized_decision_margin
from watermarklab.experiments.review_audit import (
    audit,metadata_tag,keyspace_assumptions_report)
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams,ProposalQHDWTHess

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def test_alpha_logo_is_not_blank_and_is_repeatable(tmp_path):
    source=ROOT/'data/watermark/Cc.logo.circle.png'
    dest=tmp_path/'logo.png'
    prepare(source,dest)
    first=dest.read_bytes()
    prepare(source,dest)
    assert dest.read_bytes()==first
    a=np.asarray(Image.open(dest))
    assert a.shape==(64,64)
    assert set(np.unique(a))=={0,255}
    assert int(np.sum(a==255))>0 and int(np.sum(a==0))>0
    m=json.loads((tmp_path/'secondary_watermark_preprocessing.json').read_text())
    assert m['source_channel']=='A' and m['one_bits']+m['zero_bits']==4096


def test_logo_rejects_missing_alpha_and_constant_mask(tmp_path):
    src=tmp_path/'no_alpha.png'
    Image.new('RGB',(64,64),0).save(src)
    with pytest.raises(ValueError,match='alpha'):prepare(src,tmp_path/'target.png')
    Image.new('RGBA',(64,64),(0,0,0,0)).save(src)
    with pytest.raises(ValueError,match='constant'):prepare(src,tmp_path/'target.png')


def test_bss_summary_disallows_zero_failed_and_duplicate_measurements():
    row={'host':'girl.bmp','attack':'jpeg','samples':'8','error_count':'0',
         'bss_unique_levels':'2','margin_unique_levels':'2','survival_rate':'0.5'}
    assert validate_summary_rows([row],['jpeg'],expected_host='girl.bmp')
    for field,value in [('samples','0'),('error_count','1'),('margin_unique_levels','0'),('survival_rate','nan')]:
        corrupt={**row,field:value}
        assert not validate_summary_rows([corrupt],['jpeg'],expected_host='girl.bmp')
    assert not validate_summary_rows([],['jpeg'])
    assert not validate_summary_rows([row,row],['jpeg'])
    assert not validate_summary_rows([row],['jpeg','rotation'])


def test_external_hmac_detects_side_metadata_change_without_core_claim():
    host=np.random.default_rng(6).integers(0,255,(64,64,3),dtype=np.uint8)
    wm=(np.random.default_rng(7).integers(0,2,(8,8))*255).astype(np.uint8)
    method=ProposalQHDWTHess(ProposalParams(repeat=1))
    _,key=method.embed(host,wm)
    import copy
    token=b'test-only-external-auth-key'
    original=metadata_tag(key,token)
    changed=copy.deepcopy(key)
    changed.flags=changed.flags[:-1]
    assert original != metadata_tag(changed,token)
    report=keyspace_assumptions_report('KB123')
    assert report['illustrative_max_bits_if_uniform_random']<32
    assert report['sha256_output_bits']==256


def test_security_audit_uses_true_unrelated_host_and_random_wrong_keys(tmp_path):
    hosts=sorted((ROOT/'data/host').glob('*.bmp'))
    one=tmp_path/'hosts';one.mkdir()
    (one/hosts[0].name).write_bytes(hosts[0].read_bytes())
    result=audit(one,ROOT/'data/watermark/wm.png',tmp_path/'audit',max_images=1,
                 host_size=64,watermark_size=8,repeat=1,max_attacks=0,
                 unrelated_host_path=hosts[1],wrong_key_trials=3)
    assert result['errors']==0
    assert result['random_wrong_key_rows']==3
    assert result['unrelated_host_rows']==1
    assert result['metadata_tampering_detected_all'] is True
    assert validate_security(tmp_path/'audit',hosts[0].name,wrong_key_trials=3)
    audit_csv=tmp_path/'audit/per_case_results.csv'
    modified=rows(audit_csv)
    modified=[r for r in modified if r['case']!='unrelated_host']
    with audit_csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=modified[0].keys());writer.writeheader();writer.writerows(modified)
    assert not validate_security(tmp_path/'audit',hosts[0].name,wrong_key_trials=3)


def test_validator_fails_on_missing_gallery_or_channel(tmp_path):
    assert not validate_gallery(tmp_path,'girl.bmp','wm.png')
    assert not validate_channel(tmp_path,'girl.bmp')
    assert not validate_bss(tmp_path,'girl.bmp')
    assert not validate_security(tmp_path,'girl.bmp')
