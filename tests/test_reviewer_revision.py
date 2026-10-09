from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from watermarklab.common.attack import reviewer_stress_attack_suite, AttackConfig, apply_attack
from watermarklab.experiments.splits import build_split
from watermarklab.experiments.review_audit import binary_bit_metrics, audit
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams, ProposalQHDWTHess, _candidate_attack_score,
    _extract_bit_from_candidate_block, FLAG_HPOS,
    prepare_binary_watermark_payload_from_array,
)

ROOT=Path(__file__).resolve().parents[1]


def test_exact_invalid_candidate_cannot_pass_bss(monkeypatch):
    from watermarklab.methods import proposal_qh_dwt_hess as p
    def fake_extract(block, mode, params, pos=None):
        return 0 if np.array_equal(block,np.ones((4,4))) else 1
    monkeypatch.setattr(p,"_extract_bit_from_candidate_block",fake_extract)
    base=np.ones((4,4))
    result=_candidate_attack_score(base,base,FLAG_HPOS,1,ProposalParams(),pos=(1,1))
    assert result[2]==0.0
    assert result[-1] is False or not result[-1]


def test_reviewers_exact_stronger_attacks_available():
    suite=reviewer_stress_attack_suite()
    lookup={x.name:x for x in suite}
    for name in ("jpeg_q40","jpeg_q30","gaussian_var_0p05","speckle_var_0p03","rotation_30deg"):
        assert name in lookup
    assert lookup["gaussian_var_0p05"].params["variance"]==0.05
    assert lookup["speckle_var_0p03"].params["variance"]==0.03


def test_jpeg2000_codec_failure_never_substitutes(monkeypatch):
    from watermarklab.common import attack as a
    def boom(*args,**kwargs):
        raise OSError("no codec")
    monkeypatch.setattr(a.Image.Image,"save",boom)
    with pytest.raises(RuntimeError,match="JPEG2000 attack unavailable"):
        a.jpeg2000(np.zeros((8,8,3),dtype=np.uint8))


def test_deterministic_disjoint_train_test_split(tmp_path):
    output=tmp_path/"split.json"
    a=build_split(ROOT/"data/host",output,seed=3101)
    b=build_split(ROOT/"data/host",output,seed=3101)
    assert a==b
    assert not set(a["train"]) & set(a["test"])
    assert output.exists()
    assert len(a["image_hashes"]) >=2


def test_binary_metrics_report_erasures_separately():
    m=binary_bit_metrics(np.array([1,0,0,1]),np.array([1,1,0,0]),np.array([1,0,1,0]))
    assert m["coverage"]==0.5
    assert m["payload_ber_valid_only"]==0.0
    assert m["payload_ber_all_positions"]==0.5


def test_payload_length_direct_binary_is_larger():
    wm=np.zeros((8,8),dtype=np.uint8)
    a=prepare_binary_watermark_payload_from_array(wm,ProposalParams(watermark_dwt_enabled=True))[1]
    b=prepare_binary_watermark_payload_from_array(wm,ProposalParams(watermark_dwt_enabled=False))[1]
    assert a.size==16 and b.size==64


def test_missing_votes_have_explicit_erasures():
    host=np.random.default_rng(3).integers(0,256,(64,64,3),dtype=np.uint8)
    wm=np.random.default_rng(4).integers(0,2,(8,8),dtype=np.uint8)*255
    obj=ProposalQHDWTHess(ProposalParams(repeat=1))
    watermarked,key=obj.embed(host,wm)
    diag=obj.extract_payload_diagnostics(watermarked,key)
    assert diag.vote_counts.size==16
    assert diag.erasures >=0
    assert diag.coverage <=1
    assert len(key.selected_bss)==len(key.flags)
    assert len(key.selected_mse)==len(key.flags)


def test_negative_control_audit_produces_independent_rows(tmp_path):
    result=audit(ROOT/"data/host",ROOT/"data/watermark/wm.png",tmp_path,max_images=2,host_size=64,watermark_size=8,repeat=1,max_attacks=0)
    assert result["measurements"]>=4
    assert (tmp_path/"manifest.json").exists()
    rows=(tmp_path/"per_case_results.csv").read_text()
    assert "original_unmarked" in rows
    assert "marked_clean" in rows
