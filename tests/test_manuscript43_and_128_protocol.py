from pathlib import Path
import numpy as np
from watermarklab.common.attack import (default_attack_suite, apply_attack, manuscript43_family,
 MANUSCRIPT43_FAMILIES)
from watermarklab.methods.proposal_qh_dwt_hess import (ProposalParams, prepare_binary_watermark_payload_from_array)
from watermarklab.common.io_utils import load_watermark_binary

ROOT=Path(__file__).resolve().parents[1]

def test_43_attacks_plus_separate_clean_and_published_group_counts():
    cases=default_attack_suite(False,'manuscript43')
    with_clean=default_attack_suite(True,'manuscript43')
    assert len(cases)==43 and len(with_clean)==44
    assert with_clean[0].name=='no_attack'
    assert [c.name for c in cases]==[n for names in MANUSCRIPT43_FAMILIES.values() for n in names]
    assert len(set(c.name for c in cases))==43
    assert {k:len(v) for k,v in MANUSCRIPT43_FAMILIES.items()}=={
        'compression':8,'noise':8,'filtering':6,'geometry':10,'light_colour':5,'combined':6}
    assert all(manuscript43_family(c.name) in MANUSCRIPT43_FAMILIES for c in cases)

def test_43_attacks_shapes_and_determinism_on_rgb_uint8():
    img=np.random.default_rng(1020).integers(0,256,size=(64,64,3),dtype=np.uint8)
    for cfg in default_attack_suite(False,'manuscript43'):
        x=apply_attack(img,cfg)
        assert x.shape==img.shape and x.dtype==np.uint8,cfg.name
        if any(v in cfg.name for v in ('gaussian','speckle','salt_pepper')):
            assert np.array_equal(x,apply_attack(img,cfg)),cfg.name

def test_payload_changes_from_1024_to_4096_bits_without_host_resizing():
    water=ROOT/'data/watermark/wm.png'
    params=ProposalParams()
    for size,expected in ((64,1024),(128,4096)):
        img=load_watermark_binary(water,size=size)
        _wm,bits,meta=prepare_binary_watermark_payload_from_array(img,params)
        assert img.shape==(size,size)
        assert np.asarray(bits).size==expected
        assert set(np.unique(bits)).issubset({0,1})
