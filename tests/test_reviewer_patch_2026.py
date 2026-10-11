"""Offline tests; pretrained WAM checkpoint not required."""
import numpy as np
import pytest
from pathlib import Path
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams, ProposalQHDWTHess, _givens4,
    _build_hpos_candidate, _extract_bit_from_candidate_block, FLAG_HPOS, HPOS_CANDIDATES)
from watermarklab.benchmark import _select_proposal_params_for_image
from watermarklab.experiments.neural_baselines import _select_methods, WAMAdapter


def test_q_rotations_preserve_orthogonality():
    for n in (4,8,16):
        G=_givens4(0,2,0.42,n)
        assert G.shape==(n,n)
        np.testing.assert_allclose(G.T@G,np.eye(n),atol=1e-12)


def test_h_coefficient_candidate_round_trip_across_sizes():
    rng=np.random.default_rng(42)
    for n in (4,8,16):
        block=rng.normal(15,1,(n,n))
        params=ProposalParams(block_size=n)
        for bit in (0,1):
            for _,pos in HPOS_CANDIDATES:
                cand=_build_hpos_candidate(block,bit,params,pos)
                assert cand.shape==(n,n)
                assert np.isfinite(cand).all()
                assert _extract_bit_from_candidate_block(cand,FLAG_HPOS,params,pos=pos)==bit


def test_full_params_unchanged_but_ablation_wins_over_saved_snapshot():
    base=ProposalParams.from_dict({'hpos_enabled':False,'ablation_name':'q4_only'})
    trained={'loaded':True,'path':'train-only.json','global_params':{
        'q4_tau':.62,'hpos_enabled':True,'q4_enabled':True,'arnold_enabled':True,
        'ablation_name':'full','structured_repetition_enabled':True}}
    selected,source=_select_proposal_params_for_image(Path('heldout.bmp'),base,trained)
    assert selected.hpos_enabled is False
    assert selected.ablation_name=='q4_only'
    assert selected.q4_tau==.62
    full,_=_select_proposal_params_for_image(Path('heldout.bmp'),ProposalParams(),trained)
    assert full.hpos_enabled is True
    assert full.ablation_name=='full'


def test_neural_selection_retires_chunkyseal():
    assert _select_methods('all')==['trustmark','wam']
    with pytest.raises(ValueError):_select_methods('chunkyseal')
    assert WAMAdapter.capacity==32


def test_size_options_are_explicit():
    for n in (4,8,16): assert ProposalQHDWTHess({'block_size':n}).params.block_size==n
    with pytest.raises(ValueError):ProposalQHDWTHess({'block_size':12})


def test_stratified_reviewer_subset_covers_all_seven_attack_families():
    from watermarklab.experiments.attack_selection import stratified_attack_subset
    from watermarklab.common.attack import default_attack_suite
    full=default_attack_suite(include_none=False,preset='reviewer_stress')
    chosen=stratified_attack_subset(full,7)
    assert len(chosen)==7
    assert len({a.group for a in chosen})==7
    assert len(stratified_attack_subset(full,0))==len(full)
