"""Fixed 4x4 h22/h21 proposal with separate larger-block size ablations."""

import numpy as np
import pytest
from scipy.linalg import hessenberg
from watermarklab.methods.proposal_qh_dwt_hess import (
    FLAG_HPOS, FLAG_Q4, FLAG_SKIP, HPOS_CANDIDATES, ProposalParams,
    ProposalQHDWTHess, _rank_hpos_by_energy, _encode_hpos, _decode_hpos,
    _hpos_group_candidates, _select_candidate_for_ablation,
)


@pytest.mark.parametrize('n', [4, 8, 16])
def test_groupwise_diagonal_and_subdiagonal_energy_maxima(n):
    for seed in range(4):
        block = np.random.default_rng(n * 10 + seed).normal(size=(n, n))
        actual = _rank_hpos_by_energy(block)
        assert len(actual) == 2
        if n == 4:
            assert actual == (("h22", (1,1)), ("h21", (1,0)))
        else:
            H = hessenberg(block, calc_q=False)
            diagonal_best = min((-float(H[r,r])**2, r, r) for r in range(n))
            subdiagonal_best = min((-float(H[r,r-1])**2, r, r-1) for r in range(1,n))
            assert [p for _,p in actual] == [(diagonal_best[1], diagonal_best[2]),
                                             (subdiagonal_best[1], subdiagonal_best[2])]
        assert actual[0][1][0] == actual[0][1][1]
        assert actual[1][1][0] == actual[1][1][1] + 1
        for _, pos in actual:
            idx = _encode_hpos(pos, n)
            assert idx >= 2
            assert _decode_hpos(idx, n) == pos


def test_global_top_two_are_not_used():
    # Even if other diagonal elements carry more energy, the manuscript
    # fixes the 4x4 candidates at h22 and h21.
    block = np.diag([99.0, 2.0, 85.0, 1.0])
    selected = _rank_hpos_by_energy(block)
    assert [p for _,p in selected] == [(1, 1), (1, 0)]
    assert selected[1][1] != (2, 2)


def test_paper_like_h21_h22_is_possible():
    # The exact paper pair is hard-coded in the primary 4x4 algorithm.
    block = np.array([[0., 1., 2., 3.],
                      [20., 100., 4., 5.],
                      [0., 2., 3., 6.],
                      [0., 0., 1., 4.]])
    selected = _rank_hpos_by_energy(block)
    assert selected[0][1][0] == selected[0][1][1]
    assert selected[1][1][0] == selected[1][1][1] + 1


@pytest.mark.parametrize('n', [4, 8, 16])
def test_two_group_maxima_are_two_independent_candidates(n):
    rng = np.random.default_rng(97+n)
    params = ProposalParams(block_size=n, q4_enabled=False)
    for _ in range(6):
        block = rng.normal(loc=100, scale=30, size=(n,n))
        candidates = _hpos_group_candidates(block, 1, params)
        groupwise_two = _rank_hpos_by_energy(block)
        assert len(candidates) == 2
        assert [c['h_group'] for c in candidates] == ['main_diagonal', 'first_subdiagonal']
        assert [c['pos'] for c in candidates] == [p for _,p in groupwise_two]
        assert all(_decode_hpos(c['hpos_idx'], n) == c['pos'] for c in candidates)
        assert all(c['hpos_idx'] >= 2 for c in candidates)
        assert all(c['flag'] == FLAG_HPOS for c in candidates)
        assert all(c['block'].shape == (n,n) for c in candidates)


def test_final_selector_does_not_collapse_h_groups_before_q_competition():
    params = ProposalParams(h_selection_policy="independent", min_survival_rate=0.0, bss_weight=1.0, mse_weight=0.0,
                            max_q_cand_mse=10.0, max_h_cand_mse=10.0)
    base = [
        {'flag': FLAG_Q4, 'h_group': 'Q', 'score': 0.5, 'mse': 0.1, 'raw': 1.0, 'ok': True},
        {'flag': FLAG_HPOS, 'h_group': 'main_diagonal', 'score': 0.5, 'mse': 0.1, 'raw': 1.0, 'ok': True},
        {'flag': FLAG_HPOS, 'h_group': 'first_subdiagonal', 'score': 0.5, 'mse': 0.1, 'raw': 1.0, 'ok': True},
    ]
    for group in ['Q', 'main_diagonal', 'first_subdiagonal']:
        cands = [dict(x, score=(1.0 if x['h_group'] == group else 0.5)) for x in base]
        assert _select_candidate_for_ablation(cands, params)['h_group'] == group


@pytest.mark.parametrize('n', [4, 8, 16])
def test_invalid_absolute_positions_rejected_but_legacy_keys_decodable(n):
    with pytest.raises(ValueError):
        _decode_hpos(2+16*(n+1), n)
    with pytest.raises(ValueError):
        _decode_hpos(2+16*(n-1), n)  # structurally zero location
    with pytest.raises(ValueError):
        _encode_hpos((n-1,0), n)
    assert _decode_hpos(0, n) == (1,0)
    assert _decode_hpos(1, n) == (1,1)


def test_tie_breaker_prefers_top_left_and_pre_watermark_is_identical():
    block = np.zeros((4,4),dtype=float)
    first = _rank_hpos_by_energy(block)
    assert [p for _,p in first] == [(1,1),(1,0)]
    assert first == _rank_hpos_by_energy(block.copy())


@pytest.mark.parametrize('n', [4, 8, 16])
def test_end_to_end_uses_absolute_H_locations_and_records_energy_policy(n):
    rng = np.random.default_rng(n*11 + 7)
    host = rng.integers(0,256, size=(128,128,3), dtype=np.uint8)
    watermark = (rng.integers(0,2,size=(16,16),dtype=np.uint8)*255).astype(np.uint8)
    params = ProposalParams(h_selection_policy="independent", block_size=n, repeat=1, q4_enabled=False,
                            min_survival_rate=0.25, max_h_cand_mse=25.0)
    model = ProposalQHDWTHess(params=params)
    image, key = model.embed(host, watermark)
    extracted = model.extract(image, key)
    assert image.shape == host.shape
    assert extracted.shape == watermark.shape
    assert key.payload_meta['h_candidate_policy'] == (
        'fixed_h22_h21_independent_H_competition' if n == 4
        else 'size_ablation_group_energy_independent_H')
    if n == 4:
        assert key.payload_meta['h_candidates'] == ['h22', 'h21']
    assert key.payload_meta['h_grouped_candidate_count'] == 2
    assert key.payload_meta['candidate_competition'] == 'Q_vs_independent_H_main_diagonal_vs_independent_H_first_subdiagonal'
    assert sum(key.payload_meta['h_group_selected'].values()) == key.payload_meta['hpos_used']
    assert key.payload_meta['h_position_encoding'].startswith('2 + 16')
    active = [i for f,i in zip(key.flags,key.hpos_list) if f == FLAG_HPOS]
    assert active, 'H-only run should embed at least one bit'
    assert all(i >= 2 for i in active)
    assert all((lambda rc: rc[0] in (rc[1], rc[1]+1))(_decode_hpos(i,n)) for i in active)
    if n == 4:
        assert all(_decode_hpos(i,n) in ((1,1),(1,0)) for i in active)
    assert not key.embedding_errors


def test_full_embed_exposes_three_separate_candidates_to_selector(monkeypatch):
    import watermarklab.methods.proposal_qh_dwt_hess as proposal_module

    observed = []
    original = proposal_module._select_candidate_for_ablation

    def record_candidates(cands, params):
        observed.append(tuple('Q' if c['flag'] == FLAG_Q4 else c['h_group'] for c in cands))
        return original(cands, params)

    monkeypatch.setattr(proposal_module, '_select_candidate_for_ablation', record_candidates)
    rng = np.random.default_rng(303)
    host = rng.integers(0, 256, size=(64,64,3), dtype=np.uint8)
    watermark = rng.integers(0,2,size=(8,8),dtype=np.uint8)*255
    model = ProposalQHDWTHess(params=ProposalParams(h_selection_policy="independent", block_size=4, repeat=1,
        q4_enabled=True, hpos_enabled=True, min_survival_rate=0.0,
        max_h_cand_mse=25.0, max_q_cand_mse=25.0))
    image, key = model.embed(host, watermark)
    assert image.shape == host.shape
    assert observed
    assert all(set(groups) == {'Q','main_diagonal','first_subdiagonal'}
               and len(groups) == 3 for groups in observed)
    assert key.payload_meta['candidate_competition'].startswith('Q_vs_independent_H')
