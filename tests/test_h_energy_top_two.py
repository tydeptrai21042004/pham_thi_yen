"""Dynamic top-two Hessenberg energy selection for 4x4, 8x8, and 16x16.

New keys encode absolute H positions while legacy 0/1 keys remain readable.
"""
import numpy as np
import pytest
from scipy.linalg import hessenberg
from watermarklab.methods.proposal_qh_dwt_hess import (
    FLAG_HPOS, FLAG_Q4, FLAG_SKIP, HPOS_CANDIDATES, ProposalParams,
    ProposalQHDWTHess, _rank_hpos_by_energy, _encode_hpos, _decode_hpos,
    _best_hpos_candidate,
)


@pytest.mark.parametrize('n', [4, 8, 16])
def test_top_two_are_global_legal_hessenberg_energy_maxima(n):
    for seed in range(4):
        block = np.random.default_rng(n * 10 + seed).normal(size=(n, n))
        H = hessenberg(block, calc_q=False)
        expected = sorted((-float(H[r,c])**2, r,c)
                          for r in range(n) for c in range(n) if r <= c+1)[:2]
        actual = _rank_hpos_by_energy(block)
        assert len(actual) == 2
        assert [p for _, p in actual] == [(r,c) for _,r,c in expected]
        for _, pos in actual:
            idx = _encode_hpos(pos, n)
            assert idx >= 2
            assert _decode_hpos(idx, n) == pos


def test_4x4_is_not_hard_coded_to_old_two_positions():
    # Hessenberg reduction of a diagonal matrix is itself, so its top two
    # largest energies are at h11 and h33, never legacy h21/h22.
    block = np.diag([99.0, 2.0, 85.0, 1.0])
    selected = _rank_hpos_by_energy(block)
    assert [p for _,p in selected] == [(0,0),(2,2)]
    assert all(p not in [legacy for _,legacy in HPOS_CANDIDATES]
               for _,p in selected)


@pytest.mark.parametrize('n', [4, 8, 16])
def test_h_candidate_is_one_of_energy_top_two(n):
    rng = np.random.default_rng(97+n)
    params = ProposalParams(block_size=n, q4_enabled=False)
    found_nonlegacy = False
    for _ in range(6):
        block = rng.normal(loc=100, scale=30, size=(n,n))
        chosen = _best_hpos_candidate(block, 1, params)
        top_two = _rank_hpos_by_energy(block)
        assert chosen['pos'] in [p for _,p in top_two]
        assert _decode_hpos(chosen['hpos_idx'], n) == chosen['pos']
        assert chosen['hpos_idx'] >= 2
        found_nonlegacy |= chosen['pos'] not in [p for _,p in HPOS_CANDIDATES]
    assert found_nonlegacy, 'position must not be confined to h21/h22'


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
    assert [p for _,p in first] == [(0,0),(0,1)]
    assert first == _rank_hpos_by_energy(block.copy())


@pytest.mark.parametrize('n', [4, 8, 16])
def test_end_to_end_uses_absolute_H_locations_and_records_energy_policy(n):
    rng = np.random.default_rng(n*11 + 7)
    host = rng.integers(0,256, size=(128,128,3), dtype=np.uint8)
    watermark = (rng.integers(0,2,size=(16,16),dtype=np.uint8)*255).astype(np.uint8)
    params = ProposalParams(block_size=n, repeat=1, q4_enabled=False,
                            min_survival_rate=0.25, max_h_cand_mse=25.0)
    model = ProposalQHDWTHess(params=params)
    image, key = model.embed(host, watermark)
    extracted = model.extract(image, key)
    assert image.shape == host.shape
    assert extracted.shape == watermark.shape
    assert key.payload_meta['h_candidate_policy'] == 'per_block_top_two_H_energy'
    assert key.payload_meta['h_energy_top2_count'] == 2
    assert key.payload_meta['h_position_encoding'].startswith('2 + 16')
    active = [i for f,i in zip(key.flags,key.hpos_list) if f == FLAG_HPOS]
    assert active, 'H-only run should embed at least one bit'
    assert all(i >= 2 for i in active)
    assert any(_decode_hpos(i,n) not in [p for _,p in HPOS_CANDIDATES] for i in active)
    assert not key.embedding_errors
