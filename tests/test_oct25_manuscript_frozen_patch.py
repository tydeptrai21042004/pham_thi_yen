"""Non-negotiable experiment invariants for the Oct-25 code-only patch."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermarklab.common.io_utils import load_watermark_binary, list_image_files
from watermarklab.common.metrics import nc
from watermarklab.experiments.splits import TRAIN_NAMES, ORIGINAL_TEST_NAMES, build_split
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams, Q4_GIVENS_PAIRS, _rank_hpos_by_energy, _hpos_group_candidates
)
from watermarklab.methods.contourlet_hess_su2016 import _qim_hmax, _extract_hmax

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_9_train_12_test_and_lena_excluded(tmp_path):
    split = build_split(ROOT / 'data/host', tmp_path / 'split.json')
    assert set(split['train_names']) == set(TRAIN_NAMES)
    assert set(split['test_names']) == set(ORIGINAL_TEST_NAMES)
    assert len(split['train']) == 9 and len(split['test']) == 12
    assert len(list_image_files(ROOT / 'data/host_train')) == 9
    assert {x.name for x in list_image_files(ROOT / 'data/host')} == set(ORIGINAL_TEST_NAMES)
    assert not any('lenna' in x.lower() or 'lena' in x.lower() for x in split['test_names'])


def test_both_binary_watermarks_are_full_64_by_64_no_resizing(tmp_path):
    w1 = load_watermark_binary(ROOT / 'data/watermark/wm.png')
    w2 = load_watermark_binary(ROOT / 'data/watermark/Cc.logo.circle.png')
    assert w1.shape == w2.shape == (64, 64)
    assert w1.size == w2.size == 4096
    assert set(np.unique(w1)).issubset({0,255})
    assert set(np.unique(w2)).issubset({0,255})
    assert not np.array_equal(w1, w2)
    path = tmp_path / 'wrong.png'
    Image.new('L', (32,32), 255).save(path)
    with pytest.raises(ValueError, match='exactly 64x64'):
        load_watermark_binary(path)


def test_primary_4x4_hessenberg_positions_are_fixed_as_written():
    for seed in range(10):
        block = np.random.default_rng(seed).normal(size=(4,4))
        assert _rank_hpos_by_energy(block) == (("h22", (1,1)), ("h21", (1,0)))
    # Source manuscript lists four one-indexed Givens pairs.
    assert Q4_GIVENS_PAIRS == ((0,2),(1,3),(0,1),(1,2))


def test_optimizer_fitness_function_unchanged_from_original_zip():
    path = ROOT / 'src/watermarklab/methods/proposal_qh_dwt_hess.py'
    module = ast.parse(path.read_text(encoding='utf8'))
    method = next(f for c in module.body if isinstance(c, ast.ClassDef) and c.name == 'ProposalQHDWTHess'
                  for f in c.body if isinstance(f, ast.FunctionDef) and f.name == '_evaluate_params_for_optimization')
    digest = hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()
    assert digest == '1d0c2e88ac1f538da03958e7339a85ef687c8ba642e7233df0c1933be5038b6a'


def test_nc_zero_denominator_never_reports_perfect_detection():
    z = np.zeros((64,64),dtype=np.uint8)
    assert np.isnan(nc(z,z))
    assert np.isnan(nc(z, np.ones_like(z)*255))


def test_su_paper_eq12_eq14_symbolic_quantization():
    for v in [-40.4, -1.0, 0, 14.9, 127.8]:
        for bit in (0,1):
            h = _qim_hmax(v,bit,16)
            assert _extract_hmax(h,16) == bit
