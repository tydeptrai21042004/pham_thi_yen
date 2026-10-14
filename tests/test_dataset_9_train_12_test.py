"""Regression checks for the user-specified 9-new-train / 12-original-test protocol."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from watermarklab.experiments.splits import (
    ORIGINAL_TEST_NAMES, PREVIOUS_TUNING_NAMES, TRAIN_NAMES, build_split,
)

ROOT = Path(__file__).resolve().parents[1]


def test_split_assignments_are_exact_and_no_old_train_remains_training(tmp_path):
    spec = build_split(ROOT/'data/host', tmp_path/'split.json', create_dirs=True)
    assert spec['schema'] == 'watermarklab-fixed-21-9-12-v3'
    assert spec['assignment'] == 'explicit_filenames_no_random_split'
    assert set(spec['train_names']) == set(TRAIN_NAMES)
    assert set(spec['test_names']) == set(ORIGINAL_TEST_NAMES)
    assert set(PREVIOUS_TUNING_NAMES) <= set(spec['test_names'])
    assert set(PREVIOUS_TUNING_NAMES).isdisjoint(spec['train_names'])
    assert set(spec['image_hashes']) == set(spec['train_names'] + spec['test_names'])
    assert len(spec['image_hashes']) == 21
    assert len(set(spec['decoded_rgb_sha256'].values())) == 21
    for label, names in [('train', spec['train_names']), ('test', spec['test_names'])]:
        actual = sorted(p.name for p in Path(spec[label+'_dir']).glob('*.bmp'))
        assert actual == sorted(names)
        for f in Path(spec[label+'_dir']).glob('*.bmp'):
            with Image.open(f) as im:
                assert im.format == 'BMP' and im.size == (512,512) and im.mode == 'RGB'


def test_all_nine_uploaded_files_have_real_bmp_signature():
    for name in TRAIN_NAMES:
        file = ROOT/'data/host_train'/name
        assert file.read_bytes()[:2] == b'BM', name
        with Image.open(file) as im:
            assert im.format == 'BMP' and im.mode == 'RGB' and im.size == (512,512)


def test_split_manifest_is_reproducible_across_seed_values(tmp_path):
    first = build_split(ROOT/'data/host', tmp_path/'a.json', seed=3101)
    second = build_split(ROOT/'data/host', tmp_path/'b.json', seed=1)
    assert first['train_names'] == second['train_names']
    assert first['test_names'] == second['test_names']
    assert first['image_hashes'] == second['image_hashes']
    assert first['seed'] != second['seed']  # archival only


def test_explicit_ratio_cannot_be_silently_overridden(tmp_path):
    with pytest.raises(ValueError, match='train_fraction'):
        build_split(ROOT/'data/host', tmp_path/'split.json', train_fraction=.25)


def test_missing_training_image_causes_error(tmp_path):
    training = tmp_path/'train'
    training.mkdir()
    for name in TRAIN_NAMES[1:]:
        shutil.copy2(ROOT/'data/host_train'/name, training/name)
    with pytest.raises(ValueError, match='missing'):
        build_split(ROOT/'data/host', tmp_path/'out.json', train_dir=training)


def test_train_test_duplicate_pixels_are_rejected(tmp_path):
    training = tmp_path/'train'
    training.mkdir()
    for name in TRAIN_NAMES:
        shutil.copy2(ROOT/'data/host_train'/name, training/name)
    shutil.copy2(ROOT/'data/host/Girl.bmp', training/TRAIN_NAMES[0])
    with pytest.raises(ValueError, match='Duplicate image pixels'):
        build_split(ROOT/'data/host', tmp_path/'out.json', train_dir=training)


def test_repeated_split_generation_cleans_stale_files(tmp_path):
    spec = build_split(ROOT/'data/host', tmp_path/'s.json', create_dirs=True)
    for label in ('train','test'):
        (Path(spec[label+'_dir'])/'leftover.bmp').write_bytes(b'not a BMP')
    refreshed = build_split(ROOT/'data/host', tmp_path/'s.json', create_dirs=True)
    for label in ('train','test'):
        assert not (Path(refreshed[label+'_dir'])/'leftover.bmp').exists()


def test_split_cli_runs_and_writes_proper_manifest(tmp_path):
    cmd = [sys.executable, '-m', 'watermarklab.experiments.splits', '--host-dir',
           str(ROOT/'data/host'), '--output', str(tmp_path/'run/split.json'), '--create-dirs']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    doc = json.loads((tmp_path/'run/split.json').read_text())
    assert len(doc['train_names']) == 9 and len(doc['test_names']) == 12


def test_train_images_are_decodable_by_actual_project_host_loader():
    from watermarklab.common.io_utils import load_host_rgb
    for name in TRAIN_NAMES:
        img = load_host_rgb(ROOT/'data/host_train'/name)
        assert img.shape == (512,512,3)
        assert str(img.dtype) == 'uint8'
