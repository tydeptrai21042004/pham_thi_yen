"""Fast production-contract tests for full two-watermark Kaggle reviewer workflow.

These checks intentionally do not claim the long benchmark has run. They test
orchestration, error detection, reproducibility and checkpoint safety.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from watermarklab.common.attack import default_attack_suite
from watermarklab.methods import BASELINE_METHOD_IDS, DEFAULT_METHOD_IDS
from watermarklab.benchmark import PROPOSAL_ABLATION_VARIANTS
from watermarklab.experiments.splits import build_split
from watermarklab.experiments.second_watermark import prepare

spec=importlib.util.spec_from_file_location('recover_tested',ROOT/'scripts/recover_missing_results.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_exact_13_methods_109_attack_9_ablations():
    assert len(BASELINE_METHOD_IDS)==12
    assert len(DEFAULT_METHOD_IDS)==13
    assert len(set(DEFAULT_METHOD_IDS))==13
    assert DEFAULT_METHOD_IDS[-1]=='proposal'
    assert len(default_attack_suite(False,'full'))==109
    assert len(PROPOSAL_ABLATION_VARIANTS)==9
    assert len(default_attack_suite(False,'reviewer_stress'))==29


def test_severe_attacks_and_geometric_attacks_present():
    attacks={a.name for a in default_attack_suite(False,'full')}
    assert {'jpeg_q30','jpeg_q40','gaussian_var_0p05','speckle_var_0p03'}<=attacks
    assert any('rotation' in x for x in attacks)
    assert any('resize' in x for x in attacks)
    assert any('crop' in x for x in attacks)


def test_split_9_train_12_test_disjoint(tmp_path):
    manifest=build_split(ROOT/'data/host',tmp_path/'split.json',train_dir=ROOT/'data/host_train',create_dirs=True)
    assert len(manifest['train_names'])==9
    assert len(manifest['test_names'])==12
    assert not (set(manifest['train_names']) & set(manifest['test_names']))
    assert (tmp_path/'split.json').exists()


def test_second_watermark_alpha_binary_independent(tmp_path):
    input_file=ROOT/'data/watermark/Cc.logo.circle.png'
    out=tmp_path/'second.png'
    path=prepare(input_file,out)
    from PIL import Image
    with Image.open(path) as f:
        pixels=np.asarray(f)
    assert pixels.shape==(64,64)
    assert set(np.unique(pixels))=={0,255}
    meta=json.loads((tmp_path/'secondary_watermark_preprocessing.json').read_text())
    assert meta['original_sha256']==mod.sha(input_file)
    assert meta['preprocessed_sha256']==mod.sha(path)
    assert meta['protocol'].startswith('separate cohort')


def row(phase='before_attack',attack='no_attack',image='host',method_id='proposal',nc='0.9'):
    return dict(phase=phase, attack=attack,image=image,method_id=method_id,
                psnr='54',ssim='0.99',nc=nc,ber='0.02',embed_time_sec='0.3',extract_time_sec='0.2')


def csvfile(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def test_single_host_phase_grid_strict(tmp_path):
    file=tmp_path/'grid.csv'
    rows=[row(),row(phase='after_attack',attack='jpeg_q30'),row(phase='after_attack',attack='rotation_10deg')]
    csvfile(file,rows)
    assert mod.phase_ok(file,'proposal','host',['jpeg_q30','rotation_10deg'])
    for broken in (rows[:-1],rows+[rows[0]],
                   [rows[0],rows[1],row(phase='after_attack',attack='other')],
                   [rows[0],{**rows[1], 'nc':'nan'},rows[2]],
                   [rows[0],{**rows[1], 'method_id':'wrong'},rows[2]],
                   [rows[0],{**rows[1], 'image':'test_other'},rows[2]]):
        csvfile(file,broken)
        assert not mod.phase_ok(file,'proposal','host',['jpeg_q30','rotation_10deg'])


def test_checkpoint_roundtrip_excludes_neural_and_single_host(tmp_path):
    out=tmp_path/'some_results';out.mkdir()
    (out/'classical').mkdir()
    (out/'classical'/'proposal_optimized_params.progress.json').write_text('{"per_image":{"a":1}}')
    (out/'native_neural').mkdir();(out/'native_neural'/'native.csv').write_text('forbidden')
    (out/'neural_4096').mkdir();(out/'neural_4096'/'old.csv').write_text('forbidden')
    (out/'single_host').mkdir();(out/'single_host'/'x.bmp').write_bytes(b'fake')
    (out/'watermark_2_Cc.zip').write_bytes(b'secondary-archive-dummy')
    mod.snapshot(out)
    archive=tmp_path/'some_results.zip'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert any('proposal_optimized_params.progress.json' in x for x in names)
        assert any('watermark_2_Cc.zip' in x for x in names)
        assert not any('/native_neural/' in x or '/neural_4096/' in x or '/single_host/' in x for x in names)
    resumed=tmp_path/'some_results'
    (resumed/'classical'/'proposal_optimized_params.progress.json').unlink()
    mod.restore(archive,resumed)
    assert (resumed/'classical'/'proposal_optimized_params.progress.json').exists()


def test_zip_slip_is_rejected(tmp_path):
    zpath=tmp_path/'sample.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        z.writestr('sample/../escape.txt','bad')
    with pytest.raises(ValueError):mod.restore(zpath,tmp_path/'sample')
    assert not (tmp_path/'escape.txt').exists()


def test_original_proposal_byte_integrity():
    original=ROOT/'src/watermarklab/methods/proposal_qh_dwt_hess.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='10aaca9b328e14515c20d6684e0b96f8b9a527d3bdbb7acd442c953e8405716d'


def test_cli_explicitly_supports_1_firefly_2_generations_and_neural_opt_in():
    exe=ROOT/'scripts/recover_missing_results.py'
    result=subprocess.run([sys.executable,str(exe),'--help'],text=True,capture_output=True)
    assert result.returncode==0
    for arg in ('--proposal-fireflies','--proposal-generations','--include-neural4096','--extended-sensitivity'):
        assert arg in result.stdout
    bad=subprocess.run([sys.executable,str(exe),'--step','verify','--out','/tmp/no-run',
                       '--proposal-fireflies','0'],text=True,capture_output=True)
    assert bad.returncode!=0
    assert 'positive integers' in bad.stderr
    native=subprocess.run([sys.executable,str(exe),'--step','verify','--out','/tmp/no-run',
                           '--include-neural'],text=True,capture_output=True)
    assert native.returncode!=0
    assert 'native' in native.stderr


def test_launcher_clones_github_and_never_calls_native_neural():
    text=(ROOT/'KAGGLE_GITHUB_ALL_EXPERIMENTS.py').read_text()
    assert "'git', 'clone'" in text
    assert "'--step', 'all'" in text
    assert "FIREFLIES = 1" in text and "GENERATIONS = 2" in text
    assert "RUN_4096_EXTENSIONS = True" in text
    assert "--include-neural4096" in text
    assert "--phase', 'neural'" not in text
    compile(text,'KAGGLE_GITHUB_ALL_EXPERIMENTS.py','exec')
