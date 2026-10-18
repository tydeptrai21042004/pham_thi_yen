"""Offline guardrails for checkpoint-only resumption; no pretrained weights needed."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_missing_results as r


def _fake_neural_job(tmp_path, method='trustmark'):
    folder=tmp_path/'neural_4096_forced'/method/'Girl'
    folder.mkdir(parents=True)
    csvfile=folder/'per_image_attack_results.csv'
    manifest=folder/'manifest.json'
    base={'method_id':method+'_4096_spatial','image':'Girl','watermark':'wm.png',
          'source_payload_bits':'4096','psnr':'55.0','ssim':'0.998',
          'bit_coverage':'0.5','ber_erasure_as_error':'0.5'}
    rows=[dict(base,phase='before_attack',attack='no_attack'),
          dict(base,phase='control',attack='original_unmarked',psnr='inf'),
          dict(base,phase='after_attack',attack='jpeg70'),
          dict(base,phase='after_attack',attack='crop')]
    with csvfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest.write_text(json.dumps({'complete_grid':True,
        'watermark_sha256':'expectedhash','expected_attacked_rows':2,
        'actual_attacked_rows':2}))
    return csvfile,manifest,rows


def _validate(csvfile,manifest):
    return r.valid_neural_host(csvfile,manifest,method='trustmark',host='Girl',
                               watermark='wm.png',watermark_sha='expectedhash',
                               attacks=['jpeg70','crop'])


def test_zero_mse_control_can_have_infinite_psnr(tmp_path):
    csvfile,manifest,rows=_fake_neural_job(tmp_path)
    assert _validate(csvfile,manifest)
    assert r.neural_metrics_valid(rows[1])
    assert math.isinf(float(rows[1]['psnr']))


def test_missing_metrics_and_noncontrol_infinity_rejected(tmp_path):
    csvfile,manifest,rows=_fake_neural_job(tmp_path)
    for bad in [dict(rows[2],psnr='inf'),dict(rows[2],ssim='nan'),
                dict(rows[2],bit_coverage=''),dict(rows[2],ber_erasure_as_error='-inf'),
                dict(rows[1],psnr='-inf')]:
        assert not r.neural_metrics_valid(bad)
    rows[2]['bit_coverage']=''
    with csvfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert not _validate(csvfile,manifest)


def test_wrong_cohort_method_or_duplicate_attack_rejected(tmp_path):
    csvfile,manifest,rows=_fake_neural_job(tmp_path)
    assert _validate(csvfile,manifest)
    rows[2]['watermark']='Cc.logo.circle_alpha_binary.png'
    with csvfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert not _validate(csvfile,manifest)
    rows[2]['watermark']='wm.png'
    rows[3]['attack']='jpeg70'
    with csvfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert not _validate(csvfile,manifest)


def test_neural_cache_and_completed_results_survive_checkpoint(tmp_path):
    out=tmp_path/'watermark_2_Cc'
    child=out/'neural_4096_forced/trustmark/Girl/trustmark'
    child.mkdir(parents=True)
    (child/'Girl_marked_cache.png').write_bytes(b'placeholder-cached-image')
    (child/'Girl_results.csv').write_text('phase,psnr\ncontrol,inf\n')
    (child/'Girl_config.json').write_text('{"persist":true}')
    (out/'recovery_status.json').write_text('{"jobs":{}}')
    r.snapshot(out)
    expected=tmp_path/'watermark_2_Cc.zip'
    target=tmp_path/'elsewhere'/'watermark_2_Cc'
    r.restore(expected,target)
    assert (target/'neural_4096_forced/trustmark/Girl/trustmark/Girl_marked_cache.png').read_bytes()==b'placeholder-cached-image'
    assert (target/'neural_4096_forced/trustmark/Girl/trustmark/Girl_config.json').is_file()


def test_benchmark_guard_stops_accidental_full_rerun(tmp_path):
    from kaggle_resume_both_2026 import validate_old_benchmark_contract
    out=tmp_path/'out'
    try:
        validate_old_benchmark_contract(out)
    except ValueError as e:
        assert 'expected 156 heldout host contracts' in str(e)
    else:
        raise AssertionError('Missing evidence was incorrectly accepted')
