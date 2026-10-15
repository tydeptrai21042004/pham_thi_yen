"""Fast offline regression tests for full runner; never require WAM checkpoint."""
from pathlib import Path
import csv
import importlib.util
import json
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full_recovery',ROOT/'scripts/recover_missing_results.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_classical_validator_requires_exact_method_host_rows(tmp_path):
    path=tmp_path/'rows.csv'
    columns=('method_id','image','phase','attack','psnr','ssim','nc','ber','embed_time_sec','extract_time_sec')
    def write(rows):
        with path.open('w',newline='') as fp:
            out=csv.DictWriter(fp,fieldnames=columns);out.writeheader();out.writerows(rows)
    row=dict(method_id='proposal',image='airplane',phase='before_attack',attack='no_attack',
             psnr='49',ssim='0.99',nc='1',ber='0',embed_time_sec='1',extract_time_sec='2')
    attack=dict(row,phase='after_attack',attack='jpeg50',nc='0.9',ber='0.1')
    write([row,attack]);assert runner.phase_ok(path,'proposal','airplane',['jpeg50'])
    write([row,attack,attack]);assert not runner.phase_ok(path,'proposal','airplane',['jpeg50'])
    write([row,dict(attack,image='baboon')]);assert not runner.phase_ok(path,'proposal','airplane',['jpeg50'])
    write([row,dict(attack,ber='')]);assert not runner.phase_ok(path,'proposal','airplane',['jpeg50'])


def test_results_checkpoint_preserves_neural_cache_and_drops_old_native(tmp_path):
    output=tmp_path/'results';output.mkdir()
    cached=output/'neural_4096_forced/trustmark/airplane/trustmark/airplane_marked_cache.png'
    cached.parent.mkdir(parents=True);cached.write_bytes(b'cache')
    legacy=output/'native_neural/trustmark/test.csv';legacy.parent.mkdir(parents=True);legacy.write_text('wrong protocol')
    runner.snapshot(output)
    path=tmp_path/'results.zip'
    with zipfile.ZipFile(path) as z:
        assert any(x.endswith('airplane_marked_cache.png') for x in z.namelist())
        assert not any('/native_neural/' in x for x in z.namelist())
    new_output=tmp_path/'other'/'results';new_output.mkdir(parents=True)
    runner.restore(path,new_output)
    assert (new_output/'neural_4096_forced/trustmark/airplane/trustmark/airplane_marked_cache.png').read_bytes()==b'cache'
    assert not (new_output/'native_neural').exists()


def test_kaggle_uses_new_base_and_only_4096bit_neural():
    text=(ROOT/'scripts/kaggle_neural_one_cell.py').read_text()
    assert "STEP = 'all'" in text
    assert "SOURCE_NAME = 'pham_thi_yen-main(20261015-090635).zip'" in text
    assert "PATCH_NAME = 'WatermarkLab_ALL_EXPERIMENTS_FIXED_PATCH.zip'" in text
    assert "'--phase','neural'" not in text
    pipeline=(ROOT/'scripts/recover_missing_results.py').read_text()
    assert "'--force-full'" in pipeline
    assert 'all_core_sections_complete' in pipeline
