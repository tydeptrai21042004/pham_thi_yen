"""Fast, offline operational regression tests for the reviewer workstreams.

No model checkpoint, network access, or modification to the proposed method is
needed. Synthetic adapter results are engineering tests, NEVER paper results.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from watermarklab.experiments import neural_4096 as neural
from watermarklab.experiments.bss_validation import run as run_bss
from watermarklab.experiments.review_audit import audit, binary_bit_metrics

ROOT = Path(__file__).resolve().parents[1]


class LocalPixelCodec:
    """A reversible toy adapter; does NOT represent TrustMark or WAM."""
    def __init__(self, capacity=61, *, bad_decode=(), fail_embed=False):
        self.capacity = capacity
        self.bad_decode = set(bad_decode)
        self.fail_embed = fail_embed
        self.decode_calls = 0
        self.embed_calls = 0

    def embed(self, patch, message):
        self.embed_calls += 1
        if self.fail_embed:
            raise RuntimeError('synthetic embed failure')
        assert message.shape == (self.capacity,)
        marked = patch.copy()
        marked.reshape(-1, 3)[:self.capacity, 0] = message * 255
        return marked

    def detect(self, patch):
        i = self.decode_calls
        self.decode_calls += 1
        if i in self.bad_decode:
            return None, False
        return (patch.reshape(-1, 3)[:self.capacity, 0] >= 128).astype('uint8'), True


def _inputs(tmp_path, *, hosts=1):
    folder = tmp_path / 'hosts'
    folder.mkdir()
    for i in range(hosts):
        # Nonzero host exposes unmarked controls as actual controls.
        Image.fromarray(np.full((512, 512, 3), 37 + i * 20, dtype=np.uint8)).save(folder / f'host{i}.png')
    wm = tmp_path / 'source_watermark.png'
    yy, xx = np.indices((64, 64))
    bits = ((xx * 13 + yy * 17 + xx * yy) % 7 < 3).astype('uint8')
    Image.fromarray(bits * 255).save(wm)
    return folder, wm, bits.ravel()


def _factory(codec):
    return lambda name: neural.SpatialChunk4096(codec, name=name, cols=8, resize_to=0)


def _rows(path):
    with path.open(encoding='utf-8', newline='') as fp:
        return list(csv.DictReader(fp))


@pytest.mark.parametrize('capacity,expected_chunks,last_data_bits', [(61, 68, 9), (32, 128, 32), (4096, 1, 4096)])
def test_payload_chunk_boundaries_exact_4096(capacity, expected_chunks, last_data_bits):
    bits = np.arange(4096, dtype=np.uint16) % 2
    chunks = neural.chunk_message(bits, capacity)
    assert len(chunks) == expected_chunks
    assert all(len(x) == capacity for x in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks)[:4096], bits)
    assert np.count_nonzero(chunks[-1][last_data_bits:]) == 0


@pytest.mark.parametrize('bad', [np.zeros(4095, dtype='uint8'), np.full(4096, 2, dtype='uint8'), np.array([], dtype='uint8')])
def test_payload_rejects_invalid_binary_inputs(bad):
    with pytest.raises(ValueError, match='4096 binary'):
        neural.chunk_message(bad, 32)


@pytest.mark.parametrize('capacity', [0, -1])
def test_payload_rejects_invalid_native_capacity(capacity):
    with pytest.raises(ValueError, match='native capacity'):
        neural.chunk_message(np.zeros(4096, dtype='uint8'), capacity)


def test_watermark_requires_original_64x64_and_both_bit_values(tmp_path):
    good_dir, good, bits = _inputs(tmp_path)
    np.testing.assert_array_equal(neural.watermark_bits(good), bits)
    for name, image in [('wrong.png', np.zeros((32, 32), dtype='uint8')),
                        ('constant.png', np.zeros((64, 64), dtype='uint8'))]:
        path = tmp_path / name
        Image.fromarray(image).save(path)
        with pytest.raises(ValueError):
            neural.watermark_bits(path)


@pytest.mark.parametrize('capacity', [61, 32])
def test_spatial_regions_partition_without_overlap(capacity):
    count = (4096 + capacity - 1) // capacity
    rectangles = neural.deterministic_regions(512, 512, count, cols=8)
    coverage = np.zeros((512, 512), dtype='uint8')
    for y0, y1, x0, x1 in rectangles:
        assert 0 <= y0 < y1 <= 512 and 0 <= x0 < x1 <= 512
        coverage[y0:y1, x0:x1] += 1
    assert len(rectangles) == count
    assert coverage.max() == 1
    assert coverage.sum() > 0
    with pytest.raises(ValueError, match='Host too small'):
        neural.deterministic_regions(2, 2, count, cols=8)


@pytest.mark.parametrize('capacity', [61, 32])
def test_spatial_roundtrip_is_clean_and_does_not_mutate_host(capacity, tmp_path):
    _, wm, bits = _inputs(tmp_path)
    original = np.full((512, 512, 3), 37, dtype='uint8')
    copy = original.copy()
    wrapper = neural.SpatialChunk4096(LocalPixelCodec(capacity), name='mock', cols=8, resize_to=0)
    marked = wrapper.embed(original, neural.watermark_bits(wm))
    rec = wrapper.detect(marked)
    np.testing.assert_array_equal(original, copy)
    assert marked.shape == original.shape and marked.dtype == np.uint8
    assert rec.good_chunks == wrapper.chunk_count
    assert rec.valid.all() and not rec.errors
    np.testing.assert_array_equal(rec.bits, bits)
    metrics = neural._metrics(bits, rec)
    assert metrics['exact_recovery'] == 1
    assert metrics['ber_erasure_as_error'] == 0


def test_decoder_missing_chunks_become_erasures_not_guessed_bits(tmp_path):
    _, wm, bits = _inputs(tmp_path)
    good_codec = LocalPixelCodec(61)
    encoder = neural.SpatialChunk4096(good_codec, name='mock', cols=8, resize_to=0)
    marked = encoder.embed(np.full((512, 512, 3), 37, dtype='uint8'), bits)
    decoder = neural.SpatialChunk4096(LocalPixelCodec(61, bad_decode={0, 67}), name='mock', cols=8, resize_to=0)
    rec = decoder.detect(marked)
    assert rec.good_chunks == 66 and len(rec.errors) == 2
    assert rec.valid.sum() == 4096 - 61 - 9  # last chunk has 9 real bits
    metrics = neural._metrics(bits, rec)
    assert metrics['exact_recovery'] == 0
    assert metrics['ber_erasure_as_error'] == pytest.approx(70/4096)
    assert metrics['ber'] == 0  # conditional BER excludes missing positions
    assert metrics['bit_coverage'] < 1


def test_decoder_failure_on_all_chunks_returns_undefined_conditional_scores(tmp_path):
    _, _, bits = _inputs(tmp_path)
    rec = neural.Recovered(np.zeros(4096, dtype='uint8'), np.zeros(4096, dtype=bool), 0, 68, ['decode failed'])
    metrics = neural._metrics(bits, rec)
    assert metrics['ber'] is None and metrics['nc'] is None and metrics['ncc'] is None
    assert metrics['ber_erasure_as_error'] == 1
    assert metrics['bit_coverage'] == 0
    assert metrics['exact_recovery'] == 0


def test_wrapper_rejects_invalid_host_shape_or_dtype(tmp_path):
    _, wm, bits = _inputs(tmp_path)
    wrapper = neural.SpatialChunk4096(LocalPixelCodec(), name='mock', cols=8, resize_to=0)
    with pytest.raises(ValueError, match='RGB uint8'):
        wrapper.embed(np.zeros((256, 256, 3), dtype='uint8'), bits)
    with pytest.raises(ValueError, match='RGB uint8'):
        wrapper.embed(np.zeros((512, 512, 3), dtype='float32'), bits)
    with pytest.raises(ValueError, match='512x512'):
        wrapper.detect(np.zeros((256, 256, 3), dtype='uint8'))


def test_attack_applied_once_to_assembled_host_and_exports_complete_rows(tmp_path, monkeypatch):
    hosts, wm, _ = _inputs(tmp_path)
    calls = []
    monkeypatch.setattr(neural, 'default_attack_suite', lambda **_: [SimpleNamespace(name='flip', group='synthetic')])
    def one_attack(img, config):
        calls.append((img.shape, config.name))
        altered = img.copy()
        altered[:, :, 0] = 255 - altered[:, :, 0]
        return altered
    monkeypatch.setattr(neural, 'apply_attack', one_attack)
    output = tmp_path / 'out'
    rows = neural.run(host_dir=hosts, watermark=wm, output=output, methods=('trustmark',),
                      wrapper_factory=_factory(LocalPixelCodec(61)))
    assert calls == [((512, 512, 3), 'flip')]
    assert {x['phase'] for x in rows} == {'before_attack', 'control', 'after_attack'}
    assert len(rows) == 3
    assert rows[0]['source_payload_bits'] == 4096
    assert rows[0]['bit_coverage'] == 1
    assert rows[2]['ber'] == 1
    assert rows[2]['ber_erasure_as_error'] == 1
    assert set(neural.RESULT_COLUMNS) == set(_rows(output / 'per_image_attack_results.csv')[0])
    assert len(_rows(output / 'comparison_4096.csv')) == 2
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['actual_attacked_rows'] == 1
    assert manifest['native_model_4096_support'] is False


def test_bad_clean_decoding_skips_attacks_unless_forced(tmp_path, monkeypatch):
    hosts, wm, _ = _inputs(tmp_path)
    called = []
    monkeypatch.setattr(neural, 'default_attack_suite', lambda **_: [SimpleNamespace(name='noop', group='synthetic')])
    monkeypatch.setattr(neural, 'apply_attack', lambda img, case: (called.append(case.name), img)[1])
    for force in (False, True):
        # reject first chunk; 61 erasures exceed 5%? no, so enforce >5% with 8 chunks
        codec = LocalPixelCodec(61, bad_decode=set(range(10)))
        output = tmp_path / f'out_{force}'
        rows = neural.run(host_dir=hosts, watermark=wm, output=output, methods=('trustmark',),
                          wrapper_factory=_factory(codec), force_full=force)
        assert rows[0]['clean_gate_passed'] is False
        assert rows[0]['bit_coverage'] < .95
        assert len(rows) == (3 if force else 2)
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['status'] == 'capacity_not_validated'
        assert manifest['clean_gate_rejected_hosts'] == 1
    assert called == ['noop']


def test_embed_failure_is_recorded_not_fabricated(tmp_path):
    hosts, wm, _ = _inputs(tmp_path)
    output = tmp_path / 'out'
    rows = neural.run(host_dir=hosts, watermark=wm, output=output, methods=('trustmark',),
                      attack_preset='none', wrapper_factory=_factory(LocalPixelCodec(fail_embed=True)))
    assert len(rows) == 1
    assert rows[0]['phase'] == 'embedding_failed'
    assert 'synthetic embed failure' in rows[0]['error']
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete_with_failures'
    assert not any(r.get('exact_recovery') == 1 for r in rows)


def test_restart_skips_same_contract_but_recomputes_changed_watermark(tmp_path):
    hosts, wm, _ = _inputs(tmp_path)
    codec = LocalPixelCodec()
    output = tmp_path / 'out'
    kwargs = dict(host_dir=hosts, watermark=wm, output=output, methods=('trustmark',),
                  attack_preset='none', wrapper_factory=_factory(codec))
    first = neural.run(**kwargs)
    calls = codec.embed_calls
    assert calls == 68
    again = neural.run(**kwargs)
    assert codec.embed_calls == calls
    assert len(first) == len(again) == 2
    pixels = np.asarray(Image.open(wm)).copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    Image.fromarray(pixels).save(wm)
    changed = neural.run(**kwargs)
    assert codec.embed_calls == calls + 68
    assert changed[0]['exact_recovery'] == 1


def test_actual_bss_small_image_outputs_consistent_files(tmp_path):
    out = tmp_path / 'bss'
    result = run_bss(ROOT / 'data/host/Girl.bmp', ROOT / 'data/watermark/wm.png', out,
                     host_size=64, watermark_size=8, repeat=1, max_attacks=1)
    assert len(result) == 1
    assert result[0]['samples'] > 0
    blocks = _rows(out / 'bss_per_block.csv')
    assert len(blocks) == result[0]['samples']
    assert {x['actual_survival'] for x in blocks} <= {'0', '1'}
    # Correlation can be undefined; an all-survival attack must not fake rho=1.
    if result[0]['survival_unique_levels'] < 2:
        assert result[0]['spearman_rho_bss'] is None
    assert json.loads((out / 'bss_summary.json').read_text())[0]['samples'] == len(blocks)


def test_actual_review_audit_has_negative_controls_and_metadata(tmp_path):
    out = tmp_path / 'audit'
    result = audit(ROOT / 'data/host', ROOT / 'data/watermark/wm.png', out,
                   max_images=1, host_size=64, watermark_size=8, repeat=1,
                   attack_preset='none', max_attacks=0)
    assert result['measurements'] >= 7
    rows = _rows(out / 'per_case_results.csv')
    cases = {x['case'] for x in rows}
    assert {'marked_clean', 'original_unmarked', 'wrong_secret_key',
            'corrupted_side_information', 'tampered_position_metadata'} <= cases
    clean = next(x for x in rows if x['case'] == 'marked_clean')
    assert int(clean['original_watermark_source_bits']) == 64
    assert int(clean['full_extraction_contract_json_bytes']) > 0
    assert len(_rows(out / 'side_information_overhead.csv')) == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'audit_complete_but_detection_not_verified'


def test_evidence_exporter_uses_existing_data_and_reports_missing(tmp_path):
    saved = tmp_path / 'saved'
    evidence = saved / 'heldout_full'
    evidence.mkdir(parents=True)
    rows = [dict(method_id=m, image=h, phase=phase, attack='jpeg_q40', nc=nc,
                 ber=1-nc, psnr=51, ssim=.99, embed_time_sec=.1, extract_time_sec=.2)
            for m, nc in [('proposal', .94), ('baseline', .72)]
            for h in ('a', 'b', 'c') for phase in ('before_attack', 'after_attack')]
    pd.DataFrame(rows).to_csv(evidence / 'per_image_attack_results.csv', index=False)
    out = tmp_path / 'out'
    process = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'export_reviewer_evidence.py'),
                              '--results-dir', str(saved), '--output', str(out)],
                             cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    index = _rows(out / 'reviewer_evidence_index.csv')
    assert len(index) == 6 and sum(v['exists'] == 'True' for v in index) == 1
    assert (out / 'classical_attackwise.csv').is_file()
    assert (out / 'classical_nc_host_ci.csv').is_file()
    assert len(_rows(out / 'classical_nc_host_ci.csv')) == 2
    attackwise = _rows(out / 'classical_attackwise.csv')
    assert len(attackwise) == 2
    assert all(int(x['n']) == 3 for x in attackwise)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['generated_from_saved_results_only'] is True
    assert len(manifest['missing']) == 5


@pytest.mark.parametrize('source,recovered,mask,expected', [
    ([1, 0, 1, 0], [1, 0, 0, 0], [True, True, False, True], (0.25, 0.0, 0.75)),
    ([1, 0], [0, 1], [False, False], (1.0, None, 0.0)),
])
def test_failure_aware_audit_metrics(source, recovered, mask, expected):
    got = binary_bit_metrics(np.array(source), np.array(recovered), np.array(mask))
    assert (got['payload_ber_all_positions'], got['payload_ber_valid_only'], got['coverage']) == expected


def test_resume_config_change_cannot_reuse_forced_failed_attacks(tmp_path, monkeypatch):
    hosts, wm, _ = _inputs(tmp_path)
    monkeypatch.setattr(neural, 'default_attack_suite', lambda **_: [SimpleNamespace(name='noop', group='synthetic')])
    monkeypatch.setattr(neural, 'apply_attack', lambda img, _: img)
    out = tmp_path / 'out'
    codecs = []
    def factory(name):
        codec = LocalPixelCodec(61, bad_decode=set(range(10)))
        codecs.append(codec)
        return neural.SpatialChunk4096(codec, name=name, cols=8, resize_to=0)
    kwargs = dict(host_dir=hosts, watermark=wm, output=out, methods=('trustmark',),
                  wrapper_factory=factory)
    forced = neural.run(**kwargs, force_full=True)
    assert len(forced) == 3 and forced[0]['clean_gate_passed'] is False
    not_forced = neural.run(**kwargs, force_full=False)
    assert len(not_forced) == 2  # old full run must be invalidated
    assert codecs[-1].embed_calls == 68
    assert json.loads((out / 'manifest.json').read_text())['actual_attacked_rows'] == 0


def test_resume_recomputes_if_model_variant_changed(tmp_path):
    hosts, wm, _ = _inputs(tmp_path)
    codec = LocalPixelCodec()
    kwargs = dict(host_dir=hosts, watermark=wm, output=tmp_path / 'out',
                  methods=('trustmark',), attack_preset='none', wrapper_factory=_factory(codec))
    neural.run(**kwargs, model_type='Q')
    neural.run(**kwargs, model_type='B')
    assert codec.embed_calls == 136


def test_failed_embed_summary_keeps_readable_csv_headers(tmp_path):
    hosts, wm, _ = _inputs(tmp_path)
    out = tmp_path / 'out'
    neural.run(host_dir=hosts, watermark=wm, output=out, methods=('trustmark',),
               attack_preset='none', wrapper_factory=_factory(LocalPixelCodec(fail_embed=True)))
    result = pd.read_csv(out / 'comparison_4096.csv')
    assert result.empty
    assert {'method_id', 'phase', 'ber_erasure_as_error_mean', 'bit_coverage_mean'} <= set(result.columns)


@pytest.mark.parametrize('module', ['watermarklab.experiments.neural_4096',
                                     'watermarklab.experiments.bss_validation',
                                     'watermarklab.experiments.review_audit'])
def test_entrypoint_help_loads_without_optional_neural_models(module):
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    done = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                          text=True, capture_output=True, timeout=20, env=env)
    assert done.returncode == 0, done.stderr
    assert '--output' in done.stdout and '--watermark' in done.stdout


def test_bss_module_is_runnable_as_command(tmp_path):
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    out = tmp_path / 'cli_bss'
    cmd = [sys.executable, '-m', 'watermarklab.experiments.bss_validation',
           '--host', str(ROOT / 'data/host/Girl.bmp'),
           '--watermark', str(ROOT / 'data/watermark/wm.png'),
           '--output', str(out), '--host-size', '64', '--watermark-size', '8',
           '--repeat', '1', '--max-attacks', '1']
    done = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=45, env=env)
    assert done.returncode == 0, done.stderr
    summary = json.loads((out / 'host_level_summary.json').read_text())
    assert len(summary) == 1
    assert json.loads((out / 'manifest.json').read_text())['n_hosts'] == 1


def test_audit_module_is_runnable_as_command(tmp_path):
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    out = tmp_path / 'cli_audit'
    cmd = [sys.executable, '-m', 'watermarklab.experiments.review_audit',
           '--host-dir', str(ROOT / 'data/host'),
           '--watermark', str(ROOT / 'data/watermark/wm.png'),
           '--output', str(out), '--host-size', '64', '--watermark-size', '8',
           '--repeat', '1', '--max-images', '1', '--attack-preset', 'none',
           '--max-attacks', '0']
    done = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=45, env=env)
    assert done.returncode == 0, done.stderr
    assert (out / 'manifest.json').is_file()
    assert (out / 'side_information_overhead.csv').is_file()
    assert json.loads((out / 'summary.json').read_text())['measurements'] > 0
