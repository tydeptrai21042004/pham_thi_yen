"""Fast regression tests for the exact W1/W2 Kaggle failure signatures.

No Kaggle data, no neural models and no GPU are required.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from watermarklab.common.attack import default_attack_suite
from watermarklab.experiments.attack_selection import stratified_attack_subset
from watermarklab.experiments.reviewer_checks import validate_channel, validate_bss

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_channel_validator_respects_executed_reviewer45_attack_grid(tmp_path):
    attack_ids = [a.name for a in default_attack_suite(False, 'reviewer45')]
    assert len(attack_ids) == 45
    rows = []
    for channel in ('Y', 'R', 'G', 'B'):
        for phase, attack in [('before_attack', 'no_attack'),
                              *[('after_attack', a) for a in attack_ids]]:
            rows.append(dict(channel=channel, phase=phase, attack=attack,
                             host='Girl.bmp', nc=0.9, ber=0.05,
                             psnr_clean=53.0, ssim_clean=0.99))
    _write_csv(tmp_path/'channel_ablation.csv', rows)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        status='PASS', expected_rows=len(rows), actual_rows=len(rows))))
    (tmp_path/'errors.json').write_text('[]')
    assert validate_channel(tmp_path, 'Girl.bmp', preset='reviewer45', max_attacks=45)
    assert not validate_channel(tmp_path, 'Girl.bmp', preset='reviewer_stress')


def test_bss_validator_respects_executed_reviewer45_attack_grid(tmp_path):
    attacks = stratified_attack_subset(default_attack_suite(False, 'reviewer45'), 45)
    assert len(attacks) == 45
    rows = [dict(host='Girl.bmp', attack=a.name, samples=100,
                 error_count=0, survival_rate=0.92,
                 margin_unique_levels=2, bss_unique_levels=2) for a in attacks]
    _write_csv(tmp_path/'bss_per_host_attack.csv', rows)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        status='PASS', n_hosts=1, hosts=['Girl.bmp'], rows=45)))
    host=tmp_path/'Girl';host.mkdir()
    (host/'measurement_status.json').write_text(json.dumps(dict(
        status='PASS', error_rows=0, measurement_rows=100)))
    assert validate_bss(tmp_path, 'Girl.bmp', max_attacks=45, preset='reviewer45')
    assert not validate_bss(tmp_path, 'Girl.bmp', max_attacks=45, preset='reviewer_stress')


def test_heldout_hashlib_is_global_not_shadowed_by_local_import():
    source=(ROOT/'scripts/recover_missing_results.py').read_text()
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    heldout=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='heldout')
    imports=[n for n in ast.walk(heldout) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(n,ast.Import) and any(x.name=='hashlib' for x in n.names)
                   for n in imports)
    assert 'hashlib.sha256' in ast.get_source_segment(source,heldout)


def test_fresh_runner_has_safe_one_option_extras_and_baseline_priority():
    path=ROOT/'scripts/run_remaining_fresh.py'
    source=path.read_text()
    assert '--focus-failed-baselines' in source
    assert "more[1]" not in source
    assert "recovery_step('extras',['--extended-sensitivity'])" in source
    assert "'su2016_contourlet_hess'" in source or 'su2016_contourlet_hess' in source
    compile(source,str(path),'exec')
