"""Strict, fail-closed validation of reviewer-specific numerical evidence.

No result is certified merely because a CSV file or an aggregate manifest exists.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from watermarklab.common.attack import default_attack_suite
from watermarklab.experiments.attack_selection import stratified_attack_subset
from watermarklab.experiments.bss_validation import validate_summary_rows


def read_rows(file: Path):
    if not Path(file).is_file():return []
    with Path(file).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def _json(file):
    try:return json.loads(Path(file).read_text())
    except (OSError, ValueError):return {}


def _numeric(row, cols):
    try:return all(math.isfinite(float(row[k])) for k in cols)
    except (ValueError, TypeError, KeyError):return False


def validate_channel(directory: Path, host: str):
    directory=Path(directory)
    rows=read_rows(directory/'channel_ablation.csv')
    manifest=_json(directory/'manifest.json')
    attacks={x.name for x in default_attack_suite(False,'reviewer_stress')}
    expected={(mode,'before_attack','no_attack') for mode in ('Y','R','G','B')}
    expected|={(mode,'after_attack',attack) for mode in ('Y','R','G','B') for attack in attacks}
    actual={(r.get('channel'),r.get('phase'),r.get('attack')) for r in rows}
    return (manifest.get('status')=='PASS' and not _json(directory/'errors.json')
            and len(rows)==len(expected)==manifest.get('expected_rows')
            and actual==expected and len(actual)==len(rows)
            and all(r.get('host')==host and _numeric(r,('nc','ber','psnr_clean','ssim_clean')) for r in rows))


def validate_bss(directory: Path, host: str, max_attacks:int=14):
    directory=Path(directory)
    rows=read_rows(directory/'bss_per_host_attack.csv')
    expected=[a.name for a in stratified_attack_subset(
        default_attack_suite(include_none=False,preset='reviewer_stress'),max_attacks)]
    state=_json(directory/'manifest.json')
    per_host=_json(directory/Path(host).stem/'measurement_status.json')
    return (state.get('status')=='PASS' and state.get('n_hosts')==1
            and state.get('hosts')==[host] and state.get('rows')==len(rows)
            and per_host.get('status')=='PASS' and per_host.get('error_rows')==0
            and per_host.get('measurement_rows',0)>0 and
            validate_summary_rows(rows,expected,expected_host=host))


def validate_security(directory: Path, host: str, wrong_key_trials: int = 8):
    directory=Path(directory)
    rows=read_rows(directory/'per_case_results.csv')
    cases=[r.get('case') for r in rows]
    required={'marked_clean','original_unmarked','unrelated_host','wrong_secret_key',
              'tampered_position_metadata','corrupted_side_information'}
    required|={f'wrong_secret_random_{i:03d}' for i in range(wrong_key_trials)}
    manifest=_json(directory/'manifest.json')
    report=_json(directory/'summary.json')
    threat=_json(directory/'keyspace_and_threat_model.json')
    tampered=[r for r in rows if r.get('case') in ('tampered_position_metadata','corrupted_side_information')]
    return (bool(rows) and len(cases)==len(set(cases)) and required.issubset(cases)
            and all(r.get('host')==host and not r.get('error') for r in rows)
            and len(tampered)==2 and all(r.get('metadata_tampering_detected') in (True,'True') for r in tampered)
            and manifest.get('unrelated_host_control_performed') is True
            and manifest.get('wrong_key_trials_per_host')==wrong_key_trials
            and report.get('unrelated_host_rows')==1
            and report.get('random_wrong_key_rows')==wrong_key_trials
            and report.get('errors')==0
            and threat.get('wrong_key_trials_per_host')==wrong_key_trials)


def validate_gallery(directory: Path, host: str, watermark: str):
    directory=Path(directory)
    state=_json(directory/'gallery_manifest.json')
    rows=read_rows(directory/'gallery_metrics.csv')
    methods=set(state.get('methods',[]))
    attacks=set(state.get('attacks',[]))
    expected={(m,a) for m in methods for a in attacks}
    return (bool(expected) and state.get('host')==host and state.get('watermark')==watermark
            and len(rows)==len(expected) and {(r.get('method'),r.get('attack')) for r in rows}==expected
            and all(_numeric(r,('nc','ber','clean_psnr')) and
                    (directory/r.get('extracted_image','')).is_file() for r in rows)
            and (directory/'extracted_watermarks_comparison.png').is_file())
