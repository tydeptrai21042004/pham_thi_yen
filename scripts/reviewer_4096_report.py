"""Reproducible Reviewer R1.2/R2.2 report from already-run 4096-bit archives.

Usage:
 python scripts/reviewer_4096_report.py --archives <six ZIP paths> --output report4096

No model imports, no proposal changes, no re-embedding. Enforces all 12 hosts
and 109 attack keys on both watermarks. Missing data raises a loud error.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

HOSTS = {"girl", "airplane", "athens", "baboon", "butfly", "house", "manhatan",
         "milkdrop", "pepper", "safari", "sailboat", "tiffany"}
ATTACKS = 109
BITS = 4096


def _value(value):
    if value is None or value == "":
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _avg(rows, key):
    vals = [_value(r.get(key)) for r in rows]
    vals = [x for x in vals if x is not None]
    return round(statistics.mean(vals), 7) if vals else None


def _read(zf, filename):
    return list(csv.DictReader(io.StringIO(zf.read(filename).decode('utf-8-sig'))))


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _watermark_class(zip_name):
    s=zip_name.lower()
    if 'trustmark_only_remaining_b' in s or 'watermark_2_cc' in s:
        return 'B'
    if 'trustmark_only_remaining_a' in s or 'watermark_all_results_9train_12test' in s:
        return 'A'
    raise ValueError(f'Unrecognized archive A/B label: {zip_name}; rename accordingly')


def audit_archives(archives: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    neural, classical, security = {}, {}, {}
    conflicts=[]
    sources = []
    for archive in archives:
        cohort=_watermark_class(archive.name)
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            sources.append(dict(archive=archive.name, cohort=cohort, n_entries=len(names)))
            for filename in names:
                if filename.endswith('/classical/heldout_full/per_image_attack_results.csv'):
                    if cohort in classical:
                        raise ValueError('Multiple classical whole-grid files for '+cohort)
                    classical[cohort] = _read(z, filename)
                elif filename.endswith('/classical/reviewer_analysis/security_measurements_raw.csv'):
                    if cohort in security:
                        raise ValueError('Multiple security raw files for '+cohort)
                    security[cohort] = _read(z, filename)
                elif ('/neural_4096_forced/trustmark/' in filename and
                      filename.endswith('/per_image_attack_results.csv')):
                    rows = _read(z,filename)
                    if not rows:
                        continue
                    host=rows[0]['image'].lower()
                    key=(cohort,host)
                    phases = {(r['phase'],r['attack']):r for r in rows}
                    if len(phases) != len(rows):
                        raise ValueError(f'Duplicate phase/attack rows in {filename}')
                    if key in neural:
                        existing=neural[key]
                        # Duplicated Girl in overlapping result archives: verify core scores.
                        for phase_attack in set(existing)&set(phases):
                            lhs,rhs=existing[phase_attack],phases[phase_attack]
                            for col in ('ber_erasure_as_error','bit_coverage','nc','psnr'):
                                a,b=_value(lhs.get(col)),_value(rhs.get(col))
                                if a != b and (a is None or b is None or abs(a-b)>1e-8):
                                    conflicts.append(f'{key} {phase_attack} {col}: {a} vs {b}')
                        # Use more complete run, not two independent trials or summed samples.
                        if len(phases)>len(existing):neural[key]=phases
                    else:
                        neural[key]=phases
    if conflicts:
        raise ValueError('Conflicting repeated TrustMark records: '+ '; '.join(conflicts[:4]))
    if set(classical) != {'A','B'} or set(security) != {'A','B'}:
        raise ValueError('Missing A/B classical or security archive')
    stats=[]; per_attack=[]; per_host=[]; security_rows=[]
    provenance=[]
    for cohort in ('A','B'):
        c=classical[cohort]
        control_cases=[x for x in c if x['phase']=='after_attack']
        methods = sorted(set(r['method_id'] for r in c))
        if len(methods) != 13 or 'proposal' not in methods:
            raise ValueError(f'{cohort} expected 13 classical method_ids including proposal, got {methods}')
        bymethod=defaultdict(list)
        for r in c:
            if r['phase'] in ('before_attack','after_attack'):
                bymethod[r['method_id']].append(r)
        # Exact proposal attack identity must be shared across classical and neural.
        ref={(r['image'].lower(),r['attack']) for r in bymethod['proposal'] if r['phase']=='after_attack'}
        if len(ref)!=len(HOSTS)*ATTACKS or {h for h,a in ref}!=HOSTS:
            raise ValueError(f'{cohort} proposal grid incomplete: {len(ref)}')
        for method, rows in bymethod.items():
            clean=[r for r in rows if r['phase']=='before_attack']
            attacked=[r for r in rows if r['phase']=='after_attack']
            observed={(r['image'].lower(),r['attack']) for r in attacked}
            if len(clean)!=len(HOSTS) or len(attacked)!=len(ref) or observed!=ref:
                raise ValueError(f'{cohort}: {method} attack/host mismatch')
            emb_bits=1024 if method=='proposal' else 4096
            stats.append(dict(watermark=cohort,method=method,method_kind='proposal' if method=='proposal' else 'classical_adaptation',
                     watermark_source_bits=4096, actually_embedded_bits=emb_bits,
                     nc_scope='64x64_reconstructed_from_LL_and_side_information' if method=='proposal' else '4096-bit_image',
                     clean_psnr_db=_avg(clean,'psnr'),clean_ssim=_avg(clean,'ssim'),
                     attacked_NC_mean=_avg(attacked,'nc'),attacked_BER_mean=_avg(attacked,'ber'),
                     attacked_erasure_BER_mean=_avg(attacked,'ber'),attacked_bit_recovery_fraction=round(1-_avg(attacked,'ber'),7),
                     attacked_coverage_mean=1.0,attacked_conditional_NC_mean=None,
                     attacked_exact_recovery_rate=None,zero_coverage_cases=0,
                     clean_gate_pass_rate=None,hosts=12,attacks_per_host=109,cases=len(attacked)))
            attacknames={r['attack'] for r in attacked}
            for attack in sorted(attacknames):
                rr=[r for r in attacked if r['attack']==attack]
                per_attack.append(dict(watermark=cohort,method=method,attack=attack,n=len(rr),
                                       NC_mean=_avg(rr,'nc'),BER_mean=_avg(rr,'ber'),
                                       coverage_mean=1.0,recovery_fraction_mean=round(1-_avg(rr,'ber'),7)))
        for host in sorted(HOSTS):
            key=(cohort,host)
            if key not in neural:
                raise ValueError(f'No TrustMark result for {cohort} {host}')
            rr=neural[key]
            attacks={a for (phase,a) in rr if phase=='after_attack'}
            if {(host,a) for a in attacks}!={r for r in ref if r[0]==host}:
                raise ValueError(f'{cohort} {host} TrustMark attack grid does not exactly match classical')
            if len(attacks)!=ATTACKS or ('before_attack','no_attack') not in rr or ('control','original_unmarked') not in rr:
                raise ValueError(f'{cohort} {host} missing clean/attack/control')
            for row in rr.values():
                if row.get('source_payload_bits') not in ('4096','4096.0') or row.get('native_chunk_capacity_bits') not in ('61','61.0'):
                    raise ValueError(f'Not 4096-bit TrustMark with native 61-bit model at {cohort} {host}')
                if row.get('method_id')!='trustmark_4096_spatial':
                    raise ValueError('Wrong TrustMark method_id')
            clean=rr['before_attack','no_attack']
            attacked=[r for (phase,_),r in rr.items() if phase=='after_attack']
            if any(_value(r.get('ber_erasure_as_error')) is None or _value(r.get('bit_coverage')) is None for r in attacked):
                raise ValueError(f'{cohort} {host}: missing numeric erasure-aware attacked metrics')
            per_host.append(dict(watermark=cohort,host=host,clean_psnr_db=_value(clean['psnr']),
                   clean_bit_coverage=_value(clean['bit_coverage']),clean_erasure_BER=_value(clean['ber_erasure_as_error']),
                   clean_gate_passed=clean.get('clean_gate_passed'),
                   mean_attacked_coverage=_avg(attacked,'bit_coverage'),
                   mean_attacked_erasure_BER=_avg(attacked,'ber_erasure_as_error'),
                   zero_coverage_attacks=sum(_value(r['bit_coverage'])==0 for r in attacked)))
        flat=[r for host in HOSTS for (phase,_),r in neural[cohort,host].items() if phase=='after_attack']
        clean=[neural[cohort,host]['before_attack','no_attack'] for host in HOSTS]
        nc=[r for r in flat if _value(r.get('nc')) is not None]
        ncc_full=[r for r in flat if _value(r.get('bit_coverage'))==1]
        erasures=_avg(flat,'ber_erasure_as_error')
        stats.append(dict(watermark=cohort,method='trustmark_4096_spatial',method_kind='neural_spatial_4096_extension',
             watermark_source_bits=4096,actually_embedded_bits=4096,nc_scope='4096-bit_partial_conditional_only',
             clean_psnr_db=_avg(clean,'psnr'),clean_ssim=_avg(clean,'ssim'),
             attacked_NC_mean=None,attacked_BER_mean=None,
             attacked_erasure_BER_mean=erasures,attacked_bit_recovery_fraction=round(1-erasures,7),
             attacked_coverage_mean=_avg(flat,'bit_coverage'),attacked_conditional_NC_mean=_avg(nc,'nc'),
             attacked_exact_recovery_rate=_avg(flat,'exact_recovery'),
             zero_coverage_cases=sum(_value(r.get('bit_coverage'))==0 for r in flat),
             clean_gate_pass_rate=sum(r.get('clean_gate_passed') in ('True','true','1') for r in clean)/12,
             hosts=12,attacks_per_host=109,cases=len(flat)))
        for attack in sorted(set(r['attack'] for r in flat)):
            rr=[r for r in flat if r['attack']==attack]
            per_attack.append(dict(watermark=cohort,method='trustmark_4096_spatial',attack=attack,n=len(rr),
                NC_mean=None,BER_mean=_avg(rr,'ber_erasure_as_error'),
                coverage_mean=_avg(rr,'bit_coverage'),
                recovery_fraction_mean=round(1-_avg(rr,'ber_erasure_as_error'),7)))
        srows=security[cohort]
        if len(srows)!=288 or len(set((r['host'],r['case']) for r in srows))!=288:
            raise ValueError(f'{cohort}: expected 288 distinct raw security rows')
        for case in sorted(set(r['case'] for r in srows)):
            rr=[r for r in srows if r['case']==case]
            security_rows.append(dict(watermark=cohort,case=case,hosts=len(rr),
                 watermark_NC_mean=_avg(rr,'watermark_nc'),
                 embedded_LL_payload_BER_mean=_avg(rr,'payload_ber_all_positions'),
                 decoded_coverage_mean=_avg(rr,'coverage'),
                 tampering_detected_n=sum(r.get('metadata_tampering_detected')=='True' for r in rr),
                 side_JSON_KiB_mean=(round(_avg(rr,'side_info_json_bytes')/1024,4) if _avg(rr,'side_info_json_bytes') is not None else None),
                 watermark_source_bits=4096,actually_embedded_bits=1024,
                 caution='metadata_reconstructs_source_details; NC_not_an_independent_presence_test'))
        provenance.append(dict(watermark=cohort,classical_method_count=len(methods),classical_attack_rows=len(control_cases),
            trustmark_attacked_rows=len(flat),trustmark_hosts=12,
            classical_attack_keys_verified=True,metadata_not_authenticity_proof=True))
    _write_csv(output/'comparison_all_methods.csv',stats,list(stats[0]))
    _write_csv(output/'attackwise_comparison.csv',per_attack,list(per_attack[0]))
    _write_csv(output/'trustmark_per_host.csv',per_host,list(per_host[0]))
    _write_csv(output/'security_negative_controls.csv',security_rows,list(security_rows[0]))
    audit={'sources':sources,'provenance':provenance,
        'interpretation':{
            'TrustMark_attacked_NC':'conditional on detected chunks; cannot compare to classical full-payload NC',
            'TrustMark_recovery_fraction':'1 - erasure-inclusive BER across all 4096 source bits',
            'proposal_effective_payload':'1024 LL bits, with auxiliary watermark-detail metadata',
            'security':'unmarked original NC=1 is documented failure; does not establish authentication',
            'retraining':'No models retrained or proposal code changed',
            'fairness':'same hosts/attacks/source bitmaps; different effective embedded information and decoder side information' }}
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('SUCCESS verified 13 classical methods and TrustMark for 12 hosts × 109 attacks × two watermarks')
    for r in stats:
        if r['method'] in ('proposal','trustmark_4096_spatial'):
            print(r['watermark'],r['method'],'clean PSNR',r['clean_psnr_db'],'NC',r['attacked_NC_mean'],
                  'recovery fraction',r['attacked_bit_recovery_fraction'],'coverage',r['attacked_coverage_mean'])
    return audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/reviewer_4096'))
    a=p.parse_args(argv)
    audit_archives(a.archives,a.output)

if __name__=='__main__':main()
