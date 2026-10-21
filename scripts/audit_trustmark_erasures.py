"""Summarize existing 4096-bit spatial TrustMark outputs WITHOUT neural reruns.

Run: python scripts/audit_trustmark_erasures.py path/to/per_image_attack_results.csv

Undefined conditional NC/BER stays undefined. A missing/failed decoder is
never converted into an all-zero, successfully extracted watermark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _num(v):
    if v is None or str(v).strip().lower() in {"", "nan", "none", "null"}:
        return None
    try:
        z = float(v)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None


def audit(path: str | Path) -> list[dict]:
    groups = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") not in {"before_attack", "after_attack"}:
                continue
            key = (row.get("method_id", ""), row.get("watermark", ""), row.get("phase", ""))
            groups[key].append(row)
    result = []
    for (method, wm, phase), rows in sorted(groups.items()):
        valid = [r for r in rows if _num(r.get("bit_coverage")) is not None and not str(r.get("error", "")).strip()]
        coverage = [_num(r["bit_coverage"]) for r in valid]
        wrong_or_erased = [_num(r.get("ber_erasure_as_error")) for r in valid]
        wrong_or_erased = [x for x in wrong_or_erased if x is not None]
        conditional_nc = [_num(r.get("nc")) for r in valid]
        conditional_nc = [x for x in conditional_nc if x is not None]
        result.append({
            "method": method, "watermark": wm, "phase": phase,
            "cases": len(rows), "valid_metric_cases":len(valid), "execution_error_cases":len(rows)-len(valid),
            "zero_coverage_cases":sum(x == 0.0 for x in coverage),
            "complete_coverage_cases":sum(x == 1.0 for x in coverage),
            "mean_coverage":sum(coverage)/len(coverage) if coverage else None,
            "mean_erasure_as_error_ber":sum(wrong_or_erased)/len(wrong_or_erased) if wrong_or_erased else None,
            "mean_conditional_nc":sum(conditional_nc)/len(conditional_nc) if conditional_nc else None,
            "conditional_nc_defined_cases":len(conditional_nc),
        })
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_file", type=Path)
    p.add_argument("--output", type=Path, help="Optional JSON report")
    args=p.parse_args()
    rows=audit(args.csv_file)
    payload=json.dumps(rows,indent=2,allow_nan=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload+'\n',encoding='utf-8')


if __name__ == "__main__":
    main()
