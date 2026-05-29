from __future__ import annotations
import argparse
import json
from pathlib import Path

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.proposal_validation import run_full_proposal_validation


def main():
    parser = argparse.ArgumentParser(description="Run proposal implementation validation checks.")
    parser.add_argument("--host", default="data/host/lenna.bmp")
    parser.add_argument("--watermark", default="data/watermark/wm.png")
    parser.add_argument("--no-end-to-end", action="store_true", help="Only run pure math/schedule checks.")
    parser.add_argument("--output", default="results/proposal_validation_report.json")
    args = parser.parse_args()

    host = wm = None
    if not args.no_end_to_end:
        host = load_host_rgb(args.host)
        wm = load_watermark_binary(args.watermark)
    report = run_full_proposal_validation(host, wm)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": report.get("ok", False), "output": str(out)}, indent=2))
    if not report.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
