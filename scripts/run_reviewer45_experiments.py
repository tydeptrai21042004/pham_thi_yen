#!/usr/bin/env python3
"""One shared 45-attack protocol for both watermarks, plus hparam sweeps.

Run from repository root.  This orchestration file does not modify old results.
Use the train split ONLY for the hyperparameter sweep and independent held-out
hosts for final evaluation.  See docs/REVIEWER45_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from watermarklab.common.attack import reviewer45_attack_suite, manuscript43_attack_suite, reviewer45_family
from watermarklab.common.io_utils import list_image_files, load_watermark_binary
from watermarklab.experiments.splits import TRAIN_NAMES, ORIGINAL_TEST_NAMES, _validate_folder, _validate_image


def protocol_manifest(output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    attacks = reviewer45_attack_suite(include_none=False)
    records = [
        {"index": i, "attack": a.name, "family": reviewer45_family(a.name), "operator": a.group,
         "parameters": a.params}
        for i, a in enumerate(attacks, 1)
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf8")
    digest = hashlib.sha256(encoded).hexdigest()
    (output_dir / "reviewer45_attack_manifest.json").write_text(json.dumps({
        "protocol": "reviewer45_v1", "attack_count": 45, "clean_control": "no_attack",
        "sha256": digest, "change_from_manuscript43": {
            "removed": ["lowpass_filter_3x3"],
            "added": ["resize_2p0", "crop_resize_50", "shear_x_0p10"],
            "reason": "lowpass_filter_3x3 duplicates average_filter_3x3 exactly",
        }, "attacks": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    with (output_dir / "reviewer45_attack_manifest.csv").open("w", newline="", encoding="utf8") as f:
        writer=csv.DictWriter(f, fieldnames=["index", "attack", "family", "operator", "parameters"])
        writer.writeheader()
        for rec in records:
            writer.writerow({**rec, "parameters": json.dumps(rec["parameters"], sort_keys=True)})
    return digest


def assert_pywt(allow_fallback: bool) -> None:
    if allow_fallback:
        print("WARNING: --allow-fallback permits different DWT backends; pilot runs only.")
        return
    try:
        import pywt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install PyWavelets before publication runs: pip install PyWavelets. "
                           "Use --allow-fallback only for a smoke test.") from exc


def validate_output(output_dir: Path, host_dir: Path, max_images: int | None, expected_methods: str | None = None) -> dict:
    import pandas as pd
    df_path = output_dir / "per_image_attack_results.csv"
    if not df_path.exists():
        raise RuntimeError(f"Benchmark did not write {df_path}")
    df = pd.read_csv(df_path)
    names = [p.stem for p in list_image_files(host_dir)]
    if max_images is not None:
        names = names[:max_images]
    attack_set = {a.name for a in reviewer45_attack_suite(False)}
    if df.empty:
        raise RuntimeError(f"Empty evaluation: {df_path}")
    if df[["nc", "ber"]].replace([float("inf"), -float("inf")], float("nan")).isna().any().any():
        raise RuntimeError(f"Non-finite NC/BER in {df_path}")
    if set(df.image.astype(str)) != set(names):
        raise RuntimeError(f"Host mismatch for {output_dir}: expected {names}; observed {sorted(set(df.image.astype(str)))}")
    if df.method_id.isna().any():
        raise RuntimeError(f"Missing method IDs in {df_path}")
    if expected_methods and expected_methods not in {"all", "*"}:
        required = {v.strip() for v in expected_methods.split(",") if v.strip()}
        # The CLI method registry accepts canonical method IDs; require them all
        # instead of incorrectly declaring a partial multi-baseline run complete.
        if not required.issubset(set(df.method_id.astype(str))):
            raise RuntimeError(f"Missing baseline method rows: {sorted(required-set(df.method_id.astype(str)))}")
    for method, subset in df.groupby("method_id"):
        for host in names:
            h = subset[subset.image.astype(str) == host]
            attacked = h[h.phase == "after_attack"]
            count = attacked.attack.value_counts().to_dict()
            if set(count) != attack_set or any(v != 1 for v in count.values()):
                missing = sorted(attack_set - set(count))
                extra = sorted(set(count) - attack_set)
                raise RuntimeError(f"INCOMPLETE {output_dir}, {method}, {host}: "
                                   f"got {len(attacked)}/45; missing={missing}; extra={extra}; "
                                   f"duplicate={[k for k,v in count.items() if v != 1]}")
            clean = h[h.phase == "before_attack"]
            if len(clean) != 1:
                raise RuntimeError(f"Expected one clean row: {method}/{host}; got {len(clean)}")
            if attacked[["nc", "ber"]].isna().any().any():
                raise RuntimeError(f"Missing NC/BER entries: {method}/{host}")
    return {"method_ids": sorted(df.method_id.unique().tolist()), "hosts": names,
            "attacked_rows": int((df.phase == "after_attack").sum()),
            "clean_rows": int((df.phase == "before_attack").sum())}


def run_main(args: list[str], dry_run: bool):
    cmd=[sys.executable, str(ROOT / "main.py"), *args]
    print("COMMAND:", " ".join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def build_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "sweep", "manifest"], default="evaluate")
    parser.add_argument("--test-hosts", type=Path, default=ROOT / "data/host", help="Exact 12 held-out test hosts; Lena excluded")
    parser.add_argument("--train-hosts", type=Path, default=ROOT / "data/host_train", help="Exact 9 tuning hosts, disjoint from test")
    parser.add_argument("--watermark-w1", type=Path, default=Path("data/watermark/wm.png"))
    parser.add_argument("--watermark-w2", type=Path, default=Path("data/watermark/Cc.logo.circle.png"))
    parser.add_argument("--output", type=Path, default=Path("results/reviewer45_v1"))
    parser.add_argument("--methods", default="proposal")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--repeat", default="full", help="Proposal repetitions: full by default; integer for diagnostics")
    parser.add_argument("--proposal-param-file", default="results/proposal_optimized_params.json")
    parser.add_argument("--proposal-param-mode", choices=["require", "ignore"], default="require",
                        help="require frozen parameters for a final held-out experiment; ignore for pilot/default params")
    parser.add_argument("--tau", default="0.4,0.5,0.6")
    parser.add_argument("--qh", default="6,7,8")
    parser.add_argument("--tau-margins", default="0.06,0.10")
    parser.add_argument("--h-margins", default="0.7,0.9")
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    return parser


def main():
    args=build_cli().parse_args()
    sha=protocol_manifest(args.output)
    print("reviewer45 protocol SHA256:", sha)
    if args.mode == "manifest":
        return
    if args.train_hosts.resolve() == args.test_hosts.resolve():
        raise ValueError("Training and testing host directories must be different")
    # Validate BOTH named groups in EVERY paper run. Never optimize on test hosts.
    train_paths = _validate_folder(args.train_hosts, TRAIN_NAMES, "train")
    test_paths = _validate_folder(args.test_hosts, ORIGINAL_TEST_NAMES, "test")
    assert len(train_paths) == 9 and len(test_paths) == 12
    decoded_hashes = [_validate_image(path) for path in train_paths + test_paths]
    if len(set(decoded_hashes)) != 21:
        raise ValueError("Train/test contains duplicate decoded host images")
    # The common image enumerator excludes Lena/Lenna, even if data/host
    # still retains that old file. This is the frozen manuscript host list.
    if {p.name.casefold() for p in list_image_files(args.test_hosts)} != {
        x.casefold() for x in ORIGINAL_TEST_NAMES
    }:
        raise ValueError("Benchmark enumeration differs from frozen 12-host manifest")
    watermarks = {}
    for label, wp in (("W1", args.watermark_w1), ("W2", args.watermark_w2)):
        if not wp.is_file():
            raise FileNotFoundError(wp)
        pixels = load_watermark_binary(wp)
        watermarks[label] = {"path": str(wp.resolve()),
                              "file_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(),
                              "binary_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                              "size": [64,64], "source_bits": 4096,
                              "white_bits": int((pixels > 0).sum())}
    if watermarks["W1"]["binary_sha256"] == watermarks["W2"]["binary_sha256"]:
        raise ValueError("W1 and W2 decode to identical binary images")
    (args.output / "frozen_input_manifest.json").write_text(json.dumps({
        "train_count": 9, "test_count": 12, "train_files": [p.name for p in train_paths],
        "test_files": [p.name for p in test_paths], "excluded": ["lena.bmp", "lenna.bmp"],
        "train_sha256_pixels": dict(zip((p.name for p in train_paths), decoded_hashes[:9])),
        "test_sha256_pixels": dict(zip((p.name for p in test_paths), decoded_hashes[9:])),
        "watermarks": watermarks,
        "proposal_optimizer_objective": "UNMODIFIED_FROM_SOURCE",
    }, indent=2) + "\n", encoding="utf8")
    if args.mode == "evaluate" and args.proposal_param_mode == "require" and not args.dry_run:
        # Never accept an old optimization file produced using test images.
        optimized_path = Path(args.proposal_param_file)
        if not optimized_path.is_file():
            raise FileNotFoundError(f"Nine-host frozen optimization file is missing: {optimized_path}")
        try:
            fitted = json.loads(optimized_path.read_text(encoding="utf8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError("Cannot read frozen optimization provenance") from exc
        expected_train = {Path(name).stem.casefold() for name in TRAIN_NAMES}
        if fitted.get("format") != "watermarklab_proposal_optimization_params_v1":
            raise RuntimeError("Frozen final comparison requires a benchmark-generated optimization file with 9-host provenance")
        fitted_names = {Path(name).stem.casefold() for name in fitted.get("per_image", {})}
        if fitted_names != expected_train:
            raise RuntimeError(f"Optimization was not completed on exactly 9 training hosts: {sorted(fitted_names)}")
        fitted_host_dir = Path(fitted.get("host_dir", ""))
        if not fitted_host_dir.is_dir():
            raise RuntimeError(f"Missing optimization source host directory: {fitted_host_dir}")
        fitted_source_names = _validate_folder(fitted_host_dir, TRAIN_NAMES, "train")
        if [_validate_image(x) for x in fitted_source_names] != decoded_hashes[:9]:
            raise RuntimeError("Frozen optimizer provenance uses different training image pixels")
    if not args.dry_run:
        assert_pywt(args.allow_fallback)
        if not args.allow_fallback:
            os.environ["WATERMARKLAB_STRICT_PYWT"] = "1"
    for label,wm in (("W1",args.watermark_w1),("W2",args.watermark_w2)):
        if not wm.is_file():
            raise FileNotFoundError(wm)
        out = args.output / ("hyperparams_"+label if args.mode == "sweep" else "evaluation_"+label)
        out.mkdir(parents=True,exist_ok=True)
        common=["--host-dir",str(args.train_hosts if args.mode=="sweep" else args.test_hosts),
                "--watermark",str(wm), "--output",str(out),
                "--attack-preset","reviewer45", "--no-save-images", "--proposal-repeat",str(args.repeat)]
        if args.max_images is not None:
            common += ["--max-images",str(args.max_images)]
        if args.mode == "evaluate":
            common += ["--methods",args.methods,"--resume-attack-journal",
                       "--proposal-param-mode",args.proposal_param_mode,
                       "--proposal-param-file",args.proposal_param_file]
        else:
            common += ["--phase","proposal-hparam-sweep","--proposal-param-mode","ignore",
                       "--hparam-sweep-q4-tau",args.tau,
                       "--hparam-sweep-h01-q",args.qh,
                       "--hparam-sweep-q4-margin",args.tau_margins,
                       "--hparam-sweep-h01-margin",args.h_margins]
            if args.max_combinations is not None:
                common += ["--hparam-sweep-max-combinations",str(args.max_combinations)]
        run_main(common,args.dry_run)
        if args.mode == "evaluate" and not args.dry_run:
            stats=validate_output(out,args.test_hosts,args.max_images,args.methods)
            (out / "reviewer45_completeness.json").write_text(json.dumps(
                {**stats,"protocol_hash":sha,"watermark":str(wm),"complete":True},indent=2)+"\n")
            print(label,"COMPLETE",stats)
        elif args.mode == "sweep" and not args.dry_run:
            fail_file=out/"proposal_hparam_sweep_failures.json"
            failures=json.loads(fail_file.read_text()) if fail_file.is_file() else ["missing failures.json"]
            if failures:
                raise RuntimeError(f"Incomplete sweep for {label}: {len(failures)} failures; {fail_file}")
            import pandas as pd
            combos=pd.read_csv(out/"proposal_hparam_sweep_summary_by_combo.csv")
            if combos.empty or combos.mean_attack_nc.isna().any():
                raise RuntimeError(f"Incomplete attacked hyperparameter results for {label}")
            hosts = list_image_files(args.train_hosts)
            expected_images = min(len(hosts), args.max_images) if args.max_images is not None else len(hosts)
            if expected_images < 1 or not {"images", "clean_rows", "attack_rows"}.issubset(combos.columns):
                raise RuntimeError(f"Invalid image/row counts in hyperparameter sweep {label}")
            incomplete = combos[(combos.images != expected_images) |
                                (combos.clean_rows != expected_images) |
                                (combos.attack_rows != 45 * expected_images)]
            if not incomplete.empty:
                raise RuntimeError(f"Incomplete 45-attack sweep for {label}: {len(incomplete)} parameter combinations")
    print("Both watermarks used identical reviewer45 attack definitions.")

if __name__ == "__main__":
    main()
