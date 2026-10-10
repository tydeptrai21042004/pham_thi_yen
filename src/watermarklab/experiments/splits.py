"""A frozen, reproducible 12-image protocol: 3 tuning / 9 held-out hosts.

The train fraction is NOT inferred from a previous Kaggle run. Image names
and hashes are saved so the protocol can be checked after publication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

SUPPORTED = {".bmp", ".png", ".jpg", ".jpeg"}
EXCLUDED = {"lenna.bmp"}


def build_split(host_dir: Path, output: Path, *, train_fraction: float = 0.25,
                seed: int = 3101, create_dirs: bool = False):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    files = sorted(p for p in Path(host_dir).iterdir() if p.is_file() and
                   p.suffix.lower() in SUPPORTED and p.name.casefold() not in EXCLUDED)
    # Rigid here by design: no silent addition of other images or Lenna.
    if len(files) != 12:
        raise ValueError(f"Expected exactly 12 allowed hosts after excluding Lenna; found {len(files)}")
    if round(len(files) * train_fraction) != 3:
        raise ValueError("This manuscript protocol requires exactly three tuning hosts")
    ranked = files.copy()
    random.Random(seed).shuffle(ranked)
    train, test = ranked[:3], ranked[3:]
    assert len(train) == 3 and len(test) == 9
    assert not {p.name.lower() for p in train} & {p.name.lower() for p in test}
    record = {
        "schema": "watermarklab-fixed-12-3-9-v2", "seed": seed,
        "train_fraction": train_fraction,
        "excluded_names": sorted(EXCLUDED),
        "train": [str(p.resolve()) for p in train],
        "test": [str(p.resolve()) for p in test],
        "train_names": [p.name for p in train],
        "test_names": [p.name for p in test],
        "image_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "policy": "No final-test image may be used for hyperparameter selection",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if create_dirs:
        for label, sample in (("train", train), ("test", test)):
            dst_dir = output.parent / "splits" / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            for old in dst_dir.iterdir():
                if old.is_file(): old.unlink()  # remove stale split images
            for source in sample:
                shutil.copy2(source, dst_dir / source.name)
        record["train_dir"] = str((output.parent / "splits" / "train").resolve())
        record["test_dir"] = str((output.parent / "splits" / "test").resolve())
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3101)
    ap.add_argument("--train-fraction", type=float, default=.25)
    ap.add_argument("--create-dirs", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_split(a.host_dir, a.output, train_fraction=a.train_fraction,
                                 seed=a.seed, create_dirs=a.create_dirs), indent=2))
