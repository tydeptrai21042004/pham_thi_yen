"""Frozen, explicit 9-tuning / 12-held-out RGB/BMP host protocol.

Only the nine separately supplied new images are used for parameter tuning.
All twelve original benchmark hosts (including the three previously used for
optimization) become held-out evaluation images. Lenna stays excluded.
This is a new dataset protocol, not a random resplit of the previous experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image

TRAIN_NAMES = (
    "NB0001.bmp", "NB0025.bmp", "NB0196.bmp", "NB0215.bmp", "NB0232.bmp",
    "NB0688.bmp", "NB1353.bmp", "fruits.bmp", "seaport.bmp",
)
ORIGINAL_TEST_NAMES = (
    "Girl.bmp", "airplane.bmp", "athens.bmp", "baboon.bmp", "butfly.bmp",
    "house.bmp", "manhatan.bmp", "milkdrop.bmp", "pepper.bmp", "safari.bmp",
    "sailboat.bmp", "tiffany.bmp",
)
PREVIOUS_TUNING_NAMES = ("baboon.bmp", "safari.bmp", "sailboat.bmp")
EXCLUDED = {"lenna.bmp"}


def _validate_folder(directory: Path, expected_names: tuple[str, ...], label: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {label} image directory: {directory}")
    found = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".bmp"]
    by_fold = {p.name.casefold(): p for p in found}
    if len(found) != len(by_fold):
        raise ValueError(f"Case-insensitive filename collision in {label}: {directory}")
    # Only required images participate; extra unrelated BMPs are never silently sampled.
    required = {name.casefold() for name in expected_names}
    actual = set(by_fold)
    if label == "test":
        actual -= EXCLUDED
    if actual != required:
        missing = sorted(required - actual)
        extras = sorted(actual - required)
        raise ValueError(f"{label} BMP host mismatch (missing={missing}, extra={extras}) in {directory}")
    return [by_fold[name.casefold()] for name in expected_names]


def _validate_image(path: Path) -> str:
    with Image.open(path) as img:
        if img.format != "BMP" or img.size != (512, 512) or img.mode != "RGB":
            raise ValueError(f"Host must be 512x512 RGB BMP: {path} (got {img.format}, {img.size}, {img.mode})")
        # Hash decoded pixels as well as bytes; equivalent images cannot be split across groups.
        return hashlib.sha256(img.tobytes()).hexdigest()


def build_split(host_dir: Path, output: Path, *, train_dir: Path | None = None,
                train_fraction: float | None = None, seed: int = 3101,
                create_dirs: bool = False):
    """Materialize the named 9/12 protocol; ``seed`` is legacy metadata, not used.

    ``train_fraction`` is retained only for compatibility with earlier callers,
    but cannot change the frozen, explicitly named assignment.
    """
    host_dir = Path(host_dir).resolve()
    train_dir = Path(train_dir).resolve() if train_dir is not None else host_dir.parent / "host_train"
    expected_fraction = 9 / 21
    if train_fraction is not None and abs(train_fraction - expected_fraction) > 1e-10:
        raise ValueError(f"Fixed 9/12 split has train_fraction=9/21, not {train_fraction}")
    train = _validate_folder(train_dir, TRAIN_NAMES, "train")
    test = _validate_folder(host_dir, ORIGINAL_TEST_NAMES, "test")
    all_files = train + test
    if len({p.name.casefold() for p in all_files}) != 21:
        raise ValueError("Training and testing hosts have overlapping filenames")
    decoded_hashes = {p.name: _validate_image(p) for p in all_files}
    if len(set(decoded_hashes.values())) != len(all_files):
        raise ValueError("Duplicate image pixels detected between dataset hosts")
    image_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in all_files}
    record = {
        "schema": "watermarklab-fixed-21-9-12-v3",
        "assignment": "explicit_filenames_no_random_split",
        "seed": seed,
        "train_fraction": expected_fraction,
        "excluded_names": sorted(EXCLUDED),
        "previous_tuning_names_moved_to_test": list(PREVIOUS_TUNING_NAMES),
        "train": [str(p) for p in train],
        "test": [str(p) for p in test],
        "train_names": [p.name for p in train],
        "test_names": [p.name for p in test],
        "image_hashes": image_hashes,
        "decoded_rgb_sha256": decoded_hashes,
        "policy": "Optimize/tune only on nine new hosts; evaluate on all twelve original hosts",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if create_dirs:
        for label, sample in (("train", train), ("test", test)):
            dst_dir = output.parent / "splits" / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            for old in dst_dir.iterdir():
                if old.is_file():
                    old.unlink()
            for source in sample:
                shutil.copy2(source, dst_dir / source.name)
        record["train_dir"] = str((output.parent / "splits" / "train").resolve())
        record["test_dir"] = str((output.parent / "splits" / "test").resolve())
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-dir", type=Path, required=True,
                    help="Original host directory: 12 test images plus excluded lenna.bmp")
    ap.add_argument("--train-dir", type=Path, default=None,
                    help="Directory of nine new training BMPs (default: sibling host_train)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3101,
                    help="Legacy metadata only; named split does not use randomness")
    ap.add_argument("--train-fraction", type=float, default=None,
                    help="Optional assertion; must equal 9/21")
    ap.add_argument("--create-dirs", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_split(a.host_dir, a.output, train_dir=a.train_dir,
                                 train_fraction=a.train_fraction, seed=a.seed,
                                 create_dirs=a.create_dirs), indent=2))
