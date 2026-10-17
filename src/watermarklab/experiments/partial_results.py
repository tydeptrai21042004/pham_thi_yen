"""Durable, input-fingerprinted checkpoints for long benchmark attack grids.

The checkpoint is evidence of completed rows, never a completion certificate.
An interrupted job recomputes the embedding and checks its bytes before
reusing any previously measured attack rows.  Mismatched inputs invalidate
cached data rather than silently blending results across protocols.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint(*objects: Any) -> str:
    payload = json.dumps(objects, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hash(image: np.ndarray) -> str:
    a = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str((a.shape, a.dtype)).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def _valid_row(row: Any, numerical: tuple[str, ...]) -> bool:
    if not isinstance(row, dict):
        return False
    try:
        return all(math.isfinite(float(row[name])) for name in numerical)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class AttackCheckpoint:
    def __init__(self, path: str | Path, signature: str, *, numeric_columns: tuple[str, ...]):
        self.path = Path(path)
        self.signature = str(signature)
        self.numeric_columns = numeric_columns
        self.state: dict[str, Any] = {"signature": self.signature, "marked_sha256": None,
                                      "clean": None, "attacks": {}}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if (saved.get("signature") == self.signature and
                    isinstance(saved.get("attacks"), dict)):
                self.state = saved
        except (OSError, ValueError, AttributeError):
            pass

    def prepare(self, marked_image: np.ndarray):
        current = image_hash(marked_image)
        if self.state.get("marked_sha256") != current:
            self.state = {"signature": self.signature, "marked_sha256": current,
                          "clean": None, "attacks": {}}
            self.flush()

    def clean(self):
        item = self.state.get("clean")
        return item if _valid_row(item, self.numeric_columns) else None

    def attack(self, name: str):
        item = self.state.get("attacks", {}).get(name)
        return item if _valid_row(item, self.numeric_columns) else None

    def set_clean(self, row: dict):
        self.state["clean"] = row
        self.flush()

    def set_attack(self, name: str, row: dict):
        self.state.setdefault("attacks", {})[name] = row
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, allow_nan=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)
