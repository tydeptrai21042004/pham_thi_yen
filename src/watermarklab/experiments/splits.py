"""Deterministic, disjoint image partitions for FA training and final evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def build_split(host_dir: Path, output: Path, *, train_fraction: float = 0.6, seed: int = 3101):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    files = sorted(p for p in Path(host_dir).iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})
    if len(files) < 2:
        raise ValueError("at least two host images required for disjoint train/test split")
    ranked=sorted(files,key=lambda p:hashlib.sha256(f"{seed}:{p.name}".encode()).hexdigest())
    cut=min(max(1,round(len(ranked)*train_fraction)),len(ranked)-1)
    record={"schema":"watermarklab-split-v1","seed":seed,"train_fraction":train_fraction,
            "train":[str(x.resolve()) for x in ranked[:cut]],"test":[str(x.resolve()) for x in ranked[cut:]],
            "image_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "policy":"disjoint host images; the test set must never be used for optimization"}
    assert set(record["train"]).isdisjoint(record["test"])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--host-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=3101)
    p.add_argument("--train-fraction",type=float,default=0.6)
    a=p.parse_args()
    print(json.dumps(build_split(a.host_dir,a.output,train_fraction=a.train_fraction,seed=a.seed),indent=2))
