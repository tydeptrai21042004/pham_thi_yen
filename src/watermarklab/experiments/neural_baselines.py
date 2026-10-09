"""Optional pretrained neural baseline evaluator (models are NOT bundled).

TrustMark: pip install trustmark
ChunkySeal: follow https://github.com/facebookresearch/videoseal installation.

Each baseline uses its native encoded payload capacity, NOT a 4096-bit
watermark. Do not combine these results with capacity-matched comparisons.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from watermarklab.common.attack import default_attack_suite, apply_attack
from watermarklab.common.metrics import psnr, ssim
from watermarklab.experiments.review_audit import load_rgb


class TrustMarkAdapter:
    """Official binary API; imports/downloads weights only when instantiated."""
    name = "TrustMark2025"

    def __init__(self, model_type: str = "Q"):
        try:
            from trustmark import TrustMark
        except ImportError as exc:
            raise RuntimeError("TrustMark optional dependency missing: pip install trustmark") from exc
        self.model = TrustMark(verbose=False, model_type=model_type, encoding_type=TrustMark.Encoding.BCH_5)
        self.capacity = int(self.model.schemaCapacity())

    def embed(self, rgb: np.ndarray, bits: np.ndarray) -> np.ndarray:
        text="".join(str(int(x)) for x in bits)
        if len(text) != self.capacity:
            raise ValueError(f"TrustMark requires {self.capacity} native information bits")
        im=self.model.encode(Image.fromarray(rgb),text,MODE="binary")
        return np.array(im.convert("RGB"),dtype=np.uint8)

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray | None,bool]:
        payload,present,_schema=self.model.decode(Image.fromarray(rgb),MODE="binary")
        if not present or not isinstance(payload,str) or len(payload)!=self.capacity or set(payload)-set("01"):
            return None,False
        return np.fromiter((int(c) for c in payload),dtype=np.uint8),True


class ChunkySealAdapter:
    """Pretrained VideoSeal image mode; requires torch and videoseal."""
    name = "ChunkySeal2025"

    def __init__(self, device: str = "cpu"):
        try:
            import torch
            import videoseal
        except ImportError as exc:
            raise RuntimeError("ChunkySeal dependencies missing: install official videoseal + torch") from exc
        self.torch=torch
        self.device=device
        self.model=videoseal.load("chunkyseal").eval().to(device)
        self.capacity=1024

    def _tensor(self,rgb):
        return self.torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).unsqueeze(0).float().to(self.device)/255.0

    def embed(self,rgb:np.ndarray,bits:np.ndarray)->np.ndarray:
        if bits.size != self.capacity:
            raise ValueError("ChunkySeal expects a 1024-bit native payload")
        with self.torch.no_grad():
            msg=self.torch.from_numpy(bits.reshape(1,-1).astype(np.float32)).to(self.device)
            result=self.model.embed(self._tensor(rgb),msgs=msg,is_video=False)
        marked=result["imgs_w"][0].detach().cpu().clamp(0,1)
        return (marked.permute(1,2,0).numpy()*255).round().astype(np.uint8)

    def detect(self,rgb:np.ndarray)->tuple[np.ndarray | None,bool]:
        with self.torch.no_grad():
            result=self.model.detect(self._tensor(rgb),is_video=False)
        predictions=result["preds"].detach().cpu().float()
        if predictions.ndim>2:
            predictions=predictions.mean(dim=tuple(range(2,predictions.ndim)))
        predictions=predictions.reshape(-1)
        if predictions.numel() != self.capacity+1:
            raise RuntimeError(f"Unexpected ChunkySeal output shape: {tuple(result['preds'].shape)}")
        bits=(predictions[1:]>0).numpy().astype(np.uint8)
        # Presence scores are model-dependent; expose measured bits only rather
        # than invent a universally calibrated decision threshold.
        return bits, True


def run(method_name:str,host:Path,out:Path,*,attack_preset:str="lite",max_attacks:int=5,seed:int=3101):
    adapter=TrustMarkAdapter() if method_name=="trustmark" else ChunkySealAdapter()
    original=load_rgb(host,None)
    bits=np.random.default_rng(seed).integers(0,2,adapter.capacity,dtype=np.uint8)
    t0=time.perf_counter(); marked=adapter.embed(original,bits); embed_secs=time.perf_counter()-t0
    out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(marked).save(out/"marked.png")
    cases=[("marked_clean",marked),("original_unmarked",original)]
    for a in default_attack_suite(include_none=False,preset=attack_preset)[:max_attacks]:
        cases.append(("attack:"+a.name,apply_attack(marked,a)))
    rows=[]
    for name,img in cases:
        t1=time.perf_counter(); decoded,present=adapter.detect(img)
        rows.append({"model":adapter.name,"case":name,"native_capacity_bits":adapter.capacity,
                     "raw_ber":float(np.mean(bits!=decoded)) if decoded is not None else None,
                     "detected":bool(present),"embed_seconds":embed_secs,
                     "decode_seconds":time.perf_counter()-t1,
                     "clean_psnr":psnr(original,marked),"clean_ssim":ssim(original,marked)})
    with (out/"results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata={"method":adapter.name,"payload_kind":"native_random_bits","capacity":adapter.capacity,
              "seed":seed,"model_weights":"downloaded by upstream library; record checkpoint hashes manually before publication",
              "attack_preset":attack_preset,"experiment":"optional_neural_not_capacity_matched"}
    (out/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return rows

if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--method",choices=("trustmark","chunkyseal"),required=True)
    a.add_argument("--host",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--attack-preset",default="lite")
    a.add_argument("--max-attacks",type=int,default=5)
    a.add_argument("--seed",type=int,default=3101)
    v=a.parse_args()
    print(json.dumps(run(v.method,v.host,v.output,attack_preset=v.attack_preset,max_attacks=v.max_attacks,seed=v.seed),indent=2))
