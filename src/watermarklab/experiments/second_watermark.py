"""Non-ambiguous deterministic alpha-to-binary conversion for the secondary logo.

The supplied Cc.logo.circle.png has zero-valued RGB and visible alpha. Converting
it to ordinary L/grayscale loses the watermark entirely. This only prepares an
input file, never changes the proposed embedding/extraction method.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def prepare(source:Path, destination:Path, *, alpha_threshold:int=127) -> Path:
    source,destination=Path(source),Path(destination)
    if not 0<=alpha_threshold<=255:raise ValueError('Invalid alpha threshold')
    with Image.open(source) as image:
        if image.size!=(64,64) or 'A' not in image.getbands():
            raise ValueError('Secondary logo must be a 64x64 image with alpha channel')
        alpha=np.asarray(image.getchannel('A'),dtype=np.uint8)
    binary=np.where(alpha>=alpha_threshold,255,0).astype(np.uint8)
    if len(np.unique(binary))!=2:
        raise ValueError('Secondary logo preprocessing produced a constant watermark')
    destination.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(binary,mode='L').save(destination)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'original':source.name,'original_sha256':digest(source),
              'preprocessed':destination.name,'preprocessed_sha256':digest(destination),
              'shape':[64,64],'alpha_threshold':alpha_threshold,
              'source_channel':'A', 'zero_bits':int(np.sum(binary==0)),
              'one_bits':int(np.sum(binary==255)),
              'protocol':'separate cohort; no mixing with original wm.png results'}
    (destination.parent/'secondary_watermark_preprocessing.json').write_text(json.dumps(manifest,indent=2))
    return destination
