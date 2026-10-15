"""Reviewer R2.2 experimental **capacity extensions** of TrustMark and WAM.

This is NOT native 4096-bit functionality of the original pretrained networks.
Each 64x64 watermark is partitioned into 68 native 61-bit TrustMark or 128
native 32-bit WAM messages. Independent region-level invocations are assembled
into ONE 512x512 host. All model outputs and failure/erasure masks are retained.
Up/down-resizing tiny regions can erase their watermarks: this is an empirical
feasibility experiment, not a claim that these pretrained models support it.

Run from repository root with PYTHONPATH=src:
 python -m watermarklab.experiments.neural_4096 --methods trustmark,wam \\
  --host-dir results/splits/test --watermark data/watermark/wm.png \\
  --output results/reviewer_final/neural_4096 --max-attacks 2

Requires official TrustMark / WAM dependencies and weights as described in
neural_baselines.py. Use --force-full only to continue stress tests when clean
recovery is poor; those outcomes must be reported as capacity-extension failures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from watermarklab.common.attack import apply_attack, default_attack_suite
from watermarklab.common.metrics import psnr, ssim
from watermarklab.experiments.neural_baselines import _nc_ncc
from watermarklab.experiments.neural_baselines import TrustMarkAdapter, WAMAdapter, _hosts
from watermarklab.experiments.review_audit import load_rgb

SOURCE_SHAPE = (64, 64)
HOST_SHAPE = (512, 512, 3)
DATA_BITS = 4096
RESULT_COLUMNS = [
    "method_id", "image", "watermark", "phase", "attack", "attack_group", "psnr", "ssim",
    "nc", "ncc", "ber", "nc_zero_fill", "ncc_zero_fill", "ber_erasure_as_error",
    "bit_coverage", "decoded_chunks", "total_chunks", "exact_recovery", "clean_gate_passed",
    "embedding_time_sec", "extraction_time_sec", "native_chunk_capacity_bits",
    "source_payload_bits", "encoded_chunk_slots", "side_info_bytes", "error",
]


def watermark_bits(path: Path) -> np.ndarray:
    """A canonical grayscale binary bitmap; fail on alpha-only/all-zero sources."""
    with Image.open(path) as img:
        if img.size != (64, 64):
            raise ValueError(f"Expected 64x64 original watermark, got {img.size}: {path}")
        if img.mode == "RGBA":
            a = np.asarray(img.getchannel("A"))
            rgb = np.asarray(img.convert("RGB"))
            if np.all(rgb == 0) and np.any(a != 255):
                raise ValueError("Alpha-only watermark requires an explicit conversion; refusing all-black RGB")
        image = np.asarray(img.convert("L"), dtype=np.uint8)
    bits = (image >= 127).astype(np.uint8).ravel()
    if bits.size != DATA_BITS or bits.min() == bits.max():
        raise ValueError("Watermark must contain 4096 bits and both bit values")
    return bits


def chunk_message(bits: np.ndarray, chunk_capacity: int) -> list[np.ndarray]:
    v = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if v.size != DATA_BITS or not np.isin(v, (0, 1)).all():
        raise ValueError("Expected 4096 binary input bits")
    if chunk_capacity < 1:
        raise ValueError("Invalid native capacity")
    count = math.ceil(v.size / chunk_capacity)
    padded = np.pad(v, (0, count * chunk_capacity - DATA_BITS))
    return list(padded.reshape(count, chunk_capacity))


def deterministic_regions(height: int, width: int, count: int, *, cols: int) -> list[tuple[int, int, int, int]]:
    """Non-overlapping rectangular regions covering the host (unused cells allowed)."""
    rows = math.ceil(count / cols)
    if min(height // rows, width // cols) < 1:
        raise ValueError("Host too small for number of regions")
    yy = [round(i * height / rows) for i in range(rows + 1)]
    xx = [round(i * width / cols) for i in range(cols + 1)]
    return [(yy[i // cols], yy[i // cols + 1], xx[i % cols], xx[i % cols + 1])
            for i in range(count)]


@dataclass
class Recovered:
    bits: np.ndarray
    valid: np.ndarray
    good_chunks: int
    chunks: int
    errors: list[str]


class SpatialChunk4096:
    """Model-independent spatial wrapper, with model-specific native adapters.

    resize_to is a controlled experimental choice: pixel regions are too small
    for typical checkpoints. Encoding/decoding both occur after upsampling,
    followed by restoration to the original region size. Does not silently
    change the host size or combine different original host images.
    """
    def __init__(self, native: Any, *, name: str, cols: int, resize_to: int = 256):
        self.native = native
        self.name = name
        self.capacity = int(native.capacity)
        self.cols = cols
        self.resize_to = int(resize_to)
        if self.capacity <= 0 or self.resize_to < 0:
            raise ValueError("Invalid wrapper settings")
        self.chunk_count = math.ceil(DATA_BITS / self.capacity)
        self.regions = deterministic_regions(512, 512, self.chunk_count, cols=cols)
        if len(set(self.regions)) != self.chunk_count:
            raise AssertionError("Region collision")

    def _prepare(self, patch: np.ndarray) -> np.ndarray:
        if not self.resize_to:
            return patch.copy()
        if patch.shape[:2] == (self.resize_to, self.resize_to):
            return patch.copy()
        return np.asarray(Image.fromarray(patch).resize((self.resize_to, self.resize_to), Image.Resampling.BICUBIC), dtype=np.uint8)

    @staticmethod
    def _restore(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if img.shape[:2] == size:
            return img
        return np.asarray(Image.fromarray(img).resize((size[1], size[0]), Image.Resampling.BICUBIC), dtype=np.uint8)

    def embed(self, host: np.ndarray, message: np.ndarray) -> np.ndarray:
        if host.shape != HOST_SHAPE or host.dtype != np.uint8:
            raise ValueError("Each host must be RGB uint8 512x512")
        blocks = chunk_message(message, self.capacity)
        output = host.copy()
        for i, ((y0, y1, x0, x1), block) in enumerate(zip(self.regions, blocks)):
            source = self._prepare(host[y0:y1, x0:x1])
            try:
                encoded = self.native.embed(source, block)
            except Exception as exc:
                raise RuntimeError(f"{self.name}: encode chunk {i+1}/{self.chunk_count} failed: {type(exc).__name__}: {exc}") from exc
            encoded = np.asarray(encoded)
            if encoded.ndim != 3 or encoded.shape[-1] != 3:
                raise ValueError(f"Unexpected native image shape for chunk {i}: {encoded.shape}")
            output[y0:y1, x0:x1] = self._restore(np.clip(encoded, 0, 255).astype(np.uint8), (y1-y0, x1-x0))
        return output

    def detect(self, image: np.ndarray) -> Recovered:
        if image.shape != HOST_SHAPE:
            raise ValueError("Decoder requires the attacked full-size 512x512 RGB image")
        recovered = np.zeros(self.chunk_count * self.capacity, dtype=np.uint8)
        valid = np.zeros_like(recovered, dtype=bool)
        errors: list[str] = []
        good = 0
        for i, (y0, y1, x0, x1) in enumerate(self.regions):
            try:
                patch = self._prepare(image[y0:y1, x0:x1])
                decoded, present = self.native.detect(patch)
                if decoded is None or present is False:
                    errors.append(f"chunk {i}: decoder rejected or returned no message")
                    continue
                v = np.asarray(decoded, dtype=np.uint8).reshape(-1)
                if v.size != self.capacity or not np.isin(v, (0, 1)).all():
                    errors.append(f"chunk {i}: invalid decoded payload")
                    continue
                a, b = i*self.capacity, (i+1)*self.capacity
                recovered[a:b] = v
                valid[a:b] = True
                good += 1
            except Exception as exc:
                errors.append(f"chunk {i}: {type(exc).__name__}: {exc}")
        return Recovered(recovered[:DATA_BITS], valid[:DATA_BITS], good, self.chunk_count, errors)

    @property
    def layout_bytes(self) -> int:
        """Public fixed layout needs no per-image position map; include config bytes."""
        return len(json.dumps({"cols":self.cols, "count":self.chunk_count,
                               "native_bits":self.capacity,"resize":self.resize_to},
                              sort_keys=True).encode())


def make_wrapper(method: str, *, resize_to: int = 256, device: str = "auto", model_type: str = "Q") -> SpatialChunk4096:
    if method == "trustmark":
        return SpatialChunk4096(TrustMarkAdapter(model_type=model_type, device=device),
                                name="trustmark_4096_spatial", cols=8, resize_to=resize_to)
    if method == "wam":
        return SpatialChunk4096(WAMAdapter(device=device), name="wam_4096_spatial",
                                cols=8, resize_to=resize_to)
    raise ValueError(f"Unknown method {method}")


def _metrics(truth: np.ndarray, rec: Recovered) -> dict:
    """Conditional NC/BER remain undefined on missing bits; no fabricated predictions."""
    known = rec.valid
    conditional_nc,conditional_ncc = _nc_ncc(truth[known], rec.bits[known]) if known.any() else (None,None)
    conditional_ber = float(np.mean(truth[known] != rec.bits[known])) if known.any() else None
    zero_nc,zero_ncc = _nc_ncc(truth,rec.bits)
    return {"nc": conditional_nc, "ncc":conditional_ncc,"ber":conditional_ber,
            "nc_zero_fill":zero_nc, "ncc_zero_fill":zero_ncc,
            "ber_erasure_as_error":float(np.mean(np.where(known, truth != rec.bits, True))),
            "bit_coverage":float(np.mean(known)),
            "exact_recovery":int(bool(np.all(known) and np.array_equal(truth, rec.bits))),
            "decoded_chunks":rec.good_chunks,"total_chunks":rec.chunks,
            "error": " | ".join(rec.errors[:3]) + (f" (+{len(rec.errors)-3} more)" if len(rec.errors)>3 else "")}


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def run(*, host_dir: Path, watermark: Path, output: Path, methods: tuple[str, ...] = ('trustmark','wam'),
        attack_preset: str = 'full', max_attacks: int = 0, max_images: int | None = None,
        resize_to: int = 256, device: str = 'auto', model_type: str = 'Q',
        max_clean_ber: float = .05, min_clean_coverage: float = .95,
        force_full: bool = False, save_images: bool = False,
        wrapper_factory: Callable[[str], SpatialChunk4096] | None = None) -> list[dict]:
    """Run all attacks on ONE complete marked host, with per-host resume.

    For full testing, host_dir must be the existing held-out split directory;
    the test hosts must never be used to choose region layout or resize_to.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    hosts = _hosts(Path(host_dir), max_images)
    truth = watermark_bits(Path(watermark))
    cases = default_attack_suite(include_none=False, preset=attack_preset)
    if max_attacks < 0: raise ValueError('max_attacks must be nonnegative')
    if max_attacks: cases = cases[:max_attacks]
    all_rows = []
    manifest = {'mode':'experimental_spatial_chunk_4096', 'native_model_4096_support': False,
                'status':'running', 'source_payload_bits':DATA_BITS,
                'watermark_sha256':hashlib.sha256(Path(watermark).read_bytes()).hexdigest(),
                'methods':list(methods),'hosts':[x.name for x in hosts],
                'attacks':[x.name for x in cases],'attack_preset':attack_preset,
                'config':{'resize_to':resize_to,'max_clean_ber':max_clean_ber,
                          'min_clean_coverage':min_clean_coverage,'force_full':force_full},
                'python':platform.python_version(), 'numpy':np.__version__,
                'warning':'Matched 4096-bit source watermark, NOT native 4096-bit pretrained-model capacity. Full clean extraction is not guaranteed.'}
    for method in methods:
        wrapper = wrapper_factory(method) if wrapper_factory else make_wrapper(method,resize_to=resize_to,device=device,model_type=model_type)
        for host_path in hosts:
            per_host = output / method / (host_path.stem + '_results.csv')
            sidecar = output / method / (host_path.stem + '_config.json')
            run_contract = {
                "host_sha256":hashlib.sha256(host_path.read_bytes()).hexdigest(),
                "watermark_sha256":manifest["watermark_sha256"],
                "method":method,"native_capacity":wrapper.capacity,"region_cols":wrapper.cols,
                "resize_to":wrapper.resize_to,"attacks":[c.name for c in cases],
                "device":device,"model_type":model_type,"force_full":force_full,
                "save_images":save_images,
                "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_adapter_sha256":hashlib.sha256(Path(__file__).with_name('neural_baselines.py').read_bytes()).hexdigest(),
                "clean_gate":{"max_ber":max_clean_ber,"min_coverage":min_clean_coverage}}
            # A cached, exact encoded image is necessary for meaningful resume.
            # Re-embedding can be nondeterministic: an old CSV plus a freshly
            # encoded image would silently mix measurements from different hosts.
            cache = output / method / (host_path.stem + '_marked_cache.png')
            previous: list[dict] = []
            cache_ok = False
            saved_contract = {}
            if sidecar.is_file():
                try:
                    saved_contract = json.loads(sidecar.read_text(encoding='utf-8'))
                    expected_sha = saved_contract.get('marked_sha256')
                    cache_ok = (cache.is_file() and bool(expected_sha) and
                                hashlib.sha256(cache.read_bytes()).hexdigest() == expected_sha and
                                all(saved_contract.get(k) == v for k,v in run_contract.items()))
                except (ValueError, OSError):
                    cache_ok = False
            if cache_ok and per_host.is_file():
                try:
                    with per_host.open(newline='',encoding='utf-8') as f:
                        previous = list(csv.DictReader(f))
                    expected_attacks = {case.name for case in cases}
                    have_attacks = [r.get('attack') for r in previous if r.get('phase')=='after_attack']
                    header_ok = (len(previous)>=2 and previous[0].get('phase')=='before_attack'
                                 and previous[1].get('phase')=='control'
                                 and all(r.get('image')==host_path.stem for r in previous))
                    previous_ok = (header_ok and len(have_attacks)==len(set(have_attacks))
                                   and set(have_attacks).issubset(expected_attacks)
                                   and all(r.get('source_payload_bits')=='4096' for r in previous)
                                   and all(r.get('method_id')==wrapper.name for r in previous)
                                   and all(r.get('watermark')==Path(watermark).name for r in previous)
                                   and all(r.get('bit_coverage') and r.get('psnr')
                                           for r in previous[:2]))
                    if not previous_ok:
                        previous=[]
                    else:
                        # Retry execution exceptions that have NO valid metrics.
                        # Genuine undetected chunks DO retain a numeric coverage.
                        previous=[r for r in previous if not (
                            r.get('phase')=='after_attack' and
                            (not r.get('bit_coverage') or not r.get('psnr')))]
                        have_attacks=[r.get('attack') for r in previous
                                      if r.get('phase')=='after_attack']
                        if (len(have_attacks)==len(cases) or
                            (not force_full and len(previous)==2 and
                             previous[0].get('clean_gate_passed') in ('False','0'))):
                            all_rows.extend(previous)
                            continue
                        print(f'[NEURAL-4096 RESUME] {method} {host_path.name}: '
                              f'{len(have_attacks)}/{len(cases)} attacks from cached embedding',flush=True)
                except (ValueError,OSError,csv.Error):
                    previous=[]
            host = load_rgb(host_path,None)
            if host.shape != HOST_SHAPE:
                raise ValueError(f'{host_path.name}: host must be exactly 512x512 RGB without resizing')
            t0=time.perf_counter()
            if cache_ok:
                with Image.open(cache) as im:
                    marked=np.asarray(im.convert('RGB'), dtype=np.uint8)
                embed_sec = float(previous[0].get('embedding_time_sec') or 0) if previous else 0.0
            else:
                # If the previous checkpoint was created by an older version
                # without this cache, discard ALL old measurements for this host.
                # This is required for reproducibility, even if re-embedding costs time.
                try:
                    marked=wrapper.embed(host,truth)
                except Exception as exc:
                    failure={"method_id":wrapper.name,"image":host_path.stem,
                             "watermark":Path(watermark).name,"phase":"embedding_failed",
                             "attack":"no_attack","source_payload_bits":DATA_BITS,
                             "native_chunk_capacity_bits":wrapper.capacity,
                             "encoded_chunk_slots":wrapper.chunk_count*wrapper.capacity,
                             "embedding_time_sec":time.perf_counter()-t0,
                             "error":f"{type(exc).__name__}: {exc}"}
                    _save_csv(per_host,[failure]);all_rows.append(failure)
                    _save_csv(output/'per_image_attack_results.csv',all_rows)
                    continue
                embed_sec=time.perf_counter()-t0
                cache.parent.mkdir(parents=True,exist_ok=True)
                cache_tmp=cache.with_name(cache.name+'.tmp')
                Image.fromarray(marked).save(cache_tmp,format='PNG')
                cache_tmp.replace(cache)
                previous=[]
                cache_ok=True
            run_contract = {**run_contract, 'marked_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
            # Save provenance immediately, even if the clean decoder is interrupted.
            side_tmp=sidecar.with_suffix('.json.tmp')
            side_tmp.write_text(json.dumps(run_contract, indent=2),encoding='utf-8')
            side_tmp.replace(sidecar)
            if save_images:
                dest=output / method / 'marked_images'
                dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(marked).save(dest/(host_path.stem+'.png'))
            rows=list(previous)
            def save_progress():
                _save_csv(per_host,rows)
                side_tmp=sidecar.with_suffix('.json.tmp')
                side_tmp.write_text(json.dumps(run_contract,indent=2))
                side_tmp.replace(sidecar)
                _save_csv(output/'per_image_attack_results.csv',all_rows+rows)

            def evaluate(image:np.ndarray, phase:str, attack_name:str, group:str) -> tuple[dict,Recovered]:
                tic=time.perf_counter()
                rec=wrapper.detect(image)
                ext=time.perf_counter()-tic
                metrics = _metrics(truth,rec)
                record={"method_id":wrapper.name,"image":host_path.stem,"watermark":Path(watermark).name,
                        "phase":phase,"attack":attack_name,"attack_group":group,
                        "psnr":psnr(host,image),"ssim":ssim(host,image),
                        "embedding_time_sec":embed_sec,"extraction_time_sec":ext,
                        "source_payload_bits":DATA_BITS,"native_chunk_capacity_bits":wrapper.capacity,
                        "encoded_chunk_slots":wrapper.chunk_count*wrapper.capacity,
                        "side_info_bytes":wrapper.layout_bytes, **metrics}
                rows.append(record)
                return record,rec
            if previous:
                clean=rows[0]
                gate=str(clean.get('clean_gate_passed')).lower() in ('true','1')
            else:
                clean,_=evaluate(marked,'before_attack','no_attack','clean')
                gate=bool(clean['bit_coverage']>=min_clean_coverage and clean['ber_erasure_as_error']<=max_clean_ber)
                clean['clean_gate_passed']=gate
                # Negative control: never include in attack averages.
                control,_=evaluate(host,'control','original_unmarked','control')
                control['clean_gate_passed']=gate
                save_progress()
            if gate or force_full:
                already={r.get('attack') for r in rows if r.get('phase')=='after_attack'}
                for case in cases:
                    if case.name in already:
                        continue
                    try:
                        attacked = apply_attack(marked,case)
                        record,_=evaluate(attacked,'after_attack',case.name,case.group)
                        record['clean_gate_passed']=gate
                    except Exception as exc:
                        rows.append({"method_id":wrapper.name,"image":host_path.stem,
                                     "watermark":Path(watermark).name,"phase":"after_attack",
                                     "attack":case.name,"attack_group":case.group,
                                     "source_payload_bits":DATA_BITS,"clean_gate_passed":gate,
                                     "error":f"attack/decoder error: {type(exc).__name__}: {exc}"})
                    save_progress()  # at most one attack is lost on interruption
            save_progress()
            all_rows.extend(rows)
            print(f'[NEURAL-4096] {method} {host_path.name}: clean coverage={float(clean["bit_coverage"]):.3f} '
                  f'BER-with-erasures={float(clean["ber_erasure_as_error"]):.3f}, '
                  f'gate={gate}, attack_cases={len(rows)-2}',flush=True)
            _save_csv(output/'per_image_attack_results.csv',all_rows)
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    _save_csv(output/'per_image_attack_results.csv',all_rows)
    import pandas as pd
    df=pd.DataFrame(all_rows)
    summaries=[]
    for (name,phase), grp in df[df.phase.isin(('before_attack','after_attack'))].groupby(['method_id','phase']):
        row={'method_id':name,'phase':phase,'n':len(grp),
             'failure_aware_protocol':'zero_fill_NC_and_erasure_as_error_BER_are_diagnostic_scores'}
        for k in ('psnr','ssim','nc','ncc','ber','nc_zero_fill','ncc_zero_fill','ber_erasure_as_error','bit_coverage',
                  'exact_recovery','decoded_chunks','embedding_time_sec','extraction_time_sec'):
            row[k+'_mean']=float(pd.to_numeric(grp[k],errors='coerce').mean())
        summaries.append(row)
    summary_columns=['method_id','phase','n','failure_aware_protocol'] + [
        k+'_mean' for k in ('psnr','ssim','nc','ncc','ber','nc_zero_fill','ncc_zero_fill','ber_erasure_as_error',
                           'bit_coverage','exact_recovery','decoded_chunks',
                           'embedding_time_sec','extraction_time_sec')]
    # Keep a readable CSV schema even if every embed fails and no summary row exists.
    pd.DataFrame(summaries,columns=summary_columns).to_csv(output/'comparison_4096.csv',index=False)
    failed_cases=[x for x in all_rows if x.get('phase')=='embedding_failed' or
                  (x.get('phase')=='after_attack' and x.get('error'))]
    clean_rejected=[x for x in all_rows if x.get('phase')=='before_attack' and
                    str(x.get('clean_gate_passed')) in ('False','0')]
    all_attacks = [(str(x.get('method_id')),str(x.get('image')),str(x.get('attack')))
                   for x in all_rows if x.get('phase')=='after_attack']
    complete_grid = (len(all_attacks) == len(methods)*len(hosts)*len(cases) and
                     len(set(all_attacks)) == len(all_attacks) and
                     not any(x.get('phase')=='embedding_failed' for x in all_rows))
    # Keep the established scientific status labels. Machine-verifiable
    # completeness is an independent field: status alone is never sufficient.
    manifest['status'] = ('complete_with_failures' if failed_cases else
                          'capacity_not_validated' if clean_rejected else 'complete')
    manifest['complete_grid'] = complete_grid
    manifest['failure_cases']=len(failed_cases)
    manifest['clean_gate_rejected_hosts']=len(clean_rejected)
    manifest['expected_attacked_rows']=len(methods)*len(hosts)*len(cases)
    manifest['actual_attacked_rows']=sum(x.get('phase')=='after_attack' for x in all_rows)
    manifest['rows']=len(all_rows)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return all_rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--watermark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',default='trustmark,wam')
    p.add_argument('--attack-preset',default='full')
    p.add_argument('--max-attacks',type=int,default=0)
    p.add_argument('--max-images',type=int)
    p.add_argument('--resize-to',type=int,default=256)
    p.add_argument('--device',default='auto')
    p.add_argument('--model-type',default='Q')
    p.add_argument('--max-clean-ber',type=float,default=.05)
    p.add_argument('--min-clean-coverage',type=float,default=.95)
    p.add_argument('--force-full',action='store_true')
    p.add_argument('--save-images',action='store_true')
    a=p.parse_args(argv)
    methods=tuple(x.strip().lower() for x in a.methods.split(',') if x.strip())
    if not methods or not set(methods).issubset({'wam','trustmark'}): p.error('Methods must be trustmark,wam')
    run(host_dir=a.host_dir,watermark=a.watermark,output=a.output,methods=methods,
        attack_preset=a.attack_preset,max_attacks=a.max_attacks,max_images=a.max_images,
        resize_to=a.resize_to,device=a.device,model_type=a.model_type,
        max_clean_ber=a.max_clean_ber,min_clean_coverage=a.min_clean_coverage,
        force_full=a.force_full,save_images=a.save_images)


if __name__=='__main__': main()
