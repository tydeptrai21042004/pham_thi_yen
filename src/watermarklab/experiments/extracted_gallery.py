"""One honest, fixed-host comparison: proposal + two classical baselines.

Never selects attack/image based on observed performance. PNG images and a
measured CSV are exported in addition to a single contact sheet.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from watermarklab.common.attack import default_attack_suite,apply_attack
from watermarklab.common.io_utils import load_host_rgb,load_watermark_binary
from watermarklab.common.metrics import nc,ber,psnr
from watermarklab.methods import build_methods
from watermarklab.benchmark import (load_proposal_param_file,load_guo_param_file,
                                    _select_proposal_params_for_image,_select_guo_lambda_for_image)

METHODS=('proposal','guo2017_dwt_qr_fa','hess_nha2023')
ATTACKS=('no_attack','jpeg_q30','gaussian_var_0p05','speckle_var_0p03','rotation_15deg')


def export(host:Path, watermark:Path, output:Path, *, proposal_params:Path|None=None,
           guo_params:Path|None=None, attack_names=ATTACKS,
           host_size:int=512, watermark_size:int=64):
    host=Path(host); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    attacks={c.name:c for c in default_attack_suite(True,preset='full')}
    for name in attack_names:
        if name not in attacks: raise KeyError(f'Unknown attack {name}')
    rgb=load_host_rgb(host, size=host_size); truth=load_watermark_binary(watermark, size=watermark_size)
    methods=build_methods(list(METHODS),proposal_options={'use_optimizer':False})
    sources={}
    if proposal_params is not None:
        source=load_proposal_param_file(proposal_params)
        if not source.get('loaded'): raise FileNotFoundError(f'Unusable proposal parameter file: {proposal_params}')
        methods['proposal'].params,sources['proposal']=_select_proposal_params_for_image(host,methods['proposal'].params,source)
    else: sources['proposal']='default_not_optimized'
    if guo_params is not None:
        source=load_guo_param_file(guo_params)
        if not source.get('loaded'): raise FileNotFoundError(f'Unusable Guo parameter file: {guo_params}')
        strength,sources['guo2017_dwt_qr_fa']=_select_guo_lambda_for_image(host,source)
        if strength is None:raise ValueError('Missing Guo global lambda')
        methods['guo2017_dwt_qr_fa'].lambda_strength=strength
    else: sources['guo2017_dwt_qr_fa']='default_not_optimized'
    sources['hess_nha2023']='baseline_default'
    rows=[]; panels=[]
    original=Image.fromarray(truth).convert('L')
    sz=original.size
    for method_id,method in methods.items():
        marked,key=method.embed(rgb,truth)
        for attack_name in attack_names:
            observed=marked if attack_name=='no_attack' else apply_attack(marked,attacks[attack_name])
            got=method.extract(observed,key,host_rgb=rgb)
            result=Image.fromarray(got).convert('L')
            if result.size!=sz: raise RuntimeError(f'Output size mismatch for {method_id}')
            saved=output/f'{method_id}__{attack_name}.png'
            result.save(saved)
            panels.append((method_id,attack_name,result.copy()))
            rows.append({'host':host.name,'watermark':Path(watermark).name,'method':method_id,
                         'attack':attack_name,'nc':nc(truth,got),'ber':ber(truth,got),
                         'clean_psnr':psnr(rgb,marked),'parameter_source':sources[method_id],
                         'extracted_image':saved.name})
    with (output/'gallery_metrics.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Nearest interpolation preserves the binary watermark visually.
    tile=195; margin=12; header=58
    sheet=Image.new('RGB',((len(METHODS)+1)*tile+margin*2,
                           len(attack_names)*(tile+16)+header+12),'white')
    draw=ImageDraw.Draw(sheet)
    cols=['Original',*METHODS]
    for i,label in enumerate(cols):draw.text((margin+i*tile,8),label,fill='black')
    for j,name in enumerate(attack_names):
        y=header+j*(tile+16)
        draw.text((margin,y),name,fill='black')
        sheet.paste(original.resize((tile-20,tile-20),Image.Resampling.NEAREST).convert('RGB'),(margin,y+20))
        for i,method_id in enumerate(METHODS,1):
            sample=next(img for mid,a,img in panels if mid==method_id and a==name)
            sheet.paste(sample.resize((tile-20,tile-20),Image.Resampling.NEAREST).convert('RGB'),
                        (margin+i*tile,y+20))
    sheet.save(output/'extracted_watermarks_comparison.png',optimize=True)
    (output/'gallery_manifest.json').write_text(json.dumps({'host':host.name,
            'watermark':Path(watermark).name,'methods':METHODS,'attacks':list(attack_names),
            'parameter_sources':sources,'selection':'predefined, not cherry-picked'},indent=2))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',type=Path,required=True)
    ap.add_argument('--watermark',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--proposal-params',type=Path)
    ap.add_argument('--guo-params',type=Path)
    ap.add_argument('--attacks',default=','.join(ATTACKS))
    ap.add_argument('--host-size',type=int,default=512)
    ap.add_argument('--watermark-size',type=int,default=64)
    a=ap.parse_args()
    print(f'Exported {len(export(a.host,a.watermark,a.output,proposal_params=a.proposal_params,
             guo_params=a.guo_params,attack_names=a.attacks.split(","),
             host_size=a.host_size,watermark_size=a.watermark_size))} gallery rows to {a.output}')
