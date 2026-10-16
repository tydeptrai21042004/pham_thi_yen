"""Controlled Y versus R/G/B single-channel host embedding ablation.

These are channel-variant experiments, not independently optimized proposals.
The trained global proposal parameters are fixed for all four runs.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from watermarklab.common.attack import default_attack_suite,apply_attack
from watermarklab.common.io_utils import load_host_rgb,load_watermark_binary,list_image_files
from watermarklab.common.metrics import nc,ber,psnr,ssim
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams, ProposalQHDWTHess
from watermarklab.benchmark import load_proposal_param_file

MODES={'Y':('YCrCb',(0,)), 'R':('RGB',(0,)), 'G':('RGB',(1,)), 'B':('RGB',(2,))}

def run(host_dir:Path, watermark:Path, output:Path, *, param_file:Path,
        preset:str='reviewer_stress', max_attacks:int=0, max_images:int|None=None):
    payload=load_proposal_param_file(param_file)
    if not payload.get('global_params'):
        raise ValueError('A trained global proposal parameter file is mandatory')
    host_files=[p for p in list_image_files(host_dir) if p.name.lower()!='lenna.bmp']
    if max_images:host_files=host_files[:max_images]
    wm=load_watermark_binary(watermark)
    attacks=default_attack_suite(False,preset)
    if max_attacks:attacks=attacks[:max_attacks]
    rows=[];errors=[]
    for mode,(color_space,channels) in MODES.items():
        params=ProposalParams.from_dict({**payload['global_params'],
            'host_embed_color_space':color_space,'host_channels':channels})
        method=ProposalQHDWTHess(params=params,use_optimizer=False)
        for host_path in host_files:
            host=load_host_rgb(host_path)
            try:
                marked,key=method.embed(host,wm)
                # Clean extraction is a first-class measurement for EVERY channel.
                clean=method.extract(marked,key,host_rgb=host)
                rows.append({'channel':mode,'host':host_path.name,'attack':'no_attack',
                             'phase':'before_attack','nc':nc(wm,clean),'ber':ber(wm,clean),
                             'psnr_clean':psnr(host,marked),'ssim_clean':ssim(host,marked)})
                for cfg in attacks:
                    try:
                        attacked=apply_attack(marked,cfg)
                        got=method.extract(attacked,key,host_rgb=host)
                        rows.append({'channel':mode,'host':host_path.name,'attack':cfg.name,
                                     'phase':'after_attack','nc':nc(wm,got),'ber':ber(wm,got),
                                     'psnr_clean':psnr(host,marked),'ssim_clean':ssim(host,marked)})
                    except Exception as e:errors.append({'mode':mode,'host':host_path.name,'attack':cfg.name,'error':repr(e)})
            except Exception as e:errors.append({'mode':mode,'host':host_path.name,'error':repr(e)})
    output.mkdir(parents=True,exist_ok=True)
    if rows:
        with (output/'channel_ablation.csv').open('w',newline='',encoding='utf-8') as fp:
            w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'errors.json').write_text(json.dumps(errors,indent=2))
    manifest={'host_count':len(host_files),'attack_count':len(attacks),'modes':list(MODES),
        'expected_rows':len(host_files)*(len(attacks)+1)*len(MODES),'actual_rows':len(rows),
        'expected_clean_rows':len(host_files)*len(MODES), 'actual_clean_rows':sum(r['phase']=='before_attack' for r in rows),
        'status':'PASS' if not errors and len(rows)==len(host_files)*(len(attacks)+1)*len(MODES) else 'FAIL',
        'notes':'Fixed trained parameters, different host color channels; same watermark and attacks'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if manifest['status']!='PASS':raise RuntimeError('Incomplete channel ablation; see errors.json')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host-dir',type=Path,required=True)
    ap.add_argument('--watermark',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--param-file',type=Path,required=True)
    ap.add_argument('--preset',default='reviewer_stress')
    ap.add_argument('--max-attacks',type=int,default=0)
    ap.add_argument('--max-images',type=int)
    a=ap.parse_args()
    print(json.dumps(run(a.host_dir,a.watermark,a.output,param_file=a.param_file,preset=a.preset,
        max_attacks=a.max_attacks,max_images=a.max_images),indent=2))
