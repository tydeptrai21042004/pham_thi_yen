#!/usr/bin/env python3
"""Regenerate tau_Q / q_H figures from real reviewer45 parameter-sweep CSVs.

Never extract numerical data from the supplied reference PDF: it is used only
as a visual side-by-side reference with explicit, historical provenance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={6.0:'#440154',7.0:'#21918c',8.0:'#fde725'}
MARKERS={0.4:'o',0.5:'s',0.6:'^'}


def read_data(paths: dict[str,Path], expected_attacks: int=45) -> pd.DataFrame:
    all_frames=[]
    req={"q4_tau_effective","h01_q_effective","clean_psnr_mean","mean_attack_nc"}
    for label,path in paths.items():
        df=pd.read_csv(path)
        if not req.issubset(df.columns):
            raise ValueError(f"{path} missing columns {sorted(req-set(df.columns))}")
        if df.empty:
            raise ValueError(f"Empty sweep: {path}")
        key = [k for k in ("q4_tau_effective", "q4_margin_effective", "h01_q_effective", "h01_margin_effective") if k in df.columns]
        if df.duplicated(key).any():
            raise ValueError(f"Duplicate parameter settings in {path}")
        df=df.copy();df["watermark"]=label
        if "attack_rows" in df.columns and "images" in df.columns:
            invalid=df[df.attack_rows != expected_attacks*df.images]
            if not invalid.empty:
                raise ValueError(f"Incomplete {expected_attacks}-attack sweep: {path}: {len(invalid)} combinations")
        if "images" in df.columns and (df.images.min() < 1 or df.images.nunique() != 1):
            raise ValueError(f"Inconsistent number of training hosts in {path}")
        if "clean_rows" in df.columns and "images" in df.columns and (df.clean_rows != df.images).any():
            raise ValueError(f"Missing clean hyperparameter results in {path}")
        if not np.isfinite(df[["clean_psnr_mean","mean_attack_nc"]].to_numpy(float)).all():
            raise ValueError(f"Missing or nonfinite PSNR/NC in {path}")
        all_frames.append(df)
    data=pd.concat(all_frames,ignore_index=True)
    return data


def scatter(ax,data,title):
    for q in sorted(data.h01_q_effective.astype(float).unique()):
        for tau in sorted(data.q4_tau_effective.astype(float).unique()):
            part=data[np.isclose(data.h01_q_effective.astype(float),q)&np.isclose(data.q4_tau_effective.astype(float),tau)]
            if len(part):
                ax.scatter(part.clean_psnr_mean,part.mean_attack_nc,
                           c=COLORS.get(float(q),'#777777'), marker=MARKERS.get(round(float(tau),2),'D'),
                           edgecolors='#222222',linewidths=.45,s=53,alpha=.94)
    ax.set_xlabel('Mean PSNR before attack (dB)')
    ax.set_ylabel('Mean NC after attacks')
    ax.set_title(title)
    ax.grid(alpha=.27,linestyle=':')
    # Small and meaningful legends, qH color + tauQ marker exactly as reference.
    ch=[Line2D([],[],marker='o',linestyle='',markersize=7,markerfacecolor=COLORS.get(float(q),'#777'),
               markeredgecolor='#222',label=fr'$q_H={q:g}$') for q in sorted(data.h01_q_effective.astype(float).unique())]
    sh=[Line2D([],[],marker=MARKERS.get(round(float(tau),2),'D'),linestyle='',markersize=7,
               markerfacecolor='white',markeredgecolor='#222',label=fr'$\tau_Q={tau:g}$')
        for tau in sorted(data.q4_tau_effective.astype(float).unique())]
    l=ax.legend(handles=ch,loc='upper left',title='$q_H$',frameon=True,fontsize=8)
    ax.add_artist(l)
    ax.legend(handles=sh,loc='lower right',title=r'$\tau_Q$',fontsize=8)


def make_plots(data,output:Path,reference:Path|None):
    output.mkdir(parents=True,exist_ok=True)
    for label,df in data.groupby('watermark'):
        fig,ax=plt.subplots(figsize=(10.5,5.25))
        scatter(ax,df,f'{label}: 45-attack hyperparameter behavior')
        fig.tight_layout()
        fig.savefig(output/f'hparam_tradeoff_tauQ_qH_{label}.pdf',bbox_inches='tight')
        fig.savefig(output/f'hparam_tradeoff_tauQ_qH_{label}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    # Aggregate paired watermarks only where BOTH were measured on same setting.
    key=['q4_tau_effective','q4_margin_effective','h01_q_effective','h01_margin_effective']
    if len(data.watermark.unique()) > 1:
        key=[k for k in key if k in data.columns]
        ct=data.groupby(key)['watermark'].nunique()
        if (ct != data.watermark.nunique()).any():
            raise ValueError('W1 and W2 have different parameter grids; cannot create a shared mean curve')
        combined=data.groupby(key,as_index=False).agg(
            clean_psnr_mean=('clean_psnr_mean','mean'),mean_attack_nc=('mean_attack_nc','mean'))
    else:
        combined=data.copy()
    fig,ax=plt.subplots(figsize=(10.5,5.25))
    scatter(ax,combined,'Hyperparameter behavior: imperceptibility vs. robustness (45 attacks)')
    fig.tight_layout()
    fig.savefig(output/'hparam_tradeoff_tauQ_qH_consistent.pdf',bbox_inches='tight')
    fig.savefig(output/'hparam_tradeoff_tauQ_qH_consistent.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    combined.to_csv(output/'hparam_plot_source_data.csv',index=False)
    if reference is not None:
        if not reference.is_file():
            raise FileNotFoundError(reference)
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError('Install pymupdf to compare with the original PDF') from exc
        doc=fitz.open(reference)
        if len(doc)!=1:
            raise ValueError('Expected exactly one page in the reference plot')
        pix=doc[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
        image=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
        fig,(ax0,ax1)=plt.subplots(1,2,figsize=(18,6),gridspec_kw={'width_ratios':[1,1.12]})
        ax0.imshow(image)
        ax0.set_axis_off()
        ax0.set_title('Original supplied figure (historical results)')
        scatter(ax1,combined,'New evaluation: common reviewer45 protocol')
        fig.text(.5,.01,'Panels may use different axis ranges; only compare numerical values with a matching dataset and attack protocol.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,1])
        fig.savefig(output/'hparam_comparison_original_vs_reviewer45.pdf',bbox_inches='tight')
        fig.savefig(output/'hparam_comparison_original_vs_reviewer45.png',dpi=170,bbox_inches='tight')
        plt.close(fig)
        doc.close()
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w1',type=Path,required=True,help='W1 proposal_hparam_sweep_summary_by_combo.csv')
    p.add_argument('--w2',type=Path,required=True,help='W2 proposal_hparam_sweep_summary_by_combo.csv')
    p.add_argument('--reference-pdf',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/reviewer45_hparam_figures'))
    p.add_argument('--expected-attacks',type=int,default=45)
    args=p.parse_args()
    df=read_data({'W1':args.w1,'W2':args.w2},args.expected_attacks)
    make_plots(df,args.output,args.reference_pdf)
    print('Saved figure:',args.output/'hparam_tradeoff_tauQ_qH_consistent.pdf')

if __name__=='__main__':
    main()
