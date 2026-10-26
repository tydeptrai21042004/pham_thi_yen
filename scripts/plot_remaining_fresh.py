"""Plot only measured, aggregated remaining results; do not invent missing curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def generate(out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classical=out/'classical'
    figures=out/'fresh_remaining_figures'
    figures.mkdir(parents=True,exist_ok=True)
    created={};missing=[]

    def graph(name, path, group, value, phase=None, exclude_clean=False):
        if not path.is_file():
            missing.append(str(path.relative_to(out))); return
        df=pd.read_csv(path)
        if phase is not None and 'phase' in df.columns:
            df=df[df.phase==phase]
        if exclude_clean and 'attack' in df.columns:
            df=df[~df['attack'].isin(('no_attack','clean'))]
        if group not in df.columns or value not in df.columns:
            missing.append(str(path.relative_to(out))+' missing '+group+'/'+value); return
        df[value]=pd.to_numeric(df[value],errors='coerce')
        counts=df.groupby(group)[value].agg(['mean','count']).sort_index()
        if counts.empty or counts['count'].sum()==0:
            missing.append(name+': no defined numeric observations');return
        fig,ax=plt.subplots(figsize=(max(7,min(15,0.8*len(counts)+4)),5))
        counts['mean'].plot(kind='bar',ax=ax)
        ax.set_ylabel(f'Measured mean {value} (defined observations only)')
        ax.set_title(name)
        ax.tick_params(axis='x',rotation=40)
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(figures/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
        created[name]={'source':str(path.relative_to(out)),
                       'counts_defined':{str(k):int(v) for k,v in counts['count'].items()}}

    graph('block_size_attacked_nc',classical/'block_size_comparison/per_case.csv','block_size','nc',exclude_clean=True)
    graph('block_size_attacked_ber',classical/'block_size_comparison/per_case.csv','block_size','ber',exclude_clean=True)
    graph('channel_attacked_nc',classical/'channel_ablation/channel_ablation.csv','channel','nc','after_attack')
    graph('channel_attacked_ber',classical/'channel_ablation/channel_ablation.csv','channel','ber','after_attack')
    graph('security_raw_nc',classical/'review_audit/per_case_results.csv','case','watermark_nc')
    graph('bss_correlation',classical/'bss_validation/bss_per_host_attack.csv',
          'attack','spearman_rho_bss')
    for name in ('gaata2022_dwt_hess_fwa','dwt_wht_svd_2024'):
        path=out/'failed45_from_scratch'/name/'per_image_attack_results.csv'
        graph('failed45_'+name+'_nc',path,'attack','nc','after_attack')
    # A selective held-out cohort is a measured comparison ONLY within its
    # own host/method scope. Never combine the 12-host Hessenberg subgroup
    # with the two-host WHT subgroup or imply a complete 13-method grid.
    for folder in sorted((classical/'heldout_selective').glob('*')):
        scope_file = folder/'scope.json'
        table = folder/'per_image_attack_results.csv'
        if not scope_file.is_file():
            continue
        scope = json.loads(scope_file.read_text())
        if scope.get('complete_for_selected_scope') is not True or not scope.get('not_a_full_13_method_comparison'):
            continue
        label = 'heldout_selective_' + folder.name
        graph(label+'_mean_nc', table, 'method_id', 'nc', 'after_attack')
        graph(label+'_mean_ber', table, 'method_id', 'ber', 'after_attack')
        created.setdefault(label+'_scope', {'methods': scope.get('methods', []),
                  'hosts': scope.get('hosts', []),
                  'not_a_full_13_method_comparison': True})
    atomic = figures/'FIGURE_MANIFEST.json'
    atomic.write_text(json.dumps({'created':created,'missing':missing,
                                  'note':'Figures generated only from available raw measurements. NaN ignored in mean and counted explicitly. Never evidence of full baseline comparison.'},indent=2))
    return len(created)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    opts=p.parse_args();print('Figures created:',generate(opts.out))
