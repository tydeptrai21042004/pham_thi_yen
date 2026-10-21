from pathlib import Path
import numpy as np
import pandas as pd
from watermarklab.common.attack import (
    manuscript43_attack_suite,reviewer45_attack_suite,reviewer45_family,
    default_attack_suite,apply_attack,REVIEWER45_GROUPS,
)


def test_exact_review_protocol():
    old=manuscript43_attack_suite(False)
    new=reviewer45_attack_suite(False)
    assert len(old)==43 and len(new)==45
    o={a.name for a in old}; n={a.name for a in new}
    assert o-n=={'lowpass_3x3'}
    assert n-o=={'resize_2p0','crop_resize_50','shear_x_0p10'}
    assert len(set(x.name for x in new))==45
    assert sum(s.stop-s.start for s in REVIEWER45_GROUPS.values())==45
    assert set(x.name for x in default_attack_suite(preset='reviewer45',include_none=False))==n
    for a in new:
        assert reviewer45_family(a.name) in REVIEWER45_GROUPS


def test_operators_are_deterministic_and_keep_shape():
    image=np.random.default_rng(172).integers(0,255,(48,64,3),dtype=np.uint8)
    for a in reviewer45_attack_suite(False):
        out=apply_attack(image,a)
        assert out.shape==image.shape and out.dtype==np.uint8,a.name
        assert np.array_equal(out,apply_attack(image,a)),a.name


def test_dropped_lowpass_is_really_same_operator():
    image=np.random.default_rng(1).integers(0,255,(32,32,3),dtype=np.uint8)
    old={x.name:x for x in manuscript43_attack_suite(False)}
    assert np.array_equal(apply_attack(image,old['lowpass_3x3']),
                          apply_attack(image,old['average_3x3']))


def test_plot_rejects_partial_sweep(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from plot_reviewer45_hparams import read_data
    path=tmp_path/'summary.csv'
    pd.DataFrame([{'q4_tau_effective':0.4,'h01_q_effective':6,
                   'clean_psnr_mean':52.2,'mean_attack_nc':0.94,
                   'images':1,'attack_rows':44}]).to_csv(path,index=False)
    import pytest
    with pytest.raises(ValueError,match='Incomplete'):
        read_data({'W1':path})


def test_plot_creates_pdf_for_paired_completed_sweeps(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from plot_reviewer45_hparams import read_data,make_plots
    source=tmp_path/'sweep.csv'
    pd.DataFrame([{'q4_tau_effective':0.4,'q4_margin_effective':0.08,
                   'h01_q_effective':6.,'h01_margin_effective':0.9,
                   'clean_psnr_mean':52.12,'mean_attack_nc':0.94,
                   'images':1,'attack_rows':45}]).to_csv(source,index=False)
    df=read_data({'W1':source,'W2':source})
    make_plots(df,tmp_path/'output',reference=None)
    assert (tmp_path/'output/hparam_tradeoff_tauQ_qH_consistent.pdf').is_file()
    assert (tmp_path/'output/hparam_tradeoff_tauQ_qH_W1.pdf').is_file()
    assert (tmp_path/'output/hparam_tradeoff_tauQ_qH_W2.pdf').is_file()


def test_reviewer45_size_report_rejects_incomplete_grid(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from report_manuscript43_resize import build
    attacks=[a.name for a in reviewer45_attack_suite(False)]
    clean=[]; attacked=[]
    for w in ['wm.png','Cc.logo.circle.png']:
        for size in [64,128]:
            clean.append(dict(host_image='Girl.bmp',watermark_file=w,watermark_size=size,
                              clean_psnr=52.0,clean_ssim=0.99,clean_nc=0.92,
                              clean_ber=0.08,proposal_repeat_factor=2.0))
            for name in attacks:
                attacked.append(dict(host_image='Girl.bmp',watermark_file=w,watermark_size=size,
                                     attack=name,nc_after_attack=0.8,ber_after_attack=0.2))
    pd.DataFrame(clean).to_csv(tmp_path/'proposal_size_sweep_clean_results.csv',index=False)
    attack_csv=tmp_path/'proposal_size_sweep_attack_results.csv'
    pd.DataFrame(attacked).to_csv(attack_csv,index=False)
    status=build(tmp_path,expected_hosts=1,expected_watermarks=2,attack_preset='reviewer45')
    assert status['status']=='PASS' and status['expected_attack_rows']==180
    assert (tmp_path/'reviewer45_paired_128_minus_64.csv').is_file()
    pd.DataFrame(attacked[:-1]).to_csv(attack_csv,index=False)
    import pytest
    with pytest.raises(RuntimeError,match='completeness'):
        build(tmp_path,expected_hosts=1,expected_watermarks=2,attack_preset='reviewer45')
