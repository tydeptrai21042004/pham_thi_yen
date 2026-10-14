"""Fast, offline contracts for the independent reviewer benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from watermarklab.common.attack import default_attack_suite
from watermarklab.experiments.splits import build_split
from watermarklab.experiments.evidence import audit
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams,ProposalQHDWTHess

ROOT=Path(__file__).resolve().parents[1]

def test_full_attack_suite_has_expected_reviewer_strengths():
    cases=default_attack_suite(False,preset='full')
    assert len(cases)==109
    assert len({x.name for x in cases})==109
    assert len({(x.group,str(sorted(x.params.items()))) for x in cases})==109
    names={x.name for x in cases}
    assert {'jpeg_q30','jpeg_q40','gaussian_var_0p05','speckle_var_0p03',
            'rotation_15deg','rotation_45deg'} <= names


def test_exact_nine_twelve_split_excludes_lenna(tmp_path):
    spec=build_split(ROOT/'data/host',tmp_path/'split.json',create_dirs=True)
    assert len(spec['image_hashes'])==21
    assert len(spec['train_names'])==9 and len(spec['test_names'])==12
    assert 'lenna.bmp' not in [x.lower() for x in spec['train_names']+spec['test_names']]
    assert set(spec['train_names']).isdisjoint(spec['test_names'])
    assert len(list((tmp_path/'splits'/'train').glob('*.bmp')))==9
    assert len(list((tmp_path/'splits'/'test').glob('*.bmp')))==12


def test_RGB_channel_variants_encode_and_decode():
    host=np.random.default_rng(10).integers(0,256,(64,64,3),dtype=np.uint8)
    wm=(np.random.default_rng(11).integers(0,2,(8,8))*255).astype(np.uint8)
    for channel in (0,1,2):
        params=ProposalParams.from_dict({'repeat':1,'host_embed_color_space':'RGB',
                                         'host_channels':[channel]})
        engine=ProposalQHDWTHess(params,use_optimizer=False)
        watermarked,key=engine.embed(host,wm)
        out=engine.extract(watermarked,key)
        assert out.shape==wm.shape


def test_strict_completeness_catches_missing_rows(tmp_path):
    df=pd.DataFrame([{'method_id':'proposal','image':'image','attack':'no_attack',
       'phase':'before_attack','psnr':51,'ssim':.99,'nc':1,'ber':0,
       'embed_time_sec':.1,'extract_time_sec':.02}])
    path=tmp_path/'results.csv';df.to_csv(path,index=False)
    import pytest
    with pytest.raises(RuntimeError,match='Incomplete experiment grid'):
        audit(path,tmp_path/'diagnostics',host_names=['image.bmp'],methods=['proposal'],
              attack_preset='lite')
    assert json.loads((tmp_path/'diagnostics'/'completeness.json').read_text())['status']=='FAIL'
