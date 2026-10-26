"""Regression: all-zero extraction has undefined correlation but isn't an attack crash."""
import json
import math
from watermarklab.experiments.partial_results import AttackCheckpoint
from watermarklab.common.metrics import nc,ncc,ber
import numpy as np

FIELDS=('psnr','ssim','nc','ncc','ber','embed_time_sec','extract_time_sec')


def test_zero_extraction_checkpoint_is_strict_json_and_resumable(tmp_path):
    expected=np.zeros((64,64),dtype=np.uint8)
    expected[0:24]=255
    extracted=np.zeros_like(expected)
    assert math.isnan(nc(expected,extracted))
    assert math.isnan(ncc(expected,extracted))
    row={'method_id':'gaata2022_dwt_hess_fwa', 'image':'Girl',
         'phase':'after_attack','attack':'average_5x5',
         'psnr':29.4,'ssim':0.8,'nc':nc(expected,extracted),
         'ncc':ncc(expected,extracted),'ber':ber(expected,extracted),
         'embed_time_sec':1.0,'extract_time_sec':0.1}
    c=AttackCheckpoint(tmp_path/'a.json','run1',numeric_columns=FIELDS)
    c.prepare(np.zeros((8,8),dtype=np.uint8))
    c.set_attack('average_5x5',row)
    obj=json.loads((tmp_path/'a.json').read_text(),parse_constant=lambda x:(_ for _ in ()).throw(AssertionError(x)))
    assert obj['attacks']['average_5x5']['nc'] is None
    assert obj['attacks']['average_5x5']['ncc'] is None
    c2=AttackCheckpoint(tmp_path/'a.json','run1',numeric_columns=FIELDS)
    recovered=c2.attack('average_5x5')
    assert recovered is not None and recovered['ber']==row['ber']
    assert recovered['nc'] is None


def test_missing_noncorrelation_metric_is_not_accepted(tmp_path):
    c=AttackCheckpoint(tmp_path/'a.json','x',numeric_columns=FIELDS)
    invalid=dict(psnr=40,ssim=0.9,nc=None,ncc=None,ber=float('nan'),
                 embed_time_sec=1,extract_time_sec=1)
    c.prepare(np.zeros((2,2),dtype=np.uint8))
    c.set_attack('jpeg',invalid)
    assert c.attack('jpeg') is None
    assert json.loads((tmp_path/'a.json').read_text())['attacks']['jpeg']['ber'] is None
