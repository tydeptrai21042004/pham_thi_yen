"""Offline regression tests: 4096-bit TrustMark protocol only, no downloaded weights."""
import numpy as np
from PIL import Image
import pytest

from watermarklab.experiments.neural_4096 import (
    watermark_bits, chunk_message, SpatialChunk4096, Recovered, _metrics,
)


class MockTrustMark:
    capacity = 61
    def embed(self, image, bits):
        output = image.copy()
        output.reshape(-1,3)[:61,0] = bits * 255
        return output
    def detect(self, image):
        return (image.reshape(-1,3)[:61,0] >= 128).astype('uint8'), True


def test_no_changes_to_4096_payload_or_native_chunking():
    wrapper = SpatialChunk4096(MockTrustMark(),name='trustmark_4096_spatial',cols=8,resize_to=0)
    assert wrapper.chunk_count == 68
    assert wrapper.capacity == 61
    assert len(wrapper.regions) == 68
    truth=(np.arange(4096)%2).astype('uint8')
    assert len(chunk_message(truth,61)) == 68
    assert np.array_equal(np.concatenate(chunk_message(truth,61))[:4096],truth)
    im=np.full((512,512,3),40,dtype='uint8')
    out=wrapper.embed(im,truth)
    decoded=wrapper.detect(out)
    assert decoded.valid.all() and np.array_equal(decoded.bits,truth)
    assert _metrics(truth,decoded)['fraction_correct_all_bits'] == 1.0


def test_erasure_no_false_perfect_all_bit_score():
    truth=(np.arange(4096)%2).astype('uint8')
    estimated=np.zeros(4096,dtype='uint8')
    valid=np.zeros(4096,dtype=bool)
    valid[:61]=True
    estimated[:61]=truth[:61]
    result=_metrics(truth,Recovered(estimated,valid,1,68,['undetected']))
    assert result['nc']==pytest.approx(1.0) # ONLY conditional
    assert result['nc_full_coverage'] is None
    assert result['fraction_correct_all_bits']==61/4096
    assert result['ber_erasure_as_error']==1-61/4096
    assert result['error']==''
    assert result['total_erased_bits']==4035
    assert 'undetected' in result['erasure_notes']


def test_total_erasure_gives_numeric_error_not_invented_nc():
    truth=(np.arange(4096)%2).astype('uint8')
    rec=Recovered(np.zeros(4096,dtype='uint8'),np.zeros(4096,dtype=bool),0,68,['all chunks rejected'])
    result=_metrics(truth,rec)
    assert result['nc'] is None
    assert result['nc_full_coverage'] is None
    assert result['fraction_correct_all_bits']==0.0
    assert result['ber_erasure_as_error']==1.0
    assert result['bit_coverage']==0.0


def test_explicit_alpha_watermark_conversion(tmp_path):
    path=tmp_path/'Cc.logo.circle.png'
    img=np.zeros((64,64,4),dtype=np.uint8)
    img[8:40,5:54,3]=255
    Image.fromarray(img,'RGBA').save(path)
    with pytest.raises(ValueError,match='Alpha-only'):
        watermark_bits(path)
    data=watermark_bits(path,binary_channel='alpha')
    assert data.shape==(4096,) and data.sum()==32*49
    with pytest.raises(ValueError,match='binary_channel'):
        watermark_bits(path,binary_channel='incorrect')
