"""No model downloads: verify bit mapping, failure accounting and runner contracts."""
import hashlib
import csv
import numpy as np
from PIL import Image
import pytest

from watermarklab.experiments.neural_4096 import (
    watermark_bits, chunk_message, deterministic_regions,
    SpatialChunk4096, _metrics, Recovered, run,
)


class PixelCodec:
    capacity = 61
    def embed(self, patch, message):
        assert patch.shape[0] > 0
        out = patch.copy()
        pixels = out.reshape(-1, 3)
        pixels[:self.capacity, 0] = message * 255
        return out

    def detect(self, patch):
        return (patch.reshape(-1, 3)[:self.capacity, 0] >= 128).astype(np.uint8), True


def make_watermark(path):
    pattern=np.fromfunction(lambda i,j: ((i*19 + j*13 + (i//4))%7)<3, (64,64),dtype=int)
    Image.fromarray(np.uint8(pattern)*255).save(path)
    return pattern.astype('uint8').ravel()


def test_load_original_watermark_and_chunk_exact(tmp_path):
    path=tmp_path/'wm.png';truth=make_watermark(path)
    assert np.array_equal(watermark_bits(path),truth)
    for native,n in [(61,68),(32,128)]:
        chunks=chunk_message(truth,native)
        assert len(chunks)==n
        assert np.array_equal(np.concatenate(chunks)[:4096],truth)
        assert sum(c.size for c in chunks)==native*n
    with pytest.raises(FileNotFoundError):
        watermark_bits(tmp_path/'missing.png')


def test_spatial_extended_messages_reassemble_from_one_image(tmp_path):
    bits=make_watermark(tmp_path/'wm.png')
    host=np.zeros((512,512,3),dtype=np.uint8)
    for capacity,cols,chunk_count in [(61,8,68),(32,8,128)]:
        codec=PixelCodec();codec.capacity=capacity
        wrapper=SpatialChunk4096(codec,name='mock',cols=cols,resize_to=0)
        assert len(wrapper.regions)==chunk_count
        marked=wrapper.embed(host,bits)
        assert marked.shape==(512,512,3)
        recovered=wrapper.detect(marked)
        assert recovered.good_chunks==chunk_count
        assert recovered.valid.all()
        assert np.array_equal(recovered.bits,bits)
        assert _metrics(bits,recovered)['ber_erasure_as_error']==0.0


def test_erasure_must_not_fake_perfect_correlation(tmp_path):
    bits=make_watermark(tmp_path/'wm.png')
    good=np.zeros(4096,dtype=bool);good[:61]=True
    estimated=np.zeros(4096,dtype=np.uint8);estimated[:61]=bits[:61]
    rec=Recovered(estimated,good,1,68,['67 missing'])
    result=_metrics(bits,rec)
    assert result['bit_coverage']==61/4096
    assert result['ber']==0.0 # conditional only
    assert result['ber_erasure_as_error']>=4035/4096
    assert result['exact_recovery']==0
    assert result['nc_zero_fill']<1


def test_end_to_end_csv_mock_same_inputs(tmp_path):
    host_dir=tmp_path/'hosts';host_dir.mkdir()
    Image.fromarray(np.full((512,512,3),48,dtype=np.uint8)).save(host_dir/'host.png')
    wm=tmp_path/'wm.png';truth=make_watermark(wm)
    out=tmp_path/'out'
    result=run(host_dir=host_dir,watermark=wm,output=out,methods=('trustmark',),
         attack_preset='none',wrapper_factory=lambda n:SpatialChunk4096(PixelCodec(),name=n,cols=8,resize_to=0))
    assert len(result)==2
    assert result[0]['phase']=='before_attack'
    assert result[0]['bit_coverage']==1
    assert result[0]['exact_recovery']==1
    assert (out/'comparison_4096.csv').is_file()
    assert (out/'trustmark'/'host_results.csv').is_file()
    with (out/'per_image_attack_results.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2


def test_proposal_not_in_payload_wrapper():
    # Architectural contract: wrapper imports the native adapters, never proposal.
    import inspect
    import watermarklab.experiments.neural_4096 as n
    assert 'ProposalQHDWTHess' not in inspect.getsource(n)
