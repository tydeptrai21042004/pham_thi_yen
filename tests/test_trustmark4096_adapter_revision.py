"""Offline contract tests for the optional TrustMark capacity adapter revision."""
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from watermarklab.experiments.neural_4096 import (
    SpatialChunk4096, Recovered, _metrics,
)
from watermarklab.experiments.neural_baselines import TrustMarkAdapter


class ConstantResidualCodec:
    capacity = 61

    def embed(self, patch, bits):
        encoded = patch.astype(np.int16)
        encoded[..., 0] = np.clip(encoded[..., 0] + 12, 0, 255)
        return encoded.astype(np.uint8)

    def detect(self, patch):
        return np.zeros(61, dtype=np.uint8), True


def test_projection_keeps_more_unmodified_host_detail():
    # High-frequency natural image variation should not be replaced by a
    # twice-resized source patch when transferring a watermark residual.
    rng = np.random.default_rng(41)
    host = rng.integers(30, 190, size=(512, 512, 3), dtype=np.uint8)
    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    codec = ConstantResidualCodec()
    legacy = SpatialChunk4096(codec, name='mock', cols=8, resize_to=256, projection='legacy')
    residual = SpatialChunk4096(codec, name='mock', cols=8, resize_to=256,
                                projection='residual', residual_gain=1.0)
    a = legacy.embed(host, bits)
    b = residual.embed(host, bits)
    assert a.shape == b.shape == host.shape
    assert np.mean((b.astype(float)-host.astype(float))**2) < np.mean(
        (a.astype(float)-host.astype(float))**2)
    assert b.dtype == np.uint8


class SecondViewCodec:
    capacity = 61

    def __init__(self):
        self.calls = 0

    def detect(self, patch):
        self.calls += 1
        if self.calls % 2:
            return None, False
        return np.ones(self.capacity, dtype=np.uint8), True

    def embed(self, patch, bits):
        return patch.copy()


def test_fallback_accepts_only_native_verified_bits():
    codec = SecondViewCodec()
    image = np.full((512, 512, 3), 120, dtype=np.uint8)
    wrapper = SpatialChunk4096(codec, name='mock', cols=8, resize_to=0,
                               decode_views='bicubic,lanczos')
    rec = wrapper.detect(image)
    assert rec.good_chunks == rec.fallback_chunks == wrapper.chunk_count == 68
    assert rec.valid.all() and rec.bits.all()
    assert codec.calls == 2 * 68


def test_all_rejected_stays_undecoded_and_zero_score():
    class Reject:
        capacity = 61
        def detect(self, patch):
            return None, False
    wrapper = SpatialChunk4096(Reject(), name='mock', cols=8, resize_to=0,
                               decode_views='bicubic,lanczos')
    rec = wrapper.detect(np.zeros((512, 512, 3), np.uint8))
    assert rec.good_chunks == rec.fallback_chunks == 0
    assert rec.rejected_chunks == 68
    assert not rec.valid.any()
    truth = np.tile([0,1],2048).astype(np.uint8)
    metrics = _metrics(truth,rec)
    assert metrics['nc'] is None
    assert metrics['nc_zero_fill'] == 0
    assert metrics['ber_erasure_as_error'] == 1.0


def test_upstream_strength_forwarded_only_if_requested(monkeypatch):
    captured = []
    class FakeTrustMark:
        class Encoding:
            BCH_5 = object()
        def __init__(self, **kwargs):
            pass
        def schemaCapacity(self):
            return 61
        def encode(self, image, payload, **kwargs):
            captured.append(kwargs)
            return image
    monkeypatch.setitem(sys.modules, 'trustmark', SimpleNamespace(TrustMark=FakeTrustMark))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    bits = np.zeros(61, dtype=np.uint8)
    native_image = np.zeros((256,256,3), dtype=np.uint8)
    TrustMarkAdapter(device='cpu', wm_strength=1.0).embed(native_image, bits)
    TrustMarkAdapter(device='cpu', wm_strength=1.5).embed(native_image, bits)
    assert captured[0] == {'MODE':'binary'}
    assert captured[1] == {'MODE':'binary', 'WM_STRENGTH':1.5}
