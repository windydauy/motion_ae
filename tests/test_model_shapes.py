"""Encoder / Decoder / AutoEncoder shape 测试。"""
import torch
import pytest

from motion_ae.models.encoder import MLPEncoder
from motion_ae.models.decoder import MLPDecoder
from motion_ae.models.autoencoder import MotionAutoEncoder


D = 68
W = 10
FLAT = W * D
LATENT = 8
LEVELS = [8] * LATENT


class TestEncoder:
    def test_output_shape(self):
        enc = MLPEncoder(FLAT, [512, 256], LATENT)
        x = torch.randn(4, FLAT)
        z = enc(x)
        assert z.shape == (4, LATENT)

    def test_gradient(self):
        enc = MLPEncoder(FLAT, [256], LATENT)
        x = torch.randn(2, FLAT, requires_grad=True)
        z = enc(x)
        z.sum().backward()
        assert x.grad is not None


class TestDecoder:
    def test_output_shape(self):
        dec = MLPDecoder(LATENT, [256, 512], FLAT)
        z = torch.randn(4, LATENT)
        out = dec(z)
        assert out.shape == (4, FLAT)


class TestAutoEncoder:
    @pytest.fixture
    def model(self):
        return MotionAutoEncoder(
            feature_dim=D,
            window_size=W,
            encoder_hidden_dims=[256],
            decoder_hidden_dims=[256],
            ifsq_levels=LEVELS,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, W, D)
        x_hat, z_d, info = model(x)
        assert x_hat.shape == (4, W, D)
        assert z_d.shape == (4, LATENT)

    def test_encode_only(self, model):
        x = torch.randn(2, W, D)
        z_dequant, z_d, info = model.encode(x)
        assert z_dequant.shape == (2, LATENT)
        assert z_d.shape == (2, LATENT)

    def test_decode_only(self, model):
        z = torch.randn(2, LATENT)
        out = model.decode(z)
        assert out.shape == (2, W, D)

    def test_end_to_end_gradient(self, model):
        x = torch.randn(2, W, D, requires_grad=True)
        x_hat, _z_d, _info = model(x)
        loss = ((x_hat - x) ** 2).mean()
        loss.backward()
        assert x.grad is not None
