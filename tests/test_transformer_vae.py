"""Transformer VAE model and loss tests."""
from __future__ import annotations

import torch

from transformer_vae.losses import TransformerVAELoss
from transformer_vae.models.motion_transformer_vae import MotionTransformerVAE


def _small_model() -> MotionTransformerVAE:
    return MotionTransformerVAE(
        nfeats=70,
        window_size=10,
        history_len=0,
        latent_dim=[1, 16],
        h_dim=32,
        ff_size=64,
        num_layers=3,
        num_heads=2,
        dropout=0.0,
    )


def _group_slices():
    return {
        "joint_pos": (0, 29),
        "joint_vel": (29, 58),
        "pelvis_rot6d_b": (58, 64),
        "pelvis_lin_vel_b": (64, 67),
        "pelvis_ang_vel_b": (67, 70),
    }


def test_transformer_vae_forward_shapes():
    model = _small_model()
    x = torch.randn(4, 10, 70)

    recon, dist, info = model(x, sample=True)

    assert recon.shape == x.shape
    assert dist.loc.shape == (1, 4, 16)
    assert dist.scale.shape == (1, 4, 16)
    assert info["z"].shape == (1, 4, 16)
    assert info["mu"].shape == (1, 4, 16)
    assert info["logvar"].shape == (1, 4, 16)


def test_transformer_vae_decode_deterministic_shapes():
    model = _small_model()
    x = torch.randn(2, 10, 70)
    history, future = model.split_window(x)
    z, _dist, _mu, _logvar = model.encode(future, history, sample=False)

    out = model.decode(z, history, nfuture=10)

    assert out.shape == (2, 10, 70)


def test_transformer_vae_loss_backpropagates():
    model = _small_model()
    criterion = TransformerVAELoss(
        group_slices=_group_slices(),
        loss_type="huber",
        kl_weight=1.0e-4,
    )
    x = torch.randn(2, 10, 70)

    recon, dist, _info = model(x, sample=True)
    loss, terms = criterion(recon, x, dist)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(terms["rec"])
    assert torch.isfinite(terms["kl"])
    assert any(param.grad is not None for param in model.parameters())
