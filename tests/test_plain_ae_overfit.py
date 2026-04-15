"""Plain AE 固定 batch 过拟合调试脚本测试。"""
from __future__ import annotations

import json

import torch

from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.models.plain_autoencoder import PlainMotionAutoEncoder
from scripts.debug_plain_ae_overfit import (
    compare_models_on_batch,
    overfit_single_batch,
    save_history_json,
)


def test_plain_autoencoder_forward_shape():
    model = PlainMotionAutoEncoder(
        feature_dim=68,
        window_size=10,
        encoder_hidden_dims=[64],
        decoder_hidden_dims=[64],
        latent_dim=4,
    )
    x = torch.randn(3, 10, 68)

    x_hat, info = model(x)

    assert x_hat.shape == x.shape
    assert info["z_c"].shape == (3, 4)


def test_overfit_single_batch_reduces_loss():
    torch.manual_seed(0)
    batch = torch.randn(16, 10, 68)
    model = PlainMotionAutoEncoder(
        feature_dim=68,
        window_size=10,
        encoder_hidden_dims=[128],
        decoder_hidden_dims=[128],
        latent_dim=8,
    )
    criterion = ReconstructionLoss(
        group_slices={
            "joint_pos": (0, 29),
            "joint_vel": (29, 58),
            "pelvis_quat_b": (58, 62),
            "pelvis_lin_vel_b": (62, 65),
            "pelvis_ang_vel_b": (65, 68),
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = overfit_single_batch(
        model=model,
        batch=batch,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        steps=20,
    )

    assert len(history) == 20
    assert history[-1] < history[0]


def test_compare_models_on_batch_returns_both_histories(tmp_path):
    torch.manual_seed(0)
    batch = torch.randn(8, 10, 68)
    criterion = ReconstructionLoss(
        group_slices={
            "joint_pos": (0, 29),
            "joint_vel": (29, 58),
            "pelvis_quat_b": (58, 62),
            "pelvis_lin_vel_b": (62, 65),
            "pelvis_ang_vel_b": (65, 68),
        }
    )
    plain_model = PlainMotionAutoEncoder(
        feature_dim=68,
        window_size=10,
        encoder_hidden_dims=[64],
        decoder_hidden_dims=[64],
        latent_dim=8,
    )
    ifsq_model = MotionAutoEncoder(
        feature_dim=68,
        window_size=10,
        encoder_hidden_dims=[64],
        decoder_hidden_dims=[64],
        ifsq_levels=[8] * 8,
    )

    result = compare_models_on_batch(
        batch=batch,
        criterion=criterion,
        plain_model=plain_model,
        ifsq_model=ifsq_model,
        device=torch.device("cpu"),
        steps=5,
    )

    assert set(result.keys()) >= {"plain_ae", "ifsq_ae"}
    assert len(result["plain_ae"]["history"]) == 5
    assert len(result["ifsq_ae"]["history"]) == 5
    assert "init_model_loss" in result["plain_ae"]
    assert "init_model_loss" in result["ifsq_ae"]

    output_path = tmp_path / "history.json"
    save_history_json(result, str(output_path))
    saved = json.loads(output_path.read_text())
    assert "plain_ae" in saved
    assert "ifsq_ae" in saved
