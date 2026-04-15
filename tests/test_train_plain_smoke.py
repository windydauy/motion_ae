"""Plain AE 全数据训练入口的兼容性 smoke test。"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_ae.config import MotionAEConfig, DataConfig
from motion_ae.dataset import MotionWindowDataset
from motion_ae.losses import ReconstructionLoss
from motion_ae.utils.normalization import FeatureNormalizer, compute_stats
from scripts.train_plain_ae import TrainerCompatiblePlainAutoEncoder


def _make_fake_npz(path: str, T: int = 50):
    rng = np.random.RandomState(7)
    quats = rng.randn(T, 37, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    np.savez(
        path,
        joint_pos=rng.randn(T, 29).astype(np.float32),
        joint_vel=rng.randn(T, 29).astype(np.float32),
        body_pos_w=rng.randn(T, 37, 3).astype(np.float32),
        body_quat_w=quats,
        body_lin_vel_w=rng.randn(T, 37, 3).astype(np.float32),
        body_ang_vel_w=rng.randn(T, 37, 3).astype(np.float32),
        fps=np.array([30]),
    )


def test_train_plain_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "motion.npz")
        _make_fake_npz(npz_path)

        cfg = MotionAEConfig()
        cfg.data = DataConfig(data_path=tmpdir)
        cfg.window_size = 10
        cfg.stride = 5
        cfg.model.encoder_hidden_dims = [64]
        cfg.model.decoder_hidden_dims = [64]
        cfg.model.ifsq_levels = [4, 4, 4, 4]

        ds = MotionWindowDataset([npz_path], cfg, normalizer=None)
        mean, std = compute_stats(ds.all_features)
        normalizer = FeatureNormalizer(mean, std)
        ds_norm = MotionWindowDataset([npz_path], cfg, normalizer=normalizer)

        loader = DataLoader(ds_norm, batch_size=4, shuffle=True)
        model = TrainerCompatiblePlainAutoEncoder(
            feature_dim=ds_norm.slices.total_dim,
            window_size=cfg.window_size,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            latent_dim=cfg.model.latent_dim,
        )
        criterion = ReconstructionLoss(group_slices=ds_norm.slices.as_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        model.train()
        for step, batch in enumerate(loader):
            x_hat, z_d, info = model(batch)
            loss, _ = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            assert z_d.shape == (batch.shape[0], cfg.model.latent_dim)
            assert info["z_c"].shape == (batch.shape[0], cfg.model.latent_dim)

            if step >= 1:
                break

        assert len(losses) >= 2
        assert all(np.isfinite(loss) for loss in losses)
