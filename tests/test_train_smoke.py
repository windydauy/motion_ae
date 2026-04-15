"""训练 smoke test：在合成数据上跑 2 步。"""
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from motion_ae.config import MotionAEConfig, DataConfig
from motion_ae.dataset import MotionWindowDataset
from motion_ae.feature_builder import FeatureSlices
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.normalization import FeatureNormalizer, compute_stats


def _make_fake_npz(path: str, T: int = 50):
    rng = np.random.RandomState(42)
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


def test_train_smoke():
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
        assert len(ds) > 0

        mean, std = compute_stats(ds.all_features)
        normalizer = FeatureNormalizer(mean, std)
        ds_norm = MotionWindowDataset([npz_path], cfg, normalizer=normalizer)

        D = ds_norm.slices.total_dim
        assert D == 68

        loader = DataLoader(ds_norm, batch_size=4, shuffle=True)
        model = MotionAutoEncoder(
            feature_dim=D,
            window_size=10,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            ifsq_levels=[4, 4, 4, 4],
        )

        criterion = ReconstructionLoss(
            group_slices=ds_norm.slices.as_dict(),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        losses = []
        for step, batch in enumerate(loader):
            x_hat, _z_d, _info = model(batch)
            loss, _loss_dict = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step >= 1:
                break

        assert len(losses) >= 2, "Should complete at least 2 steps"
        assert all(l > 0 for l in losses), "Loss should be positive"
        # 验证 loss 是有限的（没有 NaN/Inf）
        assert all(np.isfinite(l) for l in losses), "Loss should be finite"
