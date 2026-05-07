"""数据集测试：使用合成 npz 数据。"""
import os
import tempfile

import numpy as np
import pytest
import torch

from motion_ae.config import MotionAEConfig, NpzKeysConfig, PelvisConfig, DataConfig
from motion_ae.dataset import (
    MotionWindowDataset,
    TensorBatchLoader,
    build_train_val_loaders,
    dataloader_io_options,
)
from motion_ae.feature_builder import build_features
from motion_ae.utils.normalization import FeatureNormalizer, compute_stats


def _make_fake_npz(path: str, T: int = 50, n_joints: int = 29, n_bodies: int = 37):
    """创建一个假的 motion.npz 用于测试。"""
    rng = np.random.RandomState(0)
    # 生成随机四元数并归一化
    quats = rng.randn(T, n_bodies, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

    np.savez(
        path,
        joint_pos=rng.randn(T, n_joints).astype(np.float32),
        joint_vel=rng.randn(T, n_joints).astype(np.float32),
        body_pos_w=rng.randn(T, n_bodies, 3).astype(np.float32),
        body_quat_w=quats,
        body_lin_vel_w=rng.randn(T, n_bodies, 3).astype(np.float32),
        body_ang_vel_w=rng.randn(T, n_bodies, 3).astype(np.float32),
        fps=np.array([30]),
    )


@pytest.fixture
def fake_npz_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        sub = os.path.join(tmpdir, "clip1")
        os.makedirs(sub)
        _make_fake_npz(os.path.join(sub, "motion.npz"), T=50)
        yield tmpdir


def _make_config(data_path: str) -> MotionAEConfig:
    cfg = MotionAEConfig()
    cfg.data = DataConfig(data_path=data_path)
    cfg.window_size = 10
    cfg.stride = 1
    return cfg


class TestBuildFeatures:
    def test_output_shape(self, fake_npz_dir):
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        data = dict(np.load(path))
        keys = NpzKeysConfig()
        pelvis_cfg = PelvisConfig()
        feats, slices = build_features(data, keys, pelvis_cfg)

        assert feats.ndim == 2
        assert feats.shape[0] == 50  # T
        # D = 29 + 29 + 6 + 3 + 3 = 70
        assert feats.shape[1] == 70
        assert slices.total_dim == 70

    def test_debug_mode(self, fake_npz_dir):
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        data = dict(np.load(path))
        keys = NpzKeysConfig()
        pelvis_cfg = PelvisConfig()
        feats, slices = build_features(data, keys, pelvis_cfg, debug=True)
        assert feats.shape[1] == 70

    def test_pelvis_rot6d_b_uses_rotation_6d(self):
        data = {
            "joint_pos": np.zeros((2, 29), dtype=np.float32),
            "joint_vel": np.zeros((2, 29), dtype=np.float32),
            "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
            "body_quat_w": np.array(
                [
                    [[1.0, 0.0, 0.0, 0.0]],
                    [[-1.0, 0.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
            "fps": np.array([30]),
        }

        feats, slices = build_features(data, NpzKeysConfig(), PelvisConfig(body_index=0))
        start, end = slices.pelvis_rot6d_b

        assert end - start == 6
        expected_identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            feats[:, start:end],
            np.tile(expected_identity_6d, (2, 1)),
            atol=1e-6,
        )


class TestMotionWindowDataset:
    def test_num_windows(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        ds = MotionWindowDataset([path], cfg)
        # T=50, window=10, stride=1 → 41 windows
        assert len(ds) == 41

    def test_window_shape(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        ds = MotionWindowDataset([path], cfg)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (10, 70)

    def test_stride(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        cfg.stride = 5
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        ds = MotionWindowDataset([path], cfg)
        # T=50, window=10, stride=5 → floor((50-10)/5) + 1 = 9 windows
        assert len(ds) == 9

    def test_short_sequence_discarded(self):
        """长度不足 window_size 的序列应产生 0 个窗口。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "motion.npz")
            _make_fake_npz(path, T=5)  # T=5 < window_size=10
            cfg = _make_config(tmpdir)
            cfg.data.npz_filename = "motion.npz"
            ds = MotionWindowDataset([path], cfg)
            assert len(ds) == 0


class TestMotionWindowDatasetNormalizationParity:
    """与逐窗从特征矩阵切出的窗口（及归一化）数值一致。"""

    def test_normalized_sample_matches_manual(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        data = dict(np.load(path))
        feats, _slices = build_features(data, NpzKeysConfig(), PelvisConfig())
        mean, std = compute_stats([feats])
        normalizer = FeatureNormalizer(mean, std)
        ds = MotionWindowDataset([path], cfg, normalizer=normalizer)
        w = cfg.window_size
        for i, start in enumerate(range(0, feats.shape[0] - w + 1, cfg.stride)):
            if i >= 3:
                break
            expected = torch.tensor(normalizer.normalize_np(feats[start : start + w]), dtype=torch.float32)
            torch.testing.assert_close(ds[i], expected, rtol=0, atol=0)

    def test_unnormalized_matches_raw_window(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        data = dict(np.load(path))
        feats, _slices = build_features(data, NpzKeysConfig(), PelvisConfig())
        ds = MotionWindowDataset([path], cfg, normalizer=None)
        w = cfg.window_size
        expected = torch.tensor(feats[0:w].astype(np.float32), dtype=torch.float32)
        torch.testing.assert_close(ds[0], expected, rtol=0, atol=0)


def test_dataloader_io_options():
    assert dataloader_io_options(torch.device("cpu"), True, 4) == (0, False)
    assert dataloader_io_options(torch.device("cpu"), False, 4) == (4, False)
    assert dataloader_io_options(torch.device("cuda:0"), False, 8) == (8, True)


class TestTensorBatchLoader:
    def test_ordered_batches_keep_last_partial_batch(self):
        data = torch.arange(10).reshape(10, 1)
        loader = TensorBatchLoader(data, batch_size=4, shuffle=False, drop_last=False)

        batches = list(loader)

        assert len(loader) == 3
        torch.testing.assert_close(batches[0], data[0:4])
        torch.testing.assert_close(batches[1], data[4:8])
        torch.testing.assert_close(batches[2], data[8:10])

    def test_ordered_batches_drop_last(self):
        data = torch.arange(10).reshape(10, 1)
        loader = TensorBatchLoader(data, batch_size=4, shuffle=False, drop_last=True)

        batches = list(loader)

        assert len(loader) == 2
        assert [tuple(batch.shape) for batch in batches] == [(4, 1), (4, 1)]
        torch.testing.assert_close(torch.cat(batches, dim=0), data[0:8])

    def test_shuffle_batches_drop_without_duplicates(self):
        data = torch.arange(10).reshape(10, 1)
        loader = TensorBatchLoader(data, batch_size=4, shuffle=True, drop_last=True)

        merged = torch.cat(list(loader), dim=0).flatten()

        assert merged.numel() == 8
        assert torch.unique(merged).numel() == 8
        assert int(merged.min()) >= 0
        assert int(merged.max()) < 10

    def test_build_train_val_loaders_uses_tensor_loader_for_preloaded_data(self, fake_npz_dir):
        cfg = _make_config(fake_npz_dir)
        cfg.training.batch_size = 4
        path = os.path.join(fake_npz_dir, "clip1", "motion.npz")
        train_ds = MotionWindowDataset([path], cfg)
        val_ds = MotionWindowDataset([path], cfg)

        train_loader, val_loader = build_train_val_loaders(
            train_ds, val_ds, cfg, torch.device("cpu"), data_on_gpu=True,
        )

        assert isinstance(train_loader, TensorBatchLoader)
        assert isinstance(val_loader, TensorBatchLoader)
        assert len(train_loader) == len(train_ds) // cfg.training.batch_size


def test_default_config_preload_to_gpu_false():
    assert MotionAEConfig().training.preload_to_gpu is False
