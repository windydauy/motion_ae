"""数据集测试：使用合成 npz 数据。"""
import os
import tempfile

import numpy as np
import pytest
import torch

from motion_ae.config import MotionAEConfig, NpzKeysConfig, PelvisConfig, DataConfig
from motion_ae.dataset import MotionWindowDataset
from motion_ae.feature_builder import build_features


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
