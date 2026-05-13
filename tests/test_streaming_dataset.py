"""Tests for large-dataset streaming motion windows."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from motion_ae.config import MotionAEConfig
from motion_ae.dataset import MotionWindowDataset
from motion_ae.streaming_dataset import build_streaming_datasets


def _make_fake_npz(path: str, seed: int, T: int = 24, include_unused: bool = True):
    rng = np.random.RandomState(seed)
    quats = rng.randn(T, 30, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    payload = {
        "joint_pos": rng.randn(T, 29).astype(np.float32),
        "joint_vel": rng.randn(T, 29).astype(np.float32),
        "body_quat_w": quats,
        "body_lin_vel_w": rng.randn(T, 30, 3).astype(np.float32),
        "body_ang_vel_w": rng.randn(T, 30, 3).astype(np.float32),
    }
    if include_unused:
        payload["body_pos_w"] = rng.randn(T, 30, 3).astype(np.float32)
        payload["fps"] = np.array([30])
    np.savez(path, **payload)


def _make_cfg(data_root: str, output_root: str) -> MotionAEConfig:
    cfg = MotionAEConfig()
    cfg.data.data_path = data_root
    cfg.data.val_ratio = 0.5
    cfg.data.loader_mode = "streaming"
    cfg.data.manifest_workers = 1
    cfg.data.streaming_cache_size = 2
    cfg.training.output_root = output_root
    cfg.training.num_workers = 0
    cfg.window_size = 10
    cfg.stride = 2
    return cfg


def test_streaming_dataset_matches_packed_windows():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "data")
        os.makedirs(os.path.join(data_root, "clip_a"))
        os.makedirs(os.path.join(data_root, "clip_b"))
        _make_fake_npz(os.path.join(data_root, "clip_a", "motion.npz"), seed=0)
        _make_fake_npz(os.path.join(data_root, "clip_b", "motion.npz"), seed=1)

        cfg = _make_cfg(data_root, os.path.join(tmpdir, "outputs"))
        train_ds, _val_ds, normalizer, _slices, meta = build_streaming_datasets(cfg)

        assert meta["num_train_windows"] > 0
        entry_path = train_ds.entries[0]["path"]
        packed_ds = MotionWindowDataset([entry_path], cfg, normalizer=normalizer)

        torch.testing.assert_close(train_ds[0], packed_ds[0], rtol=0, atol=0)
        torch.testing.assert_close(train_ds[min(2, len(train_ds) - 1)], packed_ds[min(2, len(packed_ds) - 1)], rtol=0, atol=0)


def test_streaming_dataset_does_not_require_unused_npz_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "data")
        os.makedirs(os.path.join(data_root, "clip_a"))
        os.makedirs(os.path.join(data_root, "clip_b"))
        _make_fake_npz(os.path.join(data_root, "clip_a", "motion.npz"), seed=0, include_unused=False)
        _make_fake_npz(os.path.join(data_root, "clip_b", "motion.npz"), seed=1, include_unused=False)

        cfg = _make_cfg(data_root, os.path.join(tmpdir, "outputs"))
        train_ds, val_ds, _normalizer, slices, _meta = build_streaming_datasets(cfg)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert slices.total_dim == 70
        assert train_ds[0].shape == (cfg.window_size, 70)
