"""Motion 滑窗数据集。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_ae.config import MotionAEConfig
from motion_ae.feature_builder import FeatureSlices, build_features
from motion_ae.utils.io import find_npz_files, load_npz
from motion_ae.utils.normalization import FeatureNormalizer


class MotionWindowDataset(Dataset):
    """将多条 motion 特征按滑窗切分为固定长度的样本。

    每个样本 shape = (window_size, D)，target 为自身（自编码）。
    """

    def __init__(
        self,
        npz_paths: List[str],
        cfg: MotionAEConfig,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.window_size = cfg.window_size
        self.stride = cfg.stride

        self.windows: List[np.ndarray] = []
        self.slices: Optional[FeatureSlices] = None
        self.all_features: List[np.ndarray] = []

        for path in npz_paths:
            npz_data = load_npz(path)
            feats, slices = build_features(
                npz_data, cfg.npz_keys, cfg.pelvis, debug=cfg.debug,
            )
            self.all_features.append(feats)
            if self.slices is None:
                self.slices = slices

            T, D = feats.shape
            for start in range(0, T - self.window_size + 1, self.stride):
                self.windows.append(feats[start : start + self.window_size])

        if cfg.debug and len(self.windows) > 0:
            print(f"[Dataset] Total windows = {len(self.windows)}, "
                  f"window shape = {self.windows[0].shape}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.windows[idx].copy()  # (W, D) float32
        if self.normalizer is not None:
            window = self.normalizer.normalize_np(window)
        return torch.from_numpy(window)    # (W, D)


def build_datasets(
    cfg: MotionAEConfig,
    stats_path: Optional[str] = None,
) -> Tuple[MotionWindowDataset, MotionWindowDataset, FeatureNormalizer, FeatureSlices]:
    """构建 train/val 数据集 + normalizer + feature slices。

    按文件级别划分 train/val。
    如果提供 `stats_path` 且文件存在，则直接复用保存好的统计量；
    否则从 train set 重新统计 mean/std。
    """
    npz_paths = find_npz_files(cfg.data.data_path, cfg.data.npz_filename)
    assert len(npz_paths) > 0, f"No npz files found in {cfg.data.data_path}"

    # 按文件级别划分
    rng = np.random.RandomState(cfg.training.seed)
    indices = rng.permutation(len(npz_paths))
    n_val = max(1, int(len(npz_paths) * cfg.data.val_ratio))
    val_indices = set(indices[:n_val].tolist())
    train_paths = [p for i, p in enumerate(npz_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(npz_paths) if i in val_indices]

    # 先构建 train set 来获取特征（不归一化），再统计 mean/std
    train_ds_raw = MotionWindowDataset(train_paths, cfg, normalizer=None)
    assert train_ds_raw.slices is not None, "No features built"

    if stats_path is not None and os.path.exists(stats_path):
        normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
    else:
        from motion_ae.utils.normalization import compute_stats

        mean, std = compute_stats(train_ds_raw.all_features)
        normalizer = FeatureNormalizer(mean, std, eps=cfg.normalization.eps)

    # 用 normalizer 重新构建数据集
    train_ds = MotionWindowDataset(train_paths, cfg, normalizer=normalizer)
    val_ds = MotionWindowDataset(val_paths, cfg, normalizer=normalizer)

    return train_ds, val_ds, normalizer, train_ds_raw.slices
