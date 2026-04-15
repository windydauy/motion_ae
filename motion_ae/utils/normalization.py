"""特征归一化：统计 mean/std，保存/加载，归一化/反归一化。"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


class FeatureNormalizer:
    """对 (D,) 维度做逐特征标准化。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), eps)
        self._mean_t: Optional[torch.Tensor] = None
        self._std_t: Optional[torch.Tensor] = None

    # ---- numpy 版本 ----
    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize_np(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    # ---- torch 版本（自动缓存到同一 device）----
    def _ensure_torch(self, device: torch.device):
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.from_numpy(self.mean).to(device)
            self._std_t = torch.from_numpy(self.std).to(device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_torch(x.device)
        return (x - self._mean_t) / self._std_t

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_torch(x.device)
        return x * self._std_t + self._mean_t

    # ---- 保存/加载 ----
    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str, eps: float = 1e-6) -> "FeatureNormalizer":
        data = np.load(path)
        return cls(data["mean"], data["std"], eps=eps)


def compute_stats(features_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """从多个 (T, D) 特征数组中计算全局 mean 和 std。

    Args:
        features_list: 每个元素 shape = (T_i, D)

    Returns:
        (mean, std)  各 shape = (D,)
    """
    all_feats = np.concatenate(features_list, axis=0)  # (N_total, D)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    return mean, std
