"""分组 MSE 指标计算。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def grouped_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_slices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """计算各特征组的 MSE。

    Args:
        pred:   (B, W, D)  或 (B, W*D)
        target: 同 pred
        group_slices: {组名: (start, end)} 指在 D 维度上的切片

    Returns:
        {组名: scalar MSE}
    """
    if pred.dim() == 3:
        B, W, D = pred.shape
        pred = pred.reshape(B, W * D)
        target = target.reshape(B, W * D)

    results: Dict[str, torch.Tensor] = {}
    for name, (start, end) in group_slices.items():
        diff = pred[..., start:end] - target[..., start:end]
        results[name] = (diff ** 2).mean()

    results["total"] = ((pred - target) ** 2).mean()
    return results
