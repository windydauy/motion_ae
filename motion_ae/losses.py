"""损失函数：可扩展的重建损失。

第一版默认使用统一权重的 MSE loss。
后续可通过 group_weights 对不同特征组施加不同权重。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """分组加权重建损失。

    Args:
        group_slices: {组名: (start, end)} 在 D 维度上的切片
        group_weights: {组名: 权重}，默认全部为 1.0
    """

    def __init__(
        self,
        group_slices: Dict[str, Tuple[int, int]],
        group_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.group_slices = group_slices
        self.group_weights = group_weights or {k: 1.0 for k in group_slices}
        unknown_groups = set(self.group_weights) - set(self.group_slices)
        if unknown_groups:
            names = ", ".join(sorted(unknown_groups))
            raise ValueError(f"group_weights contains unknown feature groups: {names}")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred:   (B, W, D) 重建结果
            target: (B, W, D) 原始输入

        Returns:
            total_loss: scalar
            loss_dict: {组名: scalar} 各组 MSE
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        for name, (start, end) in self.group_slices.items():
            group_mse = ((pred[..., start:end] - target[..., start:end]) ** 2).mean()
            weight = self.group_weights.get(name, 1.0)
            total_loss = total_loss + weight * group_mse
            loss_dict[name] = group_mse

        loss_dict["total"] = total_loss
        return total_loss, loss_dict
