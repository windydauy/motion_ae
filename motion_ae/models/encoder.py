"""MLP Encoder：将 flatten 后的窗口特征编码为连续 latent z_c。"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Input:  (B, input_dim)   — flatten 后的窗口特征 10*D
    Output: (B, latent_dim)  — 连续 latent z_c
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        act_cls = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            z_c: (B, latent_dim)
        """
        assert x.dim() == 2 and x.shape[1] == self.input_dim, \
            f"Expected (B, {self.input_dim}), got {x.shape}"
        return self.net(x)
