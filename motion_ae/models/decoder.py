"""MLP Decoder：从 dequantized latent 重建窗口特征。"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Input:  (B, latent_dim)   — dequantized continuous latent
    Output: (B, output_dim)   — 重建的 flatten 窗口特征 10*D
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        act_cls = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            x_hat: (B, output_dim)
        """
        assert z.dim() == 2 and z.shape[1] == self.latent_dim, \
            f"Expected (B, {self.latent_dim}), got {z.shape}"
        return self.net(z)
