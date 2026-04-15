"""AutoEncoder + iFSQ 组合模型。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from motion_ae.models.decoder import MLPDecoder
from motion_ae.models.encoder import MLPEncoder
from motion_ae.models.ifsq import iFSQ


class MotionAutoEncoder(nn.Module):
    """
    完整流程：
        x (B, W, D) → flatten → Encoder → z_c → iFSQ → z_dequant → Decoder → x_hat (B, W, D)
    """

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        ifsq_levels: List[int],
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.flat_dim = window_size * feature_dim
        self.latent_dim = len(ifsq_levels)

        self.encoder = MLPEncoder(
            input_dim=self.flat_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=self.latent_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self.quantizer = iFSQ(ifsq_levels)
        self.decoder = MLPDecoder(
            latent_dim=self.latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.flat_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: (B, W, D)

        Returns:
            x_hat:  (B, W, D)  重建结果
            z_d:    (B, latent_dim)  离散索引
            info:   dict  包含 z_c, z_dequant 等
        """
        B, W, D = x.shape
        assert W == self.window_size and D == self.feature_dim, \
            f"Expected (B, {self.window_size}, {self.feature_dim}), got {x.shape}"

        x_flat = x.reshape(B, self.flat_dim)              # (B, W*D)
        z_c = self.encoder(x_flat)                         # (B, latent_dim)
        z_dequant, z_d, info = self.quantizer(z_c)         # (B, latent_dim)
        x_hat_flat = self.decoder(z_dequant)               # (B, W*D)
        x_hat = x_hat_flat.reshape(B, W, D)               # (B, W, D)

        info["x_hat_flat"] = x_hat_flat
        return x_hat, z_d, info

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """仅编码 + 量化，不解码。"""
        B, W, D = x.shape
        x_flat = x.reshape(B, self.flat_dim)
        z_c = self.encoder(x_flat)
        z_dequant, z_d, info = self.quantizer(z_c)
        return z_dequant, z_d, info

    def decode(self, z_dequant: torch.Tensor) -> torch.Tensor:
        """从 dequantized latent 解码。"""
        x_hat_flat = self.decoder(z_dequant)
        return x_hat_flat.reshape(-1, self.window_size, self.feature_dim)
