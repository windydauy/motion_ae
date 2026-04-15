"""不带量化器的 Plain AutoEncoder。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from motion_ae.models.decoder import MLPDecoder
from motion_ae.models.encoder import MLPEncoder


class PlainMotionAutoEncoder(nn.Module):
    """仅使用 MLP Encoder/Decoder 的窗口级自编码器。"""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.flat_dim = feature_dim * window_size
        self.latent_dim = latent_dim

        self.encoder = MLPEncoder(
            input_dim=self.flat_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.flat_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向：flatten -> encoder -> decoder -> reshape。"""
        batch_size, window_size, feature_dim = x.shape
        assert window_size == self.window_size and feature_dim == self.feature_dim, (
            f"Expected (B, {self.window_size}, {self.feature_dim}), got {x.shape}"
        )

        x_flat = x.reshape(batch_size, self.flat_dim)
        z_c = self.encoder(x_flat)
        x_hat_flat = self.decoder(z_c)
        x_hat = x_hat_flat.reshape(batch_size, self.window_size, self.feature_dim)
        info = {
            "z_c": z_c,
            "x_hat_flat": x_hat_flat,
        }
        return x_hat, info
