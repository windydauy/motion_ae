"""Transformer backbone AutoEncoder + iFSQ."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from motion_ae.models.ifsq import iFSQ
from transformer_vae.models.position_encoding import build_position_encoding
from transformer_vae.models.transformer_layers import (
    SkipTransformerEncoder,
    TransformerEncoderLayer,
)


class MotionTransformerAutoEncoder(nn.Module):
    """Transformer window autoencoder with iFSQ bottleneck.

    This is an AutoEncoder, not a VAE: the transformer encoder emits one
    deterministic continuous latent vector, iFSQ quantizes it, and a transformer
    decoder reconstructs the full motion window.
    """

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        ifsq_levels: List[int],
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
        position_embedding: str = "learned",
    ):
        super().__init__()
        if num_layers % 2 != 1:
            raise ValueError("Transformer iFSQ backbone requires an odd num_layers")

        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.latent_dim = len(ifsq_levels)
        self.h_dim = int(h_dim)

        self.motion_embedding = nn.Linear(self.feature_dim, self.h_dim)
        self.encoder_pos = build_position_encoding(
            self.h_dim,
            position_embedding=position_embedding,
            max_len=self.window_size + 1,
        )
        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        self.encoder = SkipTransformerEncoder(
            encoder_layer,
            num_layers,
            nn.LayerNorm(self.h_dim),
        )
        self.latent_token = nn.Parameter(torch.randn(1, self.h_dim))
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        self.quantizer = iFSQ(ifsq_levels)

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)
        self.decoder_pos = build_position_encoding(
            self.h_dim,
            position_embedding=position_embedding,
            max_len=self.window_size + 1,
        )
        decoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        self.decoder = SkipTransformerEncoder(
            decoder_layer,
            num_layers,
            nn.LayerNorm(self.h_dim),
        )
        self.output_queries = nn.Parameter(torch.zeros(self.window_size, self.h_dim))
        self.final_layer = nn.Linear(self.h_dim, self.feature_dim)

    def _encode_continuous(self, x: Tensor) -> Tensor:
        bs, window, dim = x.shape
        if window != self.window_size or dim != self.feature_dim:
            raise ValueError(
                f"Expected [B, {self.window_size}, {self.feature_dim}], got {tuple(x.shape)}"
            )

        motion_tokens = self.motion_embedding(x).permute(1, 0, 2)
        latent_token = self.latent_token[:, None, :].expand(1, bs, -1)
        xseq = torch.cat([latent_token, motion_tokens], dim=0)
        xseq = self.encoder_pos(xseq)
        hidden = self.encoder(xseq)[0]
        return self.encoder_latent_proj(hidden)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        z_c = self._encode_continuous(x)
        z_dequant, z_d, info = self.quantizer(z_c)
        x_hat = self.decode(z_dequant)
        return x_hat, z_d, info

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        z_c = self._encode_continuous(x)
        z_dequant, z_d, info = self.quantizer(z_c)
        return z_dequant, z_d, info

    def decode(self, z_dequant: Tensor) -> Tensor:
        if z_dequant.dim() != 2 or z_dequant.shape[1] != self.latent_dim:
            raise ValueError(f"Expected [B, {self.latent_dim}], got {tuple(z_dequant.shape)}")
        bs = int(z_dequant.shape[0])
        z_token = self.decoder_latent_proj(z_dequant).unsqueeze(0)
        queries = self.output_queries[:, None, :].expand(-1, bs, -1)
        xseq = torch.cat([z_token, queries], dim=0)
        xseq = self.decoder_pos(xseq)
        output = self.decoder(xseq)[-self.window_size :]
        return self.final_layer(output).permute(1, 0, 2)
