"""MLD-style Transformer VAE for motion feature windows."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from transformer_vae.models.position_encoding import build_position_encoding
from transformer_vae.models.transformer_layers import (
    SkipTransformerDecoder,
    SkipTransformerEncoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class MotionTransformerVAE(nn.Module):
    """History-conditioned Transformer VAE.

    The default project use is full-window reconstruction with
    ``history_len=0`` and ``future_len=window_size``.
    """

    def __init__(
        self,
        nfeats: int,
        window_size: int = 10,
        history_len: int = 0,
        latent_dim: List[int] | Tuple[int, int] = (1, 128),
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
    ) -> None:
        super().__init__()
        self.nfeats = int(nfeats)
        self.window_size = int(window_size)
        self.history_len = int(history_len)
        self.future_len = self.window_size - self.history_len
        if self.future_len <= 0:
            raise ValueError("window_size must be larger than history_len")

        self.latent_size = int(latent_dim[0])
        self.latent_dim = int(latent_dim[-1])
        self.h_dim = int(h_dim)
        self.arch = arch

        self.query_pos_encoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)

        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(self.h_dim))
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        if self.arch == "all_encoder":
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(self.h_dim))
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, nn.LayerNorm(self.h_dim))
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.h_dim))
        self.skel_embedding = nn.Linear(self.nfeats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, self.nfeats)

        self.register_buffer("latent_mean", torch.tensor(0.0))
        self.register_buffer("latent_std", torch.tensor(1.0))

    def split_window(self, motion: Tensor) -> Tuple[Tensor, Tensor]:
        if motion.dim() != 3:
            raise ValueError(f"Expected [B, W, D], got {tuple(motion.shape)}")
        if motion.shape[1] != self.window_size or motion.shape[2] != self.nfeats:
            raise ValueError(
                f"Expected [B, {self.window_size}, {self.nfeats}], got {tuple(motion.shape)}"
            )
        history = motion[:, : self.history_len, :]
        future = motion[:, self.history_len :, :]
        return history, future

    def encode(
        self,
        future_motion: Tensor,
        history_motion: Optional[Tensor] = None,
        scale_latent: bool = False,
        sample: bool = True,
    ) -> Tuple[Tensor, Normal, Tensor, Tensor]:
        bs, _nfuture, nfeats = future_motion.shape
        if nfeats != self.nfeats:
            raise ValueError(f"Expected feature dim {self.nfeats}, got {nfeats}")
        if history_motion is None:
            history_motion = future_motion.new_zeros(bs, 0, nfeats)

        x = torch.cat((history_motion, future_motion), dim=1)
        x = self.skel_embedding(x).permute(1, 0, 2)

        dist_token = self.global_motion_token[:, None, :].expand(-1, bs, -1)
        xseq = torch.cat((dist_token, x), dim=0)
        xseq = self.query_pos_encoder(xseq)
        dist_hidden = self.encoder(xseq)[: dist_token.shape[0]]
        dist_params = self.encoder_latent_proj(dist_hidden)

        mu = dist_params[: self.latent_size]
        logvar = torch.clamp(dist_params[self.latent_size :], min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        latent = dist.rsample() if sample else mu
        if scale_latent:
            latent = latent / self.latent_std
        return latent, dist, mu, logvar

    def decode(
        self,
        z: Tensor,
        history_motion: Optional[Tensor] = None,
        nfuture: Optional[int] = None,
        scale_latent: bool = False,
    ) -> Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected z [latent_size, B, latent_dim], got {tuple(z.shape)}")
        nfuture = int(nfuture or self.future_len)
        bs = int(z.shape[1])
        device = z.device
        if history_motion is None:
            history_motion = z.new_zeros(bs, 0, self.nfeats)

        if scale_latent:
            z = z * self.latent_std
        z = self.decoder_latent_proj(z)
        queries = torch.zeros(nfuture, bs, self.h_dim, device=device, dtype=z.dtype)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)

        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]
        else:
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z)[-nfuture:]

        return self.final_layer(output).permute(1, 0, 2)

    def reconstruct(self, motion: Tensor, sample: bool = True) -> Tuple[Tensor, Normal, Dict[str, Tensor]]:
        history, future = self.split_window(motion)
        z, dist, mu, logvar = self.encode(future, history, sample=sample)
        future_hat = self.decode(z, history, nfuture=future.shape[1])
        if self.history_len > 0:
            recon = torch.cat([history, future_hat], dim=1)
        else:
            recon = future_hat
        return recon, dist, {"z": z, "mu": mu, "logvar": logvar}

    def forward(self, motion: Tensor, sample: bool = True) -> Tuple[Tensor, Normal, Dict[str, Tensor]]:
        return self.reconstruct(motion, sample=sample)
