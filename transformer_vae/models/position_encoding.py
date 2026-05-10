"""1D position encodings adapted from TextOpRobotMDAR."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionEmbeddingSine1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        return x + self.pe[: x.shape[0], :]


class PositionEmbeddingLearned1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        return x + self.pe[: x.shape[0], :]


def build_position_encoding(
    d_model: int,
    position_embedding: str = "sine",
    max_len: int = 500,
) -> nn.Module:
    if position_embedding in {"v2", "sine"}:
        return PositionEmbeddingSine1D(d_model, max_len=max_len)
    if position_embedding in {"v3", "learned"}:
        return PositionEmbeddingLearned1D(d_model, max_len=max_len)
    raise ValueError(f"Unsupported position embedding: {position_embedding}")
