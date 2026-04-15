"""iFSQ — improved Finite Scalar Quantization.

公式（来自论文/用户提供的图片）：
    z_d = round( (L-1)/2 * (f(z_c) + 1) )
    f(x) = 2 * sigmoid(1.6 * x) - 1

四步流程：
    1. Bounded mapping:  f(z_c) ∈ (-1, 1)
    2. Scale to grid:    scaled  ∈ [0, L-1]
    3. Quantize:         z_d = round(scaled)     （前向真实离散化）
    4. Dequantize:       z_dequant = 2*z_d/(L-1) - 1 ∈ [-1, 1]

反向使用 STE：round 的梯度视为恒等。
Decoder 的输入为 z_dequant（连续值），而非整数 z_d。

每个 latent 维度可以有独立的量化级别数 L_i。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class _STERound(torch.autograd.Function):
    """Straight-Through Estimator for round: forward = round, backward = identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


ste_round = _STERound.apply


class iFSQ(nn.Module):
    """improved Finite Scalar Quantization 模块。

    支持每个维度不同的量化级别数。

    Args:
        levels: 每个 latent 维度的量化级别数列表，例如 [8, 8, 8, 8]。
                latent_dim = len(levels)。
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.latent_dim = len(levels)

        # 注册为 buffer 以便自动随模型迁移 device
        self.register_buffer(
            "_levels_t",
            torch.tensor(levels, dtype=torch.float32),
        )

    # ----------------------------------------------------------
    # 子步骤（拆分便于替换和调试）
    # ----------------------------------------------------------

    def bounded_map(self, z_c: torch.Tensor) -> torch.Tensor:
        """f(z_c) = 2 * sigmoid(1.6 * z_c) - 1,  映射到 (-1, 1)。"""
        return 2.0 * torch.sigmoid(1.6 * z_c) - 1.0

    def scale_to_grid(self, f_zc: torch.Tensor) -> torch.Tensor:
        """将 (-1, 1) 映射到 [0, L-1]。"""
        L = self._levels_t  # (latent_dim,)
        return (L - 1.0) / 2.0 * (f_zc + 1.0)

    def quantize(self, scaled: torch.Tensor) -> torch.Tensor:
        """真实离散化 + STE。返回整数级别 z_d ∈ {0, 1, ..., L-1}。"""
        z_d = ste_round(scaled)
        L = self._levels_t  # (latent_dim,)
        zeros = torch.zeros_like(L)
        z_d = torch.max(z_d, zeros)
        z_d = torch.min(z_d, L - 1.0)
        return z_d

    def dequantize(self, z_d: torch.Tensor) -> torch.Tensor:
        """将整数 z_d 映射回 [-1, 1] 的连续表示。"""
        L = self._levels_t
        return 2.0 * z_d / (L - 1.0) - 1.0

    # ----------------------------------------------------------
    # 前向
    # ----------------------------------------------------------

    def forward(self, z_c: torch.Tensor):
        """
        Args:
            z_c: (B, latent_dim) — encoder 输出的连续 latent

        Returns:
            z_dequant: (B, latent_dim) — dequantized continuous latent，用于 decoder 输入
            z_d:       (B, latent_dim) — 离散整数索引
            info: dict，包含中间值供调试
        """
        assert z_c.dim() == 2 and z_c.shape[1] == self.latent_dim, \
            f"Expected (B, {self.latent_dim}), got {z_c.shape}"

        f_zc = self.bounded_map(z_c)        # (B, latent_dim), ∈ (-1, 1)
        scaled = self.scale_to_grid(f_zc)   # (B, latent_dim), ∈ [0, L-1]
        z_d = self.quantize(scaled)          # (B, latent_dim), 整数
        z_dequant = self.dequantize(z_d)     # (B, latent_dim), ∈ [-1, 1]

        info = {
            "z_c": z_c,
            "f_zc": f_zc,
            "scaled": scaled,
            "z_d": z_d,
            "z_dequant": z_dequant,
        }
        return z_dequant, z_d, info

    @property
    def codebook_size(self) -> int:
        """编码本总大小 = ∏ L_i。"""
        size = 1
        for l in self.levels:
            size *= l
        return size
