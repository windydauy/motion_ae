"""Losses for transformer motion VAE."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class TransformerVAELoss(nn.Module):
    def __init__(
        self,
        group_slices: Dict[str, Tuple[int, int]],
        group_weights: Optional[Dict[str, float]] = None,
        rec_weight: float = 1.0,
        kl_weight: float = 1.0e-4,
        loss_type: str = "huber",
        beta: float = 1.0,
    ):
        super().__init__()
        self.group_slices = group_slices
        self.group_weights = group_weights or {name: 1.0 for name in group_slices}
        unknown_groups = set(self.group_weights) - set(self.group_slices)
        if unknown_groups:
            names = ", ".join(sorted(unknown_groups))
            raise ValueError(f"group_weights contains unknown feature groups: {names}")
        self.rec_weight = float(rec_weight)
        self.kl_weight = float(kl_weight)
        self.loss_type = loss_type
        self.beta = float(beta)

    def _criterion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return torch.mean((pred - target) ** 2)
        if self.loss_type == "huber":
            return nn.functional.huber_loss(pred, target, reduction="mean", delta=self.beta)
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dist: Normal,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = torch.tensor(0.0, device=pred.device)
        terms: Dict[str, torch.Tensor] = {}
        for name, (start, end) in self.group_slices.items():
            group_loss = self._criterion(pred[..., start:end], target[..., start:end])
            rec = rec + self.group_weights.get(name, 1.0) * group_loss
            terms[f"rec_{name}"] = group_loss

        ref = Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        kl = kl_divergence(dist, ref).mean()
        total = self.rec_weight * rec + self.kl_weight * kl
        terms["rec"] = rec
        terms["kl"] = kl
        terms["total"] = total
        return total, terms
