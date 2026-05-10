"""Evaluation utilities for transformer motion VAE."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from motion_ae.utils.metrics import grouped_mse
from transformer_vae.losses import TransformerVAELoss
from transformer_vae.models.motion_transformer_vae import MotionTransformerVAE


@torch.no_grad()
def evaluate(
    model: MotionTransformerVAE,
    loader,
    criterion: TransformerVAELoss,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    sample: bool = False,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        recon, dist, _info = model(batch, sample=sample)
        _loss, loss_terms = criterion(recon, batch, dist)
        mse_terms = grouped_mse(recon, batch, criterion.group_slices)

        bs = int(batch.shape[0])
        n_samples += bs
        for key, value in loss_terms.items():
            sums[f"loss/{key}"] = sums.get(f"loss/{key}", 0.0) + float(value.detach().item()) * bs
        for key, value in mse_terms.items():
            sums[f"mse/{key}"] = sums.get(f"mse/{key}", 0.0) + float(value.detach().item()) * bs

    if n_samples == 0:
        return {}
    return {key: value / n_samples for key, value in sums.items()}
