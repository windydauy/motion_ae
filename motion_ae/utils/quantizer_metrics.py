"""Lightweight iFSQ monitoring utilities."""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F


def init_quantizer_sums(levels: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create tensor accumulators for iFSQ statistics."""
    latent_dim = int(levels.numel())
    max_level = int(levels.max().item())
    return {
        "levels": levels.detach().to(device=device, dtype=torch.float32),
        "count": torch.tensor(0.0, device=device),
        "zc_abs_sum": torch.tensor(0.0, device=device),
        "zc_abs_max": torch.tensor(0.0, device=device),
        "boundary_sum": torch.tensor(0.0, device=device),
        "scaled_dist_sum": torch.tensor(0.0, device=device),
        "scaled_dist_sq_sum": torch.tensor(0.0, device=device),
        "scaled_dist_min": torch.tensor(float("inf"), device=device),
        "hist": torch.zeros(latent_dim, max_level, device=device),
    }


@torch.no_grad()
def update_quantizer_sums(
    sums: Dict[str, torch.Tensor],
    info: Mapping[str, torch.Tensor],
) -> None:
    """Accumulate z_c, z_d and scaled-grid statistics from one batch."""
    z_c = info["z_c"].detach()
    z_d = info["z_d"].detach()
    scaled = info["scaled"].detach()
    levels = sums["levels"]

    batch_size = int(z_d.shape[0])
    sums["count"] += float(batch_size)

    zc_abs = z_c.abs()
    sums["zc_abs_sum"] += zc_abs.sum()
    sums["zc_abs_max"] = torch.maximum(sums["zc_abs_max"], zc_abs.max())

    boundary = (z_d <= 0.0) | (z_d >= (levels - 1.0))
    sums["boundary_sum"] += boundary.float().sum()

    frac = scaled - torch.floor(scaled)
    dist_to_integer = torch.minimum(frac, 1.0 - frac)
    sums["scaled_dist_sum"] += dist_to_integer.sum()
    sums["scaled_dist_sq_sum"] += dist_to_integer.square().sum()
    sums["scaled_dist_min"] = torch.minimum(sums["scaled_dist_min"], dist_to_integer.min())

    z_idx = z_d.long().clamp_min(0)
    max_level = int(sums["hist"].shape[1])
    z_idx = torch.minimum(z_idx, torch.full_like(z_idx, max_level - 1))
    sums["hist"] += F.one_hot(z_idx, num_classes=max_level).sum(dim=0).to(sums["hist"].dtype)


@torch.no_grad()
def reduce_quantizer_sums(sums: Dict[str, torch.Tensor]) -> None:
    """All-reduce quantizer accumulators in-place for DDP."""
    for key, value in sums.items():
        if key == "levels":
            continue
        if key == "zc_abs_max":
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        elif key == "scaled_dist_min":
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
        else:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)


@torch.no_grad()
def finalize_quantizer_sums(
    sums: Dict[str, torch.Tensor],
    prefix: str = "quantizer",
) -> Dict[str, float]:
    """Convert accumulated iFSQ statistics into scalar log metrics."""
    count = float(sums["count"].item())
    if count <= 0:
        return {}

    levels = sums["levels"]
    latent_dim = int(levels.numel())
    num_values = count * latent_dim

    dist_mean = sums["scaled_dist_sum"] / num_values
    dist_var = sums["scaled_dist_sq_sum"] / num_values - dist_mean.square()
    dist_std = torch.sqrt(torch.clamp(dist_var, min=0.0))

    hist = sums["hist"]
    probs = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    perplexity = entropy.exp()

    metrics: Dict[str, float] = {
        f"{prefix}/z_c_abs_mean": float((sums["zc_abs_sum"] / num_values).item()),
        f"{prefix}/z_c_abs_max": float(sums["zc_abs_max"].item()),
        f"{prefix}/z_d_boundary_frac": float((sums["boundary_sum"] / num_values).item()),
        f"{prefix}/scaled_dist_to_int_mean": float(dist_mean.item()),
        f"{prefix}/scaled_dist_to_int_std": float(dist_std.item()),
        f"{prefix}/scaled_dist_to_int_min": float(sums["scaled_dist_min"].item()),
        f"{prefix}/code_entropy_mean": float(entropy.mean().item()),
        f"{prefix}/code_entropy_min": float(entropy.min().item()),
        f"{prefix}/code_perplexity_mean": float(perplexity.mean().item()),
        f"{prefix}/code_perplexity_min": float(perplexity.min().item()),
    }

    for idx in range(latent_dim):
        metrics[f"{prefix}/code_entropy_dim_{idx:02d}"] = float(entropy[idx].item())
        metrics[f"{prefix}/code_perplexity_dim_{idx:02d}"] = float(perplexity[idx].item())

    return metrics
