"""评估逻辑：在验证集 / 测试集上计算分组 MSE。"""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(
    model: MotionAutoEncoder,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    device: torch.device,
) -> Dict[str, float]:
    """在给定 DataLoader 上评估模型。

    Returns:
        metrics: {组名: MSE, "total": 总 MSE}
    """
    model.eval()
    sum_losses: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        x_hat, _z_d, _info = model(batch)
        _loss, loss_dict = criterion(x_hat, batch)

        for k, v in loss_dict.items():
            sum_losses[k] = sum_losses.get(k, 0.0) + v.item()
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in sum_losses.items()}

    logger.info("=== Evaluation Results ===")
    for k, v in sorted(avg.items()):
        logger.info(f"  {k:25s}: {v:.6f}")

    return avg
