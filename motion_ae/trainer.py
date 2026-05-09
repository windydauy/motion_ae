"""训练逻辑。"""
from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, Optional

import torch

from motion_ae.config import MotionAEConfig
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.logging import get_logger
from motion_ae.utils.quantizer_metrics import (
    finalize_quantizer_sums,
    init_quantizer_sums,
    update_quantizer_sums,
)
from motion_ae.utils.tracking import NullTracker

logger = get_logger(__name__)


class Trainer:
    """训练管理器。"""

    def __init__(
        self,
        model: MotionAutoEncoder,
        criterion: ReconstructionLoss,
        train_loader: Any,
        val_loader: Any,
        cfg: MotionAEConfig,
        device: torch.device,
        checkpoint_dir: str,
        tracker=None,
        run_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_dir = run_dir or checkpoint_dir
        self.tracker = tracker or NullTracker()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler_t_max = max(1, cfg.training.scheduler_t_max or cfg.training.num_epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 0.5 * (
                1.0 + math.cos(math.pi * min(epoch, scheduler_t_max) / scheduler_t_max)
            ),
        )

        self.best_val_loss = float("inf")
        self.start_epoch = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _batch_to_device(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device == self.device:
            return batch
        return batch.to(self.device, non_blocking=True)

    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pt") -> str:
        """保存 checkpoint。"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 恢复训练状态。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from {path}, epoch {self.start_epoch}")

    def _train_epoch(self) -> Dict[str, float]:
        """训练单个 epoch。"""
        self.model.train()
        loss_sums: Dict[str, torch.Tensor] = {}
        quant_sums = init_quantizer_sums(self.model.quantizer._levels_t, self.device)
        n_batches = 0

        for batch in self.train_loader:
            batch = self._batch_to_device(batch)
            x_hat, _z_d, info = self.model(batch)
            loss, loss_dict = self.criterion(x_hat, batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.grad_clip_norm,
                )
            self.optimizer.step()
            update_quantizer_sums(quant_sums, info)

            loss_dict.setdefault("total", loss)
            for key, value in loss_dict.items():
                detached = value.detach()
                if key in loss_sums:
                    loss_sums[key] = loss_sums[key] + detached
                else:
                    loss_sums[key] = detached
            n_batches += 1

        metrics = self._average_loss_sums(loss_sums, n_batches)
        metrics.update(finalize_quantizer_sums(quant_sums))
        return metrics

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        """验证单个 epoch。"""
        self.model.eval()
        loss_sums: Dict[str, torch.Tensor] = {}
        quant_sums = init_quantizer_sums(self.model.quantizer._levels_t, self.device)
        n_batches = 0

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            x_hat, _z_d, info = self.model(batch)
            loss, loss_dict = self.criterion(x_hat, batch)
            update_quantizer_sums(quant_sums, info)

            loss_dict.setdefault("total", loss)
            for key, value in loss_dict.items():
                detached = value.detach()
                if key in loss_sums:
                    loss_sums[key] = loss_sums[key] + detached
                else:
                    loss_sums[key] = detached
            n_batches += 1

        metrics = self._average_loss_sums(loss_sums, n_batches)
        metrics.update(finalize_quantizer_sums(quant_sums))
        return metrics

    @staticmethod
    def _average_loss_sums(
        loss_sums: Dict[str, torch.Tensor],
        n_batches: int,
    ) -> Dict[str, float]:
        if n_batches == 0:
            return {"total": 0.0}
        return {
            key: (value / n_batches).item()
            for key, value in loss_sums.items()
        }

    def train(self) -> Dict[str, float]:
        """执行完整训练流程。"""
        logger.info(f"Start training for {self.cfg.training.num_epochs} epochs")
        logger.info(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        self.tracker.watch(self.model)

        last_metrics: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:4d}/{self.cfg.training.num_epochs} | "
                f"train_loss={train_metrics['total']:.6f} | "
                f"val_loss={val_metrics['total']:.6f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            log_payload = {
                "epoch": epoch,
                "lr": lr,
                "best_val_loss": self.best_val_loss,
                "timing/epoch_seconds": elapsed,
            }
            log_payload.update({f"train/{key}": value for key, value in train_metrics.items()})
            log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})

            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                best_path = self.save_checkpoint(epoch, filename="best_model.pt")
                logger.info(f"  New best val_loss={self.best_val_loss:.6f}")
                log_payload["best_checkpoint"] = best_path

            if (epoch + 1) % self.cfg.training.save_every == 0:
                self.save_checkpoint(epoch, filename=f"checkpoint_epoch{epoch}.pt")

            log_payload["best_val_loss"] = self.best_val_loss
            self.tracker.log(log_payload, step=epoch)
            last_metrics = {**train_metrics, **{f"val_{key}": value for key, value in val_metrics.items()}}

        last_path = self.save_checkpoint(self.cfg.training.num_epochs - 1, filename="last_checkpoint.pt")
        self.tracker.update_summary(
            {
                "best_val_loss": self.best_val_loss,
                "last_checkpoint": last_path,
                "run_dir": self.run_dir,
            }
        )
        self.tracker.finish()
        logger.info("Training complete!")
        return last_metrics
