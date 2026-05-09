"""DDP 训练逻辑。"""
from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from motion_ae.config import MotionAEConfig
from motion_ae.losses import ReconstructionLoss
from motion_ae.utils.logging import get_logger
from motion_ae.utils.quantizer_metrics import (
    finalize_quantizer_sums,
    init_quantizer_sums,
    reduce_quantizer_sums,
    update_quantizer_sums,
)
from motion_ae.utils.tracking import NullTracker

logger = get_logger(__name__)


class DistributedTrainer:
    """DDP trainer：每个 rank 训练自己的数据 shard，指标跨 rank 汇总。"""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: ReconstructionLoss,
        train_loader: Any,
        val_loader: Any,
        cfg: MotionAEConfig,
        device: torch.device,
        checkpoint_dir: str,
        *,
        rank: int,
        world_size: int,
        tracker=None,
        run_dir: Optional[str] = None,
    ):
        self.model = model
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_dir = run_dir or checkpoint_dir
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.tracker = tracker if self.is_main else NullTracker()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _wrapped_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _batch_to_device(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device == self.device:
            return batch
        return batch.to(self.device, non_blocking=True)

    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pt") -> Optional[str]:
        if not self.is_main:
            return None

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._wrapped_model().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "world_size": self.world_size,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._wrapped_model().load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if self.is_main:
            logger.info(f"Resumed from {path}, epoch {self.start_epoch}")

    def _run_epoch(self, *, train: bool) -> Dict[str, float]:
        self.model.train(train)
        loss_sums: Dict[str, torch.Tensor] = {
            key: torch.tensor(0.0, device=self.device)
            for key in self.criterion.group_slices
        }
        loss_sums["total"] = torch.tensor(0.0, device=self.device)
        sample_count = torch.tensor(0.0, device=self.device)
        quant_sums = init_quantizer_sums(
            self._wrapped_model().quantizer._levels_t,
            self.device,
        )

        for batch in self.train_loader if train else self.val_loader:
            batch = self._batch_to_device(batch)
            x_hat, _z_d, info = self.model(batch)
            loss, loss_dict = self.criterion(x_hat, batch)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.grad_clip_norm,
                    )
                self.optimizer.step()
            update_quantizer_sums(quant_sums, info)

            bs = float(batch.shape[0])
            sample_count += bs
            loss_dict.setdefault("total", loss)
            for key, value in loss_dict.items():
                weighted = value.detach() * bs
                loss_sums[key] = loss_sums[key] + weighted

        metrics = self._distributed_average(loss_sums, sample_count)
        reduce_quantizer_sums(quant_sums)
        metrics.update(finalize_quantizer_sums(quant_sums))
        return metrics

    def _distributed_average(
        self,
        loss_sums: Dict[str, torch.Tensor],
        sample_count: torch.Tensor,
    ) -> Dict[str, float]:
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        if sample_count.item() == 0:
            return {"total": 0.0}

        out: Dict[str, float] = {}
        for key, value in loss_sums.items():
            reduced = value.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            out[key] = (reduced / sample_count).item()
        return out

    def train(self) -> Dict[str, float]:
        if self.is_main:
            logger.info(f"Start DDP training for {self.cfg.training.num_epochs} epochs")
            logger.info(
                "World size: %d | local train batches: %d | local val batches: %d",
                self.world_size,
                len(self.train_loader),
                len(self.val_loader),
            )
            self.tracker.watch(self.model)

        last_metrics: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            t0 = time.time()
            train_metrics = self._run_epoch(train=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(train=False)
            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            if self.is_main:
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
                    "ddp/world_size": self.world_size,
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
                last_metrics = {
                    **train_metrics,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }

        last_path = self.save_checkpoint(self.cfg.training.num_epochs - 1, filename="last_checkpoint.pt")
        if self.is_main:
            self.tracker.update_summary(
                {
                    "best_val_loss": self.best_val_loss,
                    "last_checkpoint": last_path,
                    "run_dir": self.run_dir,
                    "ddp/world_size": self.world_size,
                }
            )
            self.tracker.finish()
            logger.info("DDP training complete!")
        return last_metrics
