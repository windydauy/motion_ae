"""Step-based trainer for transformer motion VAE."""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch

from motion_ae.utils.logging import get_logger
from motion_ae.utils.tracking import NullTracker
from transformer_vae.evaluator import evaluate
from transformer_vae.losses import TransformerVAELoss
from transformer_vae.models.motion_transformer_vae import MotionTransformerVAE

logger = get_logger(__name__)


class TransformerVAETrainer:
    def __init__(
        self,
        model: MotionTransformerVAE,
        criterion: TransformerVAELoss,
        train_loader,
        val_loader,
        cfg,
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
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.step = 0
        self.best_val_loss = float("inf")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _set_lr(self) -> float:
        base_lr = float(self.cfg.training.learning_rate)
        if self.cfg.training.anneal_lr:
            frac = max(0.0, 1.0 - self.step / max(float(self.cfg.training.max_steps), 1.0))
            lr = base_lr * frac
        else:
            lr = base_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _batch_to_device(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device == self.device:
            return batch
        return batch.to(self.device, non_blocking=True)

    def _next_batch(self, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(self.train_loader)
            return next(iterator), iterator

    @staticmethod
    def _has_bad_grad(model: torch.nn.Module) -> bool:
        for param in model.parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return True
        return False

    def save_checkpoint(self, filename: str) -> str:
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "vae": self.model.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("vae", ckpt.get("model_state_dict"))
        if state is None:
            raise KeyError(f"No VAE state found in checkpoint: {path}")
        self.model.load_state_dict(state)
        opt_state = ckpt.get("optimizer", ckpt.get("optimizer_state_dict"))
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        self.step = int(ckpt.get("step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        logger.info("Resumed from %s at step %d", path, self.step)

    def train(self) -> Dict[str, float]:
        logger.info("Start transformer VAE training for %d steps", self.cfg.training.max_steps)
        logger.info("Train batches: %d, Val batches: %d", len(self.train_loader), len(self.val_loader))
        self.tracker.watch(self.model)

        train_iter = iter(self.train_loader)
        last_metrics: Dict[str, float] = {}

        while self.step < self.cfg.training.max_steps:
            self.model.train()
            lr = self._set_lr()
            batch, train_iter = self._next_batch(train_iter)
            batch = self._batch_to_device(batch)

            recon, dist, _info = self.model(batch, sample=True)
            loss, loss_terms = self.criterion(recon, batch, dist)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            skipped_step = self._has_bad_grad(self.model)
            grad_norm = torch.tensor(0.0)
            if not skipped_step:
                if self.cfg.training.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.grad_clip_norm,
                    )
                self.optimizer.step()

            self.step += 1
            log_payload = {
                "step": self.step,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "skipped_step": float(skipped_step),
            }
            log_payload.update({f"train/{key}": float(value.detach().cpu().item()) for key, value in loss_terms.items()})

            should_eval = self.step % self.cfg.training.eval_every == 0 or self.step == self.cfg.training.max_steps
            if should_eval:
                val_metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.criterion,
                    self.device,
                    max_batches=self.cfg.training.eval_steps,
                    sample=False,
                )
                log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})
                val_loss = val_metrics.get("loss/total", math.inf)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = self.save_checkpoint("best_model.pt")
                    log_payload["best_checkpoint"] = best_path
                logger.info(
                    "Step %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                    self.step,
                    self.cfg.training.max_steps,
                    float(loss_terms["total"].detach().item()),
                    val_loss,
                    lr,
                )
                last_metrics = val_metrics
            elif self.step % 100 == 0 or self.step == 1:
                logger.info(
                    "Step %d/%d | train_loss=%.6f | lr=%.2e",
                    self.step,
                    self.cfg.training.max_steps,
                    float(loss_terms["total"].detach().item()),
                    lr,
                )

            if self.step % self.cfg.training.save_every == 0 or self.step == self.cfg.training.max_steps:
                self.save_checkpoint(f"ckpt_{self.step}.pth")

            log_payload["best_val_loss"] = self.best_val_loss
            self.tracker.log(log_payload, step=self.step)

        last_path = self.save_checkpoint("last_checkpoint.pt")
        self.tracker.update_summary(
            {
                "best_val_loss": self.best_val_loss,
                "last_checkpoint": last_path,
                "run_dir": self.run_dir,
            }
        )
        self.tracker.finish()
        logger.info("Transformer VAE training complete")
        return last_metrics
