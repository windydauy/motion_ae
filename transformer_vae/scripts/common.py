"""Shared script helpers for transformer VAE entrypoints."""
from __future__ import annotations

import argparse
from typing import Optional

from transformer_vae.config import TransformerVAEConfig
from transformer_vae.losses import TransformerVAELoss
from transformer_vae.models.motion_transformer_vae import MotionTransformerVAE


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/transformer_vae.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logger", type=str, default=None, choices={"wandb", "none", "disabled"})
    parser.add_argument("--wandb_mode", type=str, default=None, choices={"online", "offline", "disabled"})
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")


def apply_common_overrides(cfg: TransformerVAEConfig, args: argparse.Namespace) -> TransformerVAEConfig:
    if args.data_path is not None:
        cfg.data.data_path = args.data_path
    if args.max_files is not None:
        cfg.data.max_files = args.max_files
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.device is not None:
        cfg.training.device = args.device
    if args.output_root is not None:
        cfg.training.output_root = args.output_root
    if args.experiment_name is not None:
        cfg.training.experiment_name = args.experiment_name
    if args.run_name is not None:
        cfg.training.run_name = args.run_name
    if args.logger is not None:
        cfg.logger.logger = args.logger
    if args.wandb_mode is not None:
        cfg.logger.wandb_mode = args.wandb_mode
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.debug:
        cfg.debug = True
    return cfg


def build_model(cfg: TransformerVAEConfig, feature_dim: int) -> MotionTransformerVAE:
    return MotionTransformerVAE(
        nfeats=feature_dim,
        window_size=cfg.window_size,
        history_len=0,
        latent_dim=cfg.model.latent_dim,
        h_dim=cfg.model.h_dim,
        ff_size=cfg.model.ff_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        arch=cfg.model.arch,
        normalize_before=cfg.model.normalize_before,
        activation=cfg.model.activation,
        position_embedding=cfg.model.position_embedding,
    )


def build_criterion(cfg: TransformerVAEConfig, feature_slices) -> TransformerVAELoss:
    return TransformerVAELoss(
        group_slices=feature_slices.as_dict(),
        group_weights=cfg.loss.group_weights,
        rec_weight=cfg.loss.rec_weight,
        kl_weight=cfg.loss.kl_weight,
        loss_type=cfg.loss.type,
        beta=cfg.loss.beta,
    )


def add_checkpoint_args(parser: argparse.ArgumentParser, *, include_resume: bool) -> None:
    parser.add_argument("--checkpoint", type=str, default=None)
    if include_resume:
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--load_run", type=str, default=None)
