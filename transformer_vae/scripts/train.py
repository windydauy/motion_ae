"""Train transformer motion VAE."""
from __future__ import annotations

import argparse
import os

import torch

from motion_ae.dataset import build_datasets, build_train_val_loaders, try_preload_datasets_to_gpu
from motion_ae.streaming_dataset import build_streaming_datasets, build_streaming_train_val_loaders
from motion_ae.utils.experiment import (
    create_run_dir,
    get_device,
    resolve_resume_checkpoint,
    save_config_snapshot,
)
from motion_ae.utils.logging import get_logger
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from transformer_vae.config import load_config
from transformer_vae.scripts.common import (
    add_checkpoint_args,
    add_common_args,
    apply_common_overrides,
    build_criterion,
    build_model,
)
from transformer_vae.trainer import TransformerVAETrainer

logger = get_logger("transformer_vae.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Motion Transformer VAE.")
    add_common_args(parser)
    add_checkpoint_args(parser, include_resume=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--no_anneal_lr", action="store_true")
    parser.add_argument("--no_dataset_cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.eval_every is not None:
        cfg.training.eval_every = args.eval_every
    if args.eval_steps is not None:
        cfg.training.eval_steps = args.eval_steps
    if args.save_every is not None:
        cfg.training.save_every = args.save_every
    if args.grad_clip_norm is not None:
        cfg.training.grad_clip_norm = args.grad_clip_norm
    if args.no_anneal_lr:
        cfg.training.anneal_lr = False
    if args.no_dataset_cache:
        cfg.training.dataset_cache = False

    set_seed(cfg.training.seed)
    device = torch.device(get_device(cfg.training.device))
    run_paths = create_run_dir(
        cfg.training.output_root,
        cfg.training.experiment_name,
        cfg.training.run_name,
    )
    run_dir = run_paths["run_dir"]
    logger.info("Device: %s", device)
    logger.info("Run directory: %s", run_dir)
    save_config_snapshot(cfg, run_paths["params_dir"])

    loader_mode = getattr(cfg.data, "loader_mode", "packed")
    if loader_mode == "streaming":
        train_ds, val_ds, normalizer, feature_slices, data_meta = build_streaming_datasets(cfg)
    elif loader_mode == "packed":
        train_ds, val_ds, normalizer, feature_slices = build_datasets(cfg)
        data_meta = {}
    else:
        raise ValueError(f"Unsupported data.loader_mode: {loader_mode}")

    stats_path = os.path.join(run_paths["artifacts_dir"], cfg.normalization.stats_file)
    normalizer.save(stats_path)
    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))
    logger.info("Feature dim: %d, slices: %s", feature_slices.total_dim, feature_slices.as_dict())

    if loader_mode == "streaming":
        train_loader, val_loader = build_streaming_train_val_loaders(train_ds, val_ds, cfg, device)
    else:
        train_on_gpu, val_on_gpu = try_preload_datasets_to_gpu(
            train_ds,
            val_ds,
            device,
            cfg.training.preload_to_gpu,
        )
        train_loader, val_loader = build_train_val_loaders(
            train_ds,
            val_ds,
            cfg,
            device,
            data_on_gpu=train_on_gpu and val_on_gpu,
        )

    model = build_model(cfg, feature_slices.total_dim)
    criterion = build_criterion(cfg, feature_slices)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", param_count)

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=os.path.basename(run_dir),
        job_type="train-transformer-vae",
        resume=args.resume,
    )
    tracker.update_summary(
        {
            "run_dir": run_dir,
            "stats_path": stats_path,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "feature_dim": feature_slices.total_dim,
            "model_parameters": param_count,
            "data_loader_mode": loader_mode,
            **{f"data/{key}": value for key, value in data_meta.items() if isinstance(value, (int, float, str))},
        }
    )

    trainer = TransformerVAETrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=run_paths["checkpoints_dir"],
        tracker=tracker,
        run_dir=run_dir,
    )
    if args.resume:
        resume_path = resolve_resume_checkpoint(
            output_root=cfg.training.output_root,
            experiment_name=cfg.training.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
        trainer.load_checkpoint(resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
