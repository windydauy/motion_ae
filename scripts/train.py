"""训练入口脚本。"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import load_config
from motion_ae.dataset import (
    build_datasets,
    build_train_val_loaders,
    try_preload_datasets_to_gpu,
)
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.trainer import Trainer
from motion_ae.utils.experiment import (
    create_run_dir,
    get_device,
    resolve_resume_checkpoint,
    save_config_snapshot,
)
from motion_ae.utils.logging import get_logger
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from scripts.cli_args import apply_cli_overrides, build_train_parser

logger = get_logger("scripts.train")


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    set_seed(cfg.training.seed)

    device_name = get_device(cfg.training.device)
    device = torch.device(device_name)
    logger.info(f"Device: {device}")

    run_paths = create_run_dir(
        output_root=cfg.training.output_root,
        experiment_name=cfg.training.experiment_name,
        run_name=cfg.training.run_name,
    )
    run_dir = run_paths["run_dir"]
    logger.info(f"Run directory: {run_dir}")

    config_snapshot_path = save_config_snapshot(cfg, run_paths["params_dir"])
    logger.info(f"Config snapshot saved to {config_snapshot_path}")

    logger.info(f"Loading data from {cfg.data.data_path}")
    train_ds, val_ds, normalizer, feature_slices = build_datasets(cfg)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    stats_path = os.path.join(run_paths["artifacts_dir"], cfg.normalization.stats_file)
    normalizer.save(stats_path)
    logger.info(f"Normalization stats saved to {stats_path}")

    train_on_gpu, val_on_gpu = try_preload_datasets_to_gpu(
        train_ds, val_ds, device, cfg.training.preload_to_gpu,
    )
    data_on_gpu = train_on_gpu and val_on_gpu
    if cfg.training.preload_to_gpu and data_on_gpu:
        logger.info("preload_to_gpu: 训练与验证张量已在 %s", device)
        logger.info("Using GPU TensorBatchLoader to avoid CPU DataLoader collation")
    train_loader, val_loader = build_train_val_loaders(
        train_ds, val_ds, cfg, device, data_on_gpu,
    )

    model = MotionAutoEncoder(
        feature_dim=feature_slices.total_dim,
        window_size=cfg.window_size,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        ifsq_levels=cfg.model.ifsq_levels,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=os.path.basename(run_dir),
        job_type="train",
        resume=args.resume,
    )
    tracker.update_summary(
        {
            "run_dir": run_dir,
            "stats_path": stats_path,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "model_parameters": param_count,
        }
    )

    criterion = ReconstructionLoss(
        group_slices=feature_slices.as_dict(),
        group_weights=cfg.loss.group_weights,
    )
    trainer = Trainer(
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
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
        tracker.update_summary({"resume_checkpoint": resume_path})

    trainer.train()


if __name__ == "__main__":
    main()
