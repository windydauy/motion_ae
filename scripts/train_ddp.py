"""DDP 训练入口：每个 rank 只加载自己的 packed dataset shard。"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import load_config
from motion_ae.dataset import (
    build_train_val_loaders,
    ensure_sharded_dataset_cache,
    load_sharded_datasets,
    try_preload_datasets_to_gpu,
)
from motion_ae.distributed_trainer import DistributedTrainer
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.factory import build_motion_autoencoder
from motion_ae.utils.experiment import (
    create_run_dir,
    resolve_resume_checkpoint,
    save_config_snapshot,
)
from motion_ae.utils.logging import get_logger
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import NullTracker, build_tracker
from scripts.cli_args import apply_cli_overrides, build_train_parser

logger = get_logger("scripts.train_ddp")


def _init_distributed(backend: str):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def _resolve_local_batch_size(cfg, world_size: int) -> None:
    mode = cfg.training.batch_size_mode
    if mode == "per_rank":
        return
    if mode != "global":
        raise ValueError(f"Unsupported batch_size_mode: {mode}")
    if cfg.training.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch_size={cfg.training.batch_size} must be divisible by world_size={world_size}"
        )
    cfg.training.batch_size = cfg.training.batch_size // world_size


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg.training.distributed = True

    rank, local_rank, world_size, device = _init_distributed(cfg.training.ddp_backend)
    _resolve_local_batch_size(cfg, world_size)
    set_seed(cfg.training.seed + rank)

    is_main = rank == 0
    if is_main:
        logger.info(
            "DDP initialized: world_size=%d backend=%s local_batch_size=%d device=%s",
            world_size,
            cfg.training.ddp_backend,
            cfg.training.batch_size,
            device,
        )

    run_paths_obj = [None]
    if is_main:
        run_paths = create_run_dir(
            output_root=cfg.training.output_root,
            experiment_name=cfg.training.experiment_name,
            run_name=cfg.training.run_name,
        )
        run_paths_obj[0] = run_paths
        logger.info(f"Run directory: {run_paths['run_dir']}")
        config_snapshot_path = save_config_snapshot(cfg, run_paths["params_dir"])
        logger.info(f"Config snapshot saved to {config_snapshot_path}")

        logger.info("Ensuring sharded dataset cache from %s", cfg.data.data_path)
        ensure_sharded_dataset_cache(cfg, world_size)

    dist.broadcast_object_list(run_paths_obj, src=0)
    run_paths = run_paths_obj[0]
    dist.barrier()

    train_ds, val_ds, normalizer, feature_slices, shard_meta = load_sharded_datasets(
        cfg,
        rank=rank,
        world_size=world_size,
    )
    if is_main:
        logger.info(
            "Global samples: train=%d val=%d | rank0 shard: train=%d val=%d",
            shard_meta["num_train_windows"],
            shard_meta["num_val_windows"],
            len(train_ds),
            len(val_ds),
        )
        stats_path = os.path.join(run_paths["artifacts_dir"], cfg.normalization.stats_file)
        normalizer.save(stats_path)
        logger.info(f"Normalization stats saved to {stats_path}")

    train_on_gpu, val_on_gpu = try_preload_datasets_to_gpu(
        train_ds, val_ds, device, cfg.training.preload_to_gpu,
    )
    data_on_gpu = train_on_gpu and val_on_gpu
    if is_main and cfg.training.preload_to_gpu and data_on_gpu:
        logger.info("preload_to_gpu: each rank shard is resident on its local device")
        logger.info("Using GPU TensorBatchLoader to avoid CPU DataLoader collation")
    train_loader, val_loader = build_train_val_loaders(
        train_ds, val_ds, cfg, device, data_on_gpu,
    )

    model = build_motion_autoencoder(cfg, feature_slices.total_dim).to(device)
    if device.type == "cuda":
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = DistributedDataParallel(model)

    param_count = sum(p.numel() for p in model.parameters())
    tracker = NullTracker()
    if is_main:
        logger.info(f"Model parameters: {param_count:,}")
        tracker = build_tracker(
            cfg=cfg,
            run_dir=run_paths["run_dir"],
            run_name=os.path.basename(run_paths["run_dir"]),
            job_type="train_ddp",
            resume=args.resume,
        )
        tracker.update_summary(
            {
                "run_dir": run_paths["run_dir"],
                "stats_path": os.path.join(run_paths["artifacts_dir"], cfg.normalization.stats_file),
                "train_samples": shard_meta["num_train_windows"],
                "val_samples": shard_meta["num_val_windows"],
                "local_train_samples": len(train_ds),
                "local_val_samples": len(val_ds),
                "model_parameters": param_count,
                "ddp/world_size": world_size,
                "ddp/batch_size_mode": cfg.training.batch_size_mode,
                "ddp/local_batch_size": cfg.training.batch_size,
            }
        )

    criterion = ReconstructionLoss(
        group_slices=feature_slices.as_dict(),
        group_weights=cfg.loss.group_weights,
    )
    trainer = DistributedTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=run_paths["checkpoints_dir"],
        tracker=tracker,
        run_dir=run_paths["run_dir"],
        rank=rank,
        world_size=world_size,
    )

    if args.resume:
        resume_path = resolve_resume_checkpoint(
            output_root=cfg.training.output_root,
            experiment_name=cfg.training.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
        if is_main:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            tracker.update_summary({"resume_checkpoint": resume_path})
        trainer.load_checkpoint(resume_path)
        dist.barrier()

    try:
        trainer.train()
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
