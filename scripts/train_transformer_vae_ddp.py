"""DDP training entrypoint for the transformer motion VAE."""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.dataset import (  # noqa: E402
    build_train_val_loaders,
    ensure_sharded_dataset_cache,
    load_sharded_datasets,
    try_preload_datasets_to_gpu,
)
from motion_ae.streaming_dataset import (  # noqa: E402
    build_streaming_datasets,
    build_streaming_train_val_loaders,
    ensure_streaming_manifest,
)
from motion_ae.utils.experiment import (  # noqa: E402
    create_run_dir,
    resolve_resume_checkpoint,
    save_config_snapshot,
)
from motion_ae.utils.logging import get_logger  # noqa: E402
from motion_ae.utils.metrics import grouped_mse  # noqa: E402
from motion_ae.utils.seed import set_seed  # noqa: E402
from motion_ae.utils.tracking import NullTracker, build_tracker  # noqa: E402
from transformer_vae.config import load_config  # noqa: E402
from transformer_vae.scripts.common import (  # noqa: E402
    add_checkpoint_args,
    add_common_args,
    apply_common_overrides,
    build_criterion,
    build_model,
)

logger = get_logger("transformer_vae.train_encoder_ddp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDP train Motion Transformer VAE.")
    add_common_args(parser)
    add_checkpoint_args(parser, include_resume=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--batch_size_mode", choices=["per_rank", "global"], default=None)
    parser.add_argument("--ddp_backend", type=str, default=None)
    parser.add_argument("--no_anneal_lr", action="store_true")
    return parser


def _init_distributed(backend: str):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device, backend


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


def _wrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _set_lr(optimizer: torch.optim.Optimizer, cfg, step: int) -> float:
    base_lr = float(cfg.training.learning_rate)
    if cfg.training.anneal_lr:
        frac = max(0.0, 1.0 - step / max(float(cfg.training.max_steps), 1.0))
        lr = base_lr * frac
    else:
        lr = base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _batch_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    if batch.device == device:
        return batch
    return batch.to(device, non_blocking=True)


def _has_bad_grad(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            return True
    return False


def _distributed_average(sums: Dict[str, torch.Tensor], count: torch.Tensor) -> Dict[str, float]:
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        return {}

    out: Dict[str, float] = {}
    for key, value in sums.items():
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        out[key] = float((reduced / count).detach().cpu().item())
    return out


def _reduce_train_terms(loss_terms: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> Dict[str, float]:
    count = torch.tensor(float(batch_size), device=device)
    sums = {
        f"loss/{key}": value.detach() * float(batch_size)
        for key, value in loss_terms.items()
    }
    return _distributed_average(sums, count)


@torch.no_grad()
def _evaluate_ddp(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    *,
    max_batches: Optional[int],
    sample: bool = False,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, torch.Tensor] = {}
    count = torch.tensor(0.0, device=device)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _batch_to_device(batch, device)
        recon, dist_obj, _info = model(batch, sample=sample)
        _loss, loss_terms = criterion(recon, batch, dist_obj)
        mse_terms = grouped_mse(recon, batch, criterion.group_slices)

        bs = float(batch.shape[0])
        count += bs
        for key, value in loss_terms.items():
            metric_key = f"loss/{key}"
            sums[metric_key] = sums.get(metric_key, torch.tensor(0.0, device=device)) + value.detach() * bs
        for key, value in mse_terms.items():
            metric_key = f"mse/{key}"
            sums[metric_key] = sums.get(metric_key, torch.tensor(0.0, device=device)) + value.detach() * bs

    return _distributed_average(sums, count)


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_val_loss: float,
    world_size: int,
) -> str:
    torch.save(
        {
            "vae": _wrapped_model(model).state_dict(),
            "model_state_dict": _wrapped_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "world_size": world_size,
        },
        path,
    )
    logger.info("Checkpoint saved: %s", path)
    return path


def _load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("vae", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError(f"No VAE state found in checkpoint: {path}")
    _wrapped_model(model).load_state_dict(state)
    opt_state = ckpt.get("optimizer", ckpt.get("optimizer_state_dict"))
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val_loss", float("inf")))


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
    if args.batch_size_mode is not None:
        cfg.training.batch_size_mode = args.batch_size_mode
    if args.ddp_backend is not None:
        cfg.training.ddp_backend = args.ddp_backend
    if args.no_anneal_lr:
        cfg.training.anneal_lr = False
    cfg.training.distributed = True
    loader_mode = getattr(cfg.data, "loader_mode", "packed")
    if loader_mode == "packed":
        cfg.training.dataset_cache = True
    elif loader_mode == "streaming":
        cfg.training.preload_to_gpu = False
    else:
        raise ValueError(f"Unsupported data.loader_mode: {loader_mode}")

    rank, local_rank, world_size, device, backend = _init_distributed(cfg.training.ddp_backend)
    _resolve_local_batch_size(cfg, world_size)
    set_seed(cfg.training.seed + rank)
    is_main = rank == 0

    if is_main:
        logger.info(
            "Transformer VAE DDP initialized: world_size=%d backend=%s local_batch_size=%d device=%s",
            world_size,
            backend,
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
        logger.info("Run directory: %s", run_paths["run_dir"])
        save_config_snapshot(cfg, run_paths["params_dir"])
        if loader_mode == "streaming":
            logger.info("Ensuring streaming manifest from %s", cfg.data.data_path)
            ensure_streaming_manifest(cfg)
        else:
            logger.info("Ensuring sharded dataset cache from %s", cfg.data.data_path)
            ensure_sharded_dataset_cache(cfg, world_size)

    dist.broadcast_object_list(run_paths_obj, src=0)
    run_paths = run_paths_obj[0]
    dist.barrier()

    if loader_mode == "streaming":
        train_ds, val_ds, normalizer, feature_slices, shard_meta = build_streaming_datasets(
            cfg,
            rank=rank,
            world_size=world_size,
        )
    else:
        train_ds, val_ds, normalizer, feature_slices, shard_meta = load_sharded_datasets(
            cfg,
            rank=rank,
            world_size=world_size,
        )
    if is_main:
        stats_path = os.path.join(run_paths["artifacts_dir"], cfg.normalization.stats_file)
        normalizer.save(stats_path)
        logger.info(
            "Global samples: train=%d val=%d | rank0 shard: train=%d val=%d | feature_dim=%d",
            shard_meta["num_train_windows"],
            shard_meta["num_val_windows"],
            len(train_ds),
            len(val_ds),
            feature_slices.total_dim,
        )
        logger.info("Normalization stats saved to %s", stats_path)

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

    model = build_model(cfg, feature_slices.total_dim).to(device)
    if device.type == "cuda":
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = DistributedDataParallel(model)
    criterion = build_criterion(cfg, feature_slices).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    step = 0
    best_val_loss = float("inf")
    if args.resume:
        resume_path = resolve_resume_checkpoint(
            output_root=cfg.training.output_root,
            experiment_name=cfg.training.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
        step, best_val_loss = _load_checkpoint(resume_path, model, optimizer, device)
        if is_main:
            logger.info("Resumed from %s at step %d", resume_path, step)
        dist.barrier()

    tracker = NullTracker()
    param_count = sum(p.numel() for p in _wrapped_model(model).parameters())
    if is_main:
        tracker = build_tracker(
            cfg=cfg,
            run_dir=run_paths["run_dir"],
            run_name=os.path.basename(run_paths["run_dir"]),
            job_type="train-transformer-vae-ddp",
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
                "feature_dim": feature_slices.total_dim,
                "model_parameters": param_count,
                "data_loader_mode": loader_mode,
                "ddp/world_size": world_size,
                "ddp/batch_size_mode": cfg.training.batch_size_mode,
                "ddp/local_batch_size": cfg.training.batch_size,
            }
        )
        tracker.watch(model)
        logger.info("Model parameters: %d", param_count)
        logger.info(
            "Start transformer VAE DDP training for %d steps | local train batches=%d val batches=%d",
            cfg.training.max_steps,
            len(train_loader),
            len(val_loader),
        )

    train_iter = iter(train_loader)
    last_metrics: Dict[str, float] = {}
    try:
        while step < cfg.training.max_steps:
            model.train()
            lr = _set_lr(optimizer, cfg, step)
            batch, train_iter = _next_batch(train_iter, train_loader)
            batch = _batch_to_device(batch, device)

            recon, dist_obj, _info = model(batch, sample=True)
            loss, loss_terms = criterion(recon, batch, dist_obj)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            skipped_step = _has_bad_grad(model)
            grad_norm = torch.tensor(0.0, device=device)
            if not skipped_step:
                if cfg.training.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.grad_clip_norm,
                    )
                optimizer.step()

            step += 1
            train_metrics = _reduce_train_terms(loss_terms, int(batch.shape[0]), device)
            skipped_tensor = torch.tensor(float(skipped_step), device=device)
            dist.all_reduce(skipped_tensor, op=dist.ReduceOp.MAX)

            log_payload = {
                "step": step,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "skipped_step": float(skipped_tensor.detach().cpu().item()),
                "ddp/world_size": world_size,
            }
            log_payload.update({f"train/{key}": value for key, value in train_metrics.items()})

            should_eval = step % cfg.training.eval_every == 0 or step == cfg.training.max_steps
            if should_eval:
                val_metrics = _evaluate_ddp(
                    model,
                    val_loader,
                    criterion,
                    device,
                    max_batches=cfg.training.eval_steps,
                    sample=False,
                )
                log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})
                val_loss = val_metrics.get("loss/total", math.inf)
                if is_main and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = _save_checkpoint(
                        os.path.join(run_paths["checkpoints_dir"], "best_model.pt"),
                        model,
                        optimizer,
                        step=step,
                        best_val_loss=best_val_loss,
                        world_size=world_size,
                    )
                    log_payload["best_checkpoint"] = best_path
                if is_main:
                    logger.info(
                        "Step %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                        step,
                        cfg.training.max_steps,
                        train_metrics.get("loss/total", math.nan),
                        val_loss,
                        lr,
                    )
                last_metrics = val_metrics
            elif is_main and (step % 100 == 0 or step == 1):
                logger.info(
                    "Step %d/%d | train_loss=%.6f | lr=%.2e",
                    step,
                    cfg.training.max_steps,
                    train_metrics.get("loss/total", math.nan),
                    lr,
                )

            if is_main and (step % cfg.training.save_every == 0 or step == cfg.training.max_steps):
                _save_checkpoint(
                    os.path.join(run_paths["checkpoints_dir"], f"ckpt_{step}.pth"),
                    model,
                    optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    world_size=world_size,
                )

            if is_main:
                log_payload["best_val_loss"] = best_val_loss
                tracker.log(log_payload, step=step)

        if is_main:
            last_path = _save_checkpoint(
                os.path.join(run_paths["checkpoints_dir"], "last_checkpoint.pt"),
                model,
                optimizer,
                step=step,
                best_val_loss=best_val_loss,
                world_size=world_size,
            )
            tracker.update_summary(
                {
                    "best_val_loss": best_val_loss,
                    "last_checkpoint": last_path,
                    "run_dir": run_paths["run_dir"],
                    **{f"last_val_{key}": value for key, value in last_metrics.items()},
                }
            )
            tracker.finish()
            logger.info("Transformer VAE DDP training complete")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
