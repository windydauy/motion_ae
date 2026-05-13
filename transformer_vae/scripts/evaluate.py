"""Evaluate transformer motion VAE."""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from motion_ae.dataset import build_datasets, dataloader_io_options, try_preload_one_dataset
from motion_ae.streaming_dataset import build_streaming_datasets, build_streaming_loader
from motion_ae.utils.experiment import get_device, resolve_eval_checkpoint, save_metrics_json
from motion_ae.utils.logging import get_logger
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from transformer_vae.config import load_config
from transformer_vae.evaluator import evaluate
from transformer_vae.scripts.common import (
    add_checkpoint_args,
    add_common_args,
    apply_common_overrides,
    build_criterion,
    build_model,
)

logger = get_logger("transformer_vae.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Motion Transformer VAE.")
    add_common_args(parser)
    add_checkpoint_args(parser, include_resume=False)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--run_eval_name", type=str, default="eval")
    parser.add_argument("--sample_latent", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    cfg = apply_common_overrides(load_config(args.config), args)
    set_seed(cfg.training.seed)
    device = torch.device(get_device(cfg.training.device))

    checkpoint_path = resolve_eval_checkpoint(
        output_root=cfg.training.output_root,
        experiment_name=cfg.training.experiment_name,
        run_name=args.run_name,
        checkpoint=args.checkpoint,
    )
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    stats_path = os.path.join(run_dir, "artifacts", cfg.normalization.stats_file)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    loader_mode = getattr(cfg.data, "loader_mode", "packed")
    if loader_mode == "streaming":
        train_ds, val_ds, _normalizer, feature_slices, _data_meta = build_streaming_datasets(
            cfg,
            stats_path=stats_path,
        )
        ds = val_ds if args.split == "val" else train_ds
        loader = build_streaming_loader(ds, cfg, device, shuffle=False, drop_last=False)
    elif loader_mode == "packed":
        train_ds, val_ds, _normalizer, feature_slices = build_datasets(cfg, stats_path=stats_path)
        ds = val_ds if args.split == "val" else train_ds
        data_on_gpu = try_preload_one_dataset(ds, device, cfg.training.preload_to_gpu)
        num_workers, pin_memory = dataloader_io_options(device, data_on_gpu, cfg.training.num_workers)
        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        raise ValueError(f"Unsupported data.loader_mode: {loader_mode}")

    model = build_model(cfg, feature_slices.total_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("vae", ckpt.get("model_state_dict")))
    model = model.to(device)
    criterion = build_criterion(cfg, feature_slices)

    metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        max_batches=args.max_batches,
        sample=args.sample_latent,
    )
    for key, value in sorted(metrics.items()):
        logger.info("%s: %.6f", key, value)

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=f"{os.path.basename(run_dir)}_{args.run_eval_name}",
        job_type="eval-transformer-vae",
        resume=False,
    )
    tracker.log({f"eval/{key}": value for key, value in metrics.items()})
    tracker.update_summary(
        {
            "eval_checkpoint": checkpoint_path,
            "eval_split": args.split,
            **{f"eval_{key}": value for key, value in metrics.items()},
        }
    )
    tracker.finish()

    metrics_path = save_metrics_json(metrics, os.path.join(eval_dir, f"{args.split}_{args.run_eval_name}.json"))
    logger.info("Evaluation metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
