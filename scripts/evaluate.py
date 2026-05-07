"""评估入口脚本。"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import load_config
from motion_ae.dataset import build_datasets, dataloader_io_options, try_preload_one_dataset
from motion_ae.evaluator import evaluate
from motion_ae.losses import ReconstructionLoss
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.experiment import get_device, resolve_eval_checkpoint, save_metrics_json
from motion_ae.utils.logging import get_logger
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from scripts.cli_args import apply_cli_overrides, build_evaluate_parser

logger = get_logger("scripts.evaluate")


def main() -> None:
    parser = build_evaluate_parser()
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint 是必需参数。")

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    set_seed(cfg.training.seed)

    device_name = get_device(cfg.training.device)
    device = torch.device(device_name)

    checkpoint_path = resolve_eval_checkpoint(
        output_root=cfg.training.output_root,
        experiment_name=cfg.training.experiment_name,
        run_name=args.run_name,
        checkpoint=args.checkpoint,
    )
    logger.info(f"Resolved checkpoint: {checkpoint_path}")

    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    stats_path = os.path.join(run_dir, "artifacts", cfg.normalization.stats_file)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    train_ds, val_ds, _normalizer, feature_slices = build_datasets(cfg, stats_path=stats_path)
    ds = val_ds if args.split == "val" else train_ds
    data_on_gpu = try_preload_one_dataset(ds, device, cfg.training.preload_to_gpu)
    num_workers, pin_memory = dataloader_io_options(
        device, data_on_gpu, cfg.training.num_workers,
    )

    logger.info(f"Evaluating on {args.split} set ({len(ds)} samples)")

    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = ReconstructionLoss(
        group_slices=feature_slices.as_dict(),
        group_weights=cfg.loss.group_weights,
    )
    metrics = evaluate(model, loader, criterion, device)

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=f"{os.path.basename(run_dir)}_{args.run_eval_name}",
        job_type="eval",
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

    metrics_path = save_metrics_json(
        metrics,
        os.path.join(eval_dir, f"{args.split}_{args.run_eval_name}.json"),
    )
    logger.info(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
