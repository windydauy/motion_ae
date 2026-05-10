"""Reconstruct one NPZ with transformer motion VAE."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_ae.dataset import MotionWindowDataset
from motion_ae.utils.experiment import get_device, resolve_eval_checkpoint
from motion_ae.utils.logging import get_logger
from motion_ae.utils.normalization import FeatureNormalizer
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from transformer_vae.config import load_config
from transformer_vae.scripts.common import (
    add_checkpoint_args,
    add_common_args,
    apply_common_overrides,
    build_model,
)

logger = get_logger("transformer_vae.infer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference with Motion Transformer VAE.")
    add_common_args(parser)
    add_checkpoint_args(parser, include_resume=False)
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="transformer_vae_infer_output.npz")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--sample_latent", action="store_true")
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
    stats_path = args.stats or os.path.join(run_dir, "artifacts", cfg.normalization.stats_file)
    normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
    ds = MotionWindowDataset([args.npz_path], cfg, normalizer=normalizer)
    if len(ds) == 0:
        raise ValueError(f"No windows produced from {args.npz_path}")

    model = build_model(cfg, ds.slices.total_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("vae", ckpt.get("model_state_dict")))
    model = model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    originals = []
    reconstructions = []
    zs = []
    mus = []
    logvars = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, _dist, info = model(batch, sample=args.sample_latent)
            originals.append(batch.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())
            zs.append(info["z"].permute(1, 0, 2).cpu().numpy())
            mus.append(info["mu"].permute(1, 0, 2).cpu().numpy())
            logvars.append(info["logvar"].permute(1, 0, 2).cpu().numpy())

    original = np.concatenate(originals, axis=0)
    reconstructed = np.concatenate(reconstructions, axis=0)
    result = {
        "original": original,
        "reconstructed": reconstructed,
        "original_denorm": normalizer.denormalize_np(original),
        "reconstructed_denorm": normalizer.denormalize_np(reconstructed),
        "z": np.concatenate(zs, axis=0),
        "mu": np.concatenate(mus, axis=0),
        "logvar": np.concatenate(logvars, axis=0),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **result)
    logger.info("Inference results saved to %s", args.output)

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=f"{os.path.basename(run_dir)}_infer_transformer_vae",
        job_type="infer-transformer-vae",
        resume=False,
    )
    tracker.update_summary(
        {
            "infer_checkpoint": checkpoint_path,
            "infer_input_npz": args.npz_path,
            "infer_output_npz": os.path.abspath(args.output),
        }
    )
    tracker.finish()


if __name__ == "__main__":
    main()
