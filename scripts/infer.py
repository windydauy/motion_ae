"""推理入口脚本。"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import load_config
from motion_ae.dataset import MotionWindowDataset
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.experiment import get_device, resolve_eval_checkpoint
from motion_ae.utils.logging import get_logger
from motion_ae.utils.normalization import FeatureNormalizer
from motion_ae.utils.seed import set_seed
from motion_ae.utils.tracking import build_tracker
from scripts.cli_args import apply_cli_overrides, build_infer_parser

logger = get_logger("scripts.infer")


def main() -> None:
    parser = build_infer_parser()
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

    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    stats_path = args.stats or os.path.join(run_dir, "artifacts", cfg.normalization.stats_file)
    normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)

    ds = MotionWindowDataset([args.npz_path], cfg, normalizer=normalizer)
    assert len(ds) > 0, f"No windows from {args.npz_path}"
    feature_dim = ds.slices.total_dim

    model = MotionAutoEncoder(
        feature_dim=feature_dim,
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
    model.eval()

    originals = []
    reconstructions = []
    z_cs = []
    z_ds = []
    z_dequants = []

    with torch.no_grad():
        for idx in range(len(ds)):
            window = ds[idx].unsqueeze(0).to(device)
            x_hat, z_d, info = model(window)
            originals.append(window.cpu().numpy())
            reconstructions.append(x_hat.cpu().numpy())
            z_cs.append(info["z_c"].cpu().numpy())
            z_ds.append(z_d.cpu().numpy())
            z_dequants.append(info["z_dequant"].cpu().numpy())

    results = {
        "original": np.concatenate(originals, axis=0),
        "reconstructed": np.concatenate(reconstructions, axis=0),
        "z_c": np.concatenate(z_cs, axis=0),
        "z_d": np.concatenate(z_ds, axis=0),
        "z_dequant": np.concatenate(z_dequants, axis=0),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **results)
    logger.info(f"Inference results saved to {args.output}")

    tracker = build_tracker(
        cfg=cfg,
        run_dir=run_dir,
        run_name=f"{os.path.basename(run_dir)}_infer",
        job_type="infer",
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
