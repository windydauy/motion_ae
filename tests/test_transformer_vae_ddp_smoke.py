"""CPU DDP smoke test for transformer VAE streaming training."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from transformer_vae.config import TransformerVAEConfig


def _make_fake_npz(path: str, seed: int, T: int = 24):
    rng = np.random.RandomState(seed)
    quats = rng.randn(T, 30, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    np.savez(
        path,
        joint_pos=rng.randn(T, 29).astype(np.float32),
        joint_vel=rng.randn(T, 29).astype(np.float32),
        body_quat_w=quats,
        body_lin_vel_w=rng.randn(T, 30, 3).astype(np.float32),
        body_ang_vel_w=rng.randn(T, 30, 3).astype(np.float32),
    )


def _write_small_config(path: str, data_root: str, output_root: str):
    cfg = TransformerVAEConfig()
    cfg.data.data_path = data_root
    cfg.data.val_ratio = 0.5
    cfg.data.loader_mode = "streaming"
    cfg.data.manifest_workers = 1
    cfg.data.streaming_cache_size = 2
    cfg.training.output_root = output_root
    cfg.training.experiment_name = "transformer_vae_ddp_smoke"
    cfg.training.run_name = "smoke"
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.max_steps = 2
    cfg.training.eval_every = 1
    cfg.training.eval_steps = 1
    cfg.training.save_every = 2
    cfg.training.device = "cpu"
    cfg.training.ddp_backend = "gloo"
    cfg.logger.logger = "none"
    cfg.model.latent_dim = [1, 16]
    cfg.model.h_dim = 32
    cfg.model.ff_size = 64
    cfg.model.num_layers = 3
    cfg.model.num_heads = 2
    cfg.model.dropout = 0.0
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def test_transformer_vae_ddp_streaming_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "data")
        for idx in range(4):
            clip_dir = os.path.join(data_root, f"clip_{idx}")
            os.makedirs(clip_dir)
            _make_fake_npz(os.path.join(clip_dir, "motion.npz"), seed=idx)

        output_root = os.path.join(tmpdir, "outputs")
        config_path = os.path.join(tmpdir, "transformer_vae.yaml")
        _write_small_config(config_path, data_root, output_root)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "2",
            "scripts/train_transformer_vae_ddp.py",
            "--config",
            config_path,
            "--logger",
            "none",
            "--device",
            "cpu",
            "--ddp_backend",
            "gloo",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, timeout=180)

        exp_dir = os.path.join(output_root, "transformer_vae_ddp_smoke")
        run_dirs = os.listdir(exp_dir)
        assert len(run_dirs) == 1
        ckpt_path = os.path.join(exp_dir, run_dirs[0], "checkpoints", "best_model.pt")
        assert os.path.exists(ckpt_path)
