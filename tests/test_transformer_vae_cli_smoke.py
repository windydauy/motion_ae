"""End-to-end smoke tests for transformer VAE scripts."""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import yaml

from transformer_vae.config import TransformerVAEConfig
from transformer_vae.scripts import evaluate as evaluate_script
from transformer_vae.scripts import infer as infer_script
from transformer_vae.scripts import train as train_script


def _make_fake_npz(path: str, seed: int, T: int = 24):
    rng = np.random.RandomState(seed)
    quats = rng.randn(T, 30, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    np.savez(
        path,
        joint_pos=rng.randn(T, 29).astype(np.float32),
        joint_vel=rng.randn(T, 29).astype(np.float32),
        body_pos_w=rng.randn(T, 30, 3).astype(np.float32),
        body_quat_w=quats,
        body_lin_vel_w=rng.randn(T, 30, 3).astype(np.float32),
        body_ang_vel_w=rng.randn(T, 30, 3).astype(np.float32),
        fps=np.array([30]),
    )


def _write_small_config(path: str, data_root: str, output_root: str):
    cfg = TransformerVAEConfig()
    cfg.data.data_path = data_root
    cfg.data.val_ratio = 0.5
    cfg.training.output_root = output_root
    cfg.training.experiment_name = "transformer_vae_smoke"
    cfg.training.run_name = "smoke"
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.max_steps = 2
    cfg.training.eval_every = 1
    cfg.training.eval_steps = 1
    cfg.training.save_every = 2
    cfg.training.dataset_cache = False
    cfg.training.device = "cpu"
    cfg.logger.logger = "none"
    cfg.model.latent_dim = [1, 16]
    cfg.model.h_dim = 32
    cfg.model.ff_size = 64
    cfg.model.num_layers = 3
    cfg.model.num_heads = 2
    cfg.model.dropout = 0.0
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def test_transformer_vae_train_eval_infer_smoke(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "data")
        clip_a = os.path.join(data_root, "clip_a")
        clip_b = os.path.join(data_root, "clip_b")
        os.makedirs(clip_a)
        os.makedirs(clip_b)
        npz_a = os.path.join(clip_a, "motion.npz")
        _make_fake_npz(npz_a, seed=0)
        _make_fake_npz(os.path.join(clip_b, "motion.npz"), seed=1)

        output_root = os.path.join(tmpdir, "outputs")
        config_path = os.path.join(tmpdir, "transformer_vae.yaml")
        _write_small_config(config_path, data_root, output_root)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                config_path,
                "--logger",
                "none",
            ],
        )
        train_script.main()

        exp_dir = os.path.join(output_root, "transformer_vae_smoke")
        run_dirs = sorted(os.listdir(exp_dir))
        assert len(run_dirs) == 1
        run_name = run_dirs[0]
        ckpt_path = os.path.join(exp_dir, run_name, "checkpoints", "best_model.pt")
        assert os.path.exists(ckpt_path)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "--config",
                config_path,
                "--run_name",
                run_name,
                "--checkpoint",
                "best_model.pt",
                "--logger",
                "none",
            ],
        )
        evaluate_script.main()
        assert os.path.exists(os.path.join(exp_dir, run_name, "eval", "val_eval.json"))

        infer_out = os.path.join(tmpdir, "infer_output.npz")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "infer.py",
                "--config",
                config_path,
                "--run_name",
                run_name,
                "--checkpoint",
                "best_model.pt",
                "--npz_path",
                npz_a,
                "--output",
                infer_out,
                "--logger",
                "none",
            ],
        )
        infer_script.main()

        assert os.path.exists(infer_out)
        result = np.load(infer_out)
        assert result["original"].shape == result["reconstructed"].shape
        assert result["mu"].shape[-1] == 16
