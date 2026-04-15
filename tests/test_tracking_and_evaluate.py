"""wandb 生命周期与评估脚本测试。"""
from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np
import torch

from motion_ae.config import MotionAEConfig, save_config
from motion_ae.dataset import build_datasets
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.tracking import build_tracker
from scripts import evaluate as evaluate_script


def _make_fake_npz(path: str, seed: int, T: int = 32):
    rng = np.random.RandomState(seed)
    quats = rng.randn(T, 37, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    np.savez(
        path,
        joint_pos=rng.randn(T, 29).astype(np.float32),
        joint_vel=rng.randn(T, 29).astype(np.float32),
        body_pos_w=rng.randn(T, 37, 3).astype(np.float32),
        body_quat_w=quats,
        body_lin_vel_w=rng.randn(T, 37, 3).astype(np.float32),
        body_ang_vel_w=rng.randn(T, 37, 3).astype(np.float32),
        fps=np.array([30]),
    )


def test_wandb_tracker_lifecycle(monkeypatch):
    calls = {"init": None, "log": [], "finish": 0, "watch": 0}

    class FakeRun:
        def __init__(self):
            self.summary = {}

    fake_run = FakeRun()

    fake_wandb = types.SimpleNamespace()

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return fake_run

    def fake_log(metrics, step=None):
        calls["log"].append((metrics, step))

    def fake_watch(model, log=None, log_freq=None, criterion=None):
        calls["watch"] += 1

    def fake_finish():
        calls["finish"] += 1

    fake_wandb.init = fake_init
    fake_wandb.log = fake_log
    fake_wandb.watch = fake_watch
    fake_wandb.finish = fake_finish

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = MotionAEConfig()
    cfg.logger.logger = "wandb"
    cfg.logger.log_project_name = "motion_ae_test"
    tracker = build_tracker(cfg, run_dir="/tmp/run", run_name="demo_run", job_type="train")
    tracker.watch(object())
    tracker.log({"train/loss": 1.0}, step=3)
    tracker.update_summary({"best_val_loss": 0.25})
    tracker.finish()

    assert calls["init"]["project"] == "motion_ae_test"
    assert calls["init"]["name"] == "demo_run"
    assert calls["log"][0][0]["train/loss"] == 1.0
    assert calls["log"][0][1] == 3
    assert calls["watch"] == 1
    assert calls["finish"] == 1
    assert fake_run.summary["best_val_loss"] == 0.25


def test_evaluate_script_from_run_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "data")
        os.makedirs(os.path.join(data_root, "clip_a"))
        os.makedirs(os.path.join(data_root, "clip_b"))
        _make_fake_npz(os.path.join(data_root, "clip_a", "motion.npz"), seed=0)
        _make_fake_npz(os.path.join(data_root, "clip_b", "motion.npz"), seed=1)

        cfg = MotionAEConfig()
        cfg.data.data_path = data_root
        cfg.data.val_ratio = 0.5
        cfg.training.output_root = os.path.join(tmpdir, "outputs")
        cfg.training.experiment_name = "exp_eval"
        cfg.training.batch_size = 4
        cfg.training.num_workers = 0
        cfg.logger.logger = "none"

        run_name = "2026-04-14_12-00-00_demo"
        run_dir = os.path.join(cfg.training.output_root, cfg.training.experiment_name, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        artifacts_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        config_path = os.path.join(tmpdir, "config.yaml")
        save_config(cfg, config_path)

        train_ds, _val_ds, normalizer, feature_slices = build_datasets(cfg)
        normalizer.save(os.path.join(artifacts_dir, cfg.normalization.stats_file))

        model = MotionAutoEncoder(
            feature_dim=feature_slices.total_dim,
            window_size=cfg.window_size,
            encoder_hidden_dims=cfg.model.encoder_hidden_dims,
            decoder_hidden_dims=cfg.model.decoder_hidden_dims,
            ifsq_levels=cfg.model.ifsq_levels,
            activation=cfg.model.activation,
            use_layer_norm=cfg.model.use_layer_norm,
        )
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(ckpt_dir, "best_model.pt"))

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
                "--split",
                "val",
            ],
        )

        evaluate_script.main()

        metrics_path = os.path.join(run_dir, "eval", "val_eval.json")
        assert os.path.exists(metrics_path)
