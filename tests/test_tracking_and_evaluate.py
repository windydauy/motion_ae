"""wandb 生命周期与评估脚本测试。"""
from __future__ import annotations

import os
import json
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
    calls = {"init": None, "log": [], "finish": 0, "watch": 0, "save": []}

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

    def fake_save(path, base_path=None, policy=None):
        calls["save"].append((path, base_path, policy))

    fake_wandb.init = fake_init
    fake_wandb.log = fake_log
    fake_wandb.watch = fake_watch
    fake_wandb.finish = fake_finish
    fake_wandb.save = fake_save

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def fake_run_git_command(_args, cwd):
        if _args[:2] == ["git", "diff"] and "--cached" not in _args:
            return "diff --git a/file.py b/file.py\n+print('worktree')\n"
        if _args[:3] == ["git", "diff", "--cached"]:
            return "diff --git a/file.py b/file.py\n+print('staged')\n"
        if _args[:3] == ["git", "rev-parse", "HEAD"]:
            return "abc123def\n"
        if _args[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return "main\n"
        if _args[:2] == ["git", "status"]:
            return " M file.py\n"
        raise AssertionError(f"Unexpected git command: {_args}")

    monkeypatch.setattr("motion_ae.utils.tracking._run_git_command", fake_run_git_command)

    cfg = MotionAEConfig()
    cfg.logger.logger = "wandb"
    cfg.logger.log_project_name = "motion_ae_test"
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = build_tracker(cfg, run_dir=tmpdir, run_name="demo_run", job_type="train")
        tracker.watch(object())
        tracker.log({"train/loss": 1.0}, step=3)
        tracker.update_summary({"best_val_loss": 0.25})
        tracker.finish()

        artifacts_dir = os.path.join(tmpdir, "artifacts")
        diff_path = os.path.join(artifacts_dir, "git_diff.patch")
        staged_path = os.path.join(artifacts_dir, "git_diff_staged.patch")
        meta_path = os.path.join(artifacts_dir, "git_meta.json")
        assert os.path.exists(diff_path)
        assert os.path.exists(staged_path)
        assert os.path.exists(meta_path)
        assert "worktree" in open(diff_path, "r", encoding="utf-8").read()
        assert "staged" in open(staged_path, "r", encoding="utf-8").read()
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        assert meta["head"] == "abc123def"
        assert meta["branch"] == "main"
        assert meta["status_short"] == "M file.py"

    assert calls["init"]["project"] == "motion_ae_test"
    assert calls["init"]["name"] == "demo_run"
    assert calls["log"][0][0]["train/loss"] == 1.0
    assert calls["log"][0][1] == 3
    assert calls["watch"] == 1
    assert calls["finish"] == 1
    assert fake_run.summary["best_val_loss"] == 0.25
    assert [os.path.basename(path) for path, _base_path, _policy in calls["save"]] == [
        "git_diff.patch",
        "git_diff_staged.patch",
        "git_meta.json",
    ]


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
