"""CLI 参数与实验目录工具测试。"""
from __future__ import annotations

import os
import tempfile

from motion_ae.config import MotionAEConfig
from motion_ae.utils.experiment import create_run_dir, resolve_eval_checkpoint, resolve_resume_checkpoint
from scripts.cli_args import apply_cli_overrides, build_evaluate_parser, build_train_parser


def test_train_cli_overrides():
    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--experiment_name",
            "exp_a",
            "--run_name",
            "run_a",
            "--data_path",
            "/tmp/data",
            "--batch_size",
            "8",
            "--num_epochs",
            "3",
            "--lr",
            "0.0003",
            "--weight_decay",
            "0.1",
            "--seed",
            "7",
            "--logger",
            "none",
            "--output_root",
            "/tmp/out",
        ]
    )

    cfg = apply_cli_overrides(MotionAEConfig(), args)
    assert cfg.training.experiment_name == "exp_a"
    assert cfg.training.run_name == "run_a"
    assert cfg.data.data_path == "/tmp/data"
    assert cfg.training.batch_size == 8
    assert cfg.training.num_epochs == 3
    assert cfg.training.learning_rate == 0.0003
    assert cfg.training.weight_decay == 0.1
    assert cfg.training.seed == 7
    assert cfg.logger.logger == "none"
    assert cfg.training.output_root == "/tmp/out"


def test_evaluate_parser_accepts_isaaclab_style_args():
    parser = build_evaluate_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "best_model.pt",
            "--experiment_name",
            "motion_exp",
            "--run_name",
            "2026-01-01_00-00-00_demo",
            "--split",
            "val",
        ]
    )
    assert args.checkpoint == "best_model.pt"
    assert args.experiment_name == "motion_exp"
    assert args.run_name == "2026-01-01_00-00-00_demo"
    assert args.split == "val"


def test_create_run_dir_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = create_run_dir(
            output_root=tmpdir,
            experiment_name="exp_test",
            run_name="run_test",
            timestamp="2026-04-14_12-00-00",
        )
        assert os.path.isdir(paths["run_dir"])
        assert os.path.isdir(paths["checkpoints_dir"])
        assert os.path.isdir(paths["artifacts_dir"])
        assert os.path.isdir(paths["params_dir"])
        assert os.path.isdir(paths["eval_dir"])
        assert paths["run_dir"].endswith("2026-04-14_12-00-00_run_test")


def test_resolve_resume_checkpoint_from_run_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "outputs", "exp", "run1", "checkpoints")
        os.makedirs(ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "last_checkpoint.pt")
        open(ckpt, "wb").close()

        resolved = resolve_resume_checkpoint(
            output_root=os.path.join(tmpdir, "outputs"),
            experiment_name="exp",
            load_run="run1",
            checkpoint=None,
        )
        assert resolved == os.path.abspath(ckpt)


def test_resolve_eval_checkpoint_from_run_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "outputs", "exp", "run1", "checkpoints")
        os.makedirs(ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "best_model.pt")
        open(ckpt, "wb").close()

        resolved = resolve_eval_checkpoint(
            output_root=os.path.join(tmpdir, "outputs"),
            experiment_name="exp",
            run_name="run1",
            checkpoint="best_model.pt",
        )
        assert resolved == os.path.abspath(ckpt)
