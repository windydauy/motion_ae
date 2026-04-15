"""实验目录、checkpoint 路径和结果落盘工具。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

from motion_ae.config import MotionAEConfig


def get_device(device_arg: str) -> str:
    """将 auto 解析为具体 device。"""
    if device_arg == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_arg


def create_run_dir(
    output_root: str,
    experiment_name: str,
    run_name: str = "",
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """创建统一的实验目录结构。"""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_name = ts if not run_name else f"{ts}_{run_name}"
    run_dir = os.path.join(output_root, experiment_name, run_dir_name)

    paths = {
        "run_dir": run_dir,
        "checkpoints_dir": os.path.join(run_dir, "checkpoints"),
        "artifacts_dir": os.path.join(run_dir, "artifacts"),
        "params_dir": os.path.join(run_dir, "params"),
        "eval_dir": os.path.join(run_dir, "eval"),
    }
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def save_config_snapshot(cfg: MotionAEConfig, params_dir: str) -> str:
    """保存解析后的配置快照。"""
    path = os.path.join(params_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)
    return path


def save_metrics_json(metrics: Dict[str, float], path: str) -> str:
    """保存评估指标。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def resolve_resume_checkpoint(
    output_root: str,
    experiment_name: str,
    load_run: Optional[str],
    checkpoint: Optional[str],
) -> str:
    """解析训练恢复路径。

    优先级：
    1. `checkpoint` 是已存在路径时直接使用。
    2. 否则要求提供 `load_run`，并在对应 run 的 `checkpoints/` 下查找。
    3. 未提供 checkpoint 文件名时，默认使用 `last_checkpoint.pt`。
    """
    if checkpoint and os.path.exists(checkpoint):
        return os.path.abspath(checkpoint)

    if not load_run:
        raise ValueError("`--resume` 需要配合 `--load_run` 或可直接访问的 `--checkpoint` 路径。")

    ckpt_name = checkpoint or "last_checkpoint.pt"
    ckpt_path = os.path.join(output_root, experiment_name, load_run, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    return os.path.abspath(ckpt_path)


def resolve_eval_checkpoint(
    output_root: str,
    experiment_name: str,
    run_name: Optional[str],
    checkpoint: str,
) -> str:
    """解析评估所需的 checkpoint 路径。"""
    if os.path.exists(checkpoint):
        return os.path.abspath(checkpoint)
    if not run_name:
        raise FileNotFoundError(
            "Checkpoint 不存在，且未提供 `--run_name` 用于从实验目录下解析 checkpoint。"
        )
    ckpt_path = os.path.join(output_root, experiment_name, run_name, "checkpoints", checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found: {ckpt_path}")
    return os.path.abspath(ckpt_path)
