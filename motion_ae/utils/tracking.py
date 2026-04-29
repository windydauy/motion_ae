"""实验追踪工具，当前主要提供 wandb 集成。"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional

from motion_ae.config import MotionAEConfig, config_to_dict


class NullTracker:
    """空 tracker，用于关闭外部日志时保持统一接口。"""

    def __init__(self):
        self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def watch(self, model: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def update_summary(self, metrics: Dict[str, Any]) -> None:
        return None


def _run_git_command(args: list[str], cwd: str) -> str:
    """运行 git 命令并返回 stdout。失败时返回错误摘要而不是抛异常。"""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"[git command failed] {' '.join(args)}\n{exc}\n"
    return result.stdout


def _save_git_state(run_dir: str) -> Dict[str, str]:
    """保存当前仓库的 diff / staged diff / meta 到 run artifacts 目录。"""
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    diff_path = os.path.join(artifacts_dir, "git_diff.patch")
    staged_path = os.path.join(artifacts_dir, "git_diff_staged.patch")
    meta_path = os.path.join(artifacts_dir, "git_meta.json")

    diff_text = _run_git_command(["git", "diff", "--", "."], cwd=run_dir)
    staged_text = _run_git_command(["git", "diff", "--cached", "--", "."], cwd=run_dir)
    head = _run_git_command(["git", "rev-parse", "HEAD"], cwd=run_dir).strip()
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=run_dir).strip()
    status_short = _run_git_command(["git", "status", "--short"], cwd=run_dir).strip()

    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(diff_text)
    with open(staged_path, "w", encoding="utf-8") as f:
        f.write(staged_text)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "head": head,
                "branch": branch,
                "status_short": status_short,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "git_diff": diff_path,
        "git_diff_staged": staged_path,
        "git_meta": meta_path,
    }


class WandbTracker:
    """wandb 封装，集中处理 init / log / finish。"""

    def __init__(
        self,
        cfg: MotionAEConfig,
        run_dir: str,
        run_name: str,
        job_type: str,
        resume: bool = False,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "当前配置默认使用 wandb，但环境中未安装 `wandb`。"
                "请先安装 wandb，或将 `logger.logger` / `--logger` 切为 `none`。"
            ) from exc

        self._wandb = wandb
        self.run = wandb.init(
            project=cfg.logger.log_project_name,
            entity=cfg.logger.wandb_entity,
            mode=cfg.logger.wandb_mode,
            tags=cfg.logger.wandb_tags,
            notes=cfg.logger.wandb_notes or None,
            name=run_name,
            dir=run_dir,
            config=config_to_dict(cfg),
            job_type=job_type,
            resume="allow" if resume else None,
            save_code=cfg.logger.save_code,
        )
        self.git_artifacts = _save_git_state(run_dir)
        for path in self.git_artifacts.values():
            self._wandb.save(path, base_path=run_dir, policy="now")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._wandb.log(metrics, step=step)

    def watch(self, model: Any) -> None:
        self._wandb.watch(model, log="all", log_freq=100, criterion=None)

    def finish(self) -> None:
        self._wandb.finish()

    def update_summary(self, metrics: Dict[str, Any]) -> None:
        for key, value in metrics.items():
            self.run.summary[key] = value


def build_tracker(
    cfg: MotionAEConfig,
    run_dir: str,
    run_name: str,
    job_type: str,
    resume: bool = False,
):
    """根据配置构建 tracker。"""
    logger_type = (cfg.logger.logger or "none").lower()
    if logger_type in {"none", "disabled"}:
        return NullTracker()
    if logger_type == "wandb":
        return WandbTracker(cfg=cfg, run_dir=run_dir, run_name=run_name, job_type=job_type, resume=resume)
    raise ValueError(f"Unsupported logger type: {cfg.logger.logger}")
