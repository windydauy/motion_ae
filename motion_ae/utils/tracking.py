"""实验追踪工具，当前主要提供 wandb 集成。"""
from __future__ import annotations

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
