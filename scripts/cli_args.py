"""命令行参数定义与配置覆盖。"""
from __future__ import annotations

import argparse

from motion_ae.config import MotionAEConfig


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径。")


def add_experiment_args(parser: argparse.ArgumentParser, include_resume: bool = True) -> None:
    group = parser.add_argument_group("experiment", description="实验目录与恢复相关参数。")
    group.add_argument("--experiment_name", type=str, default=None, help="实验名，对应输出根目录下的一级目录。")
    group.add_argument("--run_name", type=str, default=None, help="运行名，新建 run 时会附加到时间戳后。")
    if include_resume:
        group.add_argument("--resume", action="store_true", help="是否从已有 checkpoint 恢复训练。")
        group.add_argument("--load_run", type=str, default=None, help="需要恢复的旧 run 目录名。")
    group.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径或文件名。")
    group.add_argument("--output_root", type=str, default=None, help="输出根目录。")


def add_logger_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("logger", description="日志与 wandb 参数。")
    group.add_argument("--logger", type=str, default=None, choices={"wandb", "none", "disabled"})
    group.add_argument("--log_project_name", type=str, default=None, help="wandb project 名称。")
    group.add_argument("--wandb_entity", type=str, default=None, help="wandb entity / team。")
    group.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices={"online", "offline", "disabled"},
        help="wandb 运行模式。",
    )
    group.add_argument("--wandb_tags", nargs="*", default=None, help="wandb tags。")
    group.add_argument("--wandb_notes", type=str, default=None, help="wandb notes。")
    group.add_argument("--save_code", action="store_true", help="是否让 wandb 保存代码快照。")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime", description="运行时参数覆盖。")
    group.add_argument("--data_path", type=str, default=None, help="训练 / 评估数据路径覆盖。")
    group.add_argument("--batch_size", type=int, default=None)
    group.add_argument("--num_epochs", type=int, default=None)
    group.add_argument("--learning_rate", "--lr", type=float, default=None)
    group.add_argument("--weight_decay", type=float, default=None)
    group.add_argument("--num_workers", type=int, default=None)
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--device", type=str, default=None, help="例如 cpu / cuda / cuda:0 / auto")
    group.add_argument("--debug", action="store_true", help="打开数据调试打印。")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("evaluate", description="评估相关参数。")
    group.add_argument("--split", type=str, default="val", choices=["train", "val"])


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("infer", description="推理相关参数。")
    group.add_argument("--npz_path", type=str, required=True)
    group.add_argument("--output", type=str, default="infer_output.npz")
    group.add_argument("--stats", type=str, default=None)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Motion AutoEncoder + iFSQ.")
    add_config_args(parser)
    add_experiment_args(parser, include_resume=True)
    add_logger_args(parser)
    add_runtime_args(parser)
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Motion AutoEncoder + iFSQ.")
    add_config_args(parser)
    add_experiment_args(parser, include_resume=False)
    add_logger_args(parser)
    add_runtime_args(parser)
    add_eval_args(parser)
    parser.add_argument("--run_eval_name", type=str, default="eval", help="当前评估任务名，仅用于产物目录。")
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference: reconstruct a single npz.")
    add_config_args(parser)
    add_experiment_args(parser, include_resume=False)
    add_logger_args(parser)
    add_runtime_args(parser)
    add_infer_args(parser)
    return parser


def apply_cli_overrides(cfg: MotionAEConfig, args: argparse.Namespace) -> MotionAEConfig:
    """将 CLI 参数覆盖写回配置。"""
    if getattr(args, "data_path", None) is not None:
        cfg.data.data_path = args.data_path

    if getattr(args, "batch_size", None) is not None:
        cfg.training.batch_size = args.batch_size
    if getattr(args, "num_epochs", None) is not None:
        cfg.training.num_epochs = args.num_epochs
    if getattr(args, "learning_rate", None) is not None:
        cfg.training.learning_rate = args.learning_rate
    if getattr(args, "weight_decay", None) is not None:
        cfg.training.weight_decay = args.weight_decay
    if getattr(args, "num_workers", None) is not None:
        cfg.training.num_workers = args.num_workers
    if getattr(args, "seed", None) is not None:
        cfg.training.seed = args.seed
    if getattr(args, "device", None) is not None:
        cfg.training.device = args.device

    if getattr(args, "experiment_name", None) is not None:
        cfg.training.experiment_name = args.experiment_name
    if getattr(args, "run_name", None) is not None:
        cfg.training.run_name = args.run_name
    if getattr(args, "output_root", None) is not None:
        cfg.training.output_root = args.output_root

    if getattr(args, "logger", None) is not None:
        cfg.logger.logger = args.logger
    if getattr(args, "log_project_name", None) is not None:
        cfg.logger.log_project_name = args.log_project_name
    if getattr(args, "wandb_entity", None) is not None:
        cfg.logger.wandb_entity = args.wandb_entity
    if getattr(args, "wandb_mode", None) is not None:
        cfg.logger.wandb_mode = args.wandb_mode
    if getattr(args, "wandb_tags", None) is not None:
        cfg.logger.wandb_tags = list(args.wandb_tags)
    if getattr(args, "wandb_notes", None) is not None:
        cfg.logger.wandb_notes = args.wandb_notes
    if getattr(args, "save_code", False):
        cfg.logger.save_code = True

    if getattr(args, "debug", False):
        cfg.debug = True

    return cfg
