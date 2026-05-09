"""配置加载、序列化与 CLI 覆盖。"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    data_path: str = ""
    npz_filename: str = "motion.npz"
    val_ratio: float = 0.1


@dataclass
class NpzKeysConfig:
    joint_pos: str = "joint_pos"
    joint_vel: str = "joint_vel"
    body_pos_w: str = "body_pos_w"
    body_quat_w: str = "body_quat_w"
    body_lin_vel_w: str = "body_lin_vel_w"
    body_ang_vel_w: str = "body_ang_vel_w"
    fps: str = "fps"


@dataclass
class PelvisConfig:
    body_index: int = 0


@dataclass
class ModelConfig:
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 512])
    ifsq_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 8, 8, 8, 8, 8])
    activation: str = "relu"
    use_layer_norm: bool = True

    @property
    def latent_dim(self) -> int:
        return len(self.ifsq_levels)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    scheduler_t_max: Optional[int] = None
    grad_clip_norm: Optional[float] = None
    num_workers: int = 4
    # 将归一化后的整块 (N,W,D) 张量预载到 GPU；显存不足时自动回退 CPU。
    preload_to_gpu: bool = False
    # 缓存解析/归一化/打包后的 train/val 窗口，后续同配置直接复用。
    dataset_cache: bool = True
    # torchrun/DDP 训练时开启；单卡入口默认不使用。
    distributed: bool = False
    ddp_backend: str = "nccl"
    # DDP 下 batch_size 默认表示每张卡的 local batch。
    batch_size_mode: str = "per_rank"
    seed: int = 42
    save_every: int = 5
    output_root: str = "outputs"
    experiment_name: str = "motion_ae"
    run_name: str = ""
    device: str = "auto"


@dataclass
class LoggerConfig:
    logger: str = "wandb"
    log_project_name: str = "motion_ae"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = ""
    save_code: bool = False


@dataclass
class LossConfig:
    type: str = "mse"
    group_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "pelvis_rot6d_b": 1.0,
            "pelvis_lin_vel_b": 1.0,
            "pelvis_ang_vel_b": 1.0,
        }
    )


@dataclass
class NormalizationConfig:
    eps: float = 1e-6
    stats_file: str = "stats.npz"


@dataclass
class MotionAEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    npz_keys: NpzKeysConfig = field(default_factory=NpzKeysConfig)
    pelvis: PelvisConfig = field(default_factory=PelvisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    window_size: int = 10
    stride: int = 1
    debug: bool = False


def _fill_dataclass(dc_cls, data: dict):
    """递归地将字典填充到 dataclass。"""
    if data is None:
        return dc_cls()
    field_types = {f.name: f.type for f in dataclasses.fields(dc_cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        if isinstance(field_type, str):
            field_type = eval(field_type)  # noqa: S307
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _fill_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return dc_cls(**kwargs)


def load_config(yaml_path: Optional[str] = None) -> MotionAEConfig:
    """从 YAML 文件加载配置，缺省字段使用默认值。"""
    if yaml_path is None or not os.path.exists(yaml_path):
        return MotionAEConfig()
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _fill_dataclass(MotionAEConfig, raw)


def config_to_dict(cfg: MotionAEConfig) -> Dict[str, Any]:
    """将 dataclass 配置转为普通字典。"""
    return asdict(cfg)


def save_config(cfg: MotionAEConfig, path: str) -> None:
    """将配置保存为 YAML。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)
