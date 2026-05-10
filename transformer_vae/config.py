"""Configuration helpers for the transformer motion VAE."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from motion_ae.config import (
    DataConfig,
    LoggerConfig,
    NormalizationConfig,
    NpzKeysConfig,
    PelvisConfig,
    TrainingConfig,
)


@dataclass
class TransformerModelConfig:
    latent_dim: List[int] = field(default_factory=lambda: [1, 128])
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 9
    num_heads: int = 4
    dropout: float = 0.1
    arch: str = "all_encoder"
    normalize_before: bool = False
    activation: str = "gelu"
    position_embedding: str = "learned"


@dataclass
class TransformerTrainingConfig(TrainingConfig):
    max_steps: int = 100000
    eval_every: int = 2000
    eval_steps: int = 10
    anneal_lr: bool = True


@dataclass
class TransformerLossConfig:
    type: str = "huber"
    beta: float = 1.0
    rec_weight: float = 1.0
    kl_weight: float = 1.0e-4
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
class TransformerVAEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    npz_keys: NpzKeysConfig = field(default_factory=NpzKeysConfig)
    pelvis: PelvisConfig = field(default_factory=PelvisConfig)
    model: TransformerModelConfig = field(default_factory=TransformerModelConfig)
    training: TransformerTrainingConfig = field(default_factory=TransformerTrainingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    loss: TransformerLossConfig = field(default_factory=TransformerLossConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    window_size: int = 10
    stride: int = 1
    debug: bool = False


def _fill_dataclass(dc_cls, data: dict):
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


def load_config(yaml_path: Optional[str] = None) -> TransformerVAEConfig:
    if yaml_path is None or not os.path.exists(yaml_path):
        return TransformerVAEConfig()
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _fill_dataclass(TransformerVAEConfig, raw)
