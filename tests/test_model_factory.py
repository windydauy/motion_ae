"""Model factory tests."""
from __future__ import annotations

from motion_ae.config import MotionAEConfig, load_config
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.models.factory import build_motion_autoencoder
from motion_ae.models.transformer_autoencoder import MotionTransformerAutoEncoder


def test_factory_defaults_to_mlp_backbone():
    cfg = MotionAEConfig()
    model = build_motion_autoencoder(cfg, feature_dim=70)
    assert isinstance(model, MotionAutoEncoder)


def test_factory_builds_transformer_backbone():
    cfg = MotionAEConfig()
    cfg.window_size = 10
    cfg.model.backbone = "transformer"
    cfg.model.ifsq_levels = [64] * 64
    cfg.model.transformer_h_dim = 32
    cfg.model.transformer_ff_size = 64
    cfg.model.transformer_num_layers = 3
    cfg.model.transformer_num_heads = 2
    cfg.model.transformer_dropout = 0.0

    model = build_motion_autoencoder(cfg, feature_dim=70)

    assert isinstance(model, MotionTransformerAutoEncoder)
    assert model.latent_dim == 64
    assert model.quantizer.levels == [64] * 64


def test_transformer_no_layer_norm_config_loads_transformer_backbone():
    cfg = load_config("configs/transformer_no_layer_norm.yaml")

    assert cfg.model.backbone == "transformer"
    assert len(cfg.model.ifsq_levels) == 64
    assert set(cfg.model.ifsq_levels) == {64}
    assert cfg.window_size == 20
    assert cfg.training.batch_size == 2048
