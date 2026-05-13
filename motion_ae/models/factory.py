"""Model construction helpers."""
from __future__ import annotations

from motion_ae.config import MotionAEConfig
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.models.transformer_autoencoder import MotionTransformerAutoEncoder


def build_motion_autoencoder(cfg: MotionAEConfig, feature_dim: int):
    backbone = getattr(cfg.model, "backbone", "mlp")
    if backbone == "mlp":
        return MotionAutoEncoder(
            feature_dim=feature_dim,
            window_size=cfg.window_size,
            encoder_hidden_dims=cfg.model.encoder_hidden_dims,
            decoder_hidden_dims=cfg.model.decoder_hidden_dims,
            ifsq_levels=cfg.model.ifsq_levels,
            activation=cfg.model.activation,
            use_layer_norm=cfg.model.use_layer_norm,
        )
    if backbone == "transformer":
        return MotionTransformerAutoEncoder(
            feature_dim=feature_dim,
            window_size=cfg.window_size,
            ifsq_levels=cfg.model.ifsq_levels,
            h_dim=cfg.model.transformer_h_dim,
            ff_size=cfg.model.transformer_ff_size,
            num_layers=cfg.model.transformer_num_layers,
            num_heads=cfg.model.transformer_num_heads,
            dropout=cfg.model.transformer_dropout,
            activation=cfg.model.activation,
            normalize_before=cfg.model.transformer_normalize_before,
            position_embedding=cfg.model.transformer_position_embedding,
        )
    raise ValueError(f"Unsupported model.backbone: {backbone}")
