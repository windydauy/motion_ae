#!/usr/bin/env python3
"""Export a MotionAE checkpoint as an encoder-only ONNX model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs/motion_ae/2026-05-08_17-48-33_opti_clean_our_20"
DEFAULT_CONFIG_PATH = DEFAULT_RUN_DIR / "params/config.yaml"
DEFAULT_CHECKPOINT_PATH = DEFAULT_RUN_DIR / "checkpoints/checkpoint_epoch29999.pt"
DEFAULT_STATS_PATH = DEFAULT_RUN_DIR / "artifacts/stats.npz"
DEFAULT_OUTPUT_PATH = DEFAULT_RUN_DIR / "artifacts/motion_ae_encoder_zdequant.onnx"

from motion_ae.config import load_config  # noqa: E402
from motion_ae.models.autoencoder import MotionAutoEncoder  # noqa: E402
from motion_ae.utils.normalization import FeatureNormalizer  # noqa: E402


class MotionAEEncoderOnnxWrapper(nn.Module):
    """ONNX-friendly MotionAE encoder that returns z_dequant, z_d, and z_c."""

    def __init__(self, model: MotionAutoEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, motion_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = motion_window.shape[0]
        x_flat = motion_window.reshape(batch_size, self.model.flat_dim)
        z_c = self.model.encoder(x_flat)
        levels = self.model.quantizer._levels_t
        f_zc = 2.0 * torch.sigmoid(1.6 * z_c) - 1.0
        scaled = (levels - 1.0) / 2.0 * (f_zc + 1.0)
        z_d = torch.round(scaled)
        z_d = torch.minimum(torch.maximum(z_d, torch.zeros_like(levels)), levels - 1.0)
        z_dequant = 2.0 * z_d / (levels - 1.0) - 1.0
        return z_dequant, z_d, z_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MotionAE encoder to ONNX.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--stats", type=str, default=str(DEFAULT_STATS_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify_batch_size", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, stats_path: str, device: torch.device) -> MotionAutoEncoder:
    cfg = load_config(config_path)
    normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
    feature_dim = int(normalizer.mean.shape[0])
    model = MotionAutoEncoder(
        feature_dim=feature_dim,
        window_size=cfg.window_size,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        ifsq_levels=cfg.model.ifsq_levels,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_onnx(wrapper: MotionAEEncoderOnnxWrapper, output_path: str, dummy_input: torch.Tensor, opset: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["motion_window"],
        output_names=["z_dequant", "z_d", "z_c"],
        dynamic_axes={
            "motion_window": {0: "batch"},
            "z_dequant": {0: "batch"},
            "z_d": {0: "batch"},
            "z_c": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def verify_onnx(wrapper: MotionAEEncoderOnnxWrapper, onnx_path: str, dummy_input: torch.Tensor, rtol: float, atol: float) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_outputs = wrapper(dummy_input)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"motion_window": dummy_input.detach().cpu().numpy()})
    for name, torch_output, ort_output in zip(("z_dequant", "z_d", "z_c"), torch_outputs, ort_outputs, strict=True):
        np.testing.assert_allclose(torch_output.detach().cpu().numpy(), ort_output, rtol=rtol, atol=atol, err_msg=name)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.config, args.checkpoint, args.stats, device)
    wrapper = MotionAEEncoderOnnxWrapper(model).to(device).eval()
    dummy_input = torch.randn(args.verify_batch_size, model.window_size, model.feature_dim, device=device)

    export_onnx(wrapper, args.output, dummy_input, args.opset)
    verify_onnx(wrapper, args.output, dummy_input, args.rtol, args.atol)

    print(f"Exported MotionAE encoder ONNX: {args.output}")
    print(f"input motion_window: [batch, {model.window_size}, {model.feature_dim}]")
    print(f"outputs: z_dequant/z_d/z_c [batch, {model.latent_dim}]")


if __name__ == "__main__":
    main()
