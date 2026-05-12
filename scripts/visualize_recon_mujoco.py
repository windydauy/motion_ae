"""Visualize AE or VAE reconstruction against original motion in MuJoCo."""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_ae.config import MotionAEConfig, load_config as load_ae_config
from motion_ae.dataset import MotionWindowDataset
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.mujoco_recon import (
    clamp_window_state,
    extract_pelvis_root_trajectory,
    features_to_qpos_windows,
    overlap_average_windows,
    read_fps,
)
from motion_ae.utils.experiment import get_device, resolve_eval_checkpoint
from motion_ae.utils.io import load_npz
from motion_ae.utils.normalization import FeatureNormalizer
from transformer_vae.config import TransformerVAEConfig, load_config as load_vae_config
from transformer_vae.scripts.common import build_model as build_vae_model

ModelType = Literal["ae", "vae"]


@dataclass
class ReconstructionWindows:
    original_features: np.ndarray
    reconstructed_features: np.ndarray
    original_qpos: np.ndarray
    reconstructed_qpos: np.ndarray
    fps: float


@dataclass
class ViewerState:
    window_idx: int = 0
    frame_idx: int = 0
    autoplay: bool = True
    quit_requested: bool = False


def project_root() -> str:
    return PROJECT_ROOT


def repo_root() -> str:
    return os.path.dirname(project_root())


def default_mujoco_xml() -> str:
    return os.path.join(
        repo_root(),
        "TextOpTracker",
        "source",
        "textop_tracker",
        "textop_tracker",
        "assets",
        "unitree_description",
        "mjcf",
        "g1_act.xml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize original motion and one AE/VAE reconstruction in MuJoCo."
    )
    parser.add_argument("--model_type", choices=("ae", "vae"), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--vae_config", type=str, default="configs/transformer_vae.yaml")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mujoco_xml", type=str, default=default_mujoco_xml())
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--sample_latent", action="store_true")
    parser.add_argument(
        "--separation",
        type=float,
        default=0.8,
        help="Y-axis distance between original ghost robot and reconstructed solid robot.",
    )
    parser.add_argument(
        "--ghost_alpha",
        type=float,
        default=0.28,
        help="Transparency for the original ghost robot.",
    )
    return parser


def _apply_common_runtime_overrides(cfg, args: argparse.Namespace):
    if args.output_root is not None:
        cfg.training.output_root = args.output_root
    if args.experiment_name is not None:
        cfg.training.experiment_name = args.experiment_name
    if args.device is not None:
        cfg.training.device = args.device
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    return cfg


def _resolve_checkpoint_for_cfg(cfg, args: argparse.Namespace) -> str:
    return resolve_eval_checkpoint(
        output_root=cfg.training.output_root,
        experiment_name=cfg.training.experiment_name,
        run_name=args.run_name,
        checkpoint=args.checkpoint,
    )


def _stats_path(checkpoint_path: str, stats_arg: str | None, stats_file: str) -> str:
    if stats_arg is not None:
        return stats_arg
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return os.path.join(run_dir, "artifacts", stats_file)


def _load_ae_model(
    cfg: MotionAEConfig,
    feature_dim: int,
    checkpoint_path: str,
    device: torch.device,
) -> MotionAutoEncoder:
    model = MotionAutoEncoder(
        feature_dim=feature_dim,
        window_size=cfg.window_size,
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        ifsq_levels=cfg.model.ifsq_levels,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_vae_model(
    cfg: TransformerVAEConfig,
    feature_dim: int,
    checkpoint_path: str,
    device: torch.device,
):
    model = build_vae_model(cfg, feature_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("vae", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _reconstruct_windows(
    model_type: ModelType,
    model,
    dataset: MotionWindowDataset,
    device: torch.device,
    batch_size: int,
    normalizer: FeatureNormalizer,
    *,
    sample_latent: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    originals = []
    reconstructions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model_type == "ae":
                recon, _z_d, _info = model(batch)
            else:
                recon, _dist, _info = model(batch, sample=sample_latent)
            originals.append(batch.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())

    original_norm = np.concatenate(originals, axis=0)
    reconstructed_norm = np.concatenate(reconstructions, axis=0)
    return (
        normalizer.denormalize_np(original_norm).astype(np.float32),
        normalizer.denormalize_np(reconstructed_norm).astype(np.float32),
    )


def build_reconstruction_windows(args: argparse.Namespace) -> ReconstructionWindows:
    if args.model_type == "ae":
        cfg = _apply_common_runtime_overrides(load_ae_config(args.config), args)
    else:
        cfg = _apply_common_runtime_overrides(load_vae_config(args.vae_config), args)

    checkpoint_path = _resolve_checkpoint_for_cfg(cfg, args)
    device = torch.device(get_device(cfg.training.device))
    stats_path = _stats_path(checkpoint_path, args.stats, cfg.normalization.stats_file)
    normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
    dataset = MotionWindowDataset([args.npz_path], cfg, normalizer=normalizer)
    if len(dataset) == 0:
        raise ValueError(f"No windows produced from {args.npz_path}")

    if args.model_type == "ae":
        model = _load_ae_model(cfg, dataset.slices.total_dim, checkpoint_path, device)
    else:
        model = _load_vae_model(cfg, dataset.slices.total_dim, checkpoint_path, device)

    original_window_features, reconstructed_window_features = _reconstruct_windows(
        args.model_type,
        model,
        dataset,
        device,
        batch_size=int(cfg.training.batch_size),
        normalizer=normalizer,
        sample_latent=bool(args.sample_latent),
    )

    npz_data = load_npz(args.npz_path)
    root_pos, root_quat = extract_pelvis_root_trajectory(npz_data, cfg.npz_keys, cfg.pelvis)
    original_sequence_features = np.asarray(dataset.all_features[0], dtype=np.float32)
    reconstructed_sequence_features = overlap_average_windows(
        reconstructed_window_features,
        stride=cfg.stride,
        total_frames=original_sequence_features.shape[0],
    )
    original_qpos = features_to_qpos_windows(
        original_sequence_features[None, :, :],
        dataset.slices,
        root_pos,
        root_quat,
        stride=1,
    )
    reconstructed_qpos = features_to_qpos_windows(
        reconstructed_sequence_features[None, :, :],
        dataset.slices,
        root_pos,
        root_quat,
        stride=1,
        qpos_dim=original_qpos.shape[-1],
    )
    fps = read_fps(npz_data, cfg.npz_keys.fps)
    return ReconstructionWindows(
        original_features=original_sequence_features,
        reconstructed_features=reconstructed_sequence_features,
        original_qpos=original_qpos,
        reconstructed_qpos=reconstructed_qpos,
        fps=fps,
    )


def run_viewer(
    original_qpos: np.ndarray,
    reconstructed_qpos: np.ndarray,
    fps: float,
    xml_path: str,
    start_window: int = 0,
    ghost_alpha: float = 0.28,
    separation: float = 0.8,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise SystemExit(
            "MuJoCo is required for visualization. Install it with `pip install mujoco`."
        ) from exc

    if original_qpos.shape != reconstructed_qpos.shape:
        raise ValueError(
            f"original/reconstructed qpos shape mismatch: "
            f"{original_qpos.shape} vs {reconstructed_qpos.shape}"
        )
    if original_qpos.ndim != 3:
        raise ValueError(f"Expected qpos windows shape (N, W, nq), got {original_qpos.shape}")

    model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore[attr-defined]
    recon_data = mujoco.MjData(model)  # type: ignore[attr-defined]
    original_data = mujoco.MjData(model)  # type: ignore[attr-defined]
    model.opt.timestep = 1.0 / max(float(fps), 1e-6)
    _color_reconstructed_robot(model)

    state = ViewerState(window_idx=start_window)
    frame_dt = 1.0 / max(float(fps), 1e-6)

    def key_callback(keycode: int) -> None:
        if keycode == ord(" ") or keycode in (ord("P"), ord("p")):
            state.autoplay = not state.autoplay
            print(f"Autoplay: {state.autoplay}")
        elif keycode == 262:  # Right
            state.frame_idx += 1
            state.autoplay = False
        elif keycode == 263:  # Left
            state.frame_idx -= 1
            state.autoplay = False
        elif keycode in (ord("N"), ord("n")):
            state.window_idx += 1
            state.frame_idx = 0
        elif keycode in (ord("M"), ord("m")):
            state.window_idx -= 1
            state.frame_idx = 0
        elif keycode in (ord("R"), ord("r")):
            state.frame_idx = 0
        elif keycode in (256, ord("Q"), ord("q")):
            state.quit_requested = True
            print("Quit requested")
            return
        state.window_idx, state.frame_idx = clamp_window_state(
            state.window_idx,
            state.frame_idx,
            original_qpos.shape[0],
            original_qpos.shape[1],
        )
        print(
            f"window={state.window_idx} frame={state.frame_idx} "
            f"autoplay={state.autoplay}"
        )

    viewer = mujoco.viewer.launch_passive(
        model,
        recon_data,
        key_callback=key_callback,
    )
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25

    try:
        while viewer.is_running() and not state.quit_requested:
            state.window_idx, state.frame_idx = clamp_window_state(
                state.window_idx,
                state.frame_idx,
                original_qpos.shape[0],
                original_qpos.shape[1],
            )
            orig = original_qpos[state.window_idx, state.frame_idx].copy()
            recon = reconstructed_qpos[state.window_idx, state.frame_idx].copy()
            orig[1] -= separation * 0.5
            recon[1] += separation * 0.5

            set_qpos(model, original_data, orig)
            set_qpos(model, recon_data, recon)
            mujoco.mj_forward(model, original_data)  # type: ignore[attr-defined]
            mujoco.mj_forward(model, recon_data)  # type: ignore[attr-defined]

            viewer.user_scn.ngeom = 0
            add_ghost_robot_pose(
                mujoco,
                model,
                original_data,
                viewer.user_scn,
                rgba=np.array([0.45, 1.0, 0.55, ghost_alpha], dtype=np.float32),
            )
            viewer.sync()
            if state.autoplay:
                state.frame_idx += 1
            time.sleep(frame_dt)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def set_qpos(model, data, qpos: np.ndarray) -> None:
    if qpos.shape[-1] > model.nq:
        raise ValueError(
            f"qpos has {qpos.shape[-1]} values but MuJoCo model only has nq={model.nq}. "
            "Use a model whose joint count matches the motion data."
        )
    data.qpos[:] = 0.0
    data.qpos[: qpos.shape[-1]] = qpos


def _color_reconstructed_robot(model) -> None:
    rgba = np.array([0.1, 0.45, 1.0, 1.0], dtype=np.float32)
    for geom_id in range(model.ngeom):
        name = ""
        try:
            import mujoco

            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        except Exception:  # pragma: no cover - cosmetic fallback
            name = ""
        if name == "floor":
            continue
        if int(model.geom_group[geom_id]) == 2 or name.endswith("_collision") is False:
            model.geom_rgba[geom_id] = rgba


def add_ghost_robot_pose(
    mujoco_module,
    model,
    data,
    scene,
    rgba: np.ndarray,
) -> None:
    """Draw the original pose as a translucent full-body robot overlay.

    Let MuJoCo populate mesh-specific visualization fields through mjv_addGeoms;
    hand-constructing mesh mjvGeom entries can leave link meshes visually
    detached in the viewer.
    """
    start = int(scene.ngeom)
    opt = mujoco_module.MjvOption()
    if hasattr(opt, "geomgroup"):
        opt.geomgroup[:] = 0
        opt.geomgroup[2] = 1
    pert = mujoco_module.MjvPerturb()
    mujoco_module.mjv_addGeoms(
        model,
        data,
        opt,
        pert,
        mujoco_module.mjtCatBit.mjCAT_DYNAMIC,
        scene,
    )
    for read_idx in range(start, int(scene.ngeom)):
        geom = scene.geoms[read_idx]
        objid = int(geom.objid)
        if objid < 0 or objid >= model.ngeom:
            continue
        name = mujoco_module.mj_id2name(
            model,
            mujoco_module.mjtObj.mjOBJ_GEOM,
            objid,
        ) or ""
        if name == "floor" or int(model.geom_group[objid]) != 2:
            geom.rgba[:] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            continue
        geom.rgba[:] = rgba
        geom.category = mujoco_module.mjtCatBit.mjCAT_DECOR


def main() -> None:
    args = build_parser().parse_args()
    windows = build_reconstruction_windows(args)
    run_viewer(
        windows.original_qpos,
        windows.reconstructed_qpos,
        windows.fps,
        args.mujoco_xml,
        start_window=args.window_index,
        ghost_alpha=args.ghost_alpha,
        separation=args.separation,
    )


if __name__ == "__main__":
    main()
