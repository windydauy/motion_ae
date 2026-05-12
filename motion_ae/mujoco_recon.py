"""Helpers for MuJoCo reconstruction visualization.

The functions in this module intentionally avoid importing MuJoCo. They convert
existing motion feature windows into qpos arrays that a viewer script can feed
into a MuJoCo model.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from motion_ae.config import NpzKeysConfig, PelvisConfig
from motion_ae.feature_builder import FeatureSlices, _ensure_wxyz

ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAACLAB_TO_MUJOCO_REINDEX = [
    ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES
]


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return (quat / norm).astype(np.float32)


def extract_pelvis_root_trajectory(
    npz_data: Dict[str, np.ndarray],
    keys: NpzKeysConfig,
    pelvis_cfg: PelvisConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pelvis world position and quaternion from raw npz data.

    Returns quaternions in MuJoCo's expected ``w, x, y, z`` order.
    """
    idx = pelvis_cfg.body_index
    root_pos = np.asarray(npz_data[keys.body_pos_w][:, idx, :], dtype=np.float32)
    root_quat = np.asarray(npz_data[keys.body_quat_w][:, idx, :], dtype=np.float32)
    root_quat = _normalize_quat(_ensure_wxyz(root_quat))
    return root_pos, root_quat


def read_fps(npz_data: Dict[str, np.ndarray], key: str, default: float = 30.0) -> float:
    """Read fps from an npz payload, accepting scalar or one-element arrays."""
    if key not in npz_data:
        return float(default)
    value = np.asarray(npz_data[key])
    if value.size == 0:
        return float(default)
    return float(value.reshape(-1)[0])


def features_to_qpos_windows(
    features: np.ndarray,
    slices: FeatureSlices,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    stride: int,
    qpos_dim: int | None = None,
) -> np.ndarray:
    """Convert denormalized feature windows to MuJoCo qpos windows.

    The features only contain joint positions and local pelvis descriptors, so
    the global root trajectory is taken from the original motion by window
    start index.
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features shape (N, W, D), got {features.shape}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    num_windows, window_size, _feature_dim = features.shape
    joint_start, joint_end = slices.joint_pos
    num_joints = joint_end - joint_start
    if num_joints <= 0:
        raise ValueError(f"Invalid joint_pos slice: {slices.joint_pos}")

    target_qpos_dim = qpos_dim or 7 + num_joints
    if target_qpos_dim < 7 + num_joints:
        raise ValueError(
            f"qpos_dim={target_qpos_dim} is too small for {num_joints} joints"
        )

    qpos = np.zeros((num_windows, window_size, target_qpos_dim), dtype=np.float32)
    for win_idx in range(num_windows):
        start = win_idx * stride
        end = start + window_size
        if end > len(root_pos) or end > len(root_quat):
            raise ValueError(
                "Root trajectory is shorter than feature windows: "
                f"window {win_idx} needs frames [{start}, {end}), "
                f"root_pos={len(root_pos)}, root_quat={len(root_quat)}"
            )
        qpos[win_idx, :, :3] = root_pos[start:end]
        qpos[win_idx, :, 3:7] = root_quat[start:end]
        joints = features[win_idx, :, joint_start:joint_end]
        if num_joints == len(ISAACLAB_TO_MUJOCO_REINDEX):
            joints = joints[:, ISAACLAB_TO_MUJOCO_REINDEX]
        qpos[win_idx, :, 7 : 7 + num_joints] = joints
    return qpos


def overlap_average_windows(
    windows: np.ndarray,
    stride: int,
    total_frames: int | None = None,
) -> np.ndarray:
    """Average overlapping window predictions back to a continuous sequence."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, D), got {windows.shape}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    num_windows, window_size, feature_dim = windows.shape
    if total_frames is None:
        total_frames = (num_windows - 1) * stride + window_size
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    sums = np.zeros((total_frames, feature_dim), dtype=np.float64)
    counts = np.zeros((total_frames, 1), dtype=np.float64)
    for win_idx in range(num_windows):
        start = win_idx * stride
        end = min(start + window_size, total_frames)
        valid = end - start
        if valid <= 0:
            continue
        sums[start:end] += windows[win_idx, :valid]
        counts[start:end] += 1.0

    missing = counts[:, 0] == 0
    if np.any(missing):
        missing_idx = np.flatnonzero(missing)[:8].tolist()
        raise ValueError(f"No window predictions cover frame indices {missing_idx}")
    return (sums / counts).astype(np.float32)


def clamp_window_state(
    window_idx: int,
    frame_idx: int,
    num_windows: int,
    window_size: int,
) -> Tuple[int, int]:
    """Clamp window index and wrap frame index for interactive playback."""
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    clamped_window = min(max(int(window_idx), 0), num_windows - 1)
    wrapped_frame = int(frame_idx) % window_size
    return clamped_window, wrapped_frame
