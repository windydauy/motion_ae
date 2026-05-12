from __future__ import annotations

import numpy as np

from motion_ae.config import NpzKeysConfig, PelvisConfig
from motion_ae.feature_builder import FeatureSlices
from motion_ae.mujoco_recon import (
    ISAACLAB_TO_MUJOCO_REINDEX,
    clamp_window_state,
    extract_pelvis_root_trajectory,
    features_to_qpos_windows,
    overlap_average_windows,
    read_fps,
)


def test_extract_pelvis_root_trajectory_normalizes_quaternion():
    keys = NpzKeysConfig()
    pelvis = PelvisConfig(body_index=1)
    body_pos = np.zeros((3, 2, 3), dtype=np.float32)
    body_quat = np.zeros((3, 2, 4), dtype=np.float32)
    body_pos[:, 1] = np.array(
        [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    body_quat[:, 1] = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    npz_data = {
        keys.body_pos_w: body_pos,
        keys.body_quat_w: body_quat,
    }

    root_pos, root_quat = extract_pelvis_root_trajectory(npz_data, keys, pelvis)

    np.testing.assert_allclose(root_pos, body_pos[:, 1])
    np.testing.assert_allclose(np.linalg.norm(root_quat, axis=-1), np.ones(3), atol=1e-6)
    np.testing.assert_allclose(root_quat[0], [1.0, 0.0, 0.0, 0.0])


def test_features_to_qpos_windows_uses_root_by_window_stride_and_joint_slice():
    slices = FeatureSlices()
    slices.joint_pos = (2, 5)
    slices.joint_vel = (5, 8)
    slices.pelvis_rot6d_b = (8, 14)
    slices.pelvis_lin_vel_b = (14, 17)
    slices.pelvis_ang_vel_b = (17, 20)
    features = np.zeros((2, 3, 20), dtype=np.float32)
    features[0, :, 2:5] = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=np.float32,
    )
    features[1, :, 2:5] = np.array(
        [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]],
        dtype=np.float32,
    )
    root_pos = np.arange(18, dtype=np.float32).reshape(6, 3)
    root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (6, 1))

    qpos = features_to_qpos_windows(features, slices, root_pos, root_quat, stride=2)

    assert qpos.shape == (2, 3, 10)
    np.testing.assert_allclose(qpos[0, :, :3], root_pos[:3])
    np.testing.assert_allclose(qpos[1, :, :3], root_pos[2:5])
    np.testing.assert_allclose(qpos[0, :, 3:7], root_quat[:3])
    np.testing.assert_allclose(qpos[1, :, 7:10], features[1, :, 2:5])


def test_features_to_qpos_windows_reorders_g1_isaaclab_joints_to_mujoco_order():
    slices = FeatureSlices()
    slices.joint_pos = (0, 29)
    slices.joint_vel = (29, 58)
    slices.pelvis_rot6d_b = (58, 64)
    slices.pelvis_lin_vel_b = (64, 67)
    slices.pelvis_ang_vel_b = (67, 70)
    features = np.zeros((1, 1, 70), dtype=np.float32)
    features[0, 0, :29] = np.arange(29, dtype=np.float32)
    root_pos = np.zeros((1, 3), dtype=np.float32)
    root_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    qpos = features_to_qpos_windows(features, slices, root_pos, root_quat, stride=1)

    np.testing.assert_allclose(qpos[0, 0, 7:], np.asarray(ISAACLAB_TO_MUJOCO_REINDEX))


def test_clamp_window_state_wraps_frame_and_clamps_window():
    assert clamp_window_state(0, -1, num_windows=4, window_size=3) == (0, 2)
    assert clamp_window_state(1, 3, num_windows=4, window_size=3) == (1, 0)
    assert clamp_window_state(-5, 1, num_windows=4, window_size=3) == (0, 1)
    assert clamp_window_state(9, 2, num_windows=4, window_size=3) == (3, 2)


def test_overlap_average_windows_reconstructs_full_sequence_from_sliding_windows():
    windows = np.array(
        [
            [[0.0], [10.0], [20.0]],
            [[12.0], [22.0], [32.0]],
            [[24.0], [34.0], [44.0]],
        ],
        dtype=np.float32,
    )

    sequence = overlap_average_windows(windows, stride=1, total_frames=5)

    np.testing.assert_allclose(
        sequence[:, 0],
        np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=np.float32),
    )


def test_read_fps_supports_scalar_array_and_default():
    assert read_fps({"fps": np.array([60], dtype=np.int64)}, "fps") == 60.0
    assert read_fps({"fps": np.array(24.0, dtype=np.float32)}, "fps") == 24.0
    assert read_fps({}, "fps", default=30.0) == 30.0
