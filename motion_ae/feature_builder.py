"""从 npz 数据构建单帧特征向量。

特征拼接顺序：
    [joint_pos | joint_vel | pelvis_quat_b | pelvis_lin_vel_b | pelvis_ang_vel_b]

其中 pelvis_* 是从世界坐标系转换到 anchor（yaw-only）坐标系后的值。

⚠️  如果你的 npz 四元数格式不是 (w, x, y, z)，请在下面的
    `_ensure_wxyz` 函数中修改。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from motion_ae.config import NpzKeysConfig, PelvisConfig
from motion_ae.utils.quaternion import quat_apply, quat_conjugate, yaw_quat


# ------------------------------------------------------------------
# 特征组维度信息（由 build_features 自动计算后填入）
# ------------------------------------------------------------------

@dataclass
class FeatureSlices:
    """记录各特征组在最终特征向量中的 [start, end) 切片。"""
    joint_pos: Tuple[int, int] = (0, 0)
    joint_vel: Tuple[int, int] = (0, 0)
    pelvis_quat_b: Tuple[int, int] = (0, 0)
    pelvis_lin_vel_b: Tuple[int, int] = (0, 0)
    pelvis_ang_vel_b: Tuple[int, int] = (0, 0)

    def as_dict(self) -> Dict[str, Tuple[int, int]]:
        return {
            "joint_pos": self.joint_pos,
            "joint_vel": self.joint_vel,
            "pelvis_quat_b": self.pelvis_quat_b,
            "pelvis_lin_vel_b": self.pelvis_lin_vel_b,
            "pelvis_ang_vel_b": self.pelvis_ang_vel_b,
        }

    @property
    def total_dim(self) -> int:
        return max(e for _, e in self.as_dict().values())


def _ensure_wxyz(quat: np.ndarray) -> np.ndarray:
    """确保四元数为 (w, x, y, z) 格式。

    ⚠️  Isaac Lab / Isaac Gym 默认使用 (w, x, y, z)。
    如果你的数据是 (x, y, z, w)，取消下面的注释。
    """
    # 如果是 (x, y, z, w) 格式，取消下面这行注释：
    # quat = np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1)
    return quat


# ------------------------------------------------------------------
# Pelvis 提取（独立函数，便于修改）
# ------------------------------------------------------------------

def extract_pelvis_data(
    npz_data: Dict[str, np.ndarray],
    keys: NpzKeysConfig,
    pelvis_cfg: PelvisConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 npz 数据中提取 pelvis 的 quat / lin_vel / ang_vel（世界坐标系）。

    Args:
        npz_data: npz 加载后的字典
        keys: 键名映射配置
        pelvis_cfg: pelvis 配置（主要是 body_index）

    Returns:
        (pelvis_quat_w, pelvis_lin_vel_w, pelvis_ang_vel_w)
        各 shape = (T, 4), (T, 3), (T, 3)
    """
    idx = pelvis_cfg.body_index

    pelvis_quat_w = npz_data[keys.body_quat_w][:, idx, :]      # (T, 4)
    pelvis_lin_vel_w = npz_data[keys.body_lin_vel_w][:, idx, :] # (T, 3)
    pelvis_ang_vel_w = npz_data[keys.body_ang_vel_w][:, idx, :] # (T, 3)

    pelvis_quat_w = _ensure_wxyz(pelvis_quat_w)

    return pelvis_quat_w, pelvis_lin_vel_w, pelvis_ang_vel_w


# ------------------------------------------------------------------
# 世界系 → Anchor 系变换
# ------------------------------------------------------------------

def world_to_anchor(
    pelvis_quat_w: np.ndarray,
    pelvis_lin_vel_w: np.ndarray,
    pelvis_ang_vel_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 pelvis 的 quat / lin_vel / ang_vel 从世界系转换到 anchor（yaw-only）系。

    Anchor frame = pelvis 的 yaw-only 朝向帧（z 轴朝上时只保留绕 z 轴的旋转）。

    Args:
        pelvis_quat_w:     (T, 4) 世界系四元数 (w, x, y, z)
        pelvis_lin_vel_w:  (T, 3) 世界系线速度
        pelvis_ang_vel_w:  (T, 3) 世界系角速度

    Returns:
        (pelvis_quat_b, pelvis_lin_vel_b, pelvis_ang_vel_b)
    """
    anchor_yaw = yaw_quat(pelvis_quat_w)                         # (T, 4)
    anchor_yaw_inv = quat_conjugate(anchor_yaw)                   # (T, 4)

    from motion_ae.utils.quaternion import quat_mul
    pelvis_quat_b = quat_mul(anchor_yaw_inv, pelvis_quat_w)      # (T, 4)
    pelvis_lin_vel_b = quat_apply(anchor_yaw_inv, pelvis_lin_vel_w)  # (T, 3)
    pelvis_ang_vel_b = quat_apply(anchor_yaw_inv, pelvis_ang_vel_w)  # (T, 3)

    return pelvis_quat_b, pelvis_lin_vel_b, pelvis_ang_vel_b


# ------------------------------------------------------------------
# 主函数：构建特征
# ------------------------------------------------------------------

def build_features(
    npz_data: Dict[str, np.ndarray],
    keys: NpzKeysConfig,
    pelvis_cfg: PelvisConfig,
    debug: bool = False,
) -> Tuple[np.ndarray, FeatureSlices]:
    """从一个 npz 数据字典构建每帧特征向量。

    Args:
        npz_data: npz 加载后的字典
        keys: 键名映射
        pelvis_cfg: pelvis 配置
        debug: 是否打印调试信息

    Returns:
        features: (T, D) float32
        slices: FeatureSlices 记录各组切片
    """
    if debug:
        print("=== Feature Builder Debug ===")
        for k in sorted(npz_data.keys()):
            print(f"  {k:25s}  shape={str(npz_data[k].shape)}")

    joint_pos = npz_data[keys.joint_pos].astype(np.float32)      # (T, J)
    joint_vel = npz_data[keys.joint_vel].astype(np.float32)      # (T, J)
    T = joint_pos.shape[0]

    pelvis_quat_w, pelvis_lin_vel_w, pelvis_ang_vel_w = extract_pelvis_data(
        npz_data, keys, pelvis_cfg
    )
    pelvis_quat_b, pelvis_lin_vel_b, pelvis_ang_vel_b = world_to_anchor(
        pelvis_quat_w, pelvis_lin_vel_w, pelvis_ang_vel_w
    )

    pelvis_quat_b = pelvis_quat_b.astype(np.float32)
    pelvis_lin_vel_b = pelvis_lin_vel_b.astype(np.float32)
    pelvis_ang_vel_b = pelvis_ang_vel_b.astype(np.float32)

    assert joint_pos.shape[0] == T
    assert joint_vel.shape[0] == T
    assert pelvis_quat_b.shape == (T, 4)
    assert pelvis_lin_vel_b.shape == (T, 3)
    assert pelvis_ang_vel_b.shape == (T, 3)

    features = np.concatenate([
        joint_pos,           # (T, J)
        joint_vel,           # (T, J)
        pelvis_quat_b,      # (T, 4)
        pelvis_lin_vel_b,    # (T, 3)
        pelvis_ang_vel_b,    # (T, 3)
    ], axis=-1)              # (T, D)

    J = joint_pos.shape[1]
    offset = 0
    s = FeatureSlices()
    s.joint_pos = (offset, offset + J);            offset += J
    s.joint_vel = (offset, offset + J);            offset += J
    s.pelvis_quat_b = (offset, offset + 4);        offset += 4
    s.pelvis_lin_vel_b = (offset, offset + 3);     offset += 3
    s.pelvis_ang_vel_b = (offset, offset + 3);     offset += 3

    if debug:
        print(f"  Single frame dim D = {features.shape[1]}")
        print(f"  Feature slices: {s.as_dict()}")
        print(f"  Total frames T = {T}")
        print()

    return features, s
