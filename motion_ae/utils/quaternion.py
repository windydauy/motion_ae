"""NumPy 四元数工具函数，用于数据预处理阶段的坐标变换。

四元数格式约定：(w, x, y, z)  —— 标量在前。
如果你的数据使用 (x, y, z, w)，请在 feature_builder 中做转换。
"""
from __future__ import annotations

import numpy as np


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭 (即逆，假设单位四元数)。

    Args:
        q: (..., 4)  格式 (w, x, y, z)

    Returns:
        (..., 4) 共轭四元数
    """
    conj = q.copy()
    conj[..., 1:] *= -1
    return conj


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 四元数乘法 q1 * q2。

    Args:
        q1, q2: (..., 4) 格式 (w, x, y, z)

    Returns:
        (..., 4) 乘积
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数旋转向量: v' = q * [0, v] * q^{-1}。

    Args:
        q: (..., 4) 格式 (w, x, y, z)
        v: (..., 3)

    Returns:
        (..., 3) 旋转后的向量
    """
    q_xyz = q[..., 1:]                     # (..., 3)
    t = 2.0 * np.cross(q_xyz, v)           # (..., 3)
    return v + q[..., 0:1] * t + np.cross(q_xyz, t)


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """将单位四元数转换为 6D 旋转表示。

    6D 表示使用旋转矩阵的前两列按列拼接：
    [R00, R10, R20, R01, R11, R21]。
    """
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)

    r01 = 2.0 * (x * y - z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r21 = 2.0 * (y * z + x * w)

    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """从四元数中提取 yaw 分量，返回仅包含 yaw 旋转的四元数。

    假设 z 轴朝上。yaw 为绕 z 轴的旋转。

    Args:
        q: (..., 4) 格式 (w, x, y, z)

    Returns:
        (..., 4) 仅 yaw 的四元数
    """
    w, _x, _y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw_mag = np.sqrt(w ** 2 + z ** 2 + 1e-12)
    yaw_w = w / yaw_mag
    yaw_z = z / yaw_mag

    result = np.zeros_like(q)
    result[..., 0] = yaw_w
    result[..., 3] = yaw_z
    return result
