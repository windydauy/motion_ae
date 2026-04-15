"""NPZ 文件加载与路径扫描工具。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def find_npz_files(root: str, npz_filename: str = "motion.npz") -> List[str]:
    """递归搜索 *root* 目录下所有名为 *npz_filename* 的文件，返回绝对路径列表。

    如果 *root* 本身就是一个 npz 文件，直接返回它。
    """
    root = str(root)
    if os.path.isfile(root) and root.endswith(".npz"):
        return [os.path.abspath(root)]

    results: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if npz_filename in filenames:
            results.append(os.path.abspath(os.path.join(dirpath, npz_filename)))
    results.sort()
    return results


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """加载单个 npz 文件，返回键值字典。"""
    data = np.load(path, allow_pickle=False)
    return dict(data)


def debug_npz(path: str) -> None:
    """打印 npz 文件中所有键名、shape 和 dtype，用于调试。"""
    data = np.load(path, allow_pickle=False)
    print(f"=== Debug NPZ: {path} ===")
    for key in sorted(data.files):
        arr = data[key]
        print(f"  {key:25s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")
    print()
