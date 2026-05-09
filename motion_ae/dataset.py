"""Motion 滑窗数据集。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from motion_ae.config import MotionAEConfig
from motion_ae.feature_builder import FeatureSlices, build_features
from motion_ae.utils.io import find_npz_files, load_npz
from motion_ae.utils.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)
CACHE_FORMAT_VERSION = 1


def _feature_slices_to_dict(slices: FeatureSlices) -> dict:
    return {
        name: list(bounds)
        for name, bounds in slices.as_dict().items()
    }


def _feature_slices_from_dict(raw: dict) -> FeatureSlices:
    slices = FeatureSlices()
    for name in slices.as_dict():
        if name in raw:
            setattr(slices, name, tuple(raw[name]))
    return slices


def _dataset_cache_payload(cfg: MotionAEConfig) -> dict:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "data_path": os.path.abspath(cfg.data.data_path),
        "npz_filename": cfg.data.npz_filename,
        "val_ratio": cfg.data.val_ratio,
        "seed": cfg.training.seed,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "npz_keys": asdict(cfg.npz_keys),
        "pelvis": asdict(cfg.pelvis),
        "normalization_eps": cfg.normalization.eps,
    }


def _dataset_cache_key(cfg: MotionAEConfig) -> str:
    payload = _dataset_cache_payload(cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _dataset_cache_path(cfg: MotionAEConfig) -> str:
    cache_dir = os.path.join(cfg.training.output_root, "_dataset_cache")
    return os.path.join(cache_dir, f"motion_windows_{_dataset_cache_key(cfg)}.pt")


def _sharded_dataset_cache_dir(cfg: MotionAEConfig, world_size: int) -> str:
    cache_dir = os.path.join(cfg.training.output_root, "_dataset_cache")
    return os.path.join(cache_dir, f"motion_windows_{_dataset_cache_key(cfg)}_w{world_size}")


def _rank_bounds(n: int, rank: int, world_size: int) -> Tuple[int, int]:
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def sharded_dataset_cache_exists(cfg: MotionAEConfig, world_size: int) -> bool:
    cache_dir = _sharded_dataset_cache_dir(cfg, world_size)
    meta_path = os.path.join(cache_dir, "meta.pt")
    if not os.path.exists(meta_path):
        return False
    for rank in range(world_size):
        if not os.path.exists(os.path.join(cache_dir, f"train_rank{rank:05d}.pt")):
            return False
        if not os.path.exists(os.path.join(cache_dir, f"val_rank{rank:05d}.pt")):
            return False
    return True


def _save_tensor_shards(
    data: torch.Tensor,
    cache_dir: str,
    split: str,
    world_size: int,
) -> List[int]:
    counts: List[int] = []
    n = len(data)
    for rank in range(world_size):
        start, end = _rank_bounds(n, rank, world_size)
        shard = data[start:end].contiguous().clone()
        path = os.path.join(cache_dir, f"{split}_rank{rank:05d}.pt")
        torch.save({"data": shard, "start": start, "end": end}, path)
        counts.append(end - start)
        del shard
    return counts


def ensure_sharded_dataset_cache(cfg: MotionAEConfig, world_size: int) -> str:
    """确保存在按 rank 切分的 packed dataset cache。

    调用方应只在 rank0 执行该函数，再用 distributed barrier 同步其他 rank。
    """
    if not cfg.training.dataset_cache:
        raise ValueError("DDP sharded preload requires training.dataset_cache=true")

    cache_dir = _sharded_dataset_cache_dir(cfg, world_size)
    if sharded_dataset_cache_exists(cfg, world_size):
        logger.info("Using existing sharded dataset cache: %s", cache_dir)
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Building sharded dataset cache in %s", cache_dir)
    full_cache_exists = os.path.exists(_dataset_cache_path(cfg))
    original_dataset_cache = cfg.training.dataset_cache
    if not full_cache_exists:
        cfg.training.dataset_cache = False
    try:
        train_ds, val_ds, normalizer, slices = build_datasets(cfg)
    finally:
        cfg.training.dataset_cache = original_dataset_cache

    t0 = time.time()
    train_counts = _save_tensor_shards(train_ds.data, cache_dir, "train", world_size)
    val_counts = _save_tensor_shards(val_ds.data, cache_dir, "val", world_size)
    meta = {
        **_dataset_cache_payload(cfg),
        "cache_key": _dataset_cache_key(cfg),
        "world_size": world_size,
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "mean": normalizer.mean,
        "std": normalizer.std,
        "slices": _feature_slices_to_dict(slices),
    }
    tmp_meta_path = os.path.join(cache_dir, "meta.pt.tmp")
    meta_path = os.path.join(cache_dir, "meta.pt")
    torch.save(meta, tmp_meta_path)
    os.replace(tmp_meta_path, meta_path)
    logger.info(
        "Saved sharded dataset cache in %.1fs (world_size=%d, train=%d, val=%d)",
        time.time() - t0,
        world_size,
        len(train_ds),
        len(val_ds),
    )
    return cache_dir


def load_sharded_datasets(
    cfg: MotionAEConfig,
    rank: int,
    world_size: int,
) -> Tuple["MotionWindowDataset", "MotionWindowDataset", FeatureNormalizer, FeatureSlices, dict]:
    """加载当前 rank 的 train/val packed dataset shard。"""
    cache_dir = _sharded_dataset_cache_dir(cfg, world_size)
    meta_path = os.path.join(cache_dir, "meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Sharded dataset cache meta not found: {meta_path}")

    t0 = time.time()
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if int(meta["world_size"]) != int(world_size):
        raise ValueError(f"Cache world_size mismatch: {meta['world_size']} vs {world_size}")
    normalizer = FeatureNormalizer(
        np.asarray(meta["mean"], dtype=np.float32),
        np.asarray(meta["std"], dtype=np.float32),
        eps=cfg.normalization.eps,
    )
    slices = _feature_slices_from_dict(meta["slices"])

    train_payload = torch.load(
        os.path.join(cache_dir, f"train_rank{rank:05d}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    val_payload = torch.load(
        os.path.join(cache_dir, f"val_rank{rank:05d}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    train_ds = MotionWindowDataset.from_tensor(
        train_payload["data"], cfg, normalizer, slices,
    )
    val_ds = MotionWindowDataset.from_tensor(
        val_payload["data"], cfg, normalizer, slices,
    )
    logger.info(
        "Loaded sharded dataset cache rank %d/%d in %.1fs (train=%d, val=%d)",
        rank,
        world_size,
        time.time() - t0,
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds, normalizer, slices, meta


def _try_load_dataset_cache(
    cfg: MotionAEConfig,
    stats_path: Optional[str] = None,
) -> Optional[Tuple["MotionWindowDataset", "MotionWindowDataset", FeatureNormalizer, FeatureSlices]]:
    if not cfg.training.dataset_cache:
        return None

    path = _dataset_cache_path(cfg)
    if not os.path.exists(path):
        return None

    t0 = time.time()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        if meta.get("format_version") != CACHE_FORMAT_VERSION:
            logger.warning("Dataset cache format mismatch, rebuilding: %s", path)
            return None

        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        if stats_path is not None and os.path.exists(stats_path):
            requested = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
            if not (
                np.array_equal(mean, requested.mean)
                and np.array_equal(std, requested.std)
            ):
                logger.warning("Dataset cache stats differ from %s, rebuilding", stats_path)
                return None

        slices = _feature_slices_from_dict(payload["slices"])
        normalizer = FeatureNormalizer(mean, std, eps=cfg.normalization.eps)
        train_ds = MotionWindowDataset.from_tensor(
            payload["train_data"], cfg, normalizer, slices,
        )
        val_ds = MotionWindowDataset.from_tensor(
            payload["val_data"], cfg, normalizer, slices,
        )
        logger.info(
            "Loaded dataset cache %s in %.1fs (train=%d, val=%d)",
            path,
            time.time() - t0,
            len(train_ds),
            len(val_ds),
        )
        return train_ds, val_ds, normalizer, slices
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load dataset cache %s, rebuilding: %s", path, exc)
        return None


def _save_dataset_cache(
    cfg: MotionAEConfig,
    train_ds: "MotionWindowDataset",
    val_ds: "MotionWindowDataset",
    normalizer: FeatureNormalizer,
    slices: FeatureSlices,
    *,
    num_files: int,
    num_train_files: int,
    num_val_files: int,
) -> None:
    if not cfg.training.dataset_cache:
        return

    path = _dataset_cache_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    payload = {
        "meta": {
            **_dataset_cache_payload(cfg),
            "cache_key": _dataset_cache_key(cfg),
            "num_files": num_files,
            "num_train_files": num_train_files,
            "num_val_files": num_val_files,
            "num_train_windows": len(train_ds),
            "num_val_windows": len(val_ds),
        },
        "mean": normalizer.mean,
        "std": normalizer.std,
        "slices": _feature_slices_to_dict(slices),
        "train_data": train_ds.data.cpu(),
        "val_data": val_ds.data.cpu(),
    }
    t0 = time.time()
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        logger.info(
            "Saved dataset cache %s in %.1fs (train=%d, val=%d)",
            path,
            time.time() - t0,
            len(train_ds),
            len(val_ds),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save dataset cache %s: %s", path, exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _load_feature_file(path: str, cfg: MotionAEConfig) -> Tuple[np.ndarray, FeatureSlices]:
    try:
        npz_data = load_npz(path)
        return build_features(
            npz_data, cfg.npz_keys, cfg.pelvis, debug=cfg.debug,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load features from {path}") from exc


def _bounded_thread_map(
    loader,
    paths: List[str],
    max_workers: int,
) -> Iterator[Tuple[np.ndarray, FeatureSlices]]:
    max_pending = max_workers * 4
    path_iter = iter(paths)
    pending = deque()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_pending, len(paths))):
            pending.append(executor.submit(loader, next(path_iter)))

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                path = next(path_iter)
            except StopIteration:
                continue
            pending.append(executor.submit(loader, path))


def _load_all_features(
    npz_paths: List[str],
    cfg: MotionAEConfig,
) -> Tuple[List[np.ndarray], FeatureSlices]:
    t0 = time.time()
    workers = max(1, int(cfg.training.num_workers))
    all_features: List[np.ndarray] = []
    slices: Optional[FeatureSlices] = None

    if len(npz_paths) == 0:
        raise ValueError("npz_paths is empty")

    logger.info("Loading %d npz feature files with %d worker(s)", len(npz_paths), workers)
    if workers == 1 or len(npz_paths) == 1:
        iterator = (_load_feature_file(path, cfg) for path in npz_paths)
        for feats, sl in iterator:
            all_features.append(feats)
            if slices is None:
                slices = sl
    else:
        loader = partial(_load_feature_file, cfg=cfg)
        for feats, sl in _bounded_thread_map(loader, npz_paths, workers):
            all_features.append(feats)
            if slices is None:
                slices = sl

    assert slices is not None, "No features built"
    logger.info(
        "Loaded %d feature files in %.1fs",
        len(all_features),
        time.time() - t0,
    )
    return all_features, slices


def _count_windows_for_features(
    all_features: List[np.ndarray],
    window_size: int,
    stride: int,
) -> int:
    n = 0
    for feats in all_features:
        t = feats.shape[0]
        if t < window_size:
            continue
        n += (t - window_size) // stride + 1
    return n


def _pack_windows_to_numpy(
    all_features: List[np.ndarray],
    window_size: int,
    stride: int,
    feature_dim: int,
    normalizer: Optional[FeatureNormalizer],
) -> np.ndarray:
    """将多条 (T, D) 特征展开为 (N, W, D) float32，可选逐帧归一化。"""
    t0 = time.time()
    n_win = _count_windows_for_features(all_features, window_size, stride)
    logger.info(
        "Packing %d windows (window_size=%d, stride=%d, feature_dim=%d)",
        n_win,
        window_size,
        stride,
        feature_dim,
    )
    if n_win == 0:
        return np.zeros((0, window_size, feature_dim), dtype=np.float32)

    out = np.empty((n_win, window_size, feature_dim), dtype=np.float32)
    idx = 0
    mean = normalizer.mean if normalizer is not None else None
    std = normalizer.std if normalizer is not None else None

    for feats in all_features:
        t, d = feats.shape
        if t < window_size:
            continue
        assert d == feature_dim, f"feature dim mismatch: {d} vs {feature_dim}"

        windows = np.lib.stride_tricks.sliding_window_view(
            feats,
            window_shape=window_size,
            axis=0,
        )[::stride]
        windows = np.moveaxis(windows, -1, 1)
        n = int(windows.shape[0])
        target = out[idx : idx + n]
        if mean is not None and std is not None:
            np.subtract(windows, mean, out=target, casting="unsafe")
            np.divide(target, std, out=target)
        else:
            target[...] = windows
        idx += n

    assert idx == n_win
    logger.info("Packed %d windows in %.1fs", n_win, time.time() - t0)
    return out


class MotionWindowDataset(Dataset):
    """将多条 motion 特征按滑窗切分为固定长度的样本。

    每个样本 shape = (window_size, D)，target 为自身（自编码）。
    内部使用单块连续 (N, W, D) 存储，避免百万级小数组与每步 copy。
    """

    def __init__(
        self,
        npz_paths: List[str],
        cfg: MotionAEConfig,
        normalizer: Optional[FeatureNormalizer] = None,
        *,
        _cached_features: Optional[Tuple[List[np.ndarray], FeatureSlices]] = None,
        _keep_features: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.window_size = cfg.window_size
        self.stride = cfg.stride

        if _cached_features is not None:
            self.all_features, self.slices = _cached_features
        else:
            self.all_features, self.slices = _load_all_features(npz_paths, cfg)

        win_np = _pack_windows_to_numpy(
            self.all_features,
            self.window_size,
            self.stride,
            self.slices.total_dim,
            normalizer,
        )
        self._data = torch.from_numpy(np.ascontiguousarray(win_np))
        if not _keep_features:
            self.all_features = []

        if cfg.debug and len(self) > 0:
            print(
                f"[Dataset] Total windows = {len(self)}, "
                f"window shape = ({self.window_size}, {self.slices.total_dim})"
            )

    @classmethod
    def from_tensor(
        cls,
        data: torch.Tensor,
        cfg: MotionAEConfig,
        normalizer: Optional[FeatureNormalizer],
        slices: FeatureSlices,
    ) -> "MotionWindowDataset":
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.cfg = cfg
        obj.normalizer = normalizer
        obj.window_size = cfg.window_size
        obj.stride = cfg.stride
        obj.all_features = []
        obj.slices = slices
        obj._data = data.float()
        return obj

    @property
    def data(self) -> torch.Tensor:
        """形状 (N, window_size, D) 的样本张量。"""
        return self._data

    def nbytes(self) -> int:
        return int(self._data.numel() * self._data.element_size())

    def to_device_(self, device: torch.device) -> None:
        """将整块数据移到 device（原地替换）。"""
        self._data = self._data.to(device)

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


class TensorBatchLoader:
    """从单块 tensor 直接产出 batch，避免 DataLoader 的逐样本 Python collate。"""

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        if data.dim() < 1:
            raise ValueError(f"Expected data with batch dimension, got shape {tuple(data.shape)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = int(self.data.shape[0])
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = int(self.data.shape[0])
        if n == 0:
            return

        if self.drop_last:
            usable = (n // self.batch_size) * self.batch_size
        else:
            usable = n
        if usable == 0:
            return

        if self.shuffle:
            order = torch.randperm(n, device=self.data.device)[:usable]
            for start in range(0, usable, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                yield self.data.index_select(0, batch_idx)
        else:
            for start in range(0, usable, self.batch_size):
                end = min(start + self.batch_size, usable)
                yield self.data[start:end]


def build_datasets(
    cfg: MotionAEConfig,
    stats_path: Optional[str] = None,
) -> Tuple[MotionWindowDataset, MotionWindowDataset, FeatureNormalizer, FeatureSlices]:
    """构建 train/val 数据集 + normalizer + feature slices。

    按文件级别划分 train/val。
    训练集 npz 只从磁盘加载一次；统计量来自内存中的特征列表。
    如果提供 `stats_path` 且文件存在，则直接复用保存好的统计量；
    否则从 train set 重新统计 mean/std。
    """
    cached = _try_load_dataset_cache(cfg, stats_path=stats_path)
    if cached is not None:
        return cached

    t0 = time.time()
    npz_paths = find_npz_files(cfg.data.data_path, cfg.data.npz_filename)
    assert len(npz_paths) > 0, f"No npz files found in {cfg.data.data_path}"
    logger.info(
        "Found %d npz files under %s in %.1fs",
        len(npz_paths),
        cfg.data.data_path,
        time.time() - t0,
    )

    rng = np.random.RandomState(cfg.training.seed)
    indices = rng.permutation(len(npz_paths))
    n_val = max(1, int(len(npz_paths) * cfg.data.val_ratio))
    val_indices = set(indices[:n_val].tolist())
    train_paths = [p for i, p in enumerate(npz_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(npz_paths) if i in val_indices]

    train_features, slices = _load_all_features(train_paths, cfg)

    if stats_path is not None and os.path.exists(stats_path):
        normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
    else:
        from motion_ae.utils.normalization import compute_stats

        mean, std = compute_stats(train_features)
        normalizer = FeatureNormalizer(mean, std, eps=cfg.normalization.eps)

    train_ds = MotionWindowDataset(
        train_paths,
        cfg,
        normalizer=normalizer,
        _cached_features=(train_features, slices),
        _keep_features=False,
    )
    # 归一化后的窗口已经在 train_ds 中，释放未归一化 train feature 列表以降低峰值内存。
    del train_features
    val_features, _val_slices = _load_all_features(val_paths, cfg)
    assert _val_slices.total_dim == slices.total_dim
    val_ds = MotionWindowDataset(
        val_paths,
        cfg,
        normalizer=normalizer,
        _cached_features=(val_features, slices),
        _keep_features=False,
    )
    del val_features

    _save_dataset_cache(
        cfg,
        train_ds,
        val_ds,
        normalizer,
        slices,
        num_files=len(npz_paths),
        num_train_files=len(train_paths),
        num_val_files=len(val_paths),
    )

    return train_ds, val_ds, normalizer, slices


def estimate_preload_bytes(train_ds: MotionWindowDataset, val_ds: MotionWindowDataset) -> int:
    return train_ds.nbytes() + val_ds.nbytes()


def dataloader_io_options(
    device: torch.device,
    data_on_gpu: bool,
    num_workers_cfg: int,
) -> Tuple[int, bool]:
    """DataLoader 的 num_workers 与 pin_memory：数据已在 GPU 时必须单进程且不可 pin。"""
    if data_on_gpu:
        return 0, False
    return num_workers_cfg, device.type == "cuda"


def build_train_val_loaders(
    train_ds: MotionWindowDataset,
    val_ds: MotionWindowDataset,
    cfg: MotionAEConfig,
    device: torch.device,
    data_on_gpu: bool,
) -> Tuple[object, object]:
    """构建训练/验证 batch 迭代器。

    数据已整块放在 GPU 时，绕过 DataLoader 的 CPU sampler 与 per-sample collate。
    否则保留原 DataLoader 路径。
    """
    if data_on_gpu:
        return (
            TensorBatchLoader(
                train_ds.data,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                drop_last=True,
            ),
            TensorBatchLoader(
                val_ds.data,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                drop_last=False,
            ),
        )

    num_workers, pin_memory = dataloader_io_options(
        device, data_on_gpu, cfg.training.num_workers,
    )
    return (
        DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def try_preload_one_dataset(
    ds: MotionWindowDataset,
    device: torch.device,
    preload_requested: bool,
    headroom_bytes: int = 512 * 1024 * 1024,
) -> bool:
    """仅将单个数据集张量预载到 GPU（用于 evaluate 等只读一个 split 的场景）。"""
    if not preload_requested or device.type != "cuda":
        return False

    needed = ds.nbytes() + headroom_bytes
    try:
        free_b, _total_b = torch.cuda.mem_get_info(device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法查询 GPU 显存，跳过 preload_to_gpu: %s", exc)
        return False

    if needed > free_b:
        logger.warning(
            "preload_to_gpu 已关闭：该 split 估计需要约 %d MiB（含 %.0f MiB 余量），"
            "当前设备可用约 %d MiB。数据保留在 CPU。",
            needed // (1024 * 1024),
            headroom_bytes / (1024 * 1024),
            free_b // (1024 * 1024),
        )
        return False

    cpu = torch.device("cpu")
    try:
        ds.to_device_(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("preload_to_gpu 传输时 OOM，数据回退 CPU。")
        ds.to_device_(cpu)
        return False
    return True


def try_preload_datasets_to_gpu(
    train_ds: MotionWindowDataset,
    val_ds: MotionWindowDataset,
    device: torch.device,
    preload_requested: bool,
    headroom_bytes: int = 512 * 1024 * 1024,
) -> Tuple[bool, bool]:
    """按配置尝试将 train/val 整块张量放到 GPU。

    Args:
        train_ds, val_ds: 数据集（CPU 张量）。
        device: 训练设备。
        preload_requested: 配置项 preload_to_gpu。
        headroom_bytes: 为模型与优化器预留的显存余量（粗估）。

    Returns:
        (train_on_gpu, val_on_gpu)：回退时二者均为 False。
    """
    if not preload_requested or device.type != "cuda":
        return False, False

    needed = estimate_preload_bytes(train_ds, val_ds) + headroom_bytes
    try:
        free_b, _total_b = torch.cuda.mem_get_info(device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法查询 GPU 显存，跳过 preload_to_gpu: %s", exc)
        return False, False

    if needed > free_b:
        logger.warning(
            "preload_to_gpu 已关闭：估计需要约 %d MiB（含 %.0f MiB 余量），"
            "当前设备可用约 %d MiB。数据保留在 CPU，使用 pin_memory + non_blocking。",
            needed // (1024 * 1024),
            headroom_bytes / (1024 * 1024),
            free_b // (1024 * 1024),
        )
        return False, False

    cpu = torch.device("cpu")
    try:
        train_ds.to_device_(device)
        val_ds.to_device_(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("preload_to_gpu 传输时 OOM，数据回退 CPU。")
        train_ds.to_device_(cpu)
        val_ds.to_device_(cpu)
        return False, False
    return True, True
