"""Streaming motion window dataset for large npz collections."""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from motion_ae.config import MotionAEConfig, PelvisConfig
from motion_ae.feature_builder import FeatureSlices, world_to_anchor
from motion_ae.utils.io import find_npz_files
from motion_ae.utils.normalization import FeatureNormalizer
from motion_ae.utils.quaternion import quat_to_rot6d

logger = logging.getLogger(__name__)

STREAMING_CACHE_FORMAT_VERSION = 1


def _feature_slices_to_dict(slices: FeatureSlices) -> dict:
    return {name: list(bounds) for name, bounds in slices.as_dict().items()}


def _feature_slices_from_dict(raw: dict) -> FeatureSlices:
    slices = FeatureSlices()
    for name in slices.as_dict():
        if name in raw:
            setattr(slices, name, tuple(raw[name]))
    return slices


def _slices_from_joint_dim(joint_dim: int) -> FeatureSlices:
    offset = 0
    slices = FeatureSlices()
    slices.joint_pos = (offset, offset + joint_dim)
    offset += joint_dim
    slices.joint_vel = (offset, offset + joint_dim)
    offset += joint_dim
    slices.pelvis_rot6d_b = (offset, offset + 6)
    offset += 6
    slices.pelvis_lin_vel_b = (offset, offset + 3)
    offset += 3
    slices.pelvis_ang_vel_b = (offset, offset + 3)
    return slices


def _streaming_cache_payload(cfg: MotionAEConfig) -> dict:
    return {
        "format_version": STREAMING_CACHE_FORMAT_VERSION,
        "data_path": os.path.abspath(cfg.data.data_path),
        "npz_filename": cfg.data.npz_filename,
        "val_ratio": cfg.data.val_ratio,
        "max_files": cfg.data.max_files,
        "seed": cfg.training.seed,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "npz_keys": asdict(cfg.npz_keys),
        "pelvis": asdict(cfg.pelvis),
        "normalization_eps": cfg.normalization.eps,
    }


def _streaming_cache_key(cfg: MotionAEConfig) -> str:
    encoded = json.dumps(
        _streaming_cache_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def streaming_manifest_path(cfg: MotionAEConfig) -> str:
    cache_dir = os.path.join(cfg.training.output_root, "_dataset_cache")
    return os.path.join(cache_dir, f"streaming_manifest_{_streaming_cache_key(cfg)}.pt")


def _load_feature_file_selective(path: str, cfg: MotionAEConfig) -> Tuple[np.ndarray, FeatureSlices]:
    keys = cfg.npz_keys
    pelvis_cfg: PelvisConfig = cfg.pelvis
    idx = int(pelvis_cfg.body_index)

    try:
        with np.load(path, allow_pickle=False) as data:
            joint_pos = data[keys.joint_pos].astype(np.float32)
            joint_vel = data[keys.joint_vel].astype(np.float32)
            pelvis_quat_w = data[keys.body_quat_w][:, idx, :]
            pelvis_lin_vel_w = data[keys.body_lin_vel_w][:, idx, :]
            pelvis_ang_vel_w = data[keys.body_ang_vel_w][:, idx, :]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load streaming features from {path}") from exc

    pelvis_quat_anchor_b, pelvis_lin_vel_b, pelvis_ang_vel_b = world_to_anchor(
        pelvis_quat_w,
        pelvis_lin_vel_w,
        pelvis_ang_vel_w,
    )
    pelvis_rot6d_b = quat_to_rot6d(pelvis_quat_anchor_b).astype(np.float32)
    features = np.concatenate(
        [
            joint_pos,
            joint_vel,
            pelvis_rot6d_b,
            pelvis_lin_vel_b.astype(np.float32),
            pelvis_ang_vel_b.astype(np.float32),
        ],
        axis=-1,
    )
    return features.astype(np.float32, copy=False), _slices_from_joint_dim(joint_pos.shape[1])


def _read_npz_header(path: str, cfg: MotionAEConfig) -> Tuple[Optional[dict], Optional[FeatureSlices]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            joint_pos = data[cfg.npz_keys.joint_pos]
            frames = int(joint_pos.shape[0])
            joint_dim = int(joint_pos.shape[1])
            # Validate keys used by the streaming feature builder without reading unused arrays.
            _ = data[cfg.npz_keys.joint_vel].shape
            body_quat_shape = data[cfg.npz_keys.body_quat_w].shape
            _ = data[cfg.npz_keys.body_lin_vel_w].shape
            _ = data[cfg.npz_keys.body_ang_vel_w].shape
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping unreadable npz %s: %s", path, exc)
        return None, None

    if int(cfg.pelvis.body_index) >= int(body_quat_shape[1]):
        logger.warning(
            "Skipping %s: pelvis body index %d is outside body_quat_w shape %s",
            path,
            cfg.pelvis.body_index,
            body_quat_shape,
        )
        return None, None
    windows = 0
    if frames >= cfg.window_size:
        windows = (frames - cfg.window_size) // cfg.stride + 1
    entry = {
        "path": os.path.abspath(path),
        "num_frames": frames,
        "num_windows": int(windows),
    }
    return entry, _slices_from_joint_dim(joint_dim)


def _bounded_thread_map(loader, paths: List[str], max_workers: int):
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


def _scan_npz_manifest(cfg: MotionAEConfig) -> Tuple[List[dict], FeatureSlices]:
    t0 = time.time()
    npz_paths = find_npz_files(cfg.data.data_path, cfg.data.npz_filename)
    if cfg.data.max_files is not None:
        npz_paths = npz_paths[: int(cfg.data.max_files)]
    if not npz_paths:
        raise FileNotFoundError(f"No npz files found in {cfg.data.data_path}")
    logger.info("Found %d npz files under %s in %.1fs", len(npz_paths), cfg.data.data_path, time.time() - t0)

    workers = max(1, int(getattr(cfg.data, "manifest_workers", 1)))
    loader = partial(_read_npz_header, cfg=cfg)
    iterator = (
        (loader(path) for path in npz_paths)
        if workers == 1 or len(npz_paths) == 1
        else _bounded_thread_map(loader, npz_paths, workers)
    )

    entries: List[dict] = []
    slices: Optional[FeatureSlices] = None
    for entry, current_slices in iterator:
        if entry is None or current_slices is None:
            continue
        if entry["num_windows"] <= 0:
            continue
        if slices is None:
            slices = current_slices
        elif slices.total_dim != current_slices.total_dim:
            raise ValueError(f"Feature dim mismatch in {entry['path']}: {current_slices.total_dim} vs {slices.total_dim}")
        entries.append(entry)

    if not entries or slices is None:
        raise ValueError(f"No valid windows found under {cfg.data.data_path}")
    logger.info("Scanned %d valid npz files in %.1fs", len(entries), time.time() - t0)
    return entries, slices


def _split_entries(entries: List[dict], cfg: MotionAEConfig) -> Tuple[List[dict], List[dict]]:
    rng = np.random.RandomState(cfg.training.seed)
    indices = rng.permutation(len(entries))
    n_val = max(1, int(len(entries) * cfg.data.val_ratio))
    if len(entries) > 1:
        n_val = min(n_val, len(entries) - 1)
    val_indices = set(indices[:n_val].tolist())
    train_entries = [copy.deepcopy(entry) for i, entry in enumerate(entries) if i not in val_indices]
    val_entries = [copy.deepcopy(entry) for i, entry in enumerate(entries) if i in val_indices]
    if not train_entries:
        raise ValueError("Streaming train split is empty; provide more files or lower data.val_ratio")
    return train_entries, val_entries


def _compute_streaming_stats(train_entries: List[dict], cfg: MotionAEConfig) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    total_count = 0
    sum_: Optional[np.ndarray] = None
    sumsq: Optional[np.ndarray] = None
    for idx, entry in enumerate(train_entries):
        features, _slices = _load_feature_file_selective(entry["path"], cfg)
        if features.shape[0] == 0:
            continue
        if sum_ is None:
            sum_ = np.zeros(features.shape[1], dtype=np.float64)
            sumsq = np.zeros(features.shape[1], dtype=np.float64)
        elif features.shape[1] != sum_.shape[0]:
            raise ValueError(f"Feature dim mismatch in {entry['path']}: {features.shape[1]} vs {sum_.shape[0]}")
        total_count += int(features.shape[0])
        sum_ += features.sum(axis=0, dtype=np.float64)
        sumsq += np.square(features, dtype=np.float64).sum(axis=0, dtype=np.float64)
        if (idx + 1) % 1000 == 0:
            logger.info("Streaming stats: processed %d/%d files", idx + 1, len(train_entries))

    if total_count == 0 or sum_ is None or sumsq is None:
        raise ValueError("No frames available for streaming stats")
    mean = sum_ / total_count
    var = np.maximum(sumsq / total_count - mean * mean, 0.0)
    std = np.sqrt(var)
    logger.info("Computed streaming stats from %d files in %.1fs", len(train_entries), time.time() - t0)
    return mean.astype(np.float32), std.astype(np.float32)


def _total_windows(entries: List[dict]) -> int:
    return int(sum(int(entry["num_windows"]) for entry in entries))


def _save_manifest(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_manifest(path: str, cfg: MotionAEConfig, stats_path: Optional[str]) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load streaming manifest %s: %s", path, exc)
        return None
    meta = payload.get("meta", {})
    if meta.get("format_version") != STREAMING_CACHE_FORMAT_VERSION:
        logger.warning("Streaming manifest format mismatch, rebuilding: %s", path)
        return None
    if meta.get("cache_key") != _streaming_cache_key(cfg):
        logger.warning("Streaming manifest key mismatch, rebuilding: %s", path)
        return None
    if stats_path is not None and os.path.exists(stats_path):
        requested = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
        if not (
            np.array_equal(np.asarray(payload["mean"], dtype=np.float32), requested.mean)
            and np.array_equal(np.asarray(payload["std"], dtype=np.float32), requested.std)
        ):
            logger.warning("Streaming manifest stats differ from %s, rebuilding", stats_path)
            return None
    return payload


def ensure_streaming_manifest(cfg: MotionAEConfig, stats_path: Optional[str] = None) -> str:
    path = streaming_manifest_path(cfg)
    cached = _load_manifest(path, cfg, stats_path)
    if cached is not None:
        logger.info("Using existing streaming manifest: %s", path)
        return path

    entries, slices = _scan_npz_manifest(cfg)
    train_entries, val_entries = _split_entries(entries, cfg)
    if stats_path is not None and os.path.exists(stats_path):
        normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)
        mean, std = normalizer.mean, normalizer.std
    else:
        mean, std = _compute_streaming_stats(train_entries, cfg)

    payload = {
        "meta": {
            **_streaming_cache_payload(cfg),
            "cache_key": _streaming_cache_key(cfg),
            "num_files": len(entries),
            "num_train_files": len(train_entries),
            "num_val_files": len(val_entries),
            "num_train_windows": _total_windows(train_entries),
            "num_val_windows": _total_windows(val_entries),
        },
        "train_entries": train_entries,
        "val_entries": val_entries,
        "mean": mean,
        "std": std,
        "slices": _feature_slices_to_dict(slices),
    }
    _save_manifest(path, payload)
    logger.info(
        "Saved streaming manifest %s (train_windows=%d, val_windows=%d)",
        path,
        payload["meta"]["num_train_windows"],
        payload["meta"]["num_val_windows"],
    )
    return path


def _shard_entries_by_windows(entries: List[dict], rank: int, world_size: int) -> List[dict]:
    if world_size <= 1:
        return [copy.deepcopy(entry) for entry in entries]
    buckets: List[List[Tuple[int, dict]]] = [[] for _ in range(world_size)]
    totals = [0 for _ in range(world_size)]
    indexed = list(enumerate(entries))
    indexed.sort(key=lambda item: int(item[1]["num_windows"]), reverse=True)
    for original_idx, entry in indexed:
        bucket_idx = min(range(world_size), key=lambda i: totals[i])
        buckets[bucket_idx].append((original_idx, entry))
        totals[bucket_idx] += int(entry["num_windows"])
    selected = sorted(buckets[rank], key=lambda item: item[0])
    return [copy.deepcopy(entry) for _idx, entry in selected]


class StreamingMotionWindowDataset(Dataset):
    """Lazily loads normalized motion windows from npz files."""

    def __init__(
        self,
        entries: List[dict],
        cfg: MotionAEConfig,
        normalizer: FeatureNormalizer,
        slices: FeatureSlices,
    ):
        self.entries = [copy.deepcopy(entry) for entry in entries]
        self.cfg = cfg
        self.normalizer = normalizer
        self.slices = slices
        self.window_size = cfg.window_size
        self.stride = cfg.stride
        self.cache_size = max(0, int(getattr(cfg.data, "streaming_cache_size", 0)))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        counts = [int(entry["num_windows"]) for entry in self.entries]
        self._starts = np.cumsum([0] + counts[:-1], dtype=np.int64)
        self._ends = np.cumsum(counts, dtype=np.int64)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        if len(self._ends) == 0:
            return 0
        return int(self._ends[-1])

    @property
    def entry_starts(self) -> np.ndarray:
        return self._starts

    def _features_for_path(self, path: str) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        features, slices = _load_feature_file_selective(path, self.cfg)
        if slices.total_dim != self.slices.total_dim:
            raise ValueError(f"Feature dim mismatch in {path}: {slices.total_dim} vs {self.slices.total_dim}")
        if self.cache_size > 0:
            self._cache[path] = features
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return features

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        entry_idx = bisect.bisect_right(self._ends, idx)
        entry = self.entries[entry_idx]
        local_idx = idx - int(self._starts[entry_idx])
        start = int(local_idx) * self.stride
        end = start + self.window_size
        features = self._features_for_path(entry["path"])
        window = features[start:end]
        if window.shape[0] != self.window_size:
            raise RuntimeError(f"Invalid window from {entry['path']}: start={start}, shape={window.shape}")
        normalized = self.normalizer.normalize_np(window)
        return torch.from_numpy(np.ascontiguousarray(normalized.astype(np.float32, copy=False)))


class ClipLocalBatchSampler(Sampler[List[int]]):
    """Yields batches that stay close to clip boundaries to improve file-cache hit rate."""

    def __init__(
        self,
        dataset: StreamingMotionWindowDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        entry_indices = np.arange(len(self.dataset.entries))
        if self.shuffle:
            np.random.shuffle(entry_indices)

        batch: List[int] = []
        for entry_idx in entry_indices:
            start = int(self.dataset.entry_starts[entry_idx])
            count = int(self.dataset.entries[entry_idx]["num_windows"])
            local = np.arange(count, dtype=np.int64)
            if self.shuffle:
                np.random.shuffle(local)
            for local_idx in local:
                batch.append(start + int(local_idx))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch


def build_streaming_datasets(
    cfg: MotionAEConfig,
    stats_path: Optional[str] = None,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[StreamingMotionWindowDataset, StreamingMotionWindowDataset, FeatureNormalizer, FeatureSlices, dict]:
    manifest_path = ensure_streaming_manifest(cfg, stats_path=stats_path)
    payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
    normalizer = FeatureNormalizer(
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
        eps=cfg.normalization.eps,
    )
    slices = _feature_slices_from_dict(payload["slices"])
    train_entries = _shard_entries_by_windows(payload["train_entries"], rank, world_size)
    val_entries = _shard_entries_by_windows(payload["val_entries"], rank, world_size)
    train_ds = StreamingMotionWindowDataset(train_entries, cfg, normalizer, slices)
    val_ds = StreamingMotionWindowDataset(val_entries, cfg, normalizer, slices)
    meta = dict(payload["meta"])
    meta.update(
        {
            "manifest_path": manifest_path,
            "local_train_windows": len(train_ds),
            "local_val_windows": len(val_ds),
            "local_train_files": len(train_entries),
            "local_val_files": len(val_entries),
        }
    )
    return train_ds, val_ds, normalizer, slices, meta


def build_streaming_loader(
    ds: StreamingMotionWindowDataset,
    cfg: MotionAEConfig,
    device: torch.device,
    *,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    sampler = ClipLocalBatchSampler(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    num_workers = max(0, int(cfg.training.num_workers))
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def build_streaming_train_val_loaders(
    train_ds: StreamingMotionWindowDataset,
    val_ds: StreamingMotionWindowDataset,
    cfg: MotionAEConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    return (
        build_streaming_loader(train_ds, cfg, device, shuffle=True, drop_last=True),
        build_streaming_loader(val_ds, cfg, device, shuffle=False, drop_last=False),
    )
