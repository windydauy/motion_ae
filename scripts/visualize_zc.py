"""Visualize train-set continuous/discrete latents for a trained Motion AE checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motion_ae.config import MotionAEConfig, load_config
from motion_ae.feature_builder import build_features
from motion_ae.models.autoencoder import MotionAutoEncoder
from motion_ae.utils.experiment import get_device
from motion_ae.utils.io import find_npz_files, load_npz
from motion_ae.utils.normalization import FeatureNormalizer


DEFAULT_RUN_DIR = (
    "/home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae_6d/"
    "2026-04-21_16-48-32_rot_6d_boxing_walk"
)
DEFAULT_CONFIG = os.path.join(DEFAULT_RUN_DIR, "params", "config.yaml")
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_RUN_DIR, "checkpoints", "last_checkpoint.pt")


def infer_motion_label(npz_path: str) -> str:
    """Infer the semantic label from the immediate sequence directory name."""
    seq_name = Path(npz_path).parent.name.upper()
    if "BOXING" in seq_name:
        return "boxing"
    if "WALK" in seq_name:
        return "walk"
    return "unknown"


def pca_project_with_variance(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Project rows of x with NumPy SVD and return explained variance ratios."""
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Cannot run PCA on an empty array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    x_centered = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        return (
            np.zeros((1, n_components), dtype=np.float32),
            np.zeros(n_components, dtype=np.float32),
        )

    _u, singular_values, vh = np.linalg.svd(x_centered, full_matrices=False)
    n_available = min(n_components, vh.shape[0])
    projection = x_centered @ vh[:n_available].T
    if n_available < n_components:
        projection = np.pad(projection, ((0, 0), (0, n_components - n_available)))

    variances = (singular_values ** 2) / max(x.shape[0] - 1, 1)
    total_variance = variances.sum()
    ratio = np.zeros(n_components, dtype=np.float32)
    if total_variance > 0:
        ratio[:n_available] = (variances[:n_available] / total_variance).astype(np.float32)

    return projection.astype(np.float32), ratio


def pca_2d(x: np.ndarray) -> np.ndarray:
    projection, _ratio = pca_project_with_variance(x, 2)
    return projection


def pca_2d_with_variance(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return pca_project_with_variance(x, 2)


def pca_3d(x: np.ndarray) -> np.ndarray:
    projection, _ratio = pca_3d_with_variance(x)
    return projection


def pca_3d_with_variance(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return pca_project_with_variance(x, 3)


def sample_indices(num_points: int, max_points: int, seed: int) -> np.ndarray:
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)

    rng = np.random.RandomState(seed)
    indices = rng.choice(num_points, size=max_points, replace=False)
    indices.sort()
    return indices.astype(np.int64)


def _pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    squared_norms = np.sum(x * x, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (x @ x.T)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def _joint_probabilities(distances: np.ndarray, perplexity: float, tol: float = 1e-5) -> np.ndarray:
    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances)

    target_entropy = np.log(min(float(perplexity), float(n - 1)))
    conditional = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d_i = distances[i, mask]
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf

        for _ in range(50):
            probs = np.exp(-d_i * beta)
            sum_probs = max(float(probs.sum()), 1e-12)
            entropy = np.log(sum_probs) + beta * float(np.sum(d_i * probs)) / sum_probs
            entropy_diff = entropy - target_entropy
            if abs(entropy_diff) <= tol:
                break
            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0

        probs = np.exp(-d_i * beta)
        conditional[i, mask] = probs / max(float(probs.sum()), 1e-12)

    joint = (conditional + conditional.T) / (2.0 * n)
    return np.maximum(joint, 1e-12)


def tsne_2d(
    x: np.ndarray,
    perplexity: float = 30.0,
    num_iters: int = 500,
    learning_rate: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Small exact t-SNE implementation used when sklearn is unavailable."""
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Cannot run t-SNE on an empty array")
    if x.shape[0] <= 2:
        return pca_2d(x)

    rng = np.random.RandomState(seed)
    perplexity = min(float(perplexity), max(1.0, (x.shape[0] - 1.0) / 3.0))
    x_reduced, _ratio = pca_project_with_variance(x, min(30, x.shape[1]))
    distances = _pairwise_squared_distances(x_reduced)
    base_p = _joint_probabilities(distances, perplexity)
    p = base_p * 12.0

    y = rng.normal(0.0, 1e-4, size=(x.shape[0], 2)).astype(np.float64)
    gains = np.ones_like(y)
    update = np.zeros_like(y)

    for iteration in range(num_iters):
        y_distances = _pairwise_squared_distances(y)
        inv = 1.0 / (1.0 + y_distances)
        np.fill_diagonal(inv, 0.0)
        q = np.maximum(inv / max(float(inv.sum()), 1e-12), 1e-12)

        weighted = (p - q) * inv
        grad = 4.0 * ((weighted.sum(axis=1)[:, None] * y) - weighted @ y)

        momentum = 0.5 if iteration < 250 else 0.8
        gains = (gains + 0.2) * ((grad > 0.0) != (update > 0.0)) + (gains * 0.8) * (
            (grad > 0.0) == (update > 0.0)
        )
        gains = np.maximum(gains, 0.01)
        update = momentum * update - learning_rate * gains * grad
        y += update
        y -= y.mean(axis=0, keepdims=True)

        if iteration == 100:
            p = base_p

    return y.astype(np.float32)


def nearest_centroid_metrics(z_c: np.ndarray, labels: np.ndarray) -> dict:
    """Measure rough semantic separation with a nearest-centroid classifier."""
    valid_labels = [label for label in ("boxing", "walk") if np.any(labels == label)]
    if len(valid_labels) < 2:
        return {}

    centroids = {
        label: z_c[labels == label].mean(axis=0)
        for label in valid_labels
    }
    label_to_index = {label: i for i, label in enumerate(valid_labels)}
    centroid_matrix = np.stack([centroids[label] for label in valid_labels], axis=0)

    valid_mask = np.isin(labels, valid_labels)
    z_valid = z_c[valid_mask]
    labels_valid = labels[valid_mask]
    distances = ((z_valid[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=-1)
    pred_indices = distances.argmin(axis=1)
    pred_labels = np.asarray([valid_labels[i] for i in pred_indices])

    accuracy = float((pred_labels == labels_valid).mean())
    per_label_accuracy = {}
    for label in valid_labels:
        label_mask = labels_valid == label
        per_label_accuracy[label] = float((pred_labels[label_mask] == label).mean())
    balanced_accuracy = float(np.mean(list(per_label_accuracy.values())))
    centroid_distance = float(np.linalg.norm(centroids["boxing"] - centroids["walk"]))

    return {
        "labels": valid_labels,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_label_accuracy": per_label_accuracy,
        "centroid_distance": centroid_distance,
        "label_to_index": label_to_index,
    }


def split_train_paths(cfg: MotionAEConfig) -> Tuple[List[str], List[str]]:
    """Replicate build_datasets() file-level train/val split."""
    npz_paths = find_npz_files(cfg.data.data_path, cfg.data.npz_filename)
    if not npz_paths:
        raise FileNotFoundError(f"No {cfg.data.npz_filename} files found in {cfg.data.data_path}")

    rng = np.random.RandomState(cfg.training.seed)
    indices = rng.permutation(len(npz_paths))
    n_val = max(1, int(len(npz_paths) * cfg.data.val_ratio))
    val_indices = set(indices[:n_val].tolist())
    train_paths = [p for i, p in enumerate(npz_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(npz_paths) if i in val_indices]
    if not train_paths:
        raise ValueError("Training split is empty; lower data.val_ratio or provide more files")
    return train_paths, val_paths


def iter_train_window_batches(
    paths: Sequence[str],
    cfg: MotionAEConfig,
    normalizer: FeatureNormalizer,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, List[str], List[str], List[str], List[int]]]:
    windows: List[np.ndarray] = []
    labels: List[str] = []
    sequences: List[str] = []
    source_paths: List[str] = []
    starts: List[int] = []

    for path in paths:
        feats, _slices = build_features(load_npz(path), cfg.npz_keys, cfg.pelvis, debug=cfg.debug)
        label = infer_motion_label(path)
        sequence = Path(path).parent.name

        for start in range(0, feats.shape[0] - cfg.window_size + 1, cfg.stride):
            window = feats[start : start + cfg.window_size]
            windows.append(normalizer.normalize_np(window).astype(np.float32))
            labels.append(label)
            sequences.append(sequence)
            source_paths.append(path)
            starts.append(start)

            if len(windows) >= batch_size:
                yield torch.from_numpy(np.stack(windows)), labels, sequences, source_paths, starts
                windows, labels, sequences, source_paths, starts = [], [], [], [], []

    if windows:
        yield torch.from_numpy(np.stack(windows)), labels, sequences, source_paths, starts


def load_model(cfg: MotionAEConfig, checkpoint_path: str, feature_dim: int, device: torch.device) -> MotionAutoEncoder:
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
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def extract_latents(
    cfg: MotionAEConfig,
    checkpoint_path: str,
    stats_path: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    train_paths, val_paths = split_train_paths(cfg)
    normalizer = FeatureNormalizer.load(stats_path, eps=cfg.normalization.eps)

    first_feats, first_slices = build_features(load_npz(train_paths[0]), cfg.npz_keys, cfg.pelvis, debug=False)
    feature_dim = first_slices.total_dim
    if normalizer.mean.shape[0] != feature_dim:
        raise ValueError(
            f"Stats dim ({normalizer.mean.shape[0]}) does not match current feature dim ({feature_dim}). "
            "Regenerate stats or use the matching run config/checkpoint."
        )
    del first_feats

    model = load_model(cfg, checkpoint_path, feature_dim, device)

    zc_chunks: List[np.ndarray] = []
    zd_chunks: List[np.ndarray] = []
    labels: List[str] = []
    sequences: List[str] = []
    source_paths: List[str] = []
    starts: List[int] = []

    with torch.no_grad():
        for batch, batch_labels, batch_sequences, batch_paths, batch_starts in iter_train_window_batches(
            train_paths, cfg, normalizer, batch_size
        ):
            batch = batch.to(device)
            z_c = model.encoder(batch.reshape(batch.shape[0], model.flat_dim))
            _z_dequant, z_d, _info = model.quantizer(z_c)
            zc_chunks.append(z_c.cpu().numpy())
            zd_chunks.append(z_d.cpu().numpy().astype(np.int64))
            labels.extend(batch_labels)
            sequences.extend(batch_sequences)
            source_paths.extend(batch_paths)
            starts.extend(batch_starts)

    z_c_all = np.concatenate(zc_chunks, axis=0)
    z_d_all = np.concatenate(zd_chunks, axis=0)
    return {
        "z_c": z_c_all,
        "z_d": z_d_all,
        "labels": np.asarray(labels),
        "sequences": np.asarray(sequences),
        "source_paths": np.asarray(source_paths),
        "starts": np.asarray(starts, dtype=np.int64),
        "train_paths": np.asarray(train_paths),
        "val_paths": np.asarray(val_paths),
        "feature_dim": feature_dim,
    }


def _label_colors() -> dict:
    return {"boxing": "#d95f02", "walk": "#1b9e77", "unknown": "#7570b3"}


def save_2d_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    output_path: str,
    title: str,
    x_label: str,
    y_label: str,
    max_plot_points: int,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    indices = np.arange(projection.shape[0])
    if max_plot_points > 0 and len(indices) > max_plot_points:
        indices = rng.choice(indices, size=max_plot_points, replace=False)
        indices.sort()

    colors = _label_colors()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)

    ax = axes[0]
    for label in sorted(set(labels.tolist())):
        mask = labels[indices] == label
        ax.scatter(
            projection[indices][mask, 0],
            projection[indices][mask, 1],
            s=8,
            alpha=0.35,
            c=colors.get(label, "#666666"),
            label=f"{label} ({int((labels == label).sum())})",
            linewidths=0,
        )
    ax.set_title(f"{title} by train windows")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, markerscale=2)

    ax = axes[1]
    for sequence in sorted(set(sequences.tolist())):
        mask = sequences == sequence
        label = labels[mask][0]
        centroid = projection[mask].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=32,
            alpha=0.85,
            c=colors.get(label, "#666666"),
            linewidths=0,
        )
    ax.set_title(f"{title} sequence centroids")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_3d_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    output_path: str,
    title: str,
    axis_prefix: str,
    max_plot_points: int,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    indices = np.arange(projection.shape[0])
    if max_plot_points > 0 and len(indices) > max_plot_points:
        indices = rng.choice(indices, size=max_plot_points, replace=False)
        indices.sort()

    colors = _label_colors()
    fig = plt.figure(figsize=(14, 6), dpi=160)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for label in sorted(set(labels.tolist())):
        mask = labels[indices] == label
        points = projection[indices][mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=8,
            alpha=0.35,
            c=colors.get(label, "#666666"),
            label=f"{label} ({int((labels == label).sum())})",
            linewidths=0,
        )
    ax.set_title(f"{title} by train windows")
    ax.set_xlabel(f"{axis_prefix}1")
    ax.set_ylabel(f"{axis_prefix}2")
    ax.set_zlabel(f"{axis_prefix}3")
    ax.legend(frameon=False, markerscale=2)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    for sequence in sorted(set(sequences.tolist())):
        mask = sequences == sequence
        label = labels[mask][0]
        centroid = projection[mask].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=32,
            alpha=0.85,
            c=colors.get(label, "#666666"),
            linewidths=0,
        )
    ax.set_title(f"{title} sequence centroids")
    ax.set_xlabel(f"{axis_prefix}1")
    ax.set_ylabel(f"{axis_prefix}2")
    ax.set_zlabel(f"{axis_prefix}3")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_summary(
    output_path: str,
    payload: dict,
    variance_ratios: dict,
    plot_points: int,
    tsne_points: int,
    separation_metrics: dict,
    plot_paths: dict,
) -> None:
    label_counts = Counter(payload["labels"].tolist())
    sequence_counts = Counter(payload["sequences"].tolist())
    sequence_label_counts: dict[str, Counter] = defaultdict(Counter)
    for sequence, label in zip(payload["sequences"].tolist(), payload["labels"].tolist()):
        sequence_label_counts[sequence][label] += 1

    summary = {
        "num_windows": int(payload["z_c"].shape[0]),
        "feature_dim": int(payload["feature_dim"]),
        "num_train_files": int(len(payload["train_paths"])),
        "num_val_files_excluded": int(len(payload["val_paths"])),
        "label_counts": dict(sorted(label_counts.items())),
        "num_sequences": int(len(sequence_counts)),
        "plot_points": int(plot_points),
        "tsne_points": int(tsne_points),
        "latent_spaces": {
            "z_c": {
                "dim": int(payload["z_c"].shape[1]),
                "pca_2d_explained_variance_ratio": [
                    float(x) for x in variance_ratios["z_c_pca_2d"]
                ],
                "pca_3d_explained_variance_ratio": [
                    float(x) for x in variance_ratios["z_c_pca_3d"]
                ],
                "nearest_centroid": separation_metrics["z_c"],
            },
            "z_d": {
                "dim": int(payload["z_d"].shape[1]),
                "pca_2d_explained_variance_ratio": [
                    float(x) for x in variance_ratios["z_d_pca_2d"]
                ],
                "pca_3d_explained_variance_ratio": [
                    float(x) for x in variance_ratios["z_d_pca_3d"]
                ],
                "nearest_centroid": separation_metrics["z_d"],
            },
        },
        "plot_paths": plot_paths,
        "sequence_label_counts": {
            seq: dict(sorted(counts.items())) for seq, counts in sorted(sequence_label_counts.items())
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize train-set z_c/z_d latents for BOXING/WALK.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stats", type=str, default=None, help="Defaults to <run_dir>/artifacts/stats.npz")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/latent_vis_train")
    parser.add_argument("--batch_size", type=int, default=None, help="Defaults to training.batch_size")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0 / auto")
    parser.add_argument("--max_plot_points", type=int, default=20000, help="0 means plot all points")
    parser.add_argument("--max_tsne_points", type=int, default=1500, help="0 means use all points; can be slow")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iters", type=int, default=500)
    parser.add_argument("--tsne_learning_rate", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.training.device = args.device

    checkpoint_path = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    stats_path = args.stats or os.path.join(run_dir, "artifacts", cfg.normalization.stats_file)
    output_dir = args.output_dir or os.path.join(run_dir, "latent_vis_train")
    batch_size = args.batch_size or cfg.training.batch_size
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(get_device(cfg.training.device))
    payload = extract_latents(cfg, checkpoint_path, stats_path, batch_size, device)

    z_c = payload["z_c"].astype(np.float32)
    z_d = payload["z_d"].astype(np.float32)
    labels = payload["labels"]
    sequences = payload["sequences"]

    zc_pca_2d, zc_pca_2d_ratio = pca_2d_with_variance(z_c)
    zc_pca_3d, zc_pca_3d_ratio = pca_3d_with_variance(z_c)
    zd_pca_2d, zd_pca_2d_ratio = pca_2d_with_variance(z_d)
    zd_pca_3d, zd_pca_3d_ratio = pca_3d_with_variance(z_d)

    tsne_indices = sample_indices(z_c.shape[0], args.max_tsne_points, cfg.training.seed)
    zc_tsne_2d = tsne_2d(
        z_c[tsne_indices],
        perplexity=args.tsne_perplexity,
        num_iters=args.tsne_iters,
        learning_rate=args.tsne_learning_rate,
        seed=cfg.training.seed,
    )
    zd_tsne_2d = tsne_2d(
        z_d[tsne_indices],
        perplexity=args.tsne_perplexity,
        num_iters=args.tsne_iters,
        learning_rate=args.tsne_learning_rate,
        seed=cfg.training.seed,
    )

    variance_ratios = {
        "z_c_pca_2d": zc_pca_2d_ratio,
        "z_c_pca_3d": zc_pca_3d_ratio,
        "z_d_pca_2d": zd_pca_2d_ratio,
        "z_d_pca_3d": zd_pca_3d_ratio,
    }
    separation_metrics = {
        "z_c": nearest_centroid_metrics(z_c, labels),
        "z_d": nearest_centroid_metrics(z_d, labels),
    }

    npz_path = os.path.join(output_dir, "train_latents_zc_zd.npz")
    summary_path = os.path.join(output_dir, "train_latent_summary.json")
    plot_paths = {
        "z_c_pca_2d": os.path.join(output_dir, "train_z_c_pca_2d.png"),
        "z_c_pca_3d": os.path.join(output_dir, "train_z_c_pca_3d.png"),
        "z_c_tsne_2d": os.path.join(output_dir, "train_z_c_tsne_2d.png"),
        "z_d_pca_2d": os.path.join(output_dir, "train_z_d_pca_2d.png"),
        "z_d_pca_3d": os.path.join(output_dir, "train_z_d_pca_3d.png"),
        "z_d_tsne_2d": os.path.join(output_dir, "train_z_d_tsne_2d.png"),
    }

    np.savez(
        npz_path,
        z_c=payload["z_c"],
        z_d=payload["z_d"],
        z_c_pca_2d=zc_pca_2d,
        z_c_pca_3d=zc_pca_3d,
        z_c_tsne_2d=zc_tsne_2d,
        z_d_pca_2d=zd_pca_2d,
        z_d_pca_3d=zd_pca_3d,
        z_d_tsne_2d=zd_tsne_2d,
        tsne_indices=tsne_indices,
        labels=labels,
        tsne_labels=labels[tsne_indices],
        sequences=sequences,
        tsne_sequences=sequences[tsne_indices],
        source_paths=payload["source_paths"],
        starts=payload["starts"],
        train_paths=payload["train_paths"],
        val_paths_excluded=payload["val_paths"],
        z_c_pca_2d_explained_variance_ratio=zc_pca_2d_ratio,
        z_c_pca_3d_explained_variance_ratio=zc_pca_3d_ratio,
        z_d_pca_2d_explained_variance_ratio=zd_pca_2d_ratio,
        z_d_pca_3d_explained_variance_ratio=zd_pca_3d_ratio,
    )

    save_2d_plot(
        zc_pca_2d,
        labels,
        sequences,
        plot_paths["z_c_pca_2d"],
        "z_c 2D PCA",
        "PC1",
        "PC2",
        args.max_plot_points,
        cfg.training.seed,
    )
    save_3d_plot(
        zc_pca_3d,
        labels,
        sequences,
        plot_paths["z_c_pca_3d"],
        "z_c 3D PCA",
        "PC",
        args.max_plot_points,
        cfg.training.seed,
    )
    save_2d_plot(
        zc_tsne_2d,
        labels[tsne_indices],
        sequences[tsne_indices],
        plot_paths["z_c_tsne_2d"],
        "z_c t-SNE",
        "t-SNE 1",
        "t-SNE 2",
        0,
        cfg.training.seed,
    )
    save_2d_plot(
        zd_pca_2d,
        labels,
        sequences,
        plot_paths["z_d_pca_2d"],
        "z_d 2D PCA",
        "PC1",
        "PC2",
        args.max_plot_points,
        cfg.training.seed,
    )
    save_3d_plot(
        zd_pca_3d,
        labels,
        sequences,
        plot_paths["z_d_pca_3d"],
        "z_d 3D PCA",
        "PC",
        args.max_plot_points,
        cfg.training.seed,
    )
    save_2d_plot(
        zd_tsne_2d,
        labels[tsne_indices],
        sequences[tsne_indices],
        plot_paths["z_d_tsne_2d"],
        "z_d t-SNE",
        "t-SNE 1",
        "t-SNE 2",
        0,
        cfg.training.seed,
    )

    plot_points = payload["z_c"].shape[0] if args.max_plot_points <= 0 else min(args.max_plot_points, payload["z_c"].shape[0])
    write_summary(
        summary_path,
        payload,
        variance_ratios,
        plot_points,
        len(tsne_indices),
        separation_metrics,
        plot_paths,
    )

    print(f"Saved latents/projections: {npz_path}")
    for name, path in plot_paths.items():
        print(f"Saved {name} plot: {path}")
    print(f"Saved summary: {summary_path}")
    print(f"Windows: {payload['z_c'].shape[0]}, latent_dim: {payload['z_c'].shape[1]}")
    print(f"Labels: {dict(sorted(Counter(labels.tolist()).items()))}")
    print(f"t-SNE points: {len(tsne_indices)}")
    if separation_metrics["z_c"]:
        print(
            "Nearest-centroid z_c balanced accuracy: "
            f"{separation_metrics['z_c']['balanced_accuracy']:.4f}"
        )
    if separation_metrics["z_d"]:
        print(
            "Nearest-centroid z_d balanced accuracy: "
            f"{separation_metrics['z_d']['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
