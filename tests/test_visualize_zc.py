from __future__ import annotations

import numpy as np

from scripts.visualize_zc import (
    infer_motion_label,
    nearest_centroid_metrics,
    pca_2d,
    pca_3d,
    sample_indices,
    tsne_2d,
)


def test_infer_motion_label_uses_sequence_directory():
    root = "/data/BOXING_WALK"

    assert infer_motion_label(f"{root}/BOXING13_Skeleton 004_z_up_x_forward_gym_0/motion.npz") == "boxing"
    assert infer_motion_label(f"{root}/TEST_WALK1_Skeleton 005_z_up_x_forward_gym_0/motion.npz") == "walk"
    assert infer_motion_label(f"{root}/SPIN1_Skeleton 005_z_up_x_forward_gym_0/motion.npz") == "unknown"


def test_pca_2d_returns_centered_projection():
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    projected = pca_2d(x)

    assert projected.shape == (4, 2)
    np.testing.assert_allclose(projected.mean(axis=0), np.zeros(2), atol=1e-6)


def test_pca_3d_returns_centered_projection():
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    projected = pca_3d(x)

    assert projected.shape == (4, 3)
    np.testing.assert_allclose(projected.mean(axis=0), np.zeros(3), atol=1e-6)


def test_sample_indices_is_deterministic_and_sorted():
    first = sample_indices(num_points=20, max_points=5, seed=42)
    second = sample_indices(num_points=20, max_points=5, seed=42)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (5,)
    assert np.all(first[:-1] <= first[1:])


def test_tsne_2d_returns_finite_projection():
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.5, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    projected = tsne_2d(x, perplexity=2.0, num_iters=25, seed=0)

    assert projected.shape == (6, 2)
    assert np.isfinite(projected).all()


def test_nearest_centroid_metrics_reports_balanced_accuracy():
    z_c = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(["boxing", "boxing", "walk", "walk"])

    metrics = nearest_centroid_metrics(z_c, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
