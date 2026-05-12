# MuJoCo Reconstruction Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI viewer that loads either an AE or Transformer VAE checkpoint, reconstructs a given `motion.npz`, and visualizes the original wireframe pose with the reconstructed solid robot in one MuJoCo window.

**Architecture:** Add a small reusable conversion module for qpos construction and indexing, plus a CLI script for model loading, reconstruction, and viewer control. MuJoCo imports stay inside the viewer path so non-MuJoCo unit tests keep running in environments without the package.

**Tech Stack:** Python, NumPy, PyTorch, existing `motion_ae`/`transformer_vae` modules, optional `mujoco` runtime viewer.

---

### Task 1: QPos Conversion Helpers

**Files:**
- Create: `motion_ae/mujoco_recon.py`
- Test: `tests/test_mujoco_recon.py`

**Step 1: Write failing tests**

Create tests for:
- `extract_pelvis_root_trajectory` returns pelvis position and normalized quaternion.
- `features_to_qpos_windows` builds `[N, W, 7 + J]` qpos with original root trajectory and feature `joint_pos`.
- `clamp_window_state` wraps frame/window indices within valid bounds.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mujoco_recon.py -q`

Expected: FAIL because `motion_ae.mujoco_recon` does not exist.

**Step 3: Implement helpers**

Implement pure NumPy helpers:
- `extract_pelvis_root_trajectory(npz_data, keys, pelvis_cfg)`
- `features_to_qpos_windows(features, slices, root_pos, root_quat, stride, qpos_dim=None)`
- `clamp_window_state(window_idx, frame_idx, num_windows, window_size)`
- `read_fps(npz_data, key, default=30.0)`

**Step 4: Run tests**

Run: `python -m pytest tests/test_mujoco_recon.py -q`

Expected: PASS.

### Task 2: Reconstruction Loading Helpers

**Files:**
- Create: `scripts/visualize_recon_mujoco.py`
- Test: `tests/test_visualize_recon_mujoco.py`

**Step 1: Write failing tests**

Create tests for:
- Parser accepts `--model_type ae`.
- Parser accepts `--model_type vae`.
- `default_mujoco_xml()` points at the existing TextOpTracker G1 XML when present.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualize_recon_mujoco.py -q`

Expected: FAIL because the script module does not exist.

**Step 3: Implement CLI and non-viewer helpers**

Implement:
- Argument parser with `--model_type {ae,vae}`, `--checkpoint`, `--npz_path`, `--config`, `--stats`, `--vae_config`, `--mujoco_xml`, `--device`, `--batch_size`, `--window_index`.
- AE model loader using `MotionAutoEncoder`.
- VAE model loader using `transformer_vae.scripts.common.build_model`.
- Reconstruction function returning original/reconstructed denormalized feature windows.

**Step 4: Run tests**

Run: `python -m pytest tests/test_visualize_recon_mujoco.py tests/test_mujoco_recon.py -q`

Expected: PASS.

### Task 3: MuJoCo Viewer

**Files:**
- Modify: `scripts/visualize_recon_mujoco.py`

**Step 1: Add viewer implementation**

Implement lazy `mujoco` imports and:
- `set_qpos(data, qpos)`
- `add_wireframe_pose(model, data, scene, rgba, width)`
- `run_viewer(original_qpos, reconstructed_qpos, fps, xml_path, start_window)`

**Step 2: Manual import check**

Run: `python -m scripts.visualize_recon_mujoco --help`

Expected: CLI help prints without importing `mujoco`.

**Step 3: Run unit tests**

Run: `python -m pytest tests/test_mujoco_recon.py tests/test_visualize_recon_mujoco.py -q`

Expected: PASS.

### Task 4: Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add usage docs**

Document AE and VAE commands and key controls.

**Step 2: Run focused tests**

Run: `python -m pytest tests/test_mujoco_recon.py tests/test_visualize_recon_mujoco.py -q`

Expected: PASS.

### Task 5: Final Verification

**Files:**
- No code changes unless failures are found.

**Step 1: Run smoke tests**

Run: `python -m pytest tests/test_mujoco_recon.py tests/test_visualize_recon_mujoco.py tests/test_transformer_vae.py tests/test_model_shapes.py -q`

Expected: PASS.

**Step 2: Check git status**

Run: `git status --short`

Expected: Only intended files plus any pre-existing user changes.
