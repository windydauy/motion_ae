# Compare Overfit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the fixed-batch debug script to compare `plain AE` and `iFSQ AE` on the same batch and save both loss histories to one JSON file.

**Architecture:** Reuse the current data loading and fixed-batch sampling path from `scripts/debug_plain_ae_overfit.py`. Introduce model-agnostic helpers that run a fixed-batch overfit loop for either model, then collect and save labeled metrics for `plain_ae` and `ifsq_ae`.

**Tech Stack:** Python, PyTorch, argparse, JSON, pytest

---

### Task 1: Add a failing comparison test

**Files:**
- Modify: `tests/test_plain_ae_overfit.py`
- Test: `tests/test_plain_ae_overfit.py`

**Step 1: Write the failing test**

Add a test that:
- builds the same random batch once
- runs a compare helper that trains `plain AE` and `iFSQ AE`
- asserts the result contains `plain_ae` and `ifsq_ae`
- asserts both branches have `history` with the requested step count

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_plain_ae_overfit.py`
Expected: FAIL because the comparison helper does not exist yet.

### Task 2: Generalize the debug script

**Files:**
- Modify: `scripts/debug_plain_ae_overfit.py`
- Modify: `motion_ae/models/plain_autoencoder.py`
- Test: `tests/test_plain_ae_overfit.py`

**Step 1: Write minimal implementation**

- Add model-agnostic helpers for:
  - fixed batch selection
  - overfit loop
  - summary generation
  - JSON payload generation and saving
- Instantiate both `PlainMotionAutoEncoder` and `MotionAutoEncoder`
- Run both branches on the same batch

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_plain_ae_overfit.py`
Expected: PASS

### Task 3: Verify CLI behavior with real data

**Files:**
- Modify: `scripts/debug_plain_ae_overfit.py`

**Step 1: Run the script**

Run:
`PYTHONPATH=. python scripts/debug_plain_ae_overfit.py --config configs/default.yaml --steps 10 --log_every 5 --batch_size 64 --num_workers 0 --device cpu`

Expected:
- terminal shows `plain-ae` and `ifsq-ae`
- a JSON history file is written
- both branches report first/last/min loss

### Task 4: Final verification

**Files:**
- Test: `tests/test_plain_ae_overfit.py`
- Test: `scripts/debug_plain_ae_overfit.py`

**Step 1: Re-run focused test**

Run: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_plain_ae_overfit.py`
Expected: PASS

**Step 2: Re-run real script smoke test**

Run:
`PYTHONPATH=. python scripts/debug_plain_ae_overfit.py --config configs/default.yaml --steps 10 --log_every 5 --batch_size 64 --num_workers 0 --device cpu`

Expected:
- both models execute
- JSON payload is readable
- output is stable and labeled
