#!/usr/bin/env bash
# Single-GPU training entrypoint for the transformer motion VAE.
#
# Example:
#   ./scripts/train_transformer_vae_1gpu.sh
#
# Common overrides:
#   GPU=1 BATCH_SIZE=128 RUN_NAME=try_128 ./scripts/train_transformer_vae_1gpu.sh
#   LOGGER=none MAX_STEPS=1000 ./scripts/train_transformer_vae_1gpu.sh

set -euo pipefail
export WANDB_API_KEY=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
# export WANDB_MODE=offline
export WANDB_USERNAME=yzh_academic-shanghai-jiao-tong-university
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

DATA_PATH="${DATA_PATH:-/pfs/pfs-ilWc5D/yzh/g1_soma/npz_part}"
CONFIG="${CONFIG:-configs/transformer_vae.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-transformer_vae}"
RUN_NAME="${RUN_NAME:-1gpu_npz_part}"

BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGER="${LOGGER:-wandb}"
WANDB_MODE="${WANDB_MODE:-online}"

exec "${PYTHON:-python}" -m transformer_vae.scripts.train \
  --config "${CONFIG}" \
  --data_path "${DATA_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --run_name "${RUN_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --num_workers "${NUM_WORKERS}" \
  --device cuda \
  --logger "${LOGGER}" \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
