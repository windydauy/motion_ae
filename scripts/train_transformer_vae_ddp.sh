#!/usr/bin/env bash
# DDP training launcher for the transformer motion VAE.
#
# Default data root:
#   /pfs/pfs-ilWc5D/yzh/g1_soma/npz_part
#
# Examples:
#   NPROC_PER_NODE=8 ./scripts/train_transformer_vae_encoder_ddp.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 BATCH_SIZE=128 ./scripts/train_transformer_vae_encoder_ddp.sh
#   LOGGER=none MAX_STEPS=1000 ./scripts/train_transformer_vae_encoder_ddp.sh

set -euo pipefail
export WANDB_API_KEY=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
# export WANDB_MODE=offline
export WANDB_USERNAME=yzh_academic-shanghai-jiao-tong-university
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6}"

NPROC="${NPROC_PER_NODE:-6}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

DATA_PATH="${DATA_PATH:-/pfs/pfs-ilWc5D/yzh/g1_soma/npz}"
CONFIG="${CONFIG:-configs/transformer_vae.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-transformer_vae_encoder_ddp}"
RUN_NAME="${RUN_NAME:-npz_all_ddp}"

BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_MODE="${BATCH_SIZE_MODE:-per_rank}"
MAX_STEPS="${MAX_STEPS:-1000000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LOGGER="${LOGGER:-wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"

exec torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train_transformer_vae_ddp.py \
  --config "${CONFIG}" \
  --data_path "${DATA_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --run_name "${RUN_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_mode "${BATCH_SIZE_MODE}" \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --num_workers "${NUM_WORKERS}" \
  --device cuda \
  --ddp_backend "${DDP_BACKEND}" \
  --logger "${LOGGER}" \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
