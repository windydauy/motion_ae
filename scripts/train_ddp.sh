#!/usr/bin/env bash
# DDP 训练入口：每个 rank 加载自己的 dataset shard。
#
# 用法：
#   NPROC_PER_NODE=8 ./scripts/train_ddp.sh --config configs/no_layer_norm.yaml \
#     --experiment_name motion_ae --run_name all_soma_ddp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

NPROC="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train_ddp.py --distributed "$@"
