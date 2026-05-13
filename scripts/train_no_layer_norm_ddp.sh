#!/usr/bin/env bash
# 在物理 GPU 2/5/6/7 上用 DDP 训练 no_layer_norm 配置。
#
# 用法：
#   ./scripts/train_no_layer_norm_ddp_2_5_6_7.sh
#   ./scripts/train_no_layer_norm_ddp_2_5_6_7.sh --run_name my_run --logger none
#
# 额外参数会原样透传给 scripts/train_ddp.py，可用于覆盖 run_name、batch_size 等配置。

set -euo pipefail

export WANDB_API_KEY=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
# export WANDB_MODE=offline
export WANDB_USERNAME=yzh_academic-shanghai-jiao-tong-university
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# 默认端口避开 29500（torchrun 常见默认）等易冲突端口；可用环境变量 MASTER_PORT 覆盖。
export MASTER_PORT="${MASTER_PORT:-29688}"

exec "${SCRIPT_DIR}/train_ddp.sh" \
  --config "${ROOT}/configs/no_layer_norm.yaml" \
  --run_name "${RUN_NAME:-no_layer_norm_ddp_gpus_0_1_2_3_4_6_7}" \
  "$@"
