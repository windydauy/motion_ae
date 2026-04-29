#!/usr/bin/env bash
set -euo pipefail

# 在 val 集上评估 AE 重建误差（默认使用你当前这次 run 的配置和 checkpoint）
#
# 用法：
#   bash scripts/eval_val_recon.sh
#   bash scripts/eval_val_recon.sh <config_path> <checkpoint_path> <run_eval_name>
#
# 示例：
#   bash scripts/eval_val_recon.sh \
#     /home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/2026-04-16_19-50-25_ifsq_no_ln_stride_5_silu_BOXING_WALK/params/config.yaml \
#     /home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/2026-04-16_19-50-25_ifsq_no_ln_stride_5_silu_BOXING_WALK/checkpoints/last_checkpoint.pt \
#     val_recon

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-/home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/2026-04-16_19-50-25_ifsq_no_ln_stride_5_silu_BOXING_WALK/params/config.yaml}"
CKPT_PATH="${2:-/home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/2026-04-16_19-50-25_ifsq_no_ln_stride_5_silu_BOXING_WALK/checkpoints/checkpoint_epoch1499.pt}"
RUN_EVAL_NAME="${3:-val_recon_1499}"

PYTHONPATH=. python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --split val \
  --run_eval_name "${RUN_EVAL_NAME}" \
  --logger disabled \
  --wandb_mode disabled \
  --num_workers 0
