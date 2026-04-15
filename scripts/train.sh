#!/usr/bin/env bash
# 训练入口：在项目根目录下调用 scripts/train.py
#
# 用法示例（先 conda activate text_tracker）：
#   ./scripts/train.sh --config configs/default.yaml --experiment_name motion_ae --run_name demo
#
# 恢复训练：
#   ./scripts/train.sh --config configs/default.yaml --resume --load_run 2026-04-14_12-00-00_demo --checkpoint last_checkpoint.pt
#
# 可选环境变量：
#   PYTHON   解释器，默认 python（建议在已激活的 conda 环境中运行）
#   MOTION_AE_ROOT  若设置则覆盖自动推断的项目根目录

set -euo pipefail
export WANDB_API_KEY=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
# export WANDB_MODE=offline
export WANDB_USERNAME=yzh_academic-shanghai-jiao-tong-university
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university  
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

exec "${PYTHON:-python}" scripts/train.py "$@"
