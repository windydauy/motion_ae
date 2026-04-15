#!/usr/bin/env bash
# 评估入口：在项目根目录下调用 scripts/evaluate.py
#
# 用法示例（checkpoint 在实验 run 目录下时需提供 run_name）：
#   ./scripts/eval.sh --config configs/default.yaml \
#     --experiment_name motion_ae \
#     --run_name 2026-04-14_12-00-00_demo \
#     --checkpoint best_model.pt \
#     --split val
#
# 若 checkpoint 为绝对路径，可省略 --run_name：
#   ./scripts/eval.sh --config configs/default.yaml --checkpoint /path/to/best_model.pt
#
# 可选环境变量：
#   PYTHON   解释器，默认 python
#   MOTION_AE_ROOT  若设置则覆盖自动推断的项目根目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOTION_AE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"

exec "${PYTHON:-python}" scripts/evaluate.py "$@"
