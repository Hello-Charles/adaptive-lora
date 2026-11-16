#!/usr/bin/env bash
# M7正式启动器：无参数只显示命令；prepare只装配数据，start才启动完整GPU实验。
# 调用者须承担start的GPU成本。失败立即退出，不自动重试，不覆盖旧结果。
# Python始终用既有隔离venv和-I；不安装依赖，不从数据目录导入代码。
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${M7_PYTHON:-/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python}
PROFILE=${M7_PROFILE:-$ROOT/configs/m7_formal_autodl.json}
ENTRY=$ROOT/scripts/run_m7_experiment.py
case "${1:-info}" in
  info)
    printf '%s\n' '已准备的M7正式入口（当前不启动GPU）：' \
      'bash scripts/m7_formal.sh prepare   # 仅准备固定数据' \
      'bash scripts/m7_formal.sh start     # 正式N250 + 三方法3epochs + 统一评估/汇总' \
      '分阶段及断点续跑命令见 M7_PROTOCOL.md；不运行单测、N1或pilot。'
    ;;
  prepare)
    exec "$PYTHON" -I "$ENTRY" prepare --profile "$PROFILE"
    ;;
  start)
    # N250是Adaptive-LoRA算法阶段，不是测试；真实选层成功后才可训练Adaptive8。
    "$PYTHON" -I "$ENTRY" calibrate --execute --profile "$PROFILE"
    for method in full random_k adaptive_k; do
      "$PYTHON" -I "$ENTRY" train --execute --method "$method" --profile "$PROFILE"
      "$PYTHON" -I "$ENTRY" evaluate --execute --method "$method" --profile "$PROFILE"
    done
    "$PYTHON" -I "$ENTRY" summarize --profile "$PROFILE"
    ;;
  *)
    printf '%s\n' '用法：bash scripts/m7_formal.sh [info|prepare|start]' >&2
    exit 2
    ;;
esac
