#!/usr/bin/env bash
# Run manually inside tmux. No formal training is launched during deployment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/server_env.sh
output="${UAV_RUN_OUTPUT:-$UAV_DATA_ROOT/runs/sac_coverage_time_factor_relative_central_history_1m_seed0}"
mkdir -p "$UAV_DATA_ROOT/logs"
python -u -m uav_research train \
  --config configs/sac_coverage_time_factor_relative_central_history_server_1m.yaml \
  --output "$output" "$@" \
  2>&1 | tee -a "$UAV_DATA_ROOT/logs/sac_coverage_1m_console.log"
