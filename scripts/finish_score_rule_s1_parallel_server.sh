#!/usr/bin/env bash
set -euo pipefail
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
score_after_worker="${1:?completed worker name}"
score_finish_seed="${2:?S1 action seed}"
score_dir="$UAV_DATA_ROOT/analysis/score_rule_s1_s2_20261116/audit"
while [[ ! -f "$score_dir/worker_${score_after_worker}.exit" ]]; do sleep 10; done
[[ "$(cat "$score_dir/worker_${score_after_worker}.exit")" == 0 ]]
.venv-dppo/bin/python -u scripts/diagnose_score_rules.py s1 --checkpoint last \
  --action-seed "$score_finish_seed" --device cuda:0
