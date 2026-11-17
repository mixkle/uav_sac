#!/usr/bin/env bash
set -euo pipefail
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
score_root="$UAV_DATA_ROOT/analysis/score_rule_s1_s2_20261116"
trap 'score_final_status=$?; echo "$score_final_status" > "$score_root/audit/finish.exit"' EXIT
while true; do
  score_complete=1
  for score_ck in best last; do
    for score_seed in 880001 880002 880003; do
      [[ -f "$score_root/s1/$score_ck/$score_seed/complete.json" ]] || score_complete=0
    done
  done
  for score_rule in R1 R2 R3; do
    if [[ -f "$score_root/audit/worker_${score_rule}.exit" ]]; then
      [[ "$(cat "$score_root/audit/worker_${score_rule}.exit")" == 0 ]] || exit 1
    else
      score_complete=0
    fi
  done
  [[ "$score_complete" == 1 ]] && break
  sleep 10
done
.venv-dppo/bin/python -u scripts/diagnose_score_rules.py verify
.venv-dppo/bin/python -u scripts/diagnose_score_rules.py report
