#!/usr/bin/env bash
set -euo pipefail
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
score_worker="${1:?worker R0/R1/R2/R3 required}"
score_root="$UAV_DATA_ROOT/analysis/score_rule_s1_s2_20261116"
mkdir -p "$score_root/audit"
trap 'score_exit=$?; echo "$score_exit" > "$score_root/audit/worker_${score_worker}.exit"' EXIT
.venv-dppo/bin/python -c 'from uav_research.score_rule_experiment import locked; from uav_research.runtime import resolve_output; assert not locked(resolve_output("analysis/score_rule_s1_s2_20261116"))["smoke"]'
for score_checkpoint in best last; do
  for score_seed in 880001 880002 880003; do
    if [[ "$score_worker" != R0 ]]; then
      .venv-dppo/bin/python -u scripts/diagnose_score_rules.py closed --device cuda:0 \
        --checkpoint "$score_checkpoint" --rule "$score_worker" --action-seed "$score_seed" --split development
    fi
    .venv-dppo/bin/python -u scripts/diagnose_score_rules.py closed --device cuda:0 \
      --checkpoint "$score_checkpoint" --rule "$score_worker" --action-seed "$score_seed" --split confirmation
  done
done
if [[ "$score_worker" == R0 ]]; then
  for score_checkpoint in best last; do
    for score_seed in 880001 880002 880003; do
      .venv-dppo/bin/python -u scripts/diagnose_score_rules.py s1 --device cuda:0 \
        --checkpoint "$score_checkpoint" --action-seed "$score_seed"
    done
  done
fi
