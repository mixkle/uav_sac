#!/usr/bin/env bash
# Frozen-policy evaluation only: three protocols, no trainer or optimizer.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/dppo_server_env.sh
out="$UAV_DATA_ROOT/analysis/t1_t2_t3_post_motion_association"
mkdir -p "$out/logs" "$PWD/.cache"
socket="$PWD/.cache/tmux.sock"
for protocol in T1 T2 T3; do
  session="association_timing_${protocol}"
  if tmux -S "$socket" has-session -t "$session" 2>/dev/null; then
    echo "Already running: $session" >&2
    exit 1
  fi
done
gpu=0
for protocol in T1 T2 T3; do
  session="association_timing_${protocol}"
  tmux -S "$socket" new-session -d -s "$session" \
    "cd /home/dym/uav_sac; source scripts/dppo_server_env.sh; CUDA_VISIBLE_DEVICES=$gpu .venv-dppo/bin/python -u scripts/diagnose_association_timing.py --protocol $protocol --device cuda:0 --output $out >$out/logs/$protocol.log 2>&1; status=\$?; echo \$status >$out/logs/$protocol.exit; exit \$status"
  echo "$session GPU=$gpu log=$out/logs/$protocol.log"
  gpu=$((gpu+1))
done
