#!/usr/bin/env bash
# Wait for the frozen evaluation workers, then audit and rebuild saved results.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/dppo_server_env.sh
out="$UAV_DATA_ROOT/analysis/t1_t2_t3_post_motion_association"
while true; do
  ready=1
  for mode in T1 T2 T3; do
    [[ -f "$out/logs/$mode.exit" ]] || ready=0
  done
  [[ "$ready" == 1 ]] && break
  sleep 10
done
for mode in T1 T2 T3; do
  if [[ "$(cat "$out/logs/$mode.exit")" != 0 ]]; then
    echo "Failed evaluation: $mode; inspect $out/logs/$mode.log" >&2
    exit 1
  fi
done
.venv-dppo/bin/python -u scripts/verify_association_timing.py --output "$out"
.venv-dppo/bin/python -u scripts/diagnose_association_timing.py --rebuild --output "$out"
.venv-dppo/bin/python -u scripts/summarize_association_timing.py
echo 'Acceptance and saved-data report complete.'
