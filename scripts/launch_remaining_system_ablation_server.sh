#!/usr/bin/env bash
# Starts only the scheduler; each worker owns one GPU and takes the next free job.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_session=abl-remaining
uav_log="$UAV_DATA_ROOT/logs/system_ablation_1m/queue.log"
mkdir -p "$(dirname -- "$uav_log")"
if tmux -S "$uav_socket" has-session -t "$uav_session" 2>/dev/null; then
  echo "Queue session already exists: $uav_session" >&2
  exit 1
fi
# Check completion and existing-output safety before detaching.
"$uav_dppo_root/.venv-dppo/bin/python" scripts/queue_system_ablation.py --dry-run "$@"
printf -v uav_command 'cd %q && source scripts/dppo_server_env.sh && UAV_ABLATION_GPUS=%q UAV_ABLATION_CPUSETS=%q %q -u scripts/queue_system_ablation.py' \
  "$uav_dppo_root" "${UAV_ABLATION_GPUS:-0,1,2,3}" "${UAV_ABLATION_CPUSETS:-0-5|6-11|12-17|18-23}" "$uav_dppo_root/.venv-dppo/bin/python"
for uav_arg in "$@"; do printf -v uav_command '%s %q' "$uav_command" "$uav_arg"; done
printf -v uav_command '%s >> %q 2>&1' "$uav_command" "$uav_log"
tmux -S "$uav_socket" new-session -d -s "$uav_session" "$uav_command"
echo "Queue started: $uav_session; log=$uav_log"
