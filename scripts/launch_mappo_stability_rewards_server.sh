#!/usr/bin/env bash
# Preflight all jobs, then launch three independent single-GPU tmux sessions.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
if (( $# > 1 )) || [[ -n "${1:-}" && "${1:-}" != --check ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
IFS=',' read -r -a uav_gpus <<< "${UAV_STABILITY_GPUS:-0,1,2}"
IFS='|' read -r -a uav_cpus <<< "${UAV_STABILITY_CPUSETS:-0-7|8-15|16-23}"
uav_groups=(disconnection robustness combined)
uav_sessions=(mappo-stab-disconnection mappo-stab-robustness mappo-stab-combined)
uav_tmux_socket="$uav_dppo_root/.cache/tmux.sock"
if (( ${#uav_gpus[@]} != 3 || ${#uav_cpus[@]} != 3 )); then
  echo "Provide exactly three GPUs and three CPU sets." >&2
  exit 2
fi
"$uav_dppo_root/.venv-dppo/bin/python" - "${uav_gpus[@]}" "${uav_cpus[@]}" <<'PY'
import os
import sys
gpus, masks = sys.argv[1:4], sys.argv[4:7]
if any(not g.isdigit() for g in gpus) or len(set(gpus)) != 3:
    raise SystemExit('GPU indices must be distinct nonnegative integers.')
sets=[]
for mask in masks:
    cpus=set()
    for part in mask.split(','):
        pair=part.split('-')
        if len(pair)>2 or any(not p.isdigit() for p in pair):
            raise SystemExit('Invalid CPU affinity syntax.')
        start,end=int(pair[0]),int(pair[-1])
        if end<start:
            raise SystemExit('Invalid CPU range.')
        cpus.update(range(start,end+1))
    if len(cpus)<8 or not cpus.issubset(os.sched_getaffinity(0)):
        raise SystemExit('Allocate at least eight available CPUs per job.')
    if any(cpus & prior for prior in sets):
        raise SystemExit('CPU affinity sets must not overlap.')
    sets.append(cpus)
PY
for uav_i in 0 1 2; do
  if tmux -S "$uav_tmux_socket" has-session -t "${uav_sessions[$uav_i]}" 2>/dev/null; then
    echo "Session already exists: ${uav_sessions[$uav_i]}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_i]}" UAV_CPU_AFFINITY="${uav_cpus[$uav_i]}" \
    bash scripts/train_mappo_stability_rewards_server.sh "${uav_groups[$uav_i]}" --check
done
if [[ "${1:-}" == --check ]]; then
  echo "All three preflights passed. Training was not started."
  exit 0
fi
for uav_i in 0 1 2; do
  printf -v uav_command 'CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash scripts/train_mappo_stability_rewards_server.sh %q' \
    "${uav_gpus[$uav_i]}" "${uav_cpus[$uav_i]}" "${uav_groups[$uav_i]}"
  tmux -S "$uav_tmux_socket" new-session -d -s "${uav_sessions[$uav_i]}" -c "$uav_dppo_root" "$uav_command"
  echo "Started ${uav_sessions[$uav_i]}: GPU ${uav_gpus[$uav_i]}, CPUs ${uav_cpus[$uav_i]}"
done
echo "Attach: tmux -S $uav_tmux_socket attach -t mappo-stab-disconnection"
