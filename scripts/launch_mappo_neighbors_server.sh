#!/usr/bin/env bash
# Independent tmux jobs; preserves the original 4/5 defaults.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
if (( $# > 1 )) || [[ -n "${1:-}" && "${1:-}" != --check ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
IFS=',' read -r -a uav_gpus <<< "${UAV_NEIGHBOR_GPUS:-0,2}"
IFS='|' read -r -a uav_cpus <<< "${UAV_NEIGHBOR_CPUSETS:-0-7|12-19}"
IFS=',' read -r -a uav_slots <<< "${UAV_NEIGHBOR_SLOTS:-4,5}"
uav_sessions=()
for uav_slot in "${uav_slots[@]}"; do
  case "$uav_slot" in
    4|5|6|7|8) uav_sessions+=("mappo-neighbors$uav_slot") ;;
    *) echo "Neighbor counts must be from 4 to 8." >&2; exit 2 ;;
  esac
done
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_job_count="${#uav_slots[@]}"
if (( uav_job_count == 0 || ${#uav_gpus[@]} != uav_job_count || ${#uav_cpus[@]} != uav_job_count )); then
  echo "Provide one GPU and one CPU set per neighbor-count job." >&2
  exit 2
fi
"$uav_dppo_root/.venv-dppo/bin/python" - "$uav_job_count" "${uav_slots[@]}" "${uav_gpus[@]}" "${uav_cpus[@]}" <<'PY'
import os
import sys
n=int(sys.argv[1]);slots=sys.argv[2:2+n]
gpus,masks=sys.argv[2+n:2+2*n],sys.argv[2+2*n:2+3*n]
if len(set(slots))!=n:
    raise SystemExit('Neighbor-count jobs must be distinct.')
if any(not x.isdigit() for x in gpus) or len(set(gpus))!=n:
    raise SystemExit('Distinct nonnegative GPU indices are required.')
sets=[]
for mask in masks:
    cpus=set()
    for part in mask.split(','):
        pair=part.split('-')
        if len(pair)>2 or any(not x.isdigit() for x in pair):
            raise SystemExit('Invalid CPU affinity syntax.')
        a,b=int(pair[0]),int(pair[-1])
        if b<a:
            raise SystemExit('Invalid CPU range.')
        cpus.update(range(a,b+1))
    if len(cpus)<8 or not cpus.issubset(os.sched_getaffinity(0)):
        raise SystemExit('Allocate at least eight available CPUs per job.')
    if any(cpus & other for other in sets):
        raise SystemExit('CPU sets must not overlap.')
    sets.append(cpus)
PY
for uav_i in "${!uav_slots[@]}"; do
  if tmux -S "$uav_socket" has-session -t "${uav_sessions[$uav_i]}" 2>/dev/null; then
    echo "Session already exists: ${uav_sessions[$uav_i]}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_i]}" UAV_CPU_AFFINITY="${uav_cpus[$uav_i]}" \
    bash scripts/train_mappo_neighbors_server.sh "${uav_slots[$uav_i]}" --check
done
if [[ "${1:-}" == --check ]]; then
  echo "All $uav_job_count preflights passed. Training was not started."
  exit 0
fi
for uav_i in "${!uav_slots[@]}"; do
  printf -v uav_command 'CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash scripts/train_mappo_neighbors_server.sh %q' \
    "${uav_gpus[$uav_i]}" "${uav_cpus[$uav_i]}" "${uav_slots[$uav_i]}"
  tmux -S "$uav_socket" new-session -d -s "${uav_sessions[$uav_i]}" -c "$uav_dppo_root" "$uav_command"
  echo "Started ${uav_sessions[$uav_i]}: GPU ${uav_gpus[$uav_i]}, CPUs ${uav_cpus[$uav_i]}"
done
echo "Attach: tmux -S $uav_socket attach -t ${uav_sessions[0]}"
