#!/usr/bin/env bash
# Manual launch: all four preflights finish before any formal training starts.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_python="$uav_dppo_root/.venv-dppo/bin/python"
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_logs="$UAV_DATA_ROOT/logs/m2_motion_neighbor_controls_10m"
uav_entry="$uav_script_dir/train_m2_controls_server.sh"
uav_check_only=false
uav_resume=()
for uav_arg in "$@"; do
  case "$uav_arg" in
    --check-only) uav_check_only=true ;;
    --auto-resume) uav_resume=(--auto-resume) ;;
    *) echo 'Usage: launch_m2_controls_server.sh [--check-only] [--auto-resume]' >&2; exit 2 ;;
  esac
done
IFS=, read -r -a uav_gpus <<< "${UAV_M2_CONTROL_GPUS:-0,1,2,3}"
IFS='|' read -r -a uav_cpus <<< "${UAV_M2_CONTROL_CPUSETS:-0-5|6-11|12-17|18-23}"
(( ${#uav_gpus[@]} == 4 && ${#uav_cpus[@]} == 4 )) || { echo 'Need four GPUs and four CPU sets' >&2; exit 2; }
command -v tmux >/dev/null
"$uav_python" - "${uav_gpus[@]}" "${uav_cpus[@]}" "$UAV_DATA_ROOT" <<'PY'
import csv,io,os,shutil,subprocess,sys
gpus,masks=sys.argv[1:5],sys.argv[5:9]
if any(not x.isdecimal() for x in gpus) or len(set(gpus))!=4:
    raise ValueError('Choose four unique physical GPU IDs')
seen=set()
for mask in masks:
    cpus=set()
    for part in mask.split(','):
        bounds=list(map(int,part.split('-')))
        if len(bounds) not in (1,2) or bounds[-1]<bounds[0]:raise ValueError('Invalid affinity')
        cpus.update(range(bounds[0],bounds[-1]+1))
    if len(cpus)<4 or not cpus.issubset(os.sched_getaffinity(0)) or seen & cpus:
        raise ValueError('Choose disjoint available CPU sets, >=4 CPUs each')
    seen.update(cpus)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'],text=True)
for gpu in gpus:
    state=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    uuid,memory=next(csv.reader(io.StringIO(state),skipinitialspace=True))
    if any(line.split(',')[0].strip()==uuid for line in processes.splitlines()) or int(memory)>512:
        raise RuntimeError(f'GPU {gpu} appears busy; no training started')
if shutil.disk_usage(sys.argv[9]).free<31*2**30:
    raise OSError('Need >=31GiB free including reserve; no automatic cleanup')
PY
uav_groups=(M2-NoHop M2-N5 M2-N7 M2-N9)
for uav_index in 0 1 2 3; do
  uav_group="${uav_groups[$uav_index]}"
  if tmux -S "$uav_socket" has-session -t "m2ctrl-$uav_group" 2>/dev/null; then
    echo "Session already exists: m2ctrl-$uav_group" >&2; exit 1
  fi
  CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_index]}" UAV_CPU_AFFINITY="${uav_cpus[$uav_index]}" \
    bash "$uav_entry" check --group "$uav_group" "${uav_resume[@]}"
done
if "$uav_check_only"; then
  echo 'All four groups passed; NO formal training started.'
  exit 0
fi
mkdir -p "$uav_logs"
for uav_index in 0 1 2 3; do
  uav_group="${uav_groups[$uav_index]}"
  printf -v uav_command 'cd %q && CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash %q train --group %q' \
    "$uav_dppo_root" "${uav_gpus[$uav_index]}" "${uav_cpus[$uav_index]}" "$uav_entry" "$uav_group"
  if (( ${#uav_resume[@]} )); then uav_command+=' --auto-resume'; fi
  printf -v uav_redirect ' >> %q 2>&1' "$uav_logs/$uav_group.log"
  tmux -S "$uav_socket" new-session -d -s "m2ctrl-$uav_group" "$uav_command$uav_redirect"
  echo "Started m2ctrl-$uav_group on physical GPU ${uav_gpus[$uav_index]}; log=$uav_logs/$uav_group.log"
done
