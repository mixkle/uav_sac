#!/usr/bin/env bash
# Run only when explicitly invoked by the user. Each job uses ONE physical GPU.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_python="$uav_dppo_root/.venv-dppo/bin/python"
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_logs="$UAV_DATA_ROOT/logs/motion_only_sinr_topk_1m"
uav_entry="$uav_script_dir/train_motion_only_server.sh"
uav_session_prefix=motion
IFS=, read -r -a uav_gpus <<< "${UAV_MOTION_GPUS:-2,3}"
IFS='|' read -r -a uav_cpus <<< "${UAV_MOTION_CPUSETS:-12-17|18-23}"
uav_check_only=false
for uav_arg in "$@"; do
  case "$uav_arg" in
    --check-only) uav_check_only=true ;;
    --resume-10m)
      uav_entry="$uav_script_dir/resume_motion_only_10m_server.sh"
      uav_logs="$UAV_DATA_ROOT/logs/motion_only_sinr_topk_10m"
      uav_session_prefix=motion-10m ;;
    *) echo 'Usage: launch_motion_only_server.sh [--check-only] [--resume-10m]' >&2; exit 2 ;;
  esac
done
(( ${#uav_gpus[@]} == 2 && ${#uav_cpus[@]} == 2 )) || { echo 'Need exactly two GPUs and CPU sets' >&2; exit 2; }
command -v tmux >/dev/null
# All checks finish before starting either formal training process.
"$uav_python" - "${uav_gpus[@]}" "${uav_cpus[@]}" "$UAV_DATA_ROOT" <<'PY'
import csv,io,os,shutil,subprocess,sys
gpus,masks=sys.argv[1:3],sys.argv[3:5]
if any(not x.isdecimal() for x in gpus) or len(set(gpus)) != 2:
    raise ValueError('Choose two unique physical GPU IDs')
seen=set()
for mask in masks:
    cpus=set()
    for part in mask.split(','):
        bounds=list(map(int,part.split('-')))
        if len(bounds) not in (1,2) or bounds[-1]<bounds[0]:raise ValueError('Invalid affinity')
        cpus.update(range(bounds[0],bounds[-1]+1))
    if len(cpus)<4 or not cpus.issubset(os.sched_getaffinity(0)) or seen & cpus:
        raise ValueError('Choose disjoint available CPU sets with >=4 CPUs')
    seen.update(cpus)
for gpu in gpus:
    state=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    uuid,memory=next(csv.reader(io.StringIO(state),skipinitialspace=True))
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    if any(line.split(',')[0].strip()==uuid for line in processes.splitlines()) or int(memory)>512:
        raise RuntimeError(f'GPU {gpu} appears busy; no jobs started')
if shutil.disk_usage(sys.argv[5]).free < 23*2**30:
    raise OSError('Need >=23GiB free; no automatic cleanup')
PY
uav_groups=(M1 M2)
for uav_index in 0 1; do
  uav_group="${uav_groups[$uav_index]}"
  if tmux -S "$uav_socket" has-session -t "$uav_session_prefix-$uav_group" 2>/dev/null; then
    echo "Session exists: $uav_session_prefix-$uav_group" >&2; exit 1
  fi
  CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_index]}" bash "$uav_entry" check --group "$uav_group"
done
if "$uav_check_only"; then
  echo 'Both groups passed preflight; no formal training started.'
  exit 0
fi
mkdir -p "$uav_logs"
for uav_index in 0 1; do
  uav_group="${uav_groups[$uav_index]}"
  printf -v uav_command 'cd %q && CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash %q train --group %q > %q 2>&1' \
    "$uav_dppo_root" "${uav_gpus[$uav_index]}" "${uav_cpus[$uav_index]}" \
    "$uav_entry" "$uav_group" "$uav_logs/$uav_group.log"
  tmux -S "$uav_socket" new-session -d -s "$uav_session_prefix-$uav_group" "$uav_command"
  echo "Started $uav_session_prefix-$uav_group on physical GPU ${uav_gpus[$uav_index]}; log=$uav_logs/$uav_group.log"
done
