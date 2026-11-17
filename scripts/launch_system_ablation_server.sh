#!/usr/bin/env bash
# Explicit run list, one job per physical GPU. No automatic formal launch.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
(( $# > 0 && $# <= 4 )) || { echo 'Usage: UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_system_ablation_server.sh G0 A1 D3 D4' >&2;exit 2; }
IFS=, read -r -a uav_gpus <<< "${UAV_ABLATION_GPUS:-0,1,2,3}"
(( ${#uav_gpus[@]} >= $# )) || { echo 'Not enough selected GPUs' >&2;exit 2; }
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_logs="$UAV_DATA_ROOT/logs/system_ablation_1m"
mkdir -p "$uav_logs"
uav_groups=("$@")
IFS="|" read -r -a uav_cpus <<< "${UAV_ABLATION_CPUSETS:-0-5|6-11|12-17|18-23}"
(( ${#uav_cpus[@]} >= $# )) || { echo 'Not enough CPU affinity sets' >&2;exit 2; }
# Fail the entire launch preflight before starting even the first process.
"$uav_dppo_root/.venv-dppo/bin/python" - "${uav_gpus[@]:0:$#}" <<'PYGPU'
import sys
x=sys.argv[1:]
if any(not s.isdecimal() for s in x) or len(set(x))!=len(x):raise ValueError('Unique physical GPU IDs required')
PYGPU
"$uav_dppo_root/.venv-dppo/bin/python" - "${uav_groups[@]}" <<'PYGROUP'
import sys
if len(set(sys.argv[1:]))!=len(sys.argv)-1:raise ValueError('Duplicate group IDs')
PYGROUP
"$uav_dppo_root/.venv-dppo/bin/python" - "${uav_cpus[@]:0:$#}" <<'PYCPU'
import os,sys
seen=set()
for mask in sys.argv[1:]:
    cpus=set()
    for part in mask.split(','):
        bounds=list(map(int,part.split('-')))
        if len(bounds) not in (1,2) or bounds[-1]<bounds[0]:raise ValueError('Invalid CPU affinity')
        cpus.update(range(bounds[0],bounds[-1]+1))
    if len(cpus)<4 or not cpus.issubset(os.sched_getaffinity(0)) or cpus&seen:raise ValueError('Need disjoint sets of >=4 available CPUs')
    seen.update(cpus)
PYCPU
for uav_index in "${!uav_groups[@]}";do
 uav_group="${uav_groups[$uav_index]}"
 [[ "$uav_group" =~ ^(G0|A[1345]|B[2345]|C[234]|D[234]|E[23]|F[12]a)$ ]] || { echo "Use canonical ID: $uav_group" >&2;exit 2; }
 if tmux -S "$uav_socket" has-session -t "abl-$uav_group" 2>/dev/null;then echo "Session exists: abl-$uav_group" >&2;exit 1;fi
 CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_index]}" bash scripts/train_system_ablation_server.sh check --group "$uav_group"
 nvidia-smi -i "${uav_gpus[$uav_index]}" --query-gpu=index,name,memory.used --format=csv,noheader
 # Four independent jobs need output space in addition to the reserved free space.
done
"$uav_dppo_root/.venv-dppo/bin/python" - "$UAV_DATA_ROOT" <<'PYDISK'
import shutil,sys
if shutil.disk_usage(sys.argv[1]).free < 23*2**30:raise OSError('Need >=23GiB free before multi-job launch')
PYDISK
for uav_index in "${!uav_groups[@]}";do
 uav_group="${uav_groups[$uav_index]}"
 printf -v uav_command 'cd %q && CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash %q train --group %q > %q 2>&1' \
 "$uav_dppo_root" "${uav_gpus[$uav_index]}" "${uav_cpus[$uav_index]}" "$uav_script_dir/train_system_ablation_server.sh" "$uav_group" "$uav_logs/$uav_group.log"
 tmux -S "$uav_socket" new-session -d -s "abl-$uav_group" "$uav_command"
 echo "Started abl-$uav_group on GPU ${uav_gpus[$uav_index]}; log=$uav_logs/$uav_group.log"
done
