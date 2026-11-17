#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uav_python="$uav_dppo_root/.venv-dppo/bin/python"
uav_affinity=()
if [[ -n "${UAV_CPU_AFFINITY:-}" ]];then uav_affinity=(taskset -c "$UAV_CPU_AFFINITY");fi
exec "${uav_affinity[@]}" "$uav_python" -u scripts/system_ablation.py "$@" --device cuda:0
