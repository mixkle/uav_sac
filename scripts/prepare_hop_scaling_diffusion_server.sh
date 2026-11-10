#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CUDA_VISIBLE_DEVICES="${UAV_PREP_GPU:-0}"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_config="$uav_dppo_root/configs/diffusion_dppo_hop_scaling_capacity10_10m_server.yaml"
uav_run="$uav_dppo_root/runs/diffusion_dppo_hop_scaling_capacity10_10m_seed0"
uav_python="$uav_dppo_root/.venv-dppo/bin/python"
case "${1:-all}" in
  collect) exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_python" -u -m uav_research dppo collect --config "$uav_config" --output "$uav_run/demonstrations" ;;
  pretrain) exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_python" -u -m uav_research dppo pretrain --config "$uav_config" --demonstrations "$uav_run/demonstrations" --output "$uav_run/pretrain" ;;
  resume-pretrain) exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_python" -u -m uav_research dppo pretrain --config "$uav_config" --demonstrations "$uav_run/demonstrations" --output "$uav_run/pretrain" --resume "$uav_run/pretrain/last.pt" ;;
  all)
    taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_python" -u -m uav_research dppo collect --config "$uav_config" --output "$uav_run/demonstrations"
    exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_python" -u -m uav_research dppo pretrain --config "$uav_config" --demonstrations "$uav_run/demonstrations" --output "$uav_run/pretrain" ;;
  *) echo 'Use all|collect|pretrain|resume-pretrain' >&2;exit 2 ;;
esac
