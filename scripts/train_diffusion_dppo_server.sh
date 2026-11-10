#!/usr/bin/env bash
# User-triggered formal two-stage run; no training starts during deployment.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
# GPU 0/1 is nearest NUMA node 0; override for another allocated GPU/CPU set.
exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research dppo all \
  --config "$uav_dppo_root/configs/diffusion_dppo_capacity10_coverage_throughput_1m_server.yaml" \
  --output "$uav_dppo_root/runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0"
