#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
case "${1:?Use diffusion or gaussian}" in
 diffusion) uav_name=diffusion_dppo_hop_scaling_capacity10_10m;uav_cli=dppo ;;
 gaussian) uav_name=gaussian_mappo_topk_hop_scaling_capacity10_10m;uav_cli=gaussian_service_mappo ;;
 *) exit 2 ;;
esac
uav_args=(--config "$uav_dppo_root/configs/${uav_name}_server.yaml" --output "$uav_dppo_root/runs/${uav_name}_seed0")
if [[ -n "${UAV_RESUME_CHECKPOINT:-}" ]];then uav_args+=(--resume "$UAV_RESUME_CHECKPOINT");fi
exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research "$uav_cli" train "${uav_args[@]}"
