#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_phase="${1:-}"
[[ "$uav_phase" == train || "$uav_phase" == check ]] || { echo 'Usage: resume_motion_only_10m_server.sh train|check --group M1|M2' >&2; exit 2; }
shift
uav_check=()
if [[ "$uav_phase" == check ]]; then uav_check=(--check); fi
uav_affinity=()
if [[ -n "${UAV_CPU_AFFINITY:-}" ]]; then uav_affinity=(taskset -c "$UAV_CPU_AFFINITY"); fi
exec "${uav_affinity[@]}" "$uav_dppo_root/.venv-dppo/bin/python" -u \
  scripts/resume_motion_only_10m.py "$@" "${uav_check[@]}" --device cuda:0
