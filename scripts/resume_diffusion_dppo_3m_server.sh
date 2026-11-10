#!/usr/bin/env bash
# Continue online DPPO only. No demonstration collection or BC is repeated.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
uav_mode="${1:-single}"
case "$uav_mode" in
  dual) uav_suffix="_dual"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" ;;
  single) uav_suffix=""; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ;;
  *) echo "Usage: bash scripts/resume_diffusion_dppo_3m_server.sh [dual|single]" >&2; exit 2 ;;
esac
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_cfg="$uav_dppo_root/configs/diffusion_dppo_capacity10_coverage_throughput_3m_server${uav_suffix}.yaml"
uav_src="$uav_dppo_root/runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/checkpoints/full_001000000"
uav_out="$uav_dppo_root/runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0"
# For an interrupted 3M continuation, use its latest complete checkpoint.
if [[ -d "$uav_out/checkpoints" ]]; then
  uav_latest="$(find "$uav_out/checkpoints" -maxdepth 1 -type d -name 'full_[0-9]*' | sort | tail -n 1)"
  if [[ -n "$uav_latest" ]]; then
    uav_src="$uav_latest"
  fi
fi
# Preparing a new fork is idempotent. Later restarts already have their prefix.
if [[ ! -f "$uav_out/continuation.json" ]]; then
  "$uav_dppo_root/.venv-dppo/bin/python" scripts/prepare_dppo_continuation.py \
    --config "$uav_cfg" --resume "$uav_src" --output "$uav_out"
fi
echo "Continuing online DPPO to cumulative 3,000,000 transitions; mode=$uav_mode GPUs=$CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $uav_src"
echo "Output: $uav_out"
exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research dppo train \
  --config "$uav_cfg" --output "$uav_out" --resume "$uav_src"
