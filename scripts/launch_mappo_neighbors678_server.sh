#!/usr/bin/env bash
# Three fresh 1M runs, one GPU per policy; user invokes this to start training.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UAV_NEIGHBOR_SLOTS=6,7,8
export UAV_NEIGHBOR_GPUS="${UAV_NEIGHBOR_GPUS:-0,1,2}"
export UAV_NEIGHBOR_CPUSETS="${UAV_NEIGHBOR_CPUSETS:-0-7|8-15|16-23}"
exec bash "$uav_script_dir/launch_mappo_neighbors_server.sh" "$@"
