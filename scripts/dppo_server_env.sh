#!/usr/bin/env bash
# Source this inside the server project. Code stays in /home/dym; data uses disk2.
set -euo pipefail
uav_dppo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$uav_dppo_root" in
  /home/dym/*) ;;
  *) echo "Server runtime must be installed below /home/dym" >&2; return 1 2>/dev/null || exit 1 ;;
esac
if ! mountpoint -q /mnt/disk2; then
  echo '/mnt/disk2 is not mounted; refusing to save artifacts on the system disk.' >&2
  return 1 2>/dev/null || exit 1
fi
if [[ "${UAV_DATA_ROOT:-}" == "$uav_dppo_root" || "${UAV_DATA_ROOT:-}" == /mnt/disk2/dym ]]; then
  # Existing tmux shells may still carry the pre-migration default.
  export UAV_DATA_ROOT=/mnt/disk2/dym/uav_sac
else
  export UAV_DATA_ROOT="${UAV_DATA_ROOT:-/mnt/disk2/dym/uav_sac}"
fi
case "$UAV_DATA_ROOT" in
  /mnt/disk2/dym/uav_sac|/mnt/disk2/dym/uav_sac/*) ;;
  *) echo 'Server UAV_DATA_ROOT must be below /mnt/disk2/dym/uav_sac' >&2; return 1 2>/dev/null || exit 1 ;;
esac
uav_data_resolved="$(realpath -m -- "$UAV_DATA_ROOT")"
case "$uav_data_resolved" in
  /mnt/disk2/dym/uav_sac|/mnt/disk2/dym/uav_sac/*) ;;
  *) echo 'Resolved data directory escapes /mnt/disk2/dym/uav_sac' >&2; return 1 2>/dev/null || exit 1 ;;
esac
export UAV_DATA_ROOT="$uav_data_resolved"
mkdir -p "$UAV_DATA_ROOT"
export TMPDIR="$UAV_DATA_ROOT/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$UAV_DATA_ROOT/.cache"
export UV_CACHE_DIR="$UAV_DATA_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$uav_dppo_root/.python"
export PIP_CACHE_DIR="$UAV_DATA_ROOT/.cache/pip"
export NUMBA_CACHE_DIR="$UAV_DATA_ROOT/.cache/numba"
export MPLCONFIGDIR="$UAV_DATA_ROOT/.cache/matplotlib"
export TORCH_HOME="$UAV_DATA_ROOT/.cache/torch"
export CUDA_CACHE_PATH="$UAV_DATA_ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$UAV_DATA_ROOT/.cache/triton"
export PYTHONPYCACHEPREFIX="$UAV_DATA_ROOT/.cache/pycache"
export PYTHONUSERBASE="$uav_dppo_root/.local"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS="${UAV_TORCH_THREADS:-4}"
export OMP_NUM_THREADS="${UAV_TORCH_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
ulimit -c 0
