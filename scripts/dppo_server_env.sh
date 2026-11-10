#!/usr/bin/env bash
# Source this inside the server project. Every explicit write stays in /home/dym.
set -euo pipefail
uav_dppo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$uav_dppo_root" in
  /home/dym/*) ;;
  *) echo "Server runtime must be installed below /home/dym" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export UAV_DATA_ROOT="$uav_dppo_root"
export TMPDIR="$uav_dppo_root/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$uav_dppo_root/.cache"
export UV_CACHE_DIR="$uav_dppo_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$uav_dppo_root/.python"
export PIP_CACHE_DIR="$uav_dppo_root/.cache/pip"
export NUMBA_CACHE_DIR="$uav_dppo_root/.cache/numba"
export MPLCONFIGDIR="$uav_dppo_root/.cache/matplotlib"
export TORCH_HOME="$uav_dppo_root/.cache/torch"
export CUDA_CACHE_PATH="$uav_dppo_root/.cache/cuda"
export TRITON_CACHE_DIR="$uav_dppo_root/.cache/triton"
export PYTHONPYCACHEPREFIX="$uav_dppo_root/.cache/pycache"
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
