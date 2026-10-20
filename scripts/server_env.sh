# Source this file on the training server before running project commands.
export UAV_DATA_ROOT=/root/autodl-tmp/uav_sac
export XDG_CACHE_HOME="$UAV_DATA_ROOT/cache"
export MPLCONFIGDIR="$UAV_DATA_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$UAV_DATA_ROOT/cache/numba"
export PIP_CACHE_DIR="$UAV_DATA_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$UAV_DATA_ROOT/cache/pycache"
export TMPDIR="$UAV_DATA_ROOT/tmp"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp
mkdir -p "$UAV_DATA_ROOT" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR"
export PATH="$UAV_DATA_ROOT/venv/bin:$PATH"
