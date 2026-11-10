#!/usr/bin/env bash
# User-triggered fresh baseline; one visible GPU, no checkpoint loading.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UAV_TORCH_THREADS=8
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
case "${1:-}" in
  ""|--check) ;;
  *) echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
case "$CUDA_VISIBLE_DEVICES" in
  ""|*,*) echo "Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU." >&2; exit 2 ;;
esac
uav_baseline_config="$uav_dppo_root/configs/mappo_coverage_throughput_capacity10_10m_server.yaml"
uav_baseline_output="$uav_dppo_root/runs/mappo_coverage_throughput_capacity10_10m_seed0"
"$uav_dppo_root/.venv-dppo/bin/python" - "$uav_baseline_config" "$uav_baseline_output" <<'PY'
import json
import os
from pathlib import Path
import torch
from uav_research.config import load_config

import sys
config_path, output = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_config(config_path)
original = load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
expected = original.as_dict()
expected['run'].update(total_steps=10_000_000, device='cuda:0')
if cfg.as_dict() != expected:
    raise SystemExit('Config differs from the original baseline beyond steps/device.')
if output.resolve().parent != (Path.cwd() / 'runs').resolve():
    raise SystemExit('Output must be inside this project runs directory.')
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit('Fresh run requires an empty/new output directory; refusing to overwrite.')
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Exactly one visible CUDA GPU is required.')
print(json.dumps(dict(
    check='passed', mode='from_scratch', resume=None, seed=cfg.run.seed,
    total_steps=cfg.run.total_steps, algorithm=cfg.algo.name,
    physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], logical_device=cfg.run.device,
    gpu=torch.cuda.get_device_name(0), cpu_threads=cfg.run.cpu_threads,
    numba_threads=cfg.run.numba_threads, output=str(output),
), ensure_ascii=False), flush=True)
PY
if [[ "${1:-}" == --check ]]; then
  exit 0
fi
mkdir -p "$uav_dppo_root/logs"
uav_baseline_log="$uav_dppo_root/logs/mappo_capacity10_10m_$(date +%Y%m%d_%H%M%S).log"
echo "Starting fresh 10M baseline. Console log: $uav_baseline_log"
exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research train \
  --config "$uav_baseline_config" \
  --output "$uav_baseline_output" \
  > >(tee -a "$uav_baseline_log") 2>&1
