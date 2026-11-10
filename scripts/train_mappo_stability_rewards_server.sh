#!/usr/bin/env bash
# One fresh 1M ablation per process / visible GPU. User-triggered only.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UAV_TORCH_THREADS=8
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_group="${1:-}"
case "$uav_group" in
  disconnection|robustness|combined|disconnection_mu05|disconnection_mu2|disconnection_mu5) ;;
  *) echo "Usage: $0 {disconnection|robustness|combined|disconnection_mu05|disconnection_mu2|disconnection_mu5} [--check]" >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n "${2:-}" && "${2:-}" != --check ]]; then
  echo "Only the optional --check flag is accepted." >&2
  exit 2
fi
case "$CUDA_VISIBLE_DEVICES" in
  ""|*,*) echo "Expose exactly one GPU through CUDA_VISIBLE_DEVICES." >&2; exit 2 ;;
esac
uav_cpu_set="${UAV_CPU_AFFINITY:-0-7}"
uav_config="$uav_dppo_root/configs/mappo_capacity10_stability_${uav_group}_1m_server.yaml"
uav_output="$uav_dppo_root/runs/mappo_capacity10_stability_${uav_group}_1m_seed0"
taskset -c "$uav_cpu_set" "$uav_dppo_root/.venv-dppo/bin/python" - "$uav_group" "$uav_config" "$uav_output" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import torch
from uav_research.config import load_config
from uav_research.coverage_stability import BASELINE_CONFIG, ablation_config

group, config, output = sys.argv[1:]
cfg = load_config(config)
expected = ablation_config(load_config(BASELINE_CONFIG), group)
if cfg.as_dict() != expected.as_dict():
    raise SystemExit('Unexpected scenario or learning-parameter changes in ablation config.')
output = Path(output)
if output.resolve().parent != (Path.cwd()/'runs').resolve():
    raise SystemExit('Output must remain within this project runs directory.')
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit('Fresh run requires a new/empty directory; refusing to load/overwrite results.')
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Exactly one visible CUDA GPU is required.')
if shutil.disk_usage('.').free < 15*2**30:
    raise SystemExit('At least 15 GiB free disk is required.')
print(json.dumps(dict(check='passed', group=group, from_scratch=True, resume=None,
    seed=cfg.run.seed, total_steps=cfg.run.total_steps,
    physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(0),
    cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=cfg.run.cpu_threads,
    numba_threads=cfg.run.numba_threads, weights=dict(coverage=1.,throughput=cfg.env.throughput_weight,
    robustness=cfg.env.robustness_weight,disconnection=cfg.env.disconnection_weight),
    output=str(output)), ensure_ascii=False), flush=True)
PY
if [[ "${2:-}" == --check ]]; then
  exit 0
fi
mkdir -p "$uav_dppo_root/logs"
uav_log="$uav_dppo_root/logs/mappo_stability_${uav_group}_1m_$(date +%Y%m%d_%H%M%S).log"
echo "Starting fresh $uav_group 1M run. Console log: $uav_log"
exec taskset -c "$uav_cpu_set" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research train \
  --config "$uav_config" --output "$uav_output" \
  > >(tee -a "$uav_log") 2>&1
