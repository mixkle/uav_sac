#!/usr/bin/env bash
# Fresh single-GPU neighbor-count ablation. No checkpoint loading.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UAV_TORCH_THREADS=8
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_neighbors="${1:-}"
case "$uav_neighbors" in
  4|5|6|7|8) ;;
  *) echo "Usage: $0 {4|5|6|7|8} [--check]" >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n "${2:-}" && "${2:-}" != --check ]]; then
  echo "Only the optional --check flag is accepted." >&2
  exit 2
fi
case "$CUDA_VISIBLE_DEVICES" in
  ""|*,*) echo "Exactly one visible GPU is required." >&2; exit 2 ;;
esac
uav_cpu_set="${UAV_CPU_AFFINITY:-0-7}"
uav_config="$uav_dppo_root/configs/mappo_capacity10_neighbors${uav_neighbors}_1m_server.yaml"
uav_output="$uav_dppo_root/runs/mappo_capacity10_neighbors${uav_neighbors}_1m_seed0"
taskset -c "$uav_cpu_set" "$uav_dppo_root/.venv-dppo/bin/python" - "$uav_neighbors" "$uav_config" "$uav_output" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import torch
from uav_research.config import load_config
slots,config,output=sys.argv[1:]
cfg=load_config(config)
expected=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml').as_dict()
expected['env']['neighbor_slots']=int(slots)
expected['run']['device']='cuda:0'
if cfg.as_dict()!=expected:
    raise SystemExit('Unexpected changes beyond neighbor_slots/device.')
output=Path(output)
if output.resolve().parent!=(Path.cwd()/'runs').resolve():
    raise SystemExit('Output must stay in this project runs directory.')
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit('Fresh run requires a new/empty directory; refusing to overwrite or resume.')
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit('Exactly one visible CUDA GPU is required.')
if shutil.disk_usage('.').free<15*2**30:
    raise SystemExit('At least 15 GiB free disk is required.')
print(json.dumps(dict(check='passed',from_scratch=True,resume=None,seed=cfg.run.seed,
    neighbor_slots=cfg.env.neighbor_slots,fixed_observation=cfg.env.fixed_obs_dim,
    actor_input=cfg.env.fixed_obs_dim+cfg.algo.ue_hidden+1,critic_input=cfg.env.state_dim,
    total_steps=cfg.run.total_steps,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    gpu=torch.cuda.get_device_name(0),cpu_affinity=sorted(os.sched_getaffinity(0)),
    cpu_threads=cfg.run.cpu_threads,numba_threads=cfg.run.numba_threads,
    reward='coverage + 0.1 * normalized_throughput',output=str(output)),ensure_ascii=False),flush=True)
PY
if [[ "${2:-}" == --check ]]; then
  exit 0
fi
mkdir -p "$uav_dppo_root/logs"
uav_log="$uav_dppo_root/logs/mappo_neighbors${uav_neighbors}_1m_$(date +%Y%m%d_%H%M%S).log"
echo "Starting fresh neighbors=$uav_neighbors 1M run. Console log: $uav_log"
exec taskset -c "$uav_cpu_set" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research train \
  --config "$uav_config" --output "$uav_output" \
  > >(tee -a "$uav_log") 2>&1
