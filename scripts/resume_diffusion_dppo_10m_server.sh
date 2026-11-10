#!/usr/bin/env bash
# Resume online DPPO from full 5M state; preserve the original run.
set -euo pipefail
case "${1:-}" in
  ""|--check) ;;
  *) echo "Usage: bash scripts/resume_diffusion_dppo_10m_server.sh [--check]" >&2; exit 2 ;;
esac
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_cfg="$uav_dppo_root/configs/diffusion_dppo_capacity10_coverage_throughput_10m_server.yaml"
uav_src="$uav_dppo_root/runs/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/checkpoints/full_005000000"
uav_out="$uav_dppo_root/runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0"
# Prevent two instances of this launcher from updating the same output.
exec 9>"$uav_dppo_root/.cache/dppo-10m.lock"
if ! flock -n 9; then
  echo "A 10M launcher already holds the output lock." >&2
  exit 1
fi
# Only fully published checkpoints are eligible for subsequent restarts.
for uav_candidate in "$uav_out"/checkpoints/full_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
  if [[ -f "$uav_candidate/index.json" && -f "$uav_candidate/metadata.pt" ]]; then
    uav_src="$uav_candidate"
  fi
done
if [[ "${1:-}" == --check ]]; then
  "$uav_dppo_root/.venv-dppo/bin/python" - "$uav_cfg" "$uav_src" "$uav_out" <<'PY'
import json
from pathlib import Path
import sys
import torch
from uav_research.config import load_config
from uav_research.diffusion_dppo.storage import load, ensure_budget

cfg = load_config(sys.argv[1])
state = load(sys.argv[2], cfg, {'dppo_full'})
assert cfg.run.total_steps == 10_000_000
assert state['env_steps'] < cfg.run.total_steps, 'Requested budget already complete'
assert state['num_envs'] == cfg.run.num_envs
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() >= 1
ensure_budget(cfg, Path(sys.argv[3]))
print(json.dumps(dict(passed=True, target_steps=cfg.run.total_steps,
    source_steps=state['env_steps'], additional_steps=cfg.run.total_steps-state['env_steps'],
    best_step=state['best_step'], resume=sys.argv[2], output=sys.argv[3],
    gpu=torch.cuda.get_device_name(0), check_only=True), indent=2), flush=True)
PY
  exit 0
fi
if [[ ! -f "$uav_out/continuation.json" ]]; then
  "$uav_dppo_root/.venv-dppo/bin/python" scripts/prepare_dppo_continuation.py \
    --config "$uav_cfg" --resume "$uav_src" --output "$uav_out"
fi
echo "Continuing online DPPO to cumulative 10,000,000 transitions; GPUs=$CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $uav_src"
echo "Output: $uav_out"
exec taskset -c "${UAV_CPU_AFFINITY:-0-11}" "$uav_dppo_root/.venv-dppo/bin/python" -u -m uav_research dppo train \
  --config "$uav_cfg" --output "$uav_out" --resume "$uav_src"
