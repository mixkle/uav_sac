#!/usr/bin/env bash
# Called inside tmux by the launcher. Never invoked automatically on upload.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
IFS=, read -r -a uav_gpus <<< "${UAV_FOLLOWUP_GPUS:-0,1,2}"
IFS='|' read -r -a uav_cpus <<< "${UAV_FOLLOWUP_CPUSETS:-0-5|6-11|12-17}"
uav_logs="$UAV_DATA_ROOT/logs/diffusion_followup_1m"
mkdir -p "$uav_logs"
echo "$(date -Is) Preparation: fixed initialization and sensitivity audits"
CUDA_VISIBLE_DEVICES="${uav_gpus[2]}" UAV_CPU_AFFINITY="${uav_cpus[2]}" \
 bash scripts/train_diffusion_followup_server.sh audit-init --group A1b > "$uav_logs/audit_init.log" 2>&1
CUDA_VISIBLE_DEVICES="${uav_gpus[1]}" UAV_CPU_AFFINITY="${uav_cpus[1]}" \
 bash scripts/train_diffusion_followup_server.sh audit-coupling --group F2c > "$uav_logs/audit_coupling.log" 2>&1
echo "$(date -Is) Preparation: frozen A3 replay and F2c 50-epoch BC in parallel"
CUDA_VISIBLE_DEVICES="${uav_gpus[0]}" UAV_CPU_AFFINITY="${uav_cpus[0]}" \
 bash scripts/train_diffusion_followup_server.sh frozen --group F2b > "$uav_logs/frozen.log" 2>&1 &
uav_frozen_pid=$!
CUDA_VISIBLE_DEVICES="${uav_gpus[1]}" UAV_CPU_AFFINITY="${uav_cpus[1]}" \
 bash scripts/train_diffusion_followup_server.sh pretrain --group F2c --auto-resume > "$uav_logs/F2c_pretrain.log" 2>&1 &
uav_bc_pid=$!
uav_status=0
while kill -0 "$uav_frozen_pid" 2>/dev/null || kill -0 "$uav_bc_pid" 2>/dev/null;do
 echo "$(date -Is) Preparation still running; see frozen.log and F2c_pretrain.log. Online groups have not started."
 sleep 30
done
wait "$uav_frozen_pid" || uav_status=1
wait "$uav_bc_pid" || uav_status=1
if ((uav_status));then echo 'Preparation failed; NO online group started. See frozen/BC logs.' >&2;exit 1;fi
uav_groups=(F2b F2c A1b)
for uav_index in 0 1 2;do
 uav_group="${uav_groups[$uav_index]}"
 CUDA_VISIBLE_DEVICES="${uav_gpus[$uav_index]}" bash scripts/train_diffusion_followup_server.sh ready --group "$uav_group"
 if tmux -S "$uav_dppo_root/.cache/tmux.sock" has-session -t "follow-$uav_group" 2>/dev/null;then
  echo "Session appeared during preparation: follow-$uav_group" >&2;exit 1
 fi
done
"$uav_dppo_root/.venv-dppo/bin/python" - <<'PY'
import os,subprocess
from pathlib import Path
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for g in os.environ.get('UAV_FOLLOWUP_GPUS','0,1,2').split(','):
 uuid=subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 if uuid in apps:raise RuntimeError('Another GPU job appeared during preparation: '+g)
for g in ['F2b','F2c','A1b']:
 p=Path(os.environ['UAV_DATA_ROOT'])/'runs/diffusion_followup_1m'/(g+'_seed0')
 if any((p/n).exists() for n in ['step0.pt','last.pt','manifest.json']):raise FileExistsError('Online output appeared: '+str(p))
PY
echo "$(date -Is) Preparation complete. Starting THREE online 1M groups."
for uav_index in 0 1 2;do
 uav_group="${uav_groups[$uav_index]}"
 printf -v uav_command 'cd %q && CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q UAV_TORCH_THREADS=%q UAV_NUMBA_THREADS=%q bash scripts/train_diffusion_followup_server.sh train --group %q > %q 2>&1' \
  "$uav_dppo_root" "${uav_gpus[$uav_index]}" "${uav_cpus[$uav_index]}" "${UAV_TORCH_THREADS:-4}" "${UAV_NUMBA_THREADS:-4}" "$uav_group" "$uav_logs/$uav_group.log"
 tmux -S "$uav_dppo_root/.cache/tmux.sock" new-session -d -s "follow-$uav_group" "$uav_command"
 echo "Started follow-$uav_group on GPU ${uav_gpus[$uav_index]}"
done
