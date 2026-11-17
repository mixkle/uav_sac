#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
uav_socket="$uav_dppo_root/.cache/tmux.sock"
uav_logs="$UAV_DATA_ROOT/logs/diffusion_followup_1m"
mkdir -p "$uav_logs"
for uav_session in followup-prepare follow-F2b follow-F2c follow-A1b;do
 if tmux -S "$uav_socket" has-session -t "$uav_session" 2>/dev/null;then echo "Session exists: $uav_session" >&2;exit 1;fi
done
"$uav_dppo_root/.venv-dppo/bin/python" - <<'PY'
import os,json,shutil,subprocess
from pathlib import Path
gpus=os.environ.get('UAV_FOLLOWUP_GPUS','0,1,2').split(',')
if len(gpus)!=3 or len(set(gpus))!=3 or any(not g.isdecimal() for g in gpus):raise ValueError('Select THREE distinct GPUs')
seen=set();masks=os.environ.get('UAV_FOLLOWUP_CPUSETS','0-5|6-11|12-17').split('|')
if len(masks)!=3:raise ValueError('Three CPU affinity sets required')
for mask in masks:
 cpus=set()
 for part in mask.split(','):
  b=list(map(int,part.split('-')))
  if len(b) not in (1,2) or b[-1]<b[0]:raise ValueError('Invalid CPU range')
  cpus.update(range(b[0],b[-1]+1))
 if len(cpus)<4 or cpus&seen or not cpus<=os.sched_getaffinity(0):raise ValueError('Need available disjoint >=4 CPU sets')
 seen|=cpus
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for g in gpus:
 uuid=subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 if uuid in apps:raise RuntimeError('GPU '+g+' has another compute job; choose idle cards')
root=Path(os.environ['UAV_DATA_ROOT'])
if shutil.disk_usage(root).free<23*2**30:raise OSError('Need >=23GiB free, including15GiB reserve')
for group in ['F2b','F2c','A1b']:
 p=root/'runs/diffusion_followup_1m'/(group+'_seed0')
 if any((p/n).exists() for n in ['manifest.json','step0.pt','best.pt','last.pt']):raise FileExistsError('Use explicit full-checkpoint resume: '+str(p))
PY
for uav_group in F2b F2c A1b;do
 bash scripts/train_diffusion_followup_server.sh check --group "$uav_group"
done
printf -v uav_command 'cd %q && UAV_FOLLOWUP_GPUS=%q UAV_FOLLOWUP_CPUSETS=%q UAV_TORCH_THREADS=%q UAV_NUMBA_THREADS=%q bash scripts/prepare_launch_diffusion_followup_server.sh >> %q 2>&1' \
 "$uav_dppo_root" "${UAV_FOLLOWUP_GPUS:-0,1,2}" "${UAV_FOLLOWUP_CPUSETS:-0-5|6-11|12-17}" "${UAV_TORCH_THREADS:-4}" "${UAV_NUMBA_THREADS:-4}" "$uav_logs/prepare.log"
tmux -S "$uav_socket" new-session -d -s followup-prepare "$uav_command"
echo "Preparation started in tmux; audits + frozen replay + F2c BC, then three simultaneous ONLINE jobs."
echo "Progress: tail -f $uav_logs/prepare.log"
