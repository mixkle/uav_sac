#!/usr/bin/env bash
# User starts online training only after diffusion preparation completes.
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
if (( $# > 1 )) || [[ -n "${1:-}" && "${1:-}" != --check ]];then
 echo 'Usage: launch_hop_scaling_comparison_server.sh [--check]' >&2;exit 2
fi
IFS=, read -r -a uav_gpus <<< "${UAV_HOP_GPUS:-0,2}"
IFS='|' read -r -a uav_cpus <<< "${UAV_HOP_CPUSETS:-0-11|12-23}"
[[ ${#uav_gpus[@]} == 2 && ${#uav_cpus[@]} == 2 && "${uav_gpus[0]}" != "${uav_gpus[1]}" ]] || { echo 'Two distinct GPUs and CPU sets required' >&2;exit 2; }
uav_tmux_socket="$uav_dppo_root/.cache/tmux.sock"
command -v tmux >/dev/null
"$uav_dppo_root/.venv-dppo/bin/python" - "${uav_cpus[@]}" <<'PYCPU'
import os,sys
sets=[]
for mask in sys.argv[1:]:
    cpus=set()
    for part in mask.split(','):
        pair=part.split('-')
        if len(pair)>2 or any(not x.isdigit() for x in pair):raise ValueError('Invalid CPU affinity')
        lo,hi=int(pair[0]),int(pair[-1])
        if hi<lo:raise ValueError('Invalid CPU range')
        cpus.update(range(lo,hi+1))
    if len(cpus)<4 or not cpus.issubset(os.sched_getaffinity(0)):raise ValueError('Need at least four available CPUs per job')
    if any(cpus&other for other in sets):raise ValueError('CPU affinity sets must not overlap')
    sets.append(cpus)
PYCPU
for uav_gpu in "${uav_gpus[@]}";do [[ "$uav_gpu" =~ ^[0-9]+$ ]] && nvidia-smi -i "$uav_gpu" --query-gpu=name --format=csv,noheader;done
"$uav_dppo_root/.venv-dppo/bin/python" - <<'PY'
from pathlib import Path
import shutil,json
from uav_research.config import load_config
from uav_research.diffusion_dppo.storage import load
from uav_research.diffusion_dppo.config import contract
from uav_research.topk_diagnostic import file_hash
root=Path('/home/dym/uav_sac')
a=load_config(root/'configs/diffusion_dppo_hop_scaling_capacity10_10m_server.yaml')
b=load_config(root/'configs/gaussian_mappo_topk_hop_scaling_capacity10_10m_server.yaml')
assert a.env==b.env and a.run==b.run and a.algo.actor_lr==b.algo.actor_lr==1e-4
if not Path(a.diffusion.pretrained_checkpoint).exists():raise SystemExit('Prepare diffusion first: bash scripts/prepare_hop_scaling_diffusion_server.sh all')
pretrained=load(a.diffusion.pretrained_checkpoint,a,{'dppo_pretrain'})
pretrain_dir=Path(a.diffusion.pretrained_checkpoint).parent
summary=json.loads((pretrain_dir/'summary.json').read_text())
dataset_path=pretrain_dir.parent/'demonstrations/dataset.json'
dataset=json.loads(dataset_path.read_text())
if summary.get('status')!='complete' or summary.get('epoch')!=a.diffusion.pretrain_epochs:
 raise ValueError('Complete diffusion pretraining before paired online launch')
if not dataset.get('complete') or dataset.get('contract')!=contract(a) or dataset.get('expected_joint_transitions')!=64000:
 raise ValueError('Formal demonstrations incomplete or mismatched')
if dataset['teacher']['sha256']!=file_hash(a.diffusion.teacher_checkpoint):raise ValueError('Teacher source changed')
if pretrained['provenance']['dataset_sha256']!=file_hash(dataset_path):raise ValueError('Pretraining data changed')
load(pretrain_dir/'last.pt',a,{'dppo_pretrain'})
for name in ('diffusion_dppo_hop_scaling_capacity10_10m','gaussian_mappo_topk_hop_scaling_capacity10_10m'):
 out=root/'runs'/f'{name}_seed0'
 if any((out/f).exists() for f in ('last.pt','best.pt','manifest.json','step0.pt')):raise FileExistsError(f'Use explicit resume: {out}')
# Two online rollouts, recent checkpoints, logs and evaluation outputs; demos
# already exist. Reserve 4 GiB for outputs plus 15 GiB untouched free space.
if shutil.disk_usage(root).free<19*2**30:raise OSError('Need at least 19 GiB free before paired online run')
print('Paired configurations and new pretrained artifact verified')
PY
for uav_session in hop-diffusion hop-gaussian;do
 if tmux -S "$uav_tmux_socket" has-session -t "$uav_session" 2>/dev/null;then echo "Existing tmux session: $uav_session" >&2;exit 1;fi
done
[[ "${1:-}" == --check ]] && exit 0
mkdir -p "$uav_dppo_root/runs/hop_scaling_launch_logs"
uav_started=()
trap 'for s in "${uav_started[@]}";do tmux -S "$uav_tmux_socket" send-keys -t "$s" C-c 2>/dev/null || true;done' ERR
for uav_index in 0 1;do
 if [[ $uav_index == 0 ]];then uav_group=diffusion;else uav_group=gaussian;fi
 printf -v uav_command 'cd %q && CUDA_VISIBLE_DEVICES=%q UAV_CPU_AFFINITY=%q bash %q %q > %q 2>&1' \
   "$uav_dppo_root" "${uav_gpus[$uav_index]}" "${uav_cpus[$uav_index]}" \
   "$uav_script_dir/train_hop_scaling_group_server.sh" "$uav_group" \
   "$uav_dppo_root/runs/hop_scaling_launch_logs/${uav_group}.log"
 tmux -S "$uav_tmux_socket" new-session -d -s "hop-$uav_group" "$uav_command"
 uav_started+=("hop-$uav_group")
done
trap - ERR
printf 'Started hop-diffusion and hop-gaussian. Logs: %s/runs/hop_scaling_launch_logs/\n' "$uav_dppo_root"
