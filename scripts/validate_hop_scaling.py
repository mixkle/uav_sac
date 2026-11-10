"""Bounded acceptance only; never loads or starts a formal 10M run."""
import argparse,copy,json
from pathlib import Path
from uav_research.config import load_config,save_config
from uav_research.diffusion_dppo.data import collect_demonstrations,pretrain
from uav_research.diffusion_dppo.runner import train_online,export_actor
from uav_research.diffusion_dppo.report import evaluate_checkpoint
from uav_research.runtime import atomic_json

p=argparse.ArgumentParser();p.add_argument('--group',choices=['diffusion','gaussian'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=2048);p.add_argument('--pretrained');a=p.parse_args()
if a.steps>2048:raise ValueError('Acceptance budget exceeds 2048')
c=load_config(f'configs/{a.group}_hop_scaling_local_probe.yaml');c.run.device=a.device;c.run.total_steps=a.steps
if a.device!='cpu':c.run.cpu_threads=4;c.diffusion.microbatch_agents=4096
c.diffusion.teacher_checkpoint=str(Path('runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt').resolve())
a.output.mkdir(parents=True,exist_ok=True);save_config(c,a.output/'acceptance_config.yaml')
pretrained=a.pretrained
if a.group=='diffusion' and not pretrained:
 collect_demonstrations(c,a.output/'demonstrations');pretrain(c,a.output/'demonstrations',a.output/'pretrain');pretrained=a.output/'pretrain/best.pt'
result=train_online(c,a.output,pretrained=pretrained)
export_actor(a.output/'last.pt',a.output/'actor.pt')
evaluate_checkpoint(a.output/'last.pt',a.output/'evaluation',a.device,seeds=[10000,10001],moment_repeats=2)
atomic_json(a.output/'acceptance.json',result)
print(json.dumps(result))
