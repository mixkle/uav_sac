"""Small deterministic CUDA continuation check; isolated acceptance outputs."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.runtime import atomic_save,atomic_json,seed_all
from uav_research.diffusion_dppo.storage import artifact,load
from uav_research.diffusion_dppo.policy import DiffusionActor
from uav_research.diffusion_dppo.runner import train_online
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--group',choices=['diffusion','gaussian'],required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
c=load_config('configs/diffusion_dppo_local_smoke.yaml');c.algo.name='diffusion_dppo' if a.group=='diffusion' else 'gaussian_service_mappo'
c.env.motion_scaling['enabled']=True;c.env.history_motion_source='post_reflection_displacement'
c.run.device='cuda:0';c.run.cpu_threads=4;c.run.numba_threads=4;c.diffusion.minimum_free_gib=0
c.algo.ppo_epochs=1;c.algo.ppo_minibatches=1
source=None
if a.group=='diffusion':
 seed_all(0);source=a.output/'synthetic_acceptance_init.pt'
 atomic_save(source,artifact(c,'pretrain',actor=DiffusionActor(c).state_dict(),epoch=0,provenance={'note':'synthetic untrained small acceptance only'}))
train_online(c,a.output/'continuous',pretrained=source)
first=train_online(c,a.output/'resumed',pretrained=source,stop_after_updates=1)
train_online(c,a.output/'resumed',resume=first['full_checkpoint'])
x=load(a.output/'continuous/last.pt')['learner'];y=load(a.output/'resumed/last.pt')['learner'];max_error=[0.]
def compare(a,b):
 if isinstance(a,torch.Tensor):
  if a.numel():max_error[0]=max(max_error[0],float((a-b).abs().max()))
  torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:compare(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for l,r in zip(a,b):compare(l,r)
 else:assert a==b
compare(x,y)
atomic_json(a.output/'resume_verified.json',dict(group=a.group,max_absolute_difference=max_error[0],atol=1e-6,rtol=1e-6,updates=x['updates'],optimizer_steps=x['optimizer_steps']))
