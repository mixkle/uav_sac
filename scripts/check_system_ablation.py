"""Main-scale CPU acceptance only; never produces weights usable as formal runs."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
from uav_research.config import load_config,save_config
from uav_research.runtime import seed_all,atomic_json,memory_stats
from uav_research.diffusion_dppo.storage import configure
from uav_research.diffusion_dppo.environment import ServiceVector
from uav_research.diffusion_dppo.learner import DiffusionLearner,DppoRollout


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
 p.add_argument('--groups',nargs='+');p.add_argument('--device',default='cpu');a=p.parse_args()
 matrix=json.loads(Path('configs/system_ablation_1m/manifest.json').read_text())
 results=[]
 for key in a.groups or list(matrix['runs']):
  cfg=load_config(matrix['runs'][key]['config'])
  cfg.run.device=a.device;cfg.run.cpu_threads=8 if a.device=='cpu' else 4;cfg.run.numba_threads=4
  cfg.diffusion.microbatch_agents=256;cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=1
  cfg.diffusion.ablation.update(initialization='random',initial_checkpoint='')
  cfg.validate();device=configure(cfg);seed_all(0)
  start=time.perf_counter();learner=DiffusionLearner(cfg,device)
  vector=ServiceVector(cfg.env,16,'batch')
  try:
   obs,state,_=vector.reset(list(range(990000,990016)))
   rollout=DppoRollout();inference=physics=0.
   max_seen,max_candidates=0,0
   for _ in range(2):
    stamp=time.perf_counter();action,chain=learner.actor.sample(obs)
    value=learner.values(state).cpu().numpy();inference+=time.perf_counter()-stamp
    max_seen=max(max_seen,int(np.diff(obs.offsets).max()))
    max_candidates=max(max_candidates,int(obs.candidate_lengths.max()))
    stamp=time.perf_counter();no,ns,reward,done,trunc,_=vector.step(action);physics+=time.perf_counter()-stamp
    nv=learner.values(ns).cpu().numpy()
    rollout.append(obs,chain,state=state,reward=reward,value=value,next_value=nv,terminated=done,truncated=trunc)
    obs,state=no,ns
   stamp=time.perf_counter();metrics=learner.update(rollout);update=time.perf_counter()-stamp
   target=a.output/key;target.mkdir(parents=True,exist_ok=True);save_config(cfg,target/'probe_config.yaml')
   result=dict(group=key,transitions=32,optimizer_steps=learner.optimizer_steps,
      inference_seconds=inference,physics_seconds=physics,update_seconds=update,
      elapsed_seconds=time.perf_counter()-start,max_seen=max_seen,max_candidates=max_candidates,
      actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
      metrics=metrics,**memory_stats(device),formal_training=False)
   atomic_json(target/'result.json',result);results.append(result)
   print(key,round(result['elapsed_seconds'],2),'seconds',flush=True)
  finally:vector.close()
 atomic_json(a.output/'summary.json',results)

if __name__=='__main__':main()
