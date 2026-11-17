"""32-transition main-scene plumbing probes, NOT weights for formal training."""
import argparse,time
from pathlib import Path
from uav_research.diffusion_followup.runtime import resolve
from uav_research.runtime import seed_all,atomic_json,memory_stats
from uav_research.config import save_config
from uav_research.diffusion_dppo.learner import DiffusionLearner,DppoRollout
from uav_research.diffusion_dppo.environment import ServiceVector
from uav_research.diffusion_dppo.storage import configure
from uav_research.system_ablation.initialization import initialize


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu')
 p.add_argument('--output',required=True,type=Path);a=p.parse_args();rows=[]
 for g in ('F2b','F2c','A1b'):
  cfg,_,_=resolve(g,a.device)
  cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=1;cfg.diffusion.microbatch_agents=256
  cfg.run.cpu_threads=8 if a.device=='cpu' else 4
  if g=='F2c':cfg.diffusion.ablation.update(initialization='random',initial_checkpoint='')
  cfg.validate();device=configure(cfg);seed_all(0)
  started=time.perf_counter();learner=DiffusionLearner(cfg,device)
  if g=='F2b':initialize(learner)
  vector=ServiceVector(cfg.env,16,'batch')
  try:
   obs,state,_=vector.reset(list(range(990000,990016)));rollout=DppoRollout()
   for _ in range(2):
    action,chain=learner.actor.sample(obs);value=learner.values(state).detach().cpu().numpy()
    new,state2,reward,done,trunc,_=vector.step(action);nv=learner.values(state2).detach().cpu().numpy()
    rollout.append(obs,chain,state=state,reward=reward,value=value,next_value=nv,terminated=done,truncated=trunc)
    obs,state=new,state2
   metrics=learner.update(rollout)
   target=a.output/g;target.mkdir(parents=True,exist_ok=True);save_config(cfg,target/'probe_config.yaml')
   result=dict(group=g,formal_training=False,transitions=32,elapsed_seconds=time.perf_counter()-started,
    optimizer_steps=learner.optimizer_steps,F2c_initialization='random plumbing probe, NOT formal BC',
    metrics=metrics,**memory_stats(device));rows.append(result);atomic_json(target/'result.json',result)
   print(g,result['elapsed_seconds'],flush=True)
  finally:vector.close()
 atomic_json(a.output/'summary.json',rows)


if __name__=='__main__':main()
