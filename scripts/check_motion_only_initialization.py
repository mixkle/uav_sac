"""Frozen fresh-network acceptance; writes diagnostics, never starts training."""
import argparse
import json
import numpy as np
import torch
from motion_only_comparison import resolve
from uav_research.runtime import seed_all, atomic_json, resolve_output
from uav_research.diffusion_dppo.storage import configure
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.environment import ServiceVector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rows=[]
    for group in ('M1','M2'):
        cfg,_=resolve(group,args.device)
        device=configure(cfg);seed_all(0)
        learner=DiffusionLearner(cfg,device)
        vector=ServiceVector(cfg.env,cfg.run.num_envs,cfg.run.vector_backend)
        try:
            obs,_,_=vector.reset(list(range(990000,990000+cfg.run.num_envs)))
            action,chain=learner.actor.sample(obs)
            with torch.no_grad():lp=learner.actor.evaluate_chain(obs,chain).cpu().numpy()
            error=np.abs(lp-chain.logp)
            np.testing.assert_allclose(lp,chain.logp,atol=.002,rtol=.0002)
            assert chain.scores.size==0 and not chain.candidate_offsets.any()
            assert not list(learner.actor.score_head.parameters())
            assert not learner.actor_opt.state and not learner.value_opt.state
            norms=np.linalg.norm(action.movement.reshape(-1,2),axis=1)
            latent=np.linalg.norm(chain.movement[:,-1],axis=1)
            if not (.1<norms.mean()<.95 and np.quantile(latent,.99)<15):
                raise ValueError('Fresh action scale is saturated or degenerate')
            rows.append(dict(group=group,chain_shape=list(chain.movement.shape),
                score_latent_coordinates=0,score_head_parameters=0,fresh_adam=True,
                initialization=cfg.diffusion.ablation['initialization'],
                diffusion_init_mode=cfg.diffusion.ablation['diffusion_init_mode'],
                command_norm_mean=float(norms.mean()),command_norm_p95=float(np.quantile(norms,.95)),
                latent_norm_mean=float(latent.mean()),latent_norm_p99=float(np.quantile(latent,.99)),
                logp_recompute_abs_max=float(error.max()),formal_training=False))
        finally:vector.close()
    atomic_json(resolve_output(args.output),dict(status='passed',groups=rows,formal_training=False))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
