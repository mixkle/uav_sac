"""Full-scene fresh CUDA/CPU acceptance without formal training or run creation."""
import argparse
import json
import numpy as np
import torch

from m2_control_experiments import GROUPS,resolve
from uav_research.runtime import seed_all,atomic_json,resolve_output
from uav_research.diffusion_dppo.storage import configure
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.environment import ServiceVector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=GROUPS)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rows=[];reference_state=None;reference_action=None;reference_value=None
    for group in ([args.group] if args.group else GROUPS):
        _,cfg,_,_=resolve(group,args.device)
        device=configure(cfg);seed_all(0)
        learner=DiffusionLearner(cfg,device)
        vector=ServiceVector(cfg.env,cfg.run.num_envs,cfg.run.vector_backend)
        try:
            obs,state,_=vector.reset(range(990000,990000+cfg.run.num_envs))
            generator=torch.Generator(device=device).manual_seed(991001)
            action,chain=learner.actor.sample(obs,generator=generator)
            with torch.no_grad():
                logp=learner.actor.evaluate_chain(obs,chain).cpu().numpy()
                value=learner.values(state).cpu().numpy()
            np.testing.assert_allclose(logp,chain.logp,atol=.002,rtol=.0002)
            assert chain.scores.size==0 and not chain.candidate_offsets.any()
            assert not list(learner.actor.score_head.parameters())
            assert not learner.actor_opt.state and not learner.value_opt.state
            assert np.isfinite(action.movement).all() and np.isfinite(logp).all()
            if reference_state is None:
                reference_state=state.copy();reference_action=action.movement.copy();reference_value=value.copy()
            np.testing.assert_array_equal(state,reference_state)
            np.testing.assert_allclose(action.movement,reference_action,atol=1e-6,rtol=1e-6)
            np.testing.assert_array_equal(value,reference_value)
            observed=obs.fixed.reshape(-1,cfg.env.fixed_obs_dim)
            start=cfg.env.self_dim+2*cfg.env.num_bs
            neighbors=observed[:,start:cfg.env.current_obs_dim].reshape(-1,cfg.env.neighbor_slots,cfg.env.neighbor_dim)
            actual_neighbors=(np.abs(neighbors).sum(-1)>0).sum(-1)
            _,_,reward,_,_,infos=vector.step(action)
            np.testing.assert_allclose(reward,[i['coverage']+.1*i['tau'] for i in infos],atol=2e-7,rtol=2e-7)
            if group=='M2-NoHop':
                for env in vector.envs:np.testing.assert_array_equal(env.speed_factor,np.ones(cfg.env.num_uavs))
            rows.append(dict(group=group,device=str(device),neighbor_slots=cfg.env.neighbor_slots,
                fixed_observation_dim=cfg.env.fixed_obs_dim,context_input=learner.actor.context[0].in_features,
                critic_input=cfg.env.state_dim,actor_parameters=sum(x.numel() for x in learner.actor.parameters()),
                critic_parameters=sum(x.numel() for x in learner.value.parameters()),
                observed_neighbor_mean=float(actual_neighbors.mean()),
                observed_more_than_three_fraction=float((actual_neighbors>3).mean()),
                logp_recompute_abs_max=float(np.abs(logp-chain.logp).max()),
                command_norm_mean=float(np.linalg.norm(action.movement,axis=-1).mean()),
                executed_speed_factor_mean=float(np.mean([env.speed_factor.mean() for env in vector.envs])),
                optimization_updates=learner.updates,formal_training_started=False))
        finally:
            vector.close()
    result=dict(status='passed',groups=rows,formal_training_started=False)
    atomic_json(resolve_output(args.output),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
