"""Matched PPO/SAC history inputs, replay semantics and exact CPU recovery."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_online_time_factor import assert_exact
from test_sac_time_factor import small
from uav_research.algorithms import make_learner
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import load_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import policy_from_checkpoint
from uav_research.replay import JointReplay
from uav_research.runtime import configure_runtime, load_checkpoint, seed_all
from uav_research.train import train
from uav_research.vector import VectorEnvironment


def config():
    c=small('time_factor')
    c.env.observation_history='one_step';c.env.observation_coordinates='relative'
    c.env.user_initialization='uniform_interior';c.env.user_initial_margin_fraction=1/7
    c.env.throughput_weight=c.env.robustness_weight=c.env.disconnection_weight=0.
    c.env.boundary_penalty_weight=0.;c.run.best_metric='task_return'
    c.run.validation_seeds=[30020];c.run.numba_threads=4
    return c.validate()


def test_formal_configuration_and_paired_actor():
    sac=load_config('tests/fixtures/configs/sac_coverage_time_factor_relative_central_history_1m.yaml')
    ppo=load_config('tests/fixtures/configs/continuous_mappo_coverage_shared_central_history_100k.yaml')
    assert sac.env==ppo.env
    ppo.run.total_steps=1000000;assert sac.run==ppo.run
    assert sac.algo.name=='sac' and sac.algo.actor_sharing=='shared'
    assert (sac.env.obs_dim,sac.env.state_dim)==(105,6292)
    assert sac.algo.target_entropy==-36 and sac.run.cpu_threads==8
    # Small matched models avoid constructing a production replay during tests.
    c=config();m=copy.deepcopy(c);m.algo.name='mappo';m.algo.mappo_state_dependent_std=True
    device=configure_runtime(c)
    seed_all(0);a=make_learner(c,device)
    seed_all(0);b=make_learner(m.validate(),device)
    assert_exact(a.actor.state_dict(),b.actor.state_dict())
    obs,_,_=UAVEnvironment(c.env).reset(30020)
    seed_all(42);x=a.act(obs)
    seed_all(42);y=b.act(obs)
    np.testing.assert_array_equal(x,y)
    for name,space in [('matd3','continuous'),('sac','discrete9')]:
        wrong=copy.deepcopy(c);wrong.algo.name=name;wrong.env.action_space=space
        with pytest.raises(ValueError,match='history'):wrong.validate()


def test_history_initialization_preserves_old_sac_and_rejects_cross_load():
    c=config();old=copy.deepcopy(c);old.env.observation_history='none'
    device=configure_runtime(c)
    seed_all(22);a=make_learner(old,device);rng=torch.get_rng_state().clone()
    seed_all(22);b=make_learner(c,device)
    assert_exact(rng,torch.get_rng_state())
    assert_exact(a.critic.state_dict(),b.critic.state_dict())
    assert_exact(a.target.state_dict(),b.target.state_dict())
    first=b.actor.net[0].weight;d=c.env.current_obs_dim
    assert torch.count_nonzero(first[:,d:d+4])==0
    e=UAVEnvironment(c.env);old_e=UAVEnvironment(replace(c.env,observation_history='none'))
    obs,_,_=e.reset(52);raw,_,_=old_e.reset(52)
    torch.testing.assert_close(a.actor.distribution(raw.to_torch('cpu')).loc,
                               b.actor.distribution(obs.to_torch('cpu')).loc)
    with pytest.raises(ValueError,match='observation_history'):a.load_state_dict(b.state_dict())
    with pytest.raises(ValueError,match='observation_history'):b.load_state_dict(a.state_dict())
    legacy=a.state_dict();legacy.pop('observation_history');a.load_state_dict(legacy)


def test_replay_history_alignment_and_sac_gradient():
    c=config();model=make_learner(c,configure_runtime(c));e=c.env
    replay=JointReplay(32,(e.num_uavs,e.obs_dim),e.state_dim,seed=55)
    with VectorEnvironment(e,2,'batch') as v:
        obs,state,_=v.reset([30020,30021])
        saved=obs.copy()
        for _ in range(4):
            actions=np.full((2,e.num_uavs,2),.2,np.float32)
            no,ns,r,done,_,_=v.step(actions)
            replay.add(obs,state,actions,r,no,ns,done)
            obs,state=no,ns
        batch=replay.sample(8,'cpu')
        torch.testing.assert_close(batch['next_obs'].fixed[...,-4:-2],batch['actions'])
        np.testing.assert_array_equal(replay.arrays['obs'][0],saved.fixed[0])
        metrics=model.update(batch)
        assert all(np.isfinite(x) for x in metrics.values())
        w=model.actor.net[0].weight;d=e.current_obs_dim
        assert w.grad[:,d:d+4].norm()>0 and w[:,d:d+4].norm()>0
        assert any(p.grad is not None and p.grad.norm()>0 for p in model.actor.encoder.parameters())


def test_exact_resume_export_and_soft_q_audit(tmp_path):
    c=config();full=train(c,tmp_path/'full')
    part=copy.deepcopy(c);part.run.total_steps=32
    first=train(part,tmp_path/'split');continued=train(c,tmp_path/'split',first['resume_checkpoint'])
    a,b=load_checkpoint(full['resume_checkpoint']),load_checkpoint(continued['resume_checkpoint'])
    for key in ['learner','environments','rng','replay_metadata']:
        assert_exact(a[key],b[key])
    cfg,policy=policy_from_checkpoint(tmp_path/'full'/'last.pt','cpu')
    export_actor(tmp_path/'full'/'last.pt',tmp_path/'actor.pt')
    deployed=DeployedActor(tmp_path/'actor.pt');obs,_,_=UAVEnvironment(c.env).reset(30020)
    np.testing.assert_array_equal(deployed.act(obs),policy.act(obs,True))
    data,_=audit_policy(cfg,policy,tmp_path/'audit',samples=8,trace_seeds=(30020,))
    np.testing.assert_allclose(data['rewards'],data['diagnostics'][...,0],atol=1e-12)
    np.testing.assert_array_equal(data['value_labels'][:,-1],data['rewards'][:,-1])
    assert np.isfinite(data['values']).all()
