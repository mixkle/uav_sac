"""Controlled initialization, independent PPO gradients, team reward and recovery."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from test_history_observation import config as history_config
from test_online_time_factor import assert_exact
from uav_research.algorithms import gae, make_learner
from uav_research.config import load_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import policy_from_checkpoint
from uav_research.models import disk_transform
from uav_research.observations import RawObservation, stack_observations
from uav_research.runtime import configure_runtime, load_checkpoint, seed_all
from uav_research.train import train
from uav_research.vector import VectorEnvironment


def config(sharing='shared'):
    cfg=history_config()
    cfg.algo.actor_sharing=sharing
    for name in ['throughput_weight','robustness_weight','disconnection_weight','boundary_penalty_weight']:
        setattr(cfg.env,name,0.)
    cfg.run.best_metric='task_return'
    cfg.run.tensorboard=False
    return cfg.validate()


def rollout(cfg, learner, ticks=4):
    records=[]
    with VectorEnvironment(cfg.env,cfg.run.num_envs) as vector:
        obs,state,_=vector.reset([30020+i for i in range(cfg.run.num_envs)])
        for _ in range(ticks):
            action,logp,latent,value=learner.collect(obs,state)
            no,ns,reward,done,truncated,_=vector.step(action)
            records.append(dict(obs=obs.copy(),state=state.copy(),latent=latent,logp=logp,value=value,
                next_value=learner.values(ns).numpy(),reward=reward,terminated=done,truncated=truncated))
            obs,state=no,ns
    return {key:stack_observations([r[key] for r in records]) if key=='obs' else
            np.stack([r[key] for r in records]) for key in records[0]}


def test_config_pair_and_scope():
    configs=[load_config(f'tests/fixtures/configs/continuous_mappo_coverage_{s}_central_history_100k.yaml')
             for s in ['shared','independent']]
    assert (configs[0].env.obs_dim, configs[0].env.state_dim)==(105,6292)
    assert configs[0].run.total_steps==configs[1].run.total_steps==100000
    configs[1].algo.actor_sharing='shared'
    assert configs[0].as_dict()==configs[1].as_dict()
    for name,space in [('sac','continuous'),('mappo','discrete9'),('matd3','continuous')]:
        cfg=config('independent');cfg.algo.name=name;cfg.env.action_space=space
        cfg.env.observation_history='none'
        with pytest.raises(ValueError,match='Independent actors require'):cfg.validate()


def test_ragged_agent_groups_empty_batches_and_extra_batch_axis():
    fixed=np.arange(2*3*5,dtype=np.float32).reshape(2,3,5)
    lengths=np.array([0,2,1,3,0,1]);users=np.arange(7*4,dtype=np.float32).reshape(7,4)
    raw=RawObservation(fixed,users,np.r_[0,lengths.cumsum()])
    parts=raw.to_torch('cpu').split_agents()
    for i,part in enumerate(parts):
        np.testing.assert_array_equal(part.fixed,fixed[:,i:i+1])
        np.testing.assert_array_equal(part.lengths,lengths.reshape(2,3)[:,i])
        expected=np.concatenate([users[raw.offsets[j]:raw.offsets[j+1]] for j in [i,i+3]])
        np.testing.assert_array_equal(part.users,expected)
    multi=stack_observations([raw,raw]).to_torch('cpu').split_agents()
    assert multi[1].fixed.shape==(2,2,1,5)
    empty=RawObservation(fixed,np.empty((0,4),np.float32),np.zeros(7,np.int64))
    for part in empty.to_torch('cpu').split_agents():
        assert len(part.users)==0 and len(part.lengths)==2


def test_initial_actor_rng_critic_math_and_gradient_isolation():
    cfg=config();configure_runtime(cfg);seed_all(0)
    shared=make_learner(cfg,'cpu');rng=torch.get_rng_state().clone()
    other=config('independent');seed_all(0);independent=make_learner(other,'cpu')
    assert_exact(torch.get_rng_state(),rng)
    assert_exact(shared.value.state_dict(),independent.value.state_dict())
    modules=independent.actor.actors
    assert len({p.data_ptr() for actor in modules for p in actor.parameters()})==sum(1 for actor in modules for _ in actor.parameters())
    for actor in modules:assert_exact(actor.state_dict(),shared.actor.state_dict())
    raw,state,_=UAVEnvironment(cfg.env).reset(30020)
    obs=stack_observations([raw,raw]).to_torch('cpu')
    sa,ia=shared.actor.distribution(obs),independent.actor.distribution(obs)
    torch.testing.assert_close(sa.loc,ia.loc,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(sa.scale,ia.scale,atol=1e-7,rtol=1e-6)
    torch.manual_seed(44);a=shared.actor.sample(obs)
    torch.manual_seed(44);b=independent.actor.sample(obs)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=3e-7,rtol=1e-5)
    z=b[2].detach();old=b[1].detach()
    lp,entropy=independent.actor.evaluate_latent(obs,z)
    torch.testing.assert_close((lp-old).exp(),torch.ones_like(old),atol=1e-6,rtol=1e-6)
    normal=ia.log_prob(z).sum(-1)
    torch.testing.assert_close(lp,normal+2*torch.log1p(z.square().sum(-1)))
    loss=-(lp[:,1]+.01*entropy[:,1]).mean()
    loss.backward()
    for i,actor in enumerate(modules):
        magnitude=sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
        assert (magnitude>0) if i==1 else magnitude==0
    assert all(p.grad is None for p in independent.value.parameters())
    # Empty observation batches also retain every UAV slot.
    empty=RawObservation(raw.fixed,np.empty((0,4),np.float32),np.zeros(cfg.env.num_uavs+1,np.int64))
    assert independent.actor.sample(empty.to_torch('cpu'))[0].shape==(cfg.env.num_uavs,2)


def test_independent_update_matches_per_actor_mean_not_joint_mean():
    cfg=config('independent');cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=1
    configure_runtime(cfg);seed_all(51);model=make_learner(cfg,'cpu');data=rollout(cfg,model)
    reference=copy.deepcopy(model);n=data['state'].shape[0]*data['state'].shape[1]
    value,next_value,reward,done,trunc=[torch.as_tensor(data[k]) for k in
                                     ['value','next_value','reward','terminated','truncated']]
    adv,_=gae(reward,value,next_value,done,cfg.algo.discount,cfg.algo.gae_lambda,trunc)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-5)
    seed_all(941);indices=torch.randperm(n)
    obs=data['obs'].flatten_batch()[indices.numpy()].to_torch('cpu')
    latent=torch.as_tensor(data['latent']).flatten(0,1)[indices]
    old=torch.as_tensor(data['logp']).flatten(0,1)[indices]
    lp,ent=reference.actor.evaluate_latent(obs,latent)
    ratio=(lp-old).exp();advantages=adv.flatten()[indices,None]
    loss=-torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean(0)-cfg.algo.entropy_coef*ent.mean(0)
    for opt in reference.actor_optimizers:opt.zero_grad(set_to_none=True)
    loss.sum().backward()
    for actor,opt in zip(reference.actor.actors,reference.actor_optimizers):
        torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg.algo.max_grad_norm);opt.step()
    seed_all(941);metrics=model.update(data)
    assert_exact(model.actor.state_dict(),reference.actor.state_dict())
    assert metrics['actor_loss']==pytest.approx(float(loss.detach().mean()),abs=1e-6)
    for opt in [*model.actor_optimizers,model.value_opt]:
        assert all(int(s['step'])==1 for s in opt.state.values())
    for actor in model.actor.actors:
        grad=actor.net[0].weight.grad[:,cfg.env.current_obs_dim:cfg.env.current_obs_dim+4]
        assert grad.norm()>0
    assert any(not torch.equal(model.actor.actors[0].state_dict()[k],model.actor.actors[1].state_dict()[k])
               for k in model.actor.actors[0].state_dict())


@pytest.mark.parametrize('backend',['batch','process'])
def test_reward_decomposition_vector_paths(backend):
    cfg=config();cfg.env.horizon=3;configure_runtime(cfg)
    scalars=[UAVEnvironment(cfg.env) for _ in range(2)]
    with VectorEnvironment(cfg.env,2,backend) as vector:
        vector.reset([81,82])
        for env,seed in zip(scalars,[81,82]):env.reset(seed)
        for _ in range(3):
            commands=np.full((2,cfg.env.num_uavs,2),.1,np.float32)
            result=vector.step(commands)
            for env,command,info,team in zip(scalars,commands,result[-1],result[2]):
                reference=env.step(command)[-1]
                np.testing.assert_array_equal(info['agent_coverage'],reference['agent_coverage'])
                assert sum(info['agent_coverage'])==pytest.approx(info['coverage'])
                assert team==pytest.approx(info['coverage'])
                assert info['coverage']==float(env.C.sum()/cfg.env.num_users)
                assert (env.C.sum(axis=0)<=1).all()
                assert info['coverage']==pytest.approx(env.C.any(axis=0).mean())
                np.testing.assert_array_equal(env.trace_frame()['agent_coverage'],info['agent_coverage'])


@pytest.mark.parametrize('sharing',['shared','independent'])
def test_training_resume_legacy_and_actor_export(tmp_path,sharing):
    cfg=config(sharing);cfg.run.total_steps=64
    full=train(cfg,tmp_path/'full')
    half=copy.deepcopy(cfg);half.run.total_steps=32
    partial=train(half,tmp_path/'split');resumed=train(cfg,tmp_path/'split',partial['resume_checkpoint'])
    a,b=[load_checkpoint(r['resume_checkpoint']) for r in [full,resumed]]
    for key in ['learner','environments','rng','state']:assert_exact(a[key],b[key])
    for key in ['fixed','users','offsets']:assert_exact(getattr(a['obs'],key),getattr(b['obs'],key))
    pcfg,policy=policy_from_checkpoint(tmp_path/'full/last.pt','cpu')
    export_actor(tmp_path/'full/last.pt',tmp_path/'actor.pt');actor=DeployedActor(tmp_path/'actor.pt')
    raw,*_=UAVEnvironment(cfg.env).reset(30020)
    np.testing.assert_array_equal(actor.act(raw),policy.act(raw,True))
    wrong=config('independent' if sharing=='shared' else 'shared')
    with pytest.raises(ValueError,match='actor_sharing'):
        make_learner(wrong,'cpu').load_state_dict(a['learner'])
    if sharing=='shared':
        legacy=copy.deepcopy(a['learner']);legacy.pop('actor_sharing')
        policy.load_state_dict(legacy)
    else:
        damaged=copy.deepcopy(a['learner']);damaged['actor_agent_ids'].reverse()
        with pytest.raises(ValueError,match='count/order'):policy.load_state_dict(damaged)


def test_actor_sharing_report_and_rebuild(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_mappo_actor_sharing import build,publish,validate_configs
    configs=[config(mode) for mode in ['shared','independent']]
    validate_configs(configs)
    wrong=copy.deepcopy(configs);wrong[1].env.throughput_weight=.1
    with pytest.raises(ValueError,match='Only actor_sharing'):validate_configs(wrong)
    runs=[tmp_path/mode for mode in ['shared','independent']]
    for cfg,run in zip(configs,runs):train(cfg,run)
    report=build(runs,tmp_path/'report',samples=8,action_seeds=[880001,880002])
    assert report['training_seed_count']==1
    with np.load(tmp_path/'report/independent/last/seed_880001/audit.npz') as d:
        np.testing.assert_allclose(d['agent_diagnostics'][...,0].sum(-1),d['rewards'],atol=1e-14)
        assert d['common_means'].shape[2]==configs[0].env.num_uavs
        assert d['common_observation_times'].tolist()==[0,configs[0].env.horizon-1]
    # Rebuild must not sample an action or change the Torch RNG.
    before=torch.get_rng_state().clone()
    publish(tmp_path/'report',report)
    assert_exact(before,torch.get_rng_state())
    assert (tmp_path/'report/figures/deployment_last.png').exists()
    assert (tmp_path/'report/independent/figures/topology_last_30020.png').exists()
