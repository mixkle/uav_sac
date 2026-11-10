import copy
import numpy as np
import pytest
import torch
from uav_research.algorithms import make_learner, soft_target, soft_update
from uav_research.config import load_config
from uav_research.runtime import configure_runtime, seed_all, load_checkpoint, isolated_rng
from uav_research.sac_diagnostics import soft_q_returns, action_moments
from uav_research.environment import UAVEnvironment
from uav_research.observations import as_tensor_observation
from uav_research.train import train
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.deployment import export_actor, DeployedActor
from test_online_time_factor import assert_exact


def small(mode):
    c=load_config('tests/fixtures/configs/smoke.yaml')
    c.algo.name='sac';c.algo.critic_parameterization=mode
    c.env.throughput_weight=.1;c.env.robustness_weight=.5;c.env.disconnection_weight=.1
    c.env.reward_primary='coverage'
    c.run.cpu_threads=8;c.run.total_steps=64;c.run.eval_interval=32
    c.run.log_interval=16;c.run.validation_seeds=[10000];c.run.evaluation_policy='sample'
    c.algo.warmup_steps=16;c.algo.batch_size=8;c.algo.replay_capacity=128
    return c.validate()


def test_paired_configs_and_scope():
    a=load_config('tests/fixtures/configs/sac_full_reward_direct_250k.yaml')
    b=load_config('tests/fixtures/configs/sac_full_reward_time_factor_250k.yaml')
    assert a.run.cpu_threads==b.run.cpu_threads==8
    assert a.run.evaluation_policy=='sample' and a.env.state_dim==6292
    assert a.algo.target_entropy==-36
    b.algo.critic_parameterization='direct';assert a.as_dict()==b.as_dict()
    for algo,space,primary in [('sac','discrete9','coverage'),('matd3','continuous','coverage'),('sac','continuous','sensing')]:
        c=small('time_factor');c.algo.name=algo;c.env.action_space=space;c.env.reward_primary=primary
        with pytest.raises(ValueError):c.validate()


def test_initialization_time_scale_action_gradient_and_target():
    c=small('time_factor');device=configure_runtime(c)
    d=small('direct')
    seed_all(99);direct=make_learner(d,device);rng=torch.get_rng_state().clone()
    seed_all(99);factor=make_learner(c,device)
    assert_exact(direct.actor.state_dict(),factor.actor.state_dict())
    assert_exact(rng,torch.get_rng_state())
    for key,value in direct.critic.state_dict().items():assert_exact(value,factor.critic.state_dict()[key])
    s=torch.randn(4,c.env.state_dim);s[:,-1]=torch.tensor([1.,0.,31/32,8/32])
    actions=torch.randn(4,c.env.num_uavs,2,requires_grad=True)
    h=factor.critic.horizon_factor(s);scale=h/factor.critic.horizon_factor.table[-1].float()
    qd=direct.critic(s,actions);qf=factor.critic(s,actions)
    for old,new in zip(qd,qf):torch.testing.assert_close(new,old*scale)
    gd=torch.autograd.grad(qd[0].sum(),actions,retain_graph=True)[0]
    gf=torch.autograd.grad(qf[0].sum(),actions)[0]
    torch.testing.assert_close(gf,gd*scale[:,None,None])
    assert qf[0][0]==0 and h[2]==1
    with torch.no_grad():factor.critic.q1[-1].weight.add_(.1)
    target_before=copy.deepcopy(factor.target.state_dict())
    soft_update(factor.target,factor.critic,.25)
    torch.testing.assert_close(factor.target.q1[-1].weight,target_before['q1.4.weight']*.75+factor.critic.q1[-1].weight*.25)
    assert_exact(factor.critic.horizon_factor.state_dict(),factor.target.horizon_factor.state_dict())
    target=soft_target(torch.tensor([-2.,3.]),torch.tensor([True,False]),torch.tensor([9.,9.]),torch.tensor([-1.,-1.]),.9,.2)
    torch.testing.assert_close(target,torch.tensor([-2.,11.28]))
    legacy=direct.state_dict();legacy.pop('critic_parameterization');direct.load_state_dict(legacy)
    with pytest.raises(ValueError,match='parameterization'):factor.load_state_dict(legacy)
    with pytest.raises(ValueError,match='parameterization'):direct.load_state_dict(factor.state_dict())


def test_soft_q_mc_labels_and_rng_isolation():
    rewards=np.array([[-.5,2.,.3],[1.,-.2,4.]])
    logp=np.array([[999.,-2.,1.],[-999.,.5,-.2]])
    g=soft_q_returns(rewards,logp,.9,.4)
    np.testing.assert_array_equal(g[:,-1],rewards[:,-1])
    np.testing.assert_allclose(g[:,:-1],rewards[:,:-1]+.9*(g[:,1:]-.4*logp[:,1:]))
    changed=logp.copy();changed[:,0]=0
    np.testing.assert_array_equal(g,soft_q_returns(rewards,changed,.9,.4))
    np.testing.assert_array_equal(soft_q_returns(rewards,logp,0,.4),rewards)
    from uav_research.critic_calibration import discounted_episode_returns
    np.testing.assert_array_equal(soft_q_returns(rewards,logp,.9,0),discounted_episode_returns(rewards,.9))
    with pytest.raises(ValueError):soft_q_returns(rewards,logp[:,:1],.9,.4)
    c=small('time_factor');model=make_learner(c,configure_runtime(c))
    obs,_,_=UAVEnvironment(c.env).reset(10000)
    before=torch.get_rng_state().clone()
    with isolated_rng(),torch.inference_mode():
        m=action_moments(model.actor,as_tensor_observation(obs,'cpu'),10,8)
        assert all(torch.isfinite(v).all() for v in m.values())
    assert_exact(before,torch.get_rng_state())


@pytest.mark.parametrize('mode',['direct','time_factor'])
def test_full_pipeline_resume_export_and_audit(tmp_path,mode,monkeypatch):
    c=small(mode)
    full=train(c,tmp_path/'full')
    split=copy.deepcopy(c);split.run.total_steps=32
    first=train(split,tmp_path/'split')
    final=train(c,tmp_path/'split',first['resume_checkpoint'])
    a,b=load_checkpoint(full['resume_checkpoint']),load_checkpoint(final['resume_checkpoint'])
    for k in ['learner','environments','rng','replay_metadata']:assert_exact(a[k],b[k])
    ec,policy=policy_from_checkpoint(tmp_path/'full'/'last.pt','cpu')
    _,summary=evaluate(ec,policy,[10000],tmp_path/'eval',trace_count=1,reward_curves=True)
    assert np.isfinite(summary['mean']['return'])
    export_actor(tmp_path/'full'/'last.pt',tmp_path/'actor.pt')
    deployed=DeployedActor(tmp_path/'actor.pt');obs,_,_=UAVEnvironment(c.env).reset(10000)
    np.testing.assert_array_equal(deployed.act(obs),policy.act(obs,True))
    wrong=copy.deepcopy(c);wrong.algo.critic_parameterization='direct' if mode=='time_factor' else 'time_factor'
    with pytest.raises(ValueError,match='Environment/algorithm changed'):train(wrong,tmp_path/'wrong',full['resume_checkpoint'])
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_sac_time_factor import audit, plot_audit
    data,metrics=audit(tmp_path/'full','last',tmp_path/'audit',moment_samples=8)
    np.testing.assert_allclose(data['rewards'].sum(1).mean(),summary['mean']['return'])
    np.testing.assert_allclose(data['component_returns'].sum(-1),data['environment_returns'])
    assert metrics['q1']['value']['stages']['all']['rmse']>=0
    plot_audit(data,tmp_path/'plots','last')
