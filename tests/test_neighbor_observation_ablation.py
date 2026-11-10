import copy
import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime, seed_all, load_checkpoint
from uav_research.observations import as_tensor_observation
from uav_research.train import train
from uav_research.evaluate import Baseline, evaluate, policy_from_checkpoint
from uav_research.deployment import export_actor, DeployedActor


def config(slots):
    return load_config(f'configs/mappo_capacity10_neighbors{slots}_1m_server.yaml')


def dense_positions(cfg):
    near=np.array([[100,100],[110,100],[90,100],[100,110],[100,90],
                   [120,100],[80,100],[100,120],[100,80]],np.float64)
    far=np.array([[2000+25*i,2000] for i in range(cfg.env.num_uavs-9)],np.float64)
    return np.concatenate([near,far])


def test_configuration_differences_and_dimensions():
    base=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    for slots,fixed in [(4,129),(5,153),(6,177),(7,201),(8,225)]:
        cfg=config(slots)
        expected=base.as_dict();expected['env']['neighbor_slots']=slots;expected['run']['device']='cuda:0'
        assert cfg.as_dict()==expected
        assert cfg.env.fixed_obs_dim==fixed and cfg.env.state_dim==6292
        assert cfg.env.robustness_weight==cfg.env.disconnection_weight==0
    for invalid in [0,-1,True,1.5]:
        bad=copy.deepcopy(base);bad.env.neighbor_slots=invalid
        with pytest.raises(ValueError,match='neighbor_slots'):bad.validate()


@pytest.mark.parametrize('slots',[4,5,6,7,8])
@pytest.mark.parametrize('backend',['numpy','numba'])
def test_neighbor_order_padding_and_physics(slots,backend):
    cfg=config(slots);cfg.env.backend=backend;cfg.run.device='cpu';configure_runtime(cfg)
    base=copy.deepcopy(cfg);base.env.neighbor_slots=3
    x,y=UAVEnvironment(cfg.env),UAVEnvironment(base.env)
    positions=dense_positions(cfg)
    users=np.column_stack([np.linspace(550,2900,150),np.full(150,1600)])
    ox,sx,_=x.set_positions(positions,users);oy,sy,_=y.set_positions(positions,users)
    np.testing.assert_array_equal(sx,sy)
    start=cfg.env.self_dim+2*cfg.env.num_bs
    neighbors=ox.fixed[0,start:cfg.env.current_obs_dim].reshape(slots,cfg.env.neighbor_dim)
    np.testing.assert_allclose(neighbors[:,0],np.arange(2,slots+2)/18)
    np.testing.assert_allclose(neighbors[:,1:3],(positions[1:slots+1]-positions[0])/3500)
    # Neighbor 2 is closer than neighbor 5, independent of UE/service ordering.
    for _ in range(3):
        actions=np.full((18,2),.02,np.float32)
        ax,ay=x.step(actions),y.step(actions)
        np.testing.assert_array_equal(ax[1],ay[1]);assert ax[2]==ay[2]
        np.testing.assert_array_equal(ax[0].fixed[:,:base.env.current_obs_dim],ay[0].fixed[:,:base.env.current_obs_dim])
        np.testing.assert_array_equal(ax[0].fixed[:,-4:],ay[0].fixed[:,-4:])
        np.testing.assert_array_equal(ax[0].users,ay[0].users)
    # An isolated UAV has zero neighbor slots even though it can sense UE.
    positions[0]=[3400,3400]
    obs,_,_=x.set_positions(positions,users)
    assert not obs.fixed[0,start:cfg.env.current_obs_dim].any()


@pytest.mark.parametrize('slots',[4,5,6,7,8])
def test_controlled_initialization_rng_and_new_feature_gradients(slots):
    cfg=config(slots);cfg.run.device='cpu';device=configure_runtime(cfg)
    base=copy.deepcopy(cfg);base.env.neighbor_slots=3
    seed_all(0);reference=make_learner(base,device);rng=torch.get_rng_state().clone()
    seed_all(0);expanded=make_learner(cfg,device)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in reference.value.state_dict().items():assert torch.equal(v,expanded.value.state_dict()[k])
    a,b=reference.actor.net[0],expanded.actor.net[0]
    point=base.env.current_obs_dim;extra=(slots-3)*cfg.env.neighbor_dim
    assert torch.equal(a.weight[:,:point],b.weight[:,:point])
    assert torch.equal(a.weight[:,point:],b.weight[:,point+extra:])
    assert not b.weight[:,point:point+extra].any()
    for layer in [2,4]:
        assert torch.equal(reference.actor.net[layer].weight,expanded.actor.net[layer].weight)
    users=np.column_stack([np.linspace(550,2900,150),np.full(150,1600)])
    obs,st,_=UAVEnvironment(cfg.env).set_positions(dense_positions(cfg),users)
    old,_,_=UAVEnvironment(base.env).set_positions(dense_positions(base),users)
    n=expanded.actor.distribution(as_tensor_observation(obs,device))
    o=reference.actor.distribution(as_tensor_observation(old,device))
    torch.testing.assert_close(n.loc,o.loc,rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(n.scale,o.scale,rtol=1e-5,atol=1e-7)
    seed_all(123);expected=reference.act(old)
    seed_all(123);actual=expanded.act(obs)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
    _,_,latent,_=expanded.collect(obs,st)
    logp,entropy=expanded.actor.evaluate_latent(as_tensor_observation(obs,device),torch.as_tensor(latent))
    (logp.mean()+.01*entropy.mean()).backward()
    assert b.weight.grad[:,point:point+extra].norm()>0
    assert sum(p.grad.norm().item() for p in expanded.actor.encoder.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in expanded.value.parameters())


@pytest.mark.parametrize('slots',[4,5,6,7,8])
def test_full_train_resume_export_and_evaluation(tmp_path,slots):
    cfg=config(slots);cfg.run.device='cpu'
    cfg.env.num_uavs=6;cfg.env.num_users=18;cfg.env.horizon=8
    cfg.run.num_envs=2;cfg.run.total_steps=32;cfg.run.eval_interval=32
    cfg.run.validation_seeds=[30000,30001];cfg.run.tensorboard=False
    cfg.algo.rollout_steps=8;cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=2
    full=train(cfg,tmp_path/'full')
    part=copy.deepcopy(cfg);part.run.total_steps=16
    interrupted=train(part,tmp_path/'split')
    resumed=train(cfg,tmp_path/'split',interrupted['resume_checkpoint'])
    from test_online_time_factor import assert_exact
    a,b=load_checkpoint(full['resume_checkpoint']),load_checkpoint(resumed['resume_checkpoint'])
    assert_exact(a['learner'],b['learner']);assert_exact(a['environments'],b['environments'])
    ecfg,model=policy_from_checkpoint(tmp_path/'full/last.pt','cpu')
    obs,_,_=UAVEnvironment(ecfg.env).reset(30000)
    export_actor(tmp_path/'full/last.pt',tmp_path/'actor.pt')
    deployed=DeployedActor(tmp_path/'actor.pt')
    np.testing.assert_array_equal(model.act(obs,True),deployed.act(obs,True))
    evaluate(ecfg,model,[30000],tmp_path/'evaluation',trace_count=1,reward_curves=True)
    assert np.isfinite(Baseline(ecfg,'heuristic').act(obs)).all()
    bad=copy.deepcopy(cfg);bad.env.neighbor_slots=5 if slots==4 else 4
    with pytest.raises(ValueError):train(bad,tmp_path/'incompatible',full['resume_checkpoint'])
