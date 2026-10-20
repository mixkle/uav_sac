from dataclasses import replace
import numpy as np
import pytest
from uav_research.config import Config, EnvConfig, config_from_dict, load_config
from uav_research.environment import UAVEnvironment
from uav_research.vector import VectorEnvironment
from uav_research.evaluate import Baseline, evaluate
from uav_research.algorithms import make_learner
from uav_research.replay import JointReplay
from uav_research.observations import stack_observations
from uav_research.runtime import configure_runtime, seed_all


def small_scene(**kwargs):
    return replace(EnvConfig(), width=1000, height=1000, num_uavs=2, num_users=1,
        bs_positions=[[0,0,0],[0,400,0]], uu_range=150, ug_range=160,
        user_speed_std=0, disconnection_weight=.7, **kwargs)


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('action_space',['continuous','discrete9'])
@pytest.mark.parametrize('positions,expected',[
    ([[50,50],[50,350]],0.),  # Separate components with different gateways.
    ([[50,50],[190,50]],0.),  # The second UAV is reachable only through a relay.
    ([[50,50],[900,900]],.5),
    ([[900,900],[920,900]],1.),  # Connected to one another, but no gateway.
])
def test_disconnection_penalty_uses_any_gateway_reachability(backend,action_space,positions,expected):
    e=small_scene(backend=backend,action_space=action_space)
    env=UAVEnvironment(e)
    _,_,info=env.set_positions(positions,[[50,50]])
    assert info['disconnection_ratio']==expected
    assert info['disconnected_uavs']==round(expected*2)
    assert info['disconnection_penalty']==pytest.approx(.7*expected)
    assert info['reward']==pytest.approx(info['coverage']+.1*info['tau']+.5*info['robustness']-.7*expected)
    frame=env.trace_frame()
    np.testing.assert_allclose(frame['reward_terms'],[info['coverage'],info['tau'],info['robustness'],expected])
    assert frame['reward_contributions'].sum()==pytest.approx(info['reward'])
    assert frame['reward']==info['reward']


@pytest.mark.parametrize('action_space',['continuous','discrete9'])
def test_penalty_counts_post_action_topology(action_space):
    env=UAVEnvironment(small_scene(action_space=action_space))
    env.set_positions([[120,0],[900,900]],[[800,500]])
    assert env.info['disconnection_ratio']==.5
    actions=np.array([3,8]) if action_space=='discrete9' else np.array([[1,0],[0,0]])
    _,_,reward,_,_,info=env.step(actions)
    assert info['disconnection_ratio']==1
    assert reward==pytest.approx(-.7)


@pytest.mark.parametrize('action_space',['continuous','discrete9'])
@pytest.mark.parametrize('backend',['batch','process'])
def test_vector_reward_and_restore_match_scalar_with_penalty(action_space,backend):
    e=small_scene(action_space=action_space)
    scalars=[UAVEnvironment(e),UAVEnvironment(e)]
    for env in scalars:env.set_positions([[120,0],[900,900]],[[800,500]])
    snapshots=[env.snapshot() for env in scalars]
    actions=np.tile([3,8],(2,1)) if action_space=='discrete9' else np.tile([[1,0],[0,0]],(2,1,1))
    with VectorEnvironment(e,2,backend) as vector:
        vector.restore(snapshots)
        actual=vector.step(actions)
        for i,env in enumerate(scalars):
            expected=env.step(actions[i])
            assert actual[2][i]==pytest.approx(expected[2])
            assert actual[-1][i]['disconnection_penalty']==expected[-1]['disconnection_penalty']
        vector.restore(snapshots)
        repeated=vector.step(actions)
        np.testing.assert_array_equal(actual[2],repeated[2])


def test_legacy_config_preserves_reward_and_invalid_weights_fail():
    old=config_from_dict({'env':{'throughput_weight':.2,'robustness_weight':.3}})
    assert old.env.disconnection_weight==0
    for bad in [-.1,float('inf'),float('nan')]:
        with pytest.raises(ValueError):Config(env=replace(EnvConfig(),disconnection_weight=bad)).validate()


def test_evaluation_records_negative_rewards_and_actual_trace_reward(tmp_path):
    cfg=load_config('configs/discrete_mappo_smoke.yaml')
    cfg.env.ug_range=1  # Below altitude: all UAVs are disconnected.
    cfg.env.disconnection_weight=.7;cfg.env.horizon=3
    configure_runtime(cfg)
    rows,_=evaluate(cfg,Baseline(cfg,'static'),[30000],tmp_path,trace_count=1,reward_curves=True)
    row=rows[0]
    assert row['return']==pytest.approx(-2.1)
    assert row['disconnection_ratio']==row['last100_disconnection_ratio']==row['final_disconnection_ratio']==1
    assert row['negative_reward_steps']==3
    assert row['zero_reward_steps']==row['zero_reward_initial_steps']==0
    assert row['first_positive_reward_step']==-1
    with np.load(tmp_path/'trace_30000.npz') as trace:
        np.testing.assert_allclose(trace['reward'],trace['reward_contributions'].sum(-1))
        assert trace['reward_terms'].shape==(4,4)
        np.testing.assert_allclose(trace['reward_terms']*trace['reward_weights'],trace['reward_contributions'])
    with np.load(tmp_path/'reward_curves.npz') as curves:
        assert curves['raw'].shape==(1,3,4)
        np.testing.assert_allclose(curves['contributions'].sum(-1),curves['reward'])
        assert curves['reward'].sum()==pytest.approx(row['return'])


@pytest.mark.parametrize('name',['sac','mappo'])
def test_learner_receives_negative_environment_rewards(name):
    cfg=load_config('configs/smoke.yaml' if name=='sac' else 'configs/discrete_mappo_smoke.yaml')
    cfg.env.disconnection_weight=.7;cfg.env.ug_range=1;cfg.env.horizon=3
    cfg.algo.batch_size=4;device=configure_runtime(cfg);seed_all(1)
    learner=make_learner(cfg,device)
    with VectorEnvironment(cfg.env,2) as vector:
        obs,state,_=vector.reset([30000,30001])
        replay=JointReplay(8,(cfg.env.num_uavs,cfg.env.obs_dim),cfg.env.state_dim) if name=='sac' else None
        records=[]
        for _ in range(3):
            if name=='sac':action=learner.act(obs)
            else:action,logp,latent,value=learner.collect(obs,state)
            following=vector.step(action);next_obs,next_state,reward,done,truncated,_=following
            np.testing.assert_allclose(reward,-.7)
            if replay is not None:replay.add(obs,state,action,reward,next_obs,next_state,done)
            else:records.append(dict(obs=obs,state=state,actions=action,logp=logp,value=value,
                next_value=learner.values(next_state).cpu().numpy(),reward=reward,terminated=done,truncated=truncated))
            obs,state=next_obs,next_state
        if replay is not None:
            batch=replay.sample(4,device)
            np.testing.assert_allclose(batch['reward'].cpu().numpy(),-.7)
        else:
            batch={k:stack_observations([r[k] for r in records]) if k=='obs' else np.stack([r[k] for r in records]) for k in records[0]}
        metrics=learner.update(batch)
        assert all(np.isfinite(list(metrics.values())))
