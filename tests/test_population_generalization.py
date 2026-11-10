import copy

import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.environment import UAVEnvironment,physical_links
from uav_research.models import GaussianActor
from uav_research.observations import stack_observations
from uav_research.population_generalization import (
    PopulationEnvironment,PopulationVector,retained_uav_ids,embed_observation,paired_action,
)
from uav_research.runtime import configure_runtime,seed_all


def source(slots=3):
    cfg=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    cfg.env.neighbor_slots=slots;cfg.run.cpu_threads=1
    configure_runtime(cfg)
    return cfg.env


@pytest.mark.parametrize('slots',[3,4,5])
@pytest.mark.parametrize('u',[14,10])
def test_embedding_identity_hops_and_missing_nodes(slots,u):
    trained=source(slots);active=copy.deepcopy(trained);active.num_uavs=u
    env=PopulationEnvironment(active,trained);obs,_,_=env.reset(10000)
    # Dense deployment exercises all neighbor slots, plus an isolated/disconnected node.
    xy=np.column_stack([100+np.arange(u)*10,np.full(u,100)])
    xy[-1]=[3400,3400]
    obs,_,_=env.set_positions(xy,env.user_xy)
    out=embed_observation(obs,active,trained,env.active_ids)
    assert out.fixed.shape==(u,trained.fixed_obs_dim)
    np.testing.assert_array_equal(out.users,obs.users)
    missing=np.setdiff1d(np.arange(18),env.active_ids)
    assert not out.fixed[:,5+missing].any()
    np.testing.assert_array_equal(out.fixed[:,5+env.active_ids],env.A)
    np.testing.assert_allclose(out.fixed[:-1,3],1/19,atol=1e-7)
    assert out.fixed[-1,3]==1 and out.fixed[-1,-2]==1
    start=trained.self_dim+2*trained.num_bs
    native=active.self_dim+2*active.num_bs
    for slot in range(slots):
        a=native+slot*active.neighbor_dim;b=start+slot*trained.neighbor_dim
        for i in range(u):
            if obs.fixed[i,a]==0:
                assert not out.fixed[i,b:b+trained.neighbor_dim].any()
            else:
                local=int(round(float(obs.fixed[i,a])*u))-1
                assert out.fixed[i,b]==pytest.approx((env.active_ids[local]+1)/18)
                np.testing.assert_array_equal(out.fixed[i,b+6+env.active_ids],env.A[local])
                assert not out.fixed[i,b+6+missing].any()
                assert out.fixed[i,b+4]==pytest.approx(1/19)
    assert not out.fixed[-1,start:trained.current_obs_dim].any()
    # The same conversion works with environment batches and differently retained IDs.
    other=PopulationEnvironment(active,trained);o,_,_=other.reset(10001)
    batched=embed_observation(stack_observations([obs,o]),active,trained,
                              np.stack([env.active_ids,other.active_ids]))
    np.testing.assert_array_equal(batched.fixed[0],out.fixed)


def test_subsets_are_nested_and_independent_of_policy():
    for seed in range(10):
        ten=retained_uav_ids(seed,10);fourteen=retained_uav_ids(seed,14)
        assert set(ten).issubset(fourteen)
        np.testing.assert_array_equal(ten,retained_uav_ids(seed,10))
        assert np.all(np.diff(ten)>0)


@pytest.mark.parametrize('u,m',[(18,150),(18,90),(18,120),(14,150),(10,150)])
def test_paired_initial_positions_and_ue_motion(u,m):
    trained=source();cfg=copy.deepcopy(trained);cfg.num_uavs=u;cfg.num_users=m
    full=UAVEnvironment(trained);old,_,_=full.reset(10001)
    env=PopulationEnvironment(cfg,trained);obs,_,_=env.reset(10001)
    ids=env.active_ids
    np.testing.assert_array_equal(env.uav_xy,full.uav_xy[ids])
    np.testing.assert_array_equal(env.user_xy,full.user_xy[:m])
    for _ in range(8):
        command=np.linspace(-.1,.1,36,dtype=np.float32).reshape(18,2)
        reference=full.step(command);actual=env.step(command[ids])
        np.testing.assert_array_equal(env.user_xy,full.user_xy[:m])
        np.testing.assert_array_equal(env.uav_xy,full.uav_xy[ids])
        assert actual[2]==pytest.approx(env.info['coverage']+.1*env.info['tau'])
        assert env.info['coverage']<=min(1,u*10/m)+1e-12
        if (u,m)==(18,150):
            for index in (1,2,3,4):np.testing.assert_array_equal(actual[index],reference[index])
            np.testing.assert_array_equal(actual[0].fixed,reference[0].fixed)


@pytest.mark.parametrize('slots',[3,4,5])
def test_original_policy_identity_and_shared_noise(slots):
    trained=source(slots);seed_all(0)
    actor=GaussianActor(trained.fixed_obs_dim,256,ue_hidden=32,count_scale=10)
    vector=PopulationVector(trained,trained,2)
    obs,_,_=vector.reset([10000,10001])
    before={k:v.clone() for k,v in actor.state_dict().items()}
    seed_all(880001);expected=actor.sample(obs.to_torch('cpu'))[0].detach().numpy()
    seed_all(880001);actual=paired_action(actor,obs,trained,trained,vector.active_ids)
    np.testing.assert_array_equal(actual,expected)
    for k,v in actor.state_dict().items():assert torch.equal(v,before[k])
    assert all(p.grad is None for p in actor.parameters())
    vector.close()


def test_removed_uavs_do_not_interfere_or_consume_capacity():
    trained=source();cfg=copy.deepcopy(trained);cfg.num_uavs=10
    full=UAVEnvironment(trained);full.reset(10000)
    reduced=PopulationEnvironment(cfg,trained);reduced.reset(10000)
    full_y=physical_links(trained,full.uav_xy,full.user_xy)[2]
    small_y=physical_links(cfg,reduced.uav_xy,reduced.user_xy)[2]
    assert (small_y>=full_y[reduced.active_ids]-1e-10).all()
    assert reduced.A.shape==(10,10) and reduced.C.shape==(10,150)


def test_original_embedding_is_passthrough():
    cfg=source();obs,_,_=UAVEnvironment(cfg).reset(10000)
    assert embed_observation(obs,cfg,cfg,np.arange(18)) is obs
    with pytest.raises(ValueError):embed_observation(obs,cfg,cfg,np.zeros(18,dtype=np.int64))


@pytest.mark.parametrize('u',[14,10])
def test_multihop_history_and_common_noise_after_removal(u):
    trained=source(4);cfg=copy.deepcopy(trained);cfg.num_uavs=u
    env=PopulationEnvironment(cfg,trained);obs,_,_=env.reset(10000)
    obs.fixed[:,3]=5/(u+1);obs.fixed[:,-2]=7/(u+1)
    embedded=embed_observation(obs,cfg,trained,env.active_ids)
    np.testing.assert_allclose(embedded.fixed[:,3],5/19,atol=1e-7)
    np.testing.assert_allclose(embedded.fixed[:,-2],7/19,atol=1e-7)
    actor=GaussianActor(trained.fixed_obs_dim,32,ue_hidden=32,count_scale=10)
    with torch.no_grad():
        for p in actor.parameters():p.zero_()
    full,_,_=UAVEnvironment(trained).reset(10000)
    seed_all(7);a=paired_action(actor,full,trained,trained,np.arange(18))
    seed_all(7);b=paired_action(actor,obs,cfg,trained,env.active_ids)
    np.testing.assert_array_equal(b,a[env.active_ids])
