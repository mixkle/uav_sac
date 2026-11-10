"""Scene isolation, release labels, paired errors, and offline resume semantics."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_mappo_release_values import (
    ReleaseCritic, scene_split, mc_tail, paired_error, scores, fit_epoch,
    save_critic, restore_critic, error_interval,
)
from check_mappo_generalization import actor_digest
from test_generalization_audit import config
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime, seed_all


def test_release_mc_alignment_signed_reward_and_pair_identity():
    rewards=np.array([[2.,-1.,3.,.5],[1.,2.,-.5,1.]])
    gamma=.9
    for k in [1,2,3]:
        tail=mc_tail(rewards,gamma,k)
        prefix=(rewards[:,:k]*gamma**np.arange(k)).sum(-1)
        np.testing.assert_allclose(mc_tail(rewards,gamma,0),prefix+gamma**k*tail)
        pred=tail+[3.,-2.]
        np.testing.assert_allclose((prefix+gamma**k*pred-mc_tail(rewards,gamma,0)),gamma**k*(pred-tail))
    np.testing.assert_array_equal(mc_tail(rewards,gamma,3),rewards[:,-1])
    with pytest.raises(ValueError):mc_tail(rewards,gamma,4)


def test_scene_split_and_pair_bias_invariance():
    protocol=dict(split_seed=960001,fit_scenes=16,selection_scenes=8,check_scenes=8)
    seeds=np.arange(610000,610032);splits=scene_split(seeds,protocol)
    assert len(np.concatenate(list(splits.values())))==32
    assert len(set(np.concatenate(list(splits.values()))))==32
    rng=np.random.default_rng(5);truth=rng.normal(size=(4,3,8,32));pred=truth+rng.normal(size=truth.shape)
    np.testing.assert_allclose(paired_error(pred,truth),paired_error(pred+12,truth),atol=1e-14)
    assert scores(pred,truth)['pair_rmse']==pytest.approx(scores(pred+12,truth)['pair_rmse'])
    # Clustered perfect calibration must improve pair RMSE on every resample.
    result=error_interval(paired_error(pred,truth),paired_error(truth,truth),4,2000)
    assert result['delta']<0 and result['high']<0
    with pytest.raises(ValueError):scene_split(np.zeros(32),protocol)


def test_exact_source_prediction_actor_isolation_and_resume(tmp_path):
    cfg=config();configure_runtime(cfg);seed_all(83)
    policy=make_learner(cfg,torch.device('cpu'))
    assert cfg.algo.critic_parameterization=='time_factor'
    policy.actor.requires_grad_(False);policy.value.requires_grad_(False)
    actor_before=actor_digest(policy.actor);value_before=actor_digest(policy.value)
    model=ReleaseCritic(policy)
    rng=np.random.default_rng(17);x=rng.normal(0,.2,size=(24,cfg.env.state_dim)).astype(np.float32)
    x[:,-1]=np.arange(24)%cfg.env.horizon/cfg.env.horizon
    x[-1,-1]=1
    t=torch.from_numpy(x)
    np.testing.assert_array_equal(model(t).detach().numpy(),policy.values(x).numpy())
    assert model(t)[-1]==0
    y=(model(t).detach().numpy()+np.linspace(-2,3,len(x))).astype(np.float64)
    protocol=dict(batch_size=8,huber_delta=10,max_grad_norm=.5)
    opt=torch.optim.Adam(model.net.parameters(),lr=5e-4,eps=1e-5)
    order=np.random.default_rng(940001)
    fit_epoch(model,opt,x,y,2.,order,protocol)
    save_critic(tmp_path/'resume.pt',model,opt,1,order,protocol,2.,'sourcehash')
    clone=ReleaseCritic(policy);clone_opt=torch.optim.Adam(clone.net.parameters(),lr=5e-4,eps=1e-5)
    clone_rng=np.random.default_rng(0)
    restore_critic(tmp_path/'resume.pt',clone,clone_opt,clone_rng)
    for _ in range(3):
        a=fit_epoch(model,opt,x,y,2.,order,protocol)
        b=fit_epoch(clone,clone_opt,x,y,2.,clone_rng,protocol)
        assert a==b
    for key,value in model.state_dict().items():torch.testing.assert_close(value,clone.state_dict()[key],rtol=0,atol=0)
    assert actor_digest(policy.actor)==actor_before and actor_digest(policy.value)==value_before
    assert all(p.grad is None for p in policy.actor.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.net.parameters())
    bad=tmp_path/'bad.pt';torch.save(dict(kind='model'),bad)
    with pytest.raises(ValueError):restore_critic(bad,clone,clone_opt,clone_rng)
