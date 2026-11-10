"""Correct block horizon, policy restoration, and paired counterfactuals."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_mappo_action_blocks import block_action,block_scores,run_block
from check_mappo_action_values import run_candidate,module_hashes,evidence
from test_generalization_audit import config
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime,seed_all
from uav_research.vector import VectorEnvironment


def test_bootstrap_horizon_and_terminal_signed_rewards():
    r=np.array([[2.,-1.,3.]])
    v=np.array([[10.,4.,7.,0.]])
    k2=block_scores(r,v,.5,2)
    np.testing.assert_allclose(k2['mc'],[2.25])
    np.testing.assert_allclose(k2['prefix'],[1.5])
    np.testing.assert_allclose(k2['tail_mc'],[.75])
    np.testing.assert_allclose(k2['bootstrap'],[1.5+.25*7-10])
    np.testing.assert_allclose(block_scores(r,v,.5,1)['bootstrap'],evidence(r,v,.5)['td1'])
    for k in [3,20]:
        terminal=block_scores(r,v,.5,k)
        np.testing.assert_allclose(terminal['bootstrap'],terminal['mc']-v[:,0])
        assert terminal['tail_mc']==0 and terminal['boundary_error']==0
    with pytest.raises(ValueError):block_scores(r,v,.5,0)


def test_exact_value_oracle_and_relative_boundary_error_identity():
    r=np.array([[.2,.5,-.1,1.2],[.3,.1,.8,.7]])
    gamma=.9
    true=np.zeros((2,5))
    for t in range(3,-1,-1):true[:,t]=r[:,t]+gamma*true[:,t+1]
    for k in [1,2,4]:
        exact=block_scores(r,true,gamma,k)
        np.testing.assert_allclose(exact['bootstrap'],exact['mc']-true[:,0],atol=1e-12)
    v=true.copy();v[:,0]=5;v[:,2]+=[1,-2]
    result=block_scores(r,v,gamma,2)
    delta=lambda key:result[key][1]-result[key][0]
    np.testing.assert_allclose(delta('bootstrap')-delta('mc'),gamma**2*delta('boundary_error'))


def test_actions_restore_original_policy_at_the_right_step():
    action=np.array([[[.6,.2],[-.2,.3]]],np.float32)
    original=action.copy()
    for k in [5,20]:
        for step in [0,k-1]:
            np.testing.assert_array_equal(block_action(action,'half',step,k),action*.5)
            assert not block_action(action,'hover',step,k).any()
        for mode in ['policy','half','hover']:
            np.testing.assert_array_equal(block_action(action,mode,k,k),original)
            np.testing.assert_array_equal(block_action(action,mode,k+1,k),original)
    np.testing.assert_array_equal(action,original)


@pytest.mark.parametrize('history',['none','one_step'])
def test_saved_state_K1_regression_prefix_pairing_and_release(history):
    cfg=config();cfg.env.observation_history=history;cfg.validate()
    configure_runtime(cfg);seed_all(77)
    policy=make_learner(cfg,torch.device('cpu'))
    policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
    before=module_hashes(policy)
    with VectorEnvironment(cfg.env,cfg.run.num_envs,'batch') as vector:
        obs,*_=vector.reset([51+i for i in range(cfg.run.num_envs)])
        for _ in range(2):obs,*_=vector.step(policy.act(obs))
        case=dict(time=2,start=0,snapshots=copy.deepcopy(vector.snapshot()),
                  torch_rng=torch.get_rng_state().clone(),records=[dict(seed=51+i) for i in range(cfg.run.num_envs)])
    checks=[]
    for mode in ['half','hover']:
        old,old_check=run_candidate(cfg,policy,case,mode,0)
        new,new_check=run_block(cfg,policy,case,mode,0,1)
        for key in ['diagnostics','values','contributions']:
            np.testing.assert_array_equal(new[key],old[key])
        assert new_check==old_check
        short,c1=run_block(cfg,policy,case,mode,1,3)
        long,c2=run_block(cfg,policy,case,mode,1,5)
        for key in ['diagnostics','prefix_effective','prefix_raw']:
            np.testing.assert_array_equal(short[key][:,:3],long[key][:,:3])
        np.testing.assert_array_equal(short['values'][:,:4],long['values'][:,:4])
        for k,data in [(3,short),(5,long)]:
            np.testing.assert_array_equal(data['prefix_effective'][:,k],data['prefix_raw'][:,k])
            if mode=='hover':assert not data['prefix_effective'][:,:k].any()
            else:np.testing.assert_array_equal(data['prefix_effective'][:,:k],data['prefix_raw'][:,:k]*.5)
            block_scores(data['diagnostics'][...,0],data['values'],cfg.algo.discount,k)
        checks.extend([c1,c2])
    assert len({c['ue_path_sha256'] for c in checks})==1
    assert len({c['final_actor_rng_sha256'] for c in checks})==1
    assert module_hashes(policy)==before
    assert all(p.grad is None for net in [policy.actor,policy.value] for p in net.parameters())
