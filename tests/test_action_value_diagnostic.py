"""Counterfactual continuation semantics, terminal MC and TD comparisons."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_mappo_action_values import (
    FIELDS, evidence, intervene, module_hashes, paired_interval, run_candidate,
)
from test_generalization_audit import config
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime, isolated_rng, seed_all
from uav_research.vector import VectorEnvironment


def test_mc_honors_terminal_and_signed_general_rewards():
    reward=np.array([[2.,-1.,3.]])
    values=np.array([[10.,4.,7.,0.]])
    result=evidence(reward,values,.5)
    np.testing.assert_allclose(result['mc_q'],[2.25])
    np.testing.assert_allclose(result['td1'],[-6.])
    # With lambda=1 and full horizon, the advantage is exactly MC minus V(s).
    full=evidence(reward,values,.5,lam=1.)
    np.testing.assert_allclose(full['gae_full'],result['mc_q']-values[:,0])
    with pytest.raises(AssertionError):
        evidence(reward,np.array([[10.,4.,7.,9.]]),.5)
    with pytest.raises(ValueError):
        evidence(reward,values[:,:-1],.5)


def test_bootstrap_action_error_is_relative_next_value_error():
    rewards=np.array([[.5,.4,.8],[.6,.3,.9]])
    values=np.array([[4.,2.,1.,0.],[4.,2.5,.5,0.]])
    scores=evidence(rewards,values,.9)
    delta=lambda key:scores[key][1]-scores[key][0]
    assert delta('initial_value')==0
    np.testing.assert_allclose(delta('td1')-delta('mc_q'),.9*delta('future_mc_error'))
    # Shared state-prediction bias must cancel in the pairwise ordering.
    shifted=values.copy();shifted[:,0]+=100
    altered=evidence(rewards,shifted,.9)
    np.testing.assert_allclose(altered['td1'][1]-altered['td1'][0],delta('td1'))


def test_override_changes_only_current_action_and_owns_its_copy():
    action=np.array([[[.6,.2],[-.1,.3]]],np.float32)
    original=action.copy()
    np.testing.assert_array_equal(intervene(action,'half',0),action*.5)
    assert not intervene(action,'hover',0).any()
    for mode in ['policy','half','hover']:
        np.testing.assert_array_equal(intervene(action,mode,1),original)
    np.testing.assert_array_equal(action,original)


@pytest.mark.parametrize('history',['none','one_step'])
def test_paired_continuations_and_exact_policy_reproduction(history):
    cfg=config();cfg.env.observation_history=history
    cfg.validate();configure_runtime(cfg);seed_all(51)
    learner=make_learner(cfg,torch.device('cpu'))
    learner.actor.eval().requires_grad_(False)
    learner.value.eval().requires_grad_(False)
    before=module_hashes(learner)
    with VectorEnvironment(cfg.env,cfg.run.num_envs,'batch') as vector:
        obs,state,_=vector.reset(list(range(31,31+cfg.run.num_envs)))
        for _ in range(2):obs,state,*_=vector.step(learner.act(obs))
        case=dict(time=2,start=0,snapshots=copy.deepcopy(vector.snapshot()),
                  torch_rng=torch.get_rng_state().clone(),
                  records=[dict(seed=31+i) for i in range(cfg.run.num_envs)])
        reference=[]
        reference_values=[learner.values(state).numpy()]
        for _ in range(2,cfg.env.horizon):
            obs,state,_,_,_,infos=vector.step(learner.act(obs))
            reference.append([[v[key] for key in FIELDS] for v in infos])
            reference_values.append(learner.values(state).numpy())
    with isolated_rng():
        actual,_=run_candidate(cfg,learner,case,'policy',0)
    np.testing.assert_array_equal(actual['diagnostics'],np.swapaxes(reference,0,1))
    np.testing.assert_array_equal(actual['values'],np.stack(reference_values,-1))
    samples=[];checks=[]
    for mode in ['policy','half','hover']:
        data,check=run_candidate(cfg,learner,case,mode,1)
        samples.append(data);checks.append(check)
        evidence(data['diagnostics'][...,0],data['values'],cfg.algo.discount)
    assert len({c['ue_path_sha256'] for c in checks})==1
    assert len({c['final_actor_rng_sha256'] for c in checks})==1
    for d in samples[1:]:np.testing.assert_array_equal(d['raw_first_actions'],samples[0]['raw_first_actions'])
    np.testing.assert_array_equal(samples[1]['first_actions'],samples[0]['first_actions']*.5)
    assert not samples[2]['first_actions'].any()
    assert module_hashes(learner)==before
    assert all(p.grad is None for net in [learner.actor,learner.value] for p in net.parameters())


def test_nested_pairing_preserves_constant_difference():
    result=paired_interval(np.ones((8,12))*.25)
    for key in ['mean','low','high','nested_low','nested_high']:
        assert result[key]==.25
