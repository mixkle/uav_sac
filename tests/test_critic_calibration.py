import copy
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime
from uav_research.critic_calibration import (discounted_episode_returns, actor_digest,
    collect_complete_episodes, rebase_normalizer_from_fit_targets, fit_critic_epoch, predict_critic)
from uav_research.critic_calibration import configure_calibration_optimizer
from uav_research.critic_calibration import restore_calibration_checkpoint
from uav_research.runtime import rng_state


def test_full_returns_end_at_last_real_reward():
    rewards=np.array([[1.,2.,3.],[4.,0.,2.]])
    original=rewards.copy()
    np.testing.assert_allclose(discounted_episode_returns(rewards,.5),[[2.75,3.5,3.],[4.5,1.,2.]])
    np.testing.assert_array_equal(rewards,original)
    np.testing.assert_array_equal(discounted_episode_returns(rewards,0),rewards)


def test_calibration_keeps_actor_and_normalizer_outputs_and_terminal_states(tmp_path):
    cfg=load_config('configs/discrete_mappo_smoke.yaml');cfg.env.horizon=4
    learner=make_learner(cfg,configure_runtime(cfg))
    for p in learner.actor.parameters():p.requires_grad_(False)
    original_actor=actor_digest(learner.actor)
    data=collect_complete_episodes(cfg,learner,[41000,41001],910001,tmp_path)
    np.testing.assert_allclose(data['states'][:,:,-1],np.tile([0,.25,.5,.75],(2,1)))
    with np.load(tmp_path/'episodes.npz') as d:
        np.testing.assert_allclose(data['returns'][:,-1],d['rewards'][:,-1])
    before=predict_critic(learner,data)
    rebase_normalizer_from_fit_targets(learner,data['returns'])
    np.testing.assert_allclose(predict_critic(learner,data),before,atol=1e-5)
    frozen_normalizer=copy.deepcopy(learner.normalizer.state_dict())
    original_value=copy.deepcopy(learner.value.state_dict())
    fit_critic_epoch(learner,data,4,np.random.default_rng(1))
    assert actor_digest(learner.actor)==original_actor
    assert any(not torch.equal(original_value[k],v) for k,v in learner.value.state_dict().items())
    assert all(torch.equal(frozen_normalizer[k],v) for k,v in learner.normalizer.state_dict().items())
    assert all(p.grad is None for p in learner.actor.parameters())


def test_optimizer_controls_preserve_weights_and_separate_history_from_lr():
    cfg=load_config('configs/discrete_mappo_smoke.yaml')
    learner=make_learner(cfg,configure_runtime(cfg))
    learner.value(torch.zeros(2,cfg.env.state_dim)).sum().backward()
    learner.value_opt.step()
    original_weights=copy.deepcopy(learner.value.state_dict())
    original_state=copy.deepcopy(learner.value_opt.state_dict())
    original_actor=actor_digest(learner.actor)
    configure_calibration_optimizer(learner,learning_rate=1e-4)
    assert learner.value_opt.param_groups[0]['lr']==1e-4
    for k, state in learner.value_opt.state_dict()['state'].items():
        assert all(torch.equal(v,original_state['state'][k][name]) for name,v in state.items())
    info=configure_calibration_optimizer(learner,reset_adam=True,learning_rate=5e-4)
    assert info['initialized_parameter_states']==0
    assert learner.value_opt.param_groups[0]['eps']==original_state['param_groups'][0]['eps']
    assert learner.value_opt.param_groups[0]['betas']==original_state['param_groups'][0]['betas']
    assert all(torch.equal(v,original_weights[k]) for k,v in learner.value.state_dict().items())
    assert actor_digest(learner.actor)==original_actor


def test_calibration_resume_matches_uninterrupted_with_legacy_and_saved_permutation_rng():
    cfg=load_config('configs/discrete_mappo_smoke.yaml')
    learner=make_learner(cfg,configure_runtime(cfg))
    generator=np.random.default_rng(17)
    data_rng=np.random.default_rng(81)
    data=dict(states=data_rng.normal(size=(2,5,cfg.env.state_dim)).astype(np.float32),
              returns=data_rng.normal(size=(2,5)))
    rebase_normalizer_from_fit_targets(learner,data['returns'])
    configure_calibration_optimizer(learner,learning_rate=1e-4)
    for _ in range(2):fit_critic_epoch(learner,data,4,generator)
    checkpoint=copy.deepcopy(dict(value=learner.value.state_dict(),normalizer=learner.normalizer.state_dict(),
        value_optimizer=learner.value_opt.state_dict(),rng=rng_state(),epoch=2,
        experiment=dict(permutation_seed=17)))
    permutation_state=copy.deepcopy(generator.bit_generator.state)
    expected_metrics=fit_critic_epoch(learner,data,4,generator)
    for use_saved_rng in (False,True):
        saved=copy.deepcopy(checkpoint)
        if use_saved_rng:saved['permutation_rng']=permutation_state
        restored=make_learner(cfg,learner.device)
        resumed_rng=restore_calibration_checkpoint(restored,saved,10)
        assert resumed_rng.bit_generator.state==permutation_state
        actual_metrics=fit_critic_epoch(restored,data,4,resumed_rng)
        assert actual_metrics==expected_metrics
        for name in ('value','normalizer'):
            assert all(torch.equal(v,getattr(learner,name).state_dict()[k])
                       for k,v in getattr(restored,name).state_dict().items())
        expected_opt=learner.value_opt.state_dict()
        for k,state in restored.value_opt.state_dict()['state'].items():
            assert all(torch.equal(v,expected_opt['state'][k][key]) for key,v in state.items())
