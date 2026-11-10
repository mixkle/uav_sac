import copy

import numpy as np
import pytest
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.critic_calibration import discounted_episode_returns
from uav_research.critic_time_factor import horizon_factors
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.runtime import configure_runtime, load_checkpoint, seed_all
from uav_research.train import train
from uav_research.vector import VectorEnvironment
from test_online_time_factor import assert_exact


def small_config(mode):
    cfg = load_config('tests/fixtures/configs/discrete_mappo_smoke.yaml')
    cfg.env.throughput_weight = .1
    cfg.env.robustness_weight = .5
    cfg.env.disconnection_weight = .1
    cfg.algo.critic_parameterization = mode
    cfg.run.cpu_threads = 8
    cfg.run.total_steps = 64
    cfg.run.eval_interval = 32
    cfg.run.validation_seeds = [10000]
    return cfg.validate()


def test_paired_configuration_and_initial_actor():
    d = load_config('tests/fixtures/configs/discrete_mappo_full_reward_direct_250k.yaml')
    f = load_config('tests/fixtures/configs/discrete_mappo_full_reward_time_factor_250k.yaml')
    changed = f.as_dict()
    changed['algo']['critic_parameterization'] = 'direct'
    assert changed == d.as_dict()
    base = load_config('tests/fixtures/configs/discrete_mappo_ablation_c_mobile_coverage.yaml').as_dict()
    base['env'].update(throughput_weight=.1, robustness_weight=.5, disconnection_weight=.1)
    assert base == d.as_dict()
    for cfg in [d, f]:
        assert cfg.env.state_dim == 6292
        assert cfg.algo.value_clipping and cfg.run.total_steps == 250000
    seed_all(0)
    direct = make_learner(d, configure_runtime(d))
    seed_all(0)
    factor = make_learner(f, configure_runtime(f))
    assert_exact(direct.actor.state_dict(), factor.actor.state_dict())
    for name in ['sac', 'matd3']:
        bad = copy.deepcopy(f)
        bad.algo.name = name
        with pytest.raises(ValueError):
            bad.validate()
    bad = copy.deepcopy(f)
    bad.env.reward_primary = 'sensing'
    with pytest.raises(ValueError):
        bad.validate()


def test_signed_and_above_one_mc_labels_are_not_clipped():
    rewards = np.array([[-.1, 2., .4], [1.5, -.05, -.1]])
    gamma = .995
    returns = discounted_episode_returns(rewards, gamma)
    h = horizon_factors(np.arange(3, 0, -1), gamma)
    np.testing.assert_allclose(returns[:, :-1], rewards[:, :-1] + gamma * returns[:, 1:])
    np.testing.assert_array_equal(returns[:, -1], rewards[:, -1])
    np.testing.assert_allclose(h * (returns / h), returns)
    assert (returns / h < 0).any() and (returns / h > 1).any()
    cfg = small_config('time_factor')
    model = make_learner(cfg, configure_runtime(cfg))
    states = torch.zeros(2, cfg.env.state_dim)
    states[:, -1] = (cfg.env.horizon - 1) / cfg.env.horizon
    for output in [-.1, 1.5]:
        with torch.no_grad():
            model.value[-1].weight.zero_()
            model.value[-1].bias.fill_(output)
        torch.testing.assert_close(model.values(states), torch.full((2,), output))
    states[:, -1] = 1.
    assert torch.equal(model.values(states), torch.zeros(2))


@pytest.mark.parametrize('backend', ['batch', 'process'])
def test_full_reward_scalar_vector_and_components(backend):
    cfg = small_config('time_factor')
    configure_runtime(cfg)
    scalar = [UAVEnvironment(cfg.env) for _ in range(2)]
    for i, env in enumerate(scalar):
        env.reset(30000 + i)
    with VectorEnvironment(cfg.env, 2, backend) as vector:
        vector.reset([30000, 30001])
        for step in range(8):
            actions = (np.arange(8).reshape(2, 4) + step) % 9
            actual = vector.step(actions)
            for i, env in enumerate(scalar):
                expected = env.step(actions[i])
                np.testing.assert_array_equal(actual[1][i], expected[1])
                assert actual[2][i] == expected[2]
                info = expected[-1]
                parts = [info['coverage'], .1 * info['tau'], .5 * info['robustness'], -.1 * info['disconnection_ratio']]
                assert sum(parts) == pytest.approx(expected[2])
                np.testing.assert_allclose(env.trace_frame()['reward_contributions'], parts)


@pytest.mark.parametrize('mode', ['direct', 'time_factor'])
def test_full_reward_training_resume_and_evaluation(tmp_path, mode, monkeypatch):
    cfg = small_config(mode)
    full = train(cfg, tmp_path / 'full')
    part = copy.deepcopy(cfg)
    part.run.total_steps = 32
    initial = train(part, tmp_path / 'split')
    resumed = train(cfg, tmp_path / 'split', initial['resume_checkpoint'])
    a, b = load_checkpoint(full['resume_checkpoint']), load_checkpoint(resumed['resume_checkpoint'])
    assert_exact(a['learner'], b['learner'])
    assert_exact(a['environments'], b['environments'])
    ecfg, policy = policy_from_checkpoint(tmp_path / 'full' / 'last.pt', 'cpu')
    episodes, summary = evaluate(ecfg, policy, [10000], tmp_path / 'evaluation', trace_count=1)
    assert np.isfinite(episodes[0]['return'])
    assert np.isfinite(summary['mean']['robustness'])
    # The independent diagnostic must use the complete reward, including its
    # signed components, rather than the old coverage-only MC assertion.
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from compare_full_reward_time_factor import audit_full_reward, critic_scores
    audit_full_reward(tmp_path / 'full', 'last', tmp_path / 'audit.npz')
    with np.load(tmp_path / 'audit.npz') as data:
        np.testing.assert_allclose(data['rewards'].sum(1).mean(), summary['mean']['return'])
        np.testing.assert_allclose(data['component_returns'].sum(-1), data['returns'])
        metrics = critic_scores(data['values'], data['returns'], data['horizon_factors'])
        assert np.isfinite(metrics['average_reward']['stages']['all']['normalized_rmse'])
    from compare_full_reward_time_factor import critic_scores
    constant = critic_scores(np.ones((2, 3)), np.ones((2, 3)), np.ones(3))
    assert constant['value']['stages']['all']['normalized_rmse'] is None
