import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_research.algorithms import make_learner
from uav_research.bridge_diagnostics import audit_policy, conditional_moments
from uav_research.config import load_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.models import disk_transform
from uav_research.observations import as_tensor_observation, stack_observations
from uav_research.runtime import configure_runtime, isolated_rng, load_checkpoint, seed_all
from uav_research.train import train
from test_online_time_factor import assert_exact


def small(label):
    cfg = load_config('tests/fixtures/configs/discrete_mappo_smoke.yaml')
    cfg.env.observation_coordinates = 'relative'
    cfg.env.throughput_weight = .1; cfg.env.robustness_weight = .5
    cfg.env.disconnection_weight = .1
    cfg.env.action_space = 'discrete9' if label == 'M0' else 'continuous'
    cfg.algo.name = 'sac' if label == 'S0' else 'mappo'
    cfg.algo.mappo_state_dependent_std = label == 'M1'
    cfg.algo.critic_parameterization = 'time_factor'
    cfg.run.cpu_threads = 8; cfg.run.total_steps = 64; cfg.run.eval_interval = 32
    cfg.run.log_interval = 16; cfg.run.validation_seeds = [30020, 30021]
    cfg.algo.warmup_steps = 16; cfg.algo.replay_capacity = 128; cfg.algo.batch_size = 8
    return cfg.validate()


def test_configs_initial_actor_and_scope():
    m0 = load_config('tests/fixtures/configs/discrete_mappo_full_reward_time_factor_relative_100k.yaml')
    m1 = load_config('tests/fixtures/configs/continuous_mappo_full_reward_time_factor_relative_100k.yaml')
    s0 = load_config('tests/fixtures/configs/sac_full_reward_time_factor_relative_100k.yaml')
    assert m1.env == s0.env and m1.run.total_steps == m0.run.total_steps == 100000
    assert m1.run.cpu_threads == m0.run.cpu_threads == 8
    assert (m1.env.obs_dim, m1.env.state_dim) == (101, 6292)
    m0.env.action_space = 'continuous'; m0.algo.mappo_state_dependent_std = True
    assert m0.as_dict() == m1.as_dict()
    # Small actor constructors use exactly the same parameters/RNG as SAC.
    seed_all(0); a = make_learner(small('S0'), configure_runtime(small('S0')))
    seed_all(0); b = make_learner(small('M1'), configure_runtime(small('M1')))
    assert_exact(a.actor.state_dict(), b.actor.state_dict())
    raw, _, _ = UAVEnvironment(small('M1').env).reset(30020)
    obs = as_tensor_observation(raw, 'cpu')
    seed_all(42); x = a.actor.sample(obs)
    seed_all(42); y = b.actor.sample(obs)
    for left, right in zip(x, y): torch.testing.assert_close(left, right, rtol=0, atol=0)
    for label in ['M0', 'S0']:
        cfg = small(label); cfg.algo.mappo_state_dependent_std = True
        with pytest.raises(ValueError, match='continuous MAPPO'): cfg.validate()
    cfg = small('M1'); cfg.algo.mappo_state_dependent_std = 1
    with pytest.raises(ValueError, match='boolean'): cfg.validate()


def test_probabilities_ratios_and_gradients():
    cfg = small('M1'); configure_runtime(cfg); seed_all(50)
    model = make_learner(cfg, torch.device('cpu'))
    raw, _, _ = UAVEnvironment(cfg.env).reset(30020)
    obs = stack_observations([raw, raw]).to_torch('cpu')
    action, old_logp, latent = model.actor.sample(obs)
    assert (action.norm(dim=-1) < 1).all()
    latent = latent.detach(); old_logp = old_logp.detach()
    new_logp, entropy = model.actor.evaluate_latent(obs, latent)
    torch.testing.assert_close(new_logp, old_logp)
    assert new_logp.shape == (2, cfg.env.num_uavs)
    torch.testing.assert_close((new_logp - old_logp).exp(), torch.ones_like(new_logp))
    normal_before = model.actor.distribution(obs).log_prob(latent).sum(-1).detach()
    manual = normal_before + 2 * torch.log1p(latent.square().sum(-1))
    torch.testing.assert_close(manual, old_logp)
    with torch.no_grad(): model.actor.net[-1].bias[:2].add_(.1)
    changed, _ = model.actor.evaluate_latent(obs, latent)
    normal_after = model.actor.distribution(obs).log_prob(latent).sum(-1)
    torch.testing.assert_close(changed-old_logp, normal_after-normal_before, atol=3e-7, rtol=1e-5)
    # Entropy uses a new sample and its exact transform determinant.
    with isolated_rng():
        torch.manual_seed(77)
        dist = model.actor.distribution(obs)
        jac = disk_transform(dist.rsample())[1]
        expected = dist.entropy().sum(-1) + jac
    with isolated_rng():
        torch.manual_seed(77)
        _, actual = model.actor.evaluate_latent(obs, latent)
    torch.testing.assert_close(expected, actual)
    for objective in ['ppo', 'entropy']:
        model.actor_opt.zero_grad(set_to_none=True)
        lp, ent = model.actor.evaluate_latent(obs, latent)
        adv = torch.linspace(-1, 1, cfg.env.num_uavs)[None]
        loss = -((lp-old_logp).exp()*adv).mean() if objective == 'ppo' else -ent.mean()
        loss.backward()
        grad = model.actor.net[-1].weight.grad
        assert grad[:2].abs().sum() > 0 and grad[2:].abs().sum() > 0
        assert sum(p.grad.abs().sum() for p in model.actor.encoder.parameters() if p.grad is not None) > 0
    actor = copy.deepcopy(model.actor.state_dict())
    state = torch.zeros(2, cfg.env.state_dim)
    model.actor_opt.zero_grad(set_to_none=True); model.value_opt.zero_grad(set_to_none=True)
    model.value_components(state)[0].square().sum().backward(); model.value_opt.step()
    assert_exact(actor, model.actor.state_dict())
    assert all(p.grad is None for p in model.actor.parameters())


def test_conditional_moments_and_rollout_rng():
    cfg = small('M0'); configure_runtime(cfg)
    raw, _, _ = UAVEnvironment(cfg.env).reset(30020)
    obs = as_tensor_observation(raw, 'cpu')
    model = make_learner(cfg, torch.device('cpu'))
    with torch.no_grad():
        model.actor.net[-1].weight.zero_(); model.actor.net[-1].bias.zero_()
        exact = conditional_moments(model.actor, obs, 'discrete9', 30)
    torch.testing.assert_close(exact[:, 0], torch.full((4,), 800.), atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(exact[:, 1], torch.full((4,), 1/9))
    torch.testing.assert_close(exact[:, 3], torch.full((4,), np.log(9), dtype=torch.float32))
    # Audit and generic evaluate reproduce the exact sampled reward trajectory.


@pytest.mark.parametrize('label', ['M0', 'M1'])
def test_training_exact_resume_export_and_audit(tmp_path, label):
    cfg = small(label)
    full = train(cfg, tmp_path/'full')
    split = copy.deepcopy(cfg); split.run.total_steps = 32
    first = train(split, tmp_path/'split')
    resumed = train(cfg, tmp_path/'split', first['resume_checkpoint'])
    x, y = [load_checkpoint(p['resume_checkpoint']) for p in [full, resumed]]
    for key in ['learner', 'environments', 'rng', 'obs', 'state']:
        if key == 'obs':
            for name in ['fixed', 'users', 'offsets']: assert_exact(getattr(x[key], name), getattr(y[key], name))
        else: assert_exact(x[key], y[key])
    assert not (Path(full['resume_checkpoint'])/'replay').exists()
    pcfg, policy = policy_from_checkpoint(tmp_path/'full/last.pt', 'cpu')
    _, summary = evaluate(pcfg, policy, pcfg.run.validation_seeds, tmp_path/'eval', trace_count=1, reward_curves=True)
    before_rng = torch.get_rng_state().clone()
    data, metrics = audit_policy(pcfg, policy, tmp_path/'audit', samples=8)
    assert_exact(before_rng, torch.get_rng_state())
    with np.load(tmp_path/'eval/reward_curves.npz') as curves:
        np.testing.assert_array_equal(data['rewards'], curves['reward'])
    assert metrics['value']['stages']['all']['rmse'] >= 0
    assert np.isfinite(summary['mean']['return'])
    export_actor(tmp_path/'full/last.pt', tmp_path/'actor.pt')
    deployed = DeployedActor(tmp_path/'actor.pt')
    assert deployed.artifact['specification']['state_dependent_std'] == (label == 'M1')
    raw, _, _ = UAVEnvironment(cfg.env).reset(30020)
    np.testing.assert_array_equal(deployed.act(raw), policy.act(raw, True))
    wrong = copy.deepcopy(cfg); wrong.env.observation_coordinates = 'absolute'
    with pytest.raises(ValueError, match='observation coordinates'):
        make_learner(wrong, configure_runtime(wrong)).load_state_dict(x['learner'])
    if label == 'M1':
        wrong = copy.deepcopy(cfg); wrong.algo.mappo_state_dependent_std = False
        with pytest.raises(ValueError, match='standard deviation'):
            make_learner(wrong, configure_runtime(wrong)).load_state_dict(x['learner'])
        with pytest.raises(ValueError, match='Environment/algorithm changed'):
            train(wrong, tmp_path/'wrong', full['resume_checkpoint'])


def test_legacy_mappo_loading():
    cfg = small('M1'); cfg.env.observation_coordinates = 'absolute'
    cfg.algo.mappo_state_dependent_std = False
    model = make_learner(cfg, configure_runtime(cfg))
    state = model.state_dict(); state.pop('observation_coordinates'); state.pop('mappo_state_dependent_std')
    model.load_state_dict(state)
    assert not model.actor.state_dependent_std


def test_three_way_report_and_rebuild(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_mappo_bridge import build, publish, validate_configs
    configs = {k: small(k) for k in ['M0', 'M1', 'S0']}
    validate_configs(configs)
    bad = copy.deepcopy(configs); bad['M1'].env.observation_coordinates = 'absolute'
    with pytest.raises(ValueError, match='physical'): validate_configs(bad)
    runs = [tmp_path/k for k in configs]
    for cfg, run in zip(configs.values(), runs): train(cfg, run)
    report = build(runs, tmp_path/'report', samples=8)
    assert report['training_seed_count'] == 1 and len(report['paired']) == 42
    assert report['groups']['M1']['checkpoints']['last']['critic']['value_semantics'].startswith('environment V')
    assert report['groups']['S0']['checkpoints']['last']['critic']['value_semantics'].startswith('soft Q')
    publish(tmp_path/'report', report)
    assert (tmp_path/'report/figures/validation_progress.png').exists()
    assert (tmp_path/'report/M1/figures/topology_last_30020.png').exists()
