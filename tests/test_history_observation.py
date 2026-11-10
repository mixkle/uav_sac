"""One-step history: timing, matched initialization, recovery and report pairing."""
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from test_mappo_bridge import small
from test_online_time_factor import assert_exact
from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.deployment import export_actor, DeployedActor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import stack_observations
from uav_research.runtime import configure_runtime, load_checkpoint, seed_all
from uav_research.train import train
from uav_research.vector import VectorEnvironment


def config():
    cfg = small('M1')
    cfg.env.observation_history = 'one_step'
    cfg.env.uav_boundary = 'reflect'
    cfg.env.user_initialization = 'uniform_interior'
    cfg.env.user_initial_margin_fraction = 1/7
    return cfg.validate()


def expected_history(env, action=None):
    action = np.zeros((env.cfg.num_uavs, 2)) if action is None else action
    return np.column_stack([action, env.B.min(1)/(env.cfg.num_uavs+1),
                            env.sensed.sum(1)/env.cfg.capacity]).astype(np.float32)


def test_configuration_and_pairing():
    base = load_config('tests/fixtures/configs/continuous_mappo_full_reward_time_factor_relative_central_100k.yaml')
    hist = load_config('tests/fixtures/configs/continuous_mappo_full_reward_time_factor_relative_central_history_100k.yaml')
    assert (base.env.obs_dim, hist.env.obs_dim, hist.env.state_dim) == (101, 105, 6292)
    hist.env.observation_history = 'none'
    assert hist.as_dict() == base.as_dict()
    hist.env.observation_history = 'invalid'
    with pytest.raises(ValueError, match='observation_history'): hist.validate()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_history_alignment_immutability_and_controlled_physics(backend):
    cfg = config(); cfg.env.backend = backend
    # All UE visible, counts above capacity; last UAV has no backhaul.
    cfg.env.capacity = 2
    cfg.env.sensing_threshold_db = -200
    env = UAVEnvironment(cfg.env)
    old = UAVEnvironment(replace(cfg.env, observation_history='none'))
    positions = np.array([[cfg.env.width-1, 10], [10, 10], [20, 20],
                          [cfg.env.width-1, cfg.env.height-1]], dtype=float)
    users = np.tile([[cfg.env.width/2, cfg.env.height/2]], (cfg.env.num_users, 1))
    obs, _, _ = env.set_positions(positions, users)
    old.set_positions(positions, users)
    np.testing.assert_array_equal(obs.fixed[:, -4:], expected_history(env))
    assert (obs.fixed[:, -1] > 1).all()
    assert (obs.fixed[:, -2] == 1).any()
    saved = obs.copy()
    for action in [np.tile([.8, 0], (cfg.env.num_uavs, 1)), np.zeros((cfg.env.num_uavs, 2))]:
        expected = expected_history(env, action)
        result = env.step(action); reference = old.step(action)
        np.testing.assert_array_equal(result[0].fixed[:, -4:], expected)
        np.testing.assert_array_equal(result[0].fixed[:, :-4], reference[0].fixed)
        np.testing.assert_array_equal(result[1], reference[1])
        assert result[2:5] == reference[2:5]
        assert_exact(env.movement_rng.bit_generator.state, old.movement_rng.bit_generator.state)
        copy_obs = result[0].copy()
        env._refresh(); env._refresh()
        np.testing.assert_array_equal(copy_obs.fixed, env.obs.fixed)
        restored = UAVEnvironment(cfg.env); restored.restore(env.snapshot())
        np.testing.assert_array_equal(restored.obs.fixed, env.obs.fixed)
    np.testing.assert_array_equal(saved.fixed, obs.fixed)  # No in-place corruption.
    env.reset(81)
    np.testing.assert_array_equal(env.obs.fixed[:, -4:], expected_history(env))
    missing = env.snapshot(); missing.pop('history')
    with pytest.raises(ValueError, match='complete finite'): env.restore(missing)
    with pytest.raises(ValueError, match='observation_history'): env.restore(old.snapshot())


@pytest.mark.parametrize('backend', ['batch', 'process'])
def test_vector_reset_terminal_and_restore(backend):
    cfg = config(); cfg.env.horizon = 2
    configure_runtime(cfg)
    scalars = [UAVEnvironment(cfg.env), UAVEnvironment(cfg.env)]
    with VectorEnvironment(cfg.env, 2, backend) as vector:
        obs, _, _ = vector.reset([51, 52])
        for e, s in zip(scalars, [51, 52]): e.reset(s)
        for _ in range(2):
            actions = np.full((2, cfg.env.num_uavs, 2), .2, np.float32)
            result = vector.step(actions)
            refs = [e.step(a) for e, a in zip(scalars, actions)]
            expected = stack_observations([r[0] for r in refs])
            np.testing.assert_array_equal(result[0].fixed, expected.fixed)
            np.testing.assert_array_equal(result[1], np.stack([r[1] for r in refs]))
        assert result[3].all()
        terminal = result[0].copy()
        snaps = vector.snapshot(); vector.restore(snaps)
        assert_exact(snaps, vector.snapshot())
        vector.reset_at(0)
        after = vector.snapshot()
        assert_exact(after[1], snaps[1])
        assert after[0]['t'] == 0
        assert (terminal.fixed[..., -4:-2] == .2).all()


def test_initial_function_rng_critic_and_history_gradient():
    cfg = config(); old_cfg = copy.deepcopy(cfg); old_cfg.env.observation_history = 'none'
    device = configure_runtime(cfg)
    seed_all(21); old = make_learner(old_cfg, device); rng = torch.get_rng_state().clone()
    seed_all(21); new = make_learner(cfg, device)
    assert_exact(torch.get_rng_state(), rng)
    assert_exact(old.value.state_dict(), new.value.state_dict())
    current = cfg.env.current_obs_dim
    w = new.actor.net[0].weight
    assert torch.count_nonzero(w[:, current:current+4]) == 0
    assert sum(p.numel() for p in new.actor.parameters()) - sum(p.numel() for p in old.actor.parameters()) == 4*cfg.algo.hidden
    a, b = UAVEnvironment(cfg.env), UAVEnvironment(old_cfg.env)
    x, s, _ = a.reset(30020); y, _, _ = b.reset(30020)
    da = new.actor.distribution(x.to_torch(device)); db = old.actor.distribution(y.to_torch(device))
    torch.testing.assert_close(da.loc, db.loc, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(da.scale, db.scale, atol=2e-7, rtol=1e-6)
    seed_all(42); aa = new.act(x); seed_all(42); ab = old.act(y)
    np.testing.assert_allclose(aa, ab, atol=2e-7, rtol=1e-6)
    assert_exact(new.values(s), old.values(s))
    before = copy.deepcopy(old.actor.state_dict())
    for _ in range(2):
        action, logp, latent, _ = new.collect(x, s)
        x, s, *_ = a.step(action)
    logp, entropy = new.actor.evaluate_latent(x.to_torch(device), torch.as_tensor(latent))
    (-(logp*torch.arange(cfg.env.num_uavs)).mean() - .01*entropy.mean()).backward()
    assert new.actor.net[0].weight.grad[:, current:current+4].abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in new.actor.encoder.parameters() if p.grad is not None) > 0
    assert_exact(old.actor.state_dict(), before)
    with pytest.raises(ValueError, match='observation_history'): old.load_state_dict(new.state_dict())
    with pytest.raises(ValueError, match='observation_history'): new.load_state_dict(old.state_dict())
    legacy = old.state_dict(); legacy.pop('observation_history'); old.load_state_dict(legacy)


def test_full_resume_export_audit_and_report(tmp_path, monkeypatch):
    cfg = config(); cfg.run.cpu_threads = 1
    full = train(cfg, tmp_path/'full')
    part = copy.deepcopy(cfg); part.run.total_steps = 32
    first = train(part, tmp_path/'split')
    resumed = train(cfg, tmp_path/'split', first['resume_checkpoint'])
    a, b = [load_checkpoint(r['resume_checkpoint']) for r in [full, resumed]]
    for key in ['learner', 'environments', 'rng', 'state']: assert_exact(a[key], b[key])
    for key in ['fixed', 'users', 'offsets']: assert_exact(getattr(a['obs'], key), getattr(b['obs'], key))
    saved_cfg, policy = policy_from_checkpoint(tmp_path/'full/last.pt', 'cpu')
    export_actor(tmp_path/'full/last.pt', tmp_path/'actor.pt')
    deployed = DeployedActor(tmp_path/'actor.pt')
    assert deployed.artifact['observation_history']['mode'] == 'one_step'
    obs, _, _ = UAVEnvironment(cfg.env).reset(30020)
    np.testing.assert_array_equal(deployed.act(obs), policy.act(obs, True))
    wrong = copy.deepcopy(cfg); wrong.env.observation_history = 'none'
    with pytest.raises(ValueError, match='Environment/algorithm changed'):
        train(wrong, tmp_path/'wrong', full['resume_checkpoint'])
    # Complete small paired report, including MC, independent action seeds,
    # link retention, common-state branch histories, and data-only rebuild.
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_mappo_history import build, publish
    from check_mappo_generalization import new_scenes, evaluate_scenes
    from check_mappo_deployment_branches import capture_states
    from report_mappo_bridge import digest
    from uav_research.runtime import atomic_json
    old = copy.deepcopy(cfg); old.env.observation_history = 'none'
    train(old, tmp_path/'baseline')
    _, old_policy = policy_from_checkpoint(tmp_path/'baseline/best.pt', 'cpu')
    scenes = new_scenes(old, [30030, 30031], 'small')
    raw, _, _ = evaluate_scenes(old, old_policy, scenes, 880001)
    cases, _ = capture_states(old, old_policy, scenes, [2, 4], 880001, raw)
    branch = tmp_path/'branch'; branch.mkdir()
    torch.save(dict(unseen=scenes), branch/'initial_scenes.pt')
    torch.save(cases, branch/'branch_states.pt')
    atomic_json(branch/'experiment.json', dict(parent=str(branch),
        source_hashes={str(tmp_path/'baseline/best.pt'):digest(tmp_path/'baseline/best.pt')}))
    comparison = build([tmp_path/'baseline', tmp_path/'full'], tmp_path/'report',
                       samples=8, diagnostic_seeds=[30030, 30031], action_seeds=[880001, 880002],
                       branch_source=branch)
    assert comparison['branch_differences']
    publish(tmp_path/'report', comparison)
    assert (tmp_path/'report/figures/coverage_last.png').exists()
    assert (tmp_path/'report/report.md').exists()
