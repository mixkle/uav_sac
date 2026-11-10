"""Coordinate semantics, unchanged physics, and checkpoint isolation."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import Baseline, evaluate, policy_from_checkpoint
from uav_research.experiments import assert_observations_equal
from uav_research.observations import stack_observations
from uav_research.runtime import configure_runtime, load_checkpoint
from uav_research.train import train
from uav_research.vector import VectorEnvironment
from test_online_time_factor import assert_exact
from test_sac_time_factor import small


def test_experiment_changes_only_coordinates_and_budget():
    old = load_config('tests/fixtures/configs/sac_full_reward_time_factor_250k.yaml')
    new = load_config('tests/fixtures/configs/sac_full_reward_time_factor_relative_100k.yaml')
    assert new.run.total_steps == 100000 and new.run.cpu_threads == 8
    assert new.env.observation_coordinates == 'relative'
    assert (new.env.obs_dim, new.env.state_dim) == (101, 6292)
    new.env.observation_coordinates = 'absolute'
    new.run.total_steps = old.run.total_steps
    assert new.as_dict() == old.as_dict()
    new.env.observation_coordinates = 'invalid'
    with pytest.raises(ValueError, match='observation_coordinates'):
        new.validate()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_relative_values_masks_and_identical_physics(backend):
    cfg = small('time_factor')
    # Rectangular area checks that x and y use different physical scales.
    e = replace(cfg.env, width=800., height=400., uu_range=150., backend=backend)
    positions = np.array([[100., 100.], [200., 100.], [100., 200.], [750., 350.]])
    users = np.tile([[150., 100.], [750., 350.]], (12, 1))
    absolute = UAVEnvironment(e)
    relative = UAVEnvironment(replace(e, observation_coordinates='relative'))
    a, sa, ia = absolute.set_positions(positions, users)
    r, sr, ir = relative.set_positions(positions, users)
    np.testing.assert_array_equal(sa, sr)
    assert ia['reward'] == ir['reward']
    np.testing.assert_array_equal(a.users, r.users)
    np.testing.assert_array_equal(a.offsets, r.offsets)
    np.testing.assert_array_equal(a.fixed[:, :e.self_dim], r.fixed[:, :e.self_dim])
    scale = np.array([e.width, e.height])
    for i in range(e.num_uavs):
        np.testing.assert_allclose(r.fixed[i, :2], positions[i] / scale)
        bs = r.fixed[i, e.self_dim:e.self_dim + 2 * e.num_bs].reshape(-1, 2)
        np.testing.assert_allclose(bs, (np.array(e.bs_positions)[:, :2] - positions[i]) / scale)
        start = e.self_dim + 2 * e.num_bs
        neighbors = r.fixed[i, start:].reshape(3, e.neighbor_dim)
        old_neighbors = a.fixed[i, start:].reshape(3, e.neighbor_dim)
        for row, old in zip(neighbors, old_neighbors):
            if row[0] == 0:
                assert not row.any()
            else:
                j = round(float(row[0]) * e.num_uavs) - 1
                np.testing.assert_allclose(row[1:3], (positions[j] - positions[i]) / scale)
                np.testing.assert_array_equal(row[3:], old[3:])
    # Tie is still resolved by UAV id; disconnected UAV keeps all neighbor slots zero.
    assert r.fixed[0, start] == 2 / 4
    assert not r.fixed[3, start:].any()
    np.testing.assert_allclose(Baseline(replace(cfg, env=e), 'heuristic').act(a),
        Baseline(replace(cfg, env=relative.cfg), 'heuristic').act(r), atol=1e-6)
    reference = UAVEnvironment(replace(relative.cfg, backend='numpy'))
    expected, _, _ = reference.set_positions(positions, users)
    assert_observations_equal(r, expected)
    # Random streams and subsequent physics remain identical under identical commands.
    absolute.reset(51); relative.reset(51)
    for _ in range(4):
        action = np.full((e.num_uavs, 2), .1, np.float32)
        x, y = absolute.step(action), relative.step(action)
        np.testing.assert_array_equal(x[1], y[1])
        assert x[2:5] == y[2:5]
        np.testing.assert_array_equal(absolute.user_xy, relative.user_xy)


@pytest.mark.parametrize('backend', ['batch', 'process'])
def test_vector_paths(backend):
    cfg = small('time_factor')
    cfg.env.observation_coordinates = 'relative'
    configure_runtime(cfg)
    vec = VectorEnvironment(cfg.env, 2, backend)
    scalar = [UAVEnvironment(cfg.env), UAVEnvironment(cfg.env)]
    try:
        obs, state, _ = vec.reset([51, 52])
        expected = [env.reset(seed) for env, seed in zip(scalar, [51, 52])]
        assert_observations_equal(obs, stack_observations([r[0] for r in expected]))
        actions = np.full((2, cfg.env.num_uavs, 2), .1, np.float32)
        result = vec.step(actions)
        expected = [env.step(action) for env, action in zip(scalar, actions)]
        assert_observations_equal(result[0], stack_observations([r[0] for r in expected]))
        np.testing.assert_array_equal(result[1], np.stack([r[1] for r in expected]))
        np.testing.assert_allclose(result[2], [r[2] for r in expected])
    finally:
        vec.close()


def test_relative_training_resume_export_and_legacy_guard(tmp_path):
    cfg = small('time_factor')
    cfg.env.observation_coordinates = 'relative'
    cfg.run.validation_seeds = [30020]
    full = train(cfg, tmp_path / 'full')
    part = copy.deepcopy(cfg); part.run.total_steps = 32
    first = train(part, tmp_path / 'split')
    resumed = train(cfg, tmp_path / 'split', first['resume_checkpoint'])
    a, b = load_checkpoint(full['resume_checkpoint']), load_checkpoint(resumed['resume_checkpoint'])
    for key in ['learner', 'environments', 'rng', 'replay_metadata']:
        assert_exact(a[key], b[key])
    saved_cfg, policy = policy_from_checkpoint(tmp_path / 'full/last.pt', 'cpu')
    assert saved_cfg.env.observation_coordinates == 'relative'
    _, metrics = evaluate(saved_cfg, policy, [30020], tmp_path / 'eval', trace_count=1)
    assert np.isfinite(metrics['mean']['return'])
    export_actor(tmp_path / 'full/last.pt', tmp_path / 'actor.pt')
    deployed = DeployedActor(tmp_path / 'actor.pt')
    assert deployed.artifact['observation_coordinates'] == 'relative'
    obs, _, _ = UAVEnvironment(cfg.env).reset(30020)
    np.testing.assert_array_equal(deployed.act(obs), policy.act(obs, True))
    legacy = copy.deepcopy(cfg); legacy.env.observation_coordinates = 'absolute'
    with pytest.raises(ValueError, match='Environment/algorithm changed'):
        train(legacy, tmp_path / 'wrong', full['resume_checkpoint'])
    learner = make_learner(legacy, configure_runtime(legacy))
    with pytest.raises(ValueError, match='observation coordinates'):
        learner.load_state_dict(a['learner'])
    old = learner.state_dict(); old.pop('observation_coordinates')
    learner.load_state_dict(old)  # Historical missing flag remains absolute.
    relative = make_learner(cfg, configure_runtime(cfg))
    with pytest.raises(ValueError, match='observation coordinates'):
        relative.load_state_dict(old)
