from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from uav_research.config import Config, EnvConfig, config_from_dict, load_config, save_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.runtime import configure_runtime, load_checkpoint
from uav_research.train import train
from uav_research.vector import VectorEnvironment


VARIANTS = {
    'a_mobile_connected': ('coverage', 1.0, .5),
    'b_static_connected': ('coverage', 0.0, .5),
    'c_mobile_coverage': ('coverage', 1.0, 0.0),
    'd_mobile_sensing': ('sensing', 1.0, 0.0),
    'e_mobile_sensing_connected': ('sensing', 1.0, .5),
    'f_static_coverage': ('coverage', 0.0, 0.0),
}


def small_config(name):
    cfg = load_config('configs/discrete_mappo_smoke.yaml')
    primary, speed, mu = VARIANTS[name]
    cfg.env.reward_primary = primary
    cfg.env.user_speed_std = speed
    cfg.env.disconnection_weight = mu
    cfg.env.throughput_weight = cfg.env.robustness_weight = 0.0
    cfg.run.cpu_threads = 1
    cfg.run.validation_seeds = [30000]
    cfg.run.total_steps = 64
    cfg.run.eval_interval = 32
    return cfg


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def test_ablation_configs_only_change_three_environment_fields():
    configs = [load_config(f'configs/discrete_mappo_ablation_{name}.yaml') for name in VARIANTS]
    base = configs[0].as_dict()
    for cfg in configs:
        data = cfg.as_dict()
        assert data['algo'] == base['algo'] and data['run'] == base['run']
        assert cfg.run.total_steps == 250000 and len(cfg.run.validation_seeds) == 32
        for key in ['reward_primary', 'user_speed_std', 'disconnection_weight']:
            data['env'].pop(key)
        reference = {k: v for k, v in base['env'].items()
                     if k not in ['reward_primary', 'user_speed_std', 'disconnection_weight']}
        assert data['env'] == reference
    for seed in [0, 10000, 30000]:
        envs = [UAVEnvironment(cfg.env) for cfg in configs]
        for env in envs:
            env.reset(seed)
            np.testing.assert_array_equal(env.uav_xy, envs[0].uav_xy)
            np.testing.assert_array_equal(env.user_xy, envs[0].user_xy)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_sensing_reward_counts_unique_users_without_backhaul_or_service(backend):
    e = EnvConfig(width=2000, height=2000, num_uavs=2, num_users=3,
                  bs_positions=[[0, 0, 0]], ug_range=160, uu_range=150,
                  reward_primary='sensing', throughput_weight=0, robustness_weight=0,
                  user_speed_std=0, backend=backend)
    env = UAVEnvironment(e)
    _, _, info = env.set_positions([[1000, 1000], [1000, 1000]],
                                   [[1000, 1000], [1001, 1000], [0, 2000]])
    np.testing.assert_array_equal(env.sensed, [[True, True, False], [True, True, False]])
    assert info['coverage'] == 0 and info['disconnection_ratio'] == 1
    assert info['discovery_fraction'] == info['reward'] == info['reward_primary'] == 2 / 3
    # Previously seen users confer no reward after the current sensing set is empty.
    env.ever_discovered[:] = True
    env.uav_xy[:] = [0, 0]
    env._refresh()
    assert env.info['cumulative_discovery_fraction'] == 1
    assert env.info['discovery_fraction'] == env.info['reward'] == 0


@pytest.mark.parametrize('name', list(VARIANTS))
def test_reward_formula_and_trace_semantics(name):
    cfg = small_config(name)
    env = UAVEnvironment(cfg.env)
    env.reset(30000)
    for _ in range(8):
        _, _, reward, _, _, info = env.step(np.arange(4, dtype=np.int64))
        primary = info[cfg.env.primary_reward_metric]
        assert info['reward_primary'] == primary
        assert reward == primary - cfg.env.disconnection_weight * info['disconnection_ratio']
        frame = env.trace_frame()
        assert frame['reward_terms'][0] == info['coverage']
        assert frame['reward_contributions'][0] == primary
        assert frame['reward_contributions'].sum() == pytest.approx(reward)


@pytest.mark.parametrize('name', ['b_static_connected', 'f_static_coverage'])
def test_stationary_users_survive_full_episode_and_restore_with_independent_rng(name):
    cfg = small_config(name)
    cfg.env.horizon = 400
    fixed = UAVEnvironment(cfg.env)
    mobile = UAVEnvironment(replace(cfg.env, user_speed_std=1.0))
    fixed.reset(31415)
    mobile.reset(31415)
    initial = fixed.user_xy.copy()
    action = np.full(4, 8, np.int64)
    for t in range(400):
        result = fixed.step(action)
        mobile.step(action)
        np.testing.assert_array_equal(fixed.user_xy, initial)
        assert fixed.info['user_displacement_mean'] == fixed.info['user_speed_mean'] == 0
        assert fixed.movement_rng.bit_generator.state == mobile.movement_rng.bit_generator.state
        if t == 199:
            snapshot = fixed.snapshot()
            fixed.step(action)
            fixed.restore(snapshot)
    assert result[3] and not result[4]
    assert mobile.info['user_displacement_mean'] > 0


def test_reward_switch_does_not_change_observation_or_dynamics():
    a = UAVEnvironment(small_config('c_mobile_coverage').env)
    b = UAVEnvironment(small_config('d_mobile_sensing').env)
    a.reset(30003)
    b.reset(30003)
    for step in range(12):
        action = (np.arange(4, dtype=np.int64) + step) % 9
        a.step(action)
        b.step(action)
        for key in ['uav_xy', 'user_xy', 'A', 'B', 'C', 'Y', 'sensed', 'state']:
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        for key in ['fixed', 'users', 'offsets']:
            np.testing.assert_array_equal(getattr(a.obs, key), getattr(b.obs, key))


@pytest.mark.parametrize('backend', ['batch', 'process'])
@pytest.mark.parametrize('name', ['b_static_connected', 'd_mobile_sensing', 'e_mobile_sensing_connected', 'f_static_coverage'])
def test_vector_paths_preserve_ablation_semantics(backend, name):
    e = small_config(name).env
    envs = [UAVEnvironment(e) for _ in range(2)]
    for i, env in enumerate(envs):
        env.reset(30000 + i)
    configure_runtime(small_config(name))
    with VectorEnvironment(e, 2, backend) as vector:
        vector.reset([30000, 30001])
        for t in range(5):
            actions = (np.arange(8).reshape(2, 4) + t) % 9
            actual = vector.step(actions)
            for i, env in enumerate(envs):
                expected = env.step(actions[i])
                np.testing.assert_array_equal(actual[1][i], expected[1])
                assert actual[2][i] == expected[2]
                for key in ['reward_primary', 'coverage', 'discovery_fraction', 'user_displacement_mean']:
                    assert actual[-1][i][key] == expected[-1][key]


@pytest.mark.parametrize('name', list(VARIANTS))
def test_full_pipeline_resume_export_and_reward_curves(tmp_path, name):
    cfg = small_config(name)
    full = train(cfg, tmp_path / 'full')
    part = deepcopy(cfg)
    part.run.total_steps = 32
    first = train(part, tmp_path / 'split')
    train(cfg, tmp_path / 'split', first['resume_checkpoint'])
    a = load_checkpoint(full['resume_checkpoint'])
    b = load_checkpoint(tmp_path / 'split' / 'resume_000000000064')
    assert_exact(a['learner'], b['learner'])
    assert_exact(a['environments'], b['environments'])
    ecfg, policy = policy_from_checkpoint(tmp_path / 'full' / 'last.pt', torch.device('cpu'))
    rows, _ = evaluate(ecfg, policy, [30000], tmp_path / 'evaluation', trace_count=1, reward_curves=True)
    assert 'last100_discovery_fraction' in rows[0]
    with np.load(tmp_path / 'evaluation' / 'trace_30000.npz') as data:
        np.testing.assert_allclose(data['reward'], data['reward_contributions'].sum(-1))
        reconstructed = (data['reward_terms'] * data['reward_weights']).sum(-1)
        reconstructed += data['sensing_reward_weight'] * data['discovery_fraction']
        np.testing.assert_allclose(reconstructed, data['reward'])
        assert str(data['reward_primary_name']) == cfg.env.reward_primary
    with np.load(tmp_path / 'evaluation' / 'reward_curves.npz') as data:
        assert data['names'][0] == cfg.env.reward_primary
        np.testing.assert_allclose(data['contributions'].sum(-1), data['reward'])
        assert data['reward'].sum() == pytest.approx(rows[0]['return'])
    export_actor(tmp_path / 'full' / 'last.pt', tmp_path / 'actor.pt')
    deployed = DeployedActor(tmp_path / 'actor.pt')
    obs, _, _ = UAVEnvironment(ecfg.env).reset(30000)
    np.testing.assert_array_equal(deployed.act(obs), policy.act(obs, True))
    bad = deepcopy(cfg)
    bad.env.reward_primary = 'coverage' if cfg.env.reward_primary == 'sensing' else 'sensing'
    with pytest.raises(ValueError, match='Environment/algorithm changed'):
        train(bad, tmp_path / 'wrong_reward', full['resume_checkpoint'])
    bad = deepcopy(cfg)
    bad.env.user_speed_std = .2
    with pytest.raises(ValueError, match='Environment/algorithm changed'):
        train(bad, tmp_path / 'wrong_motion', full['resume_checkpoint'])


def test_legacy_default_and_invalid_objective():
    assert config_from_dict({}).env.reward_primary == 'coverage'
    with pytest.raises(ValueError, match='reward_primary'):
        Config(env=replace(EnvConfig(), reward_primary='cumulative')).validate()


def test_comparison_accepts_four_or_six_groups_and_checks_new_groups(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_ue_reward_ablations import GROUPS, check_runs, selected_groups

    assert GROUPS == VARIANTS
    paths = []
    for name in VARIANTS:
        cfg = load_config(f'configs/discrete_mappo_ablation_{name}.yaml')
        run = tmp_path / name
        run.mkdir()
        save_config(cfg, run / 'config.yaml')
        (run / 'manifest.json').write_text(json.dumps({'status': 'completed', 'env_steps': 250000}))
        paths.append(run)
    assert len(check_runs(paths[:4])) == 4
    assert len(check_runs(paths)) == 6
    for count in [0, 1, 3, 5, 7]:
        with pytest.raises(ValueError, match='Provide runs'):
            selected_groups([None] * count)
    with pytest.raises(ValueError, match='e_mobile_sensing_connected'):
        check_runs(paths[:4] + [paths[3], paths[5]])
    bad = load_config(paths[5] / 'config.yaml')
    bad.env.user_speed_std = 1.0
    save_config(bad, paths[5] / 'config.yaml')
    with pytest.raises(ValueError, match='f_static_coverage'):
        check_runs(paths)
