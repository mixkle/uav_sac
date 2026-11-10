"""Checks for exogenous scene reconstruction and frozen evaluation equivalence."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_mappo_generalization import (
    FIELDS, actor_digest, cluster_values, evaluate_scenes,
    independent_difference, new_scenes, reconstruct_seen,
)
from test_mappo_bridge import small
from test_online_time_factor import assert_exact
from uav_research.algorithms import make_learner
from uav_research.evaluate import evaluate
from uav_research.runtime import configure_runtime, seed_all
from uav_research.vector import VectorEnvironment


def config():
    cfg = small('M1')
    cfg.env.user_initialization = 'uniform_interior'
    cfg.env.user_initial_margin_fraction = 1 / 7
    cfg.env.uav_initialization = 'uniform'
    cfg.env.horizon = 8
    cfg.run.evaluation_action_seed = 880001
    return cfg.validate()


def test_reconstruct_actual_training_scenes_and_motion():
    cfg = config()
    seeds = [91 + i for i in range(cfg.run.num_envs)]
    ticks = 2 * cfg.env.horizon + 3
    originals = {}
    with VectorEnvironment(cfg.env, cfg.run.num_envs, 'batch') as vector:
        vector.reset(seeds)
        for tick in range(ticks):
            if tick % cfg.env.horizon == 0:
                for worker, saved in enumerate(vector.snapshot()):
                    originals[(worker, tick // cfg.env.horizon)] = saved
            # Deliberately use nonzero actions; reconstruction must ignore them
            # while reproducing the complete UE paths and scene resets.
            actions = np.full((cfg.run.num_envs, cfg.env.num_uavs, 2), .3, np.float32)
            _, _, _, done, _, _ = vector.step(actions)
            for worker in np.flatnonzero(done):
                vector.reset_at(int(worker))
        full = dict(env_steps=ticks * cfg.run.num_envs, environments=vector.snapshot())
    scenes, checks = reconstruct_seen(cfg, full, seeds, [0, 1])
    assert len(scenes) == 2 * cfg.run.num_envs
    assert all(row['exact_initial_layout_and_ue_rng_match'] for row in checks)
    for scene in scenes:
        assert_exact(scene['snapshot'], originals[(scene['worker'], scene['episode'])])
    damaged = dict(full, environments=[dict(s) for s in full['environments']])
    damaged['environments'][0]['user_xy'] = damaged['environments'][0]['user_xy'] + 1
    import pytest
    with pytest.raises(AssertionError):
        reconstruct_seen(cfg, damaged, seeds, [0, 1])


def test_frozen_snapshot_evaluation_reproduces_standard_evaluate(tmp_path):
    cfg = config()
    configure_runtime(cfg)
    seed_all(31)
    policy = make_learner(cfg, torch.device('cpu'))
    policy.actor.eval()
    policy.actor.requires_grad_(False)
    before = actor_digest(policy.actor)
    seeds = [30020, 30021]
    scenes = new_scenes(cfg, seeds, 'validation')
    raw, initial, _ = evaluate_scenes(cfg, policy, scenes, cfg.run.evaluation_action_seed)
    rows, summary = evaluate(cfg, policy, seeds, tmp_path, reward_curves=True)
    with np.load(tmp_path / 'reward_curves.npz') as saved:
        for name in ['coverage', 'discovery_fraction', 'disconnection_ratio', 'task_reward']:
            j = list(saved['diagnostic_names']).index(name)
            np.testing.assert_array_equal(raw[..., FIELDS.index(name)], saved['diagnostics'][..., j])
    np.testing.assert_allclose(raw[..., FIELDS.index('coverage')].mean(), summary['mean']['coverage'])
    assert initial.shape == (len(seeds), len(FIELDS))
    assert actor_digest(policy.actor) == before
    assert all(p.grad is None for p in policy.actor.parameters())
    repeated, _, _ = evaluate_scenes(cfg, policy, scenes, cfg.run.evaluation_action_seed)
    np.testing.assert_array_equal(raw, repeated)


def test_cluster_inference_and_difference_direction():
    records = [dict(worker=0), dict(worker=1), dict(worker=0), dict(worker=1)]
    np.testing.assert_array_equal(cluster_values(np.array([1., 4., 3., 6.]), records, True), [2., 5.])
    result = independent_difference(np.ones(16) * .5, np.ones(64) * .4)
    np.testing.assert_allclose([result[k] for k in ['delta', 'low', 'high']], -.1)
