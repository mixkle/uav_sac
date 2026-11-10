"""Counterfactual masks, trajectory pairing, and branch-return checks."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_mappo_deployment_branches import (
    ALL_FIELDS, FIELDS, apply_freeze, assert_no_intervention_equivalent, critical_relays, paired_difference,
    run_branch, stage_metrics,
)
from check_mappo_generalization import actor_digest
from test_generalization_audit import config
from uav_research.algorithms import make_learner
from uav_research.kernels import shortest_hops
from uav_research.runtime import configure_runtime, seed_all
from uav_research.vector import VectorEnvironment


def graph_env(adjacency, direct, counts):
    u = len(counts)
    active = np.ones(u, bool)
    hops = shortest_hops(adjacency, direct, active)
    posts = []
    for failed in range(u):
        mask = active.copy()
        mask[failed] = False
        posts.append(shortest_hops(adjacency, direct, mask))
    return SimpleNamespace(cfg=SimpleNamespace(num_uavs=u), B=hops,
                           post_hops=np.stack(posts), counts=np.array(counts))


def test_relay_role_excludes_own_users_and_uses_gateway_redundancy():
    a = np.eye(5, dtype=bool)
    for i, j in [(0, 1), (1, 2), (3, 4)]:
        a[i, j] = a[j, i] = True
    d = np.zeros((5, 2), bool)
    d[0, 0] = d[3, 1] = True
    mask, dependent, users = critical_relays(graph_env(a, d, [0, 2, 7, 0, 0]))
    np.testing.assert_array_equal(mask, [True, True, False, True, False])
    np.testing.assert_array_equal(dependent, [2, 1, 0, 1, 0])
    np.testing.assert_array_equal(users, [9, 7, 0, 0, 0])
    a = np.eye(4, dtype=bool)
    for i, j in [(0, 1), (2, 1), (1, 3)]:
        a[i, j] = a[j, i] = True
    d = np.zeros((4, 2), bool)
    d[0, 0] = d[2, 1] = True
    mask, _, _ = critical_relays(graph_env(a, d, [0, 0, 0, 5]))
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_branch_restoration_noise_and_frozen_positions(tmp_path):
    cfg = config()
    configure_runtime(cfg)
    seed_all(51)
    policy = make_learner(cfg, torch.device('cpu'))
    policy.actor.eval()
    policy.actor.requires_grad_(False)
    before = actor_digest(policy.actor)
    with VectorEnvironment(cfg.env, cfg.run.num_envs, 'batch') as vec:
        seeds = list(range(31, 31 + cfg.run.num_envs))
        obs, _, _ = vec.reset(seeds)
        for _ in range(3):
            obs, _, _, _, _, _ = vec.step(policy.act(obs))
        saved = vec.snapshot()
        rng = torch.get_rng_state().clone()
        expected = []
        for _ in range(3, cfg.env.horizon):
            obs, _, _, _, _, infos = vec.step(policy.act(obs))
            expected.append([[i[k] for k in FIELDS] for i in infos])
    relays = np.zeros((cfg.run.num_envs, cfg.env.num_uavs), bool)
    relays[:, 0] = True
    random = np.roll(relays, 1, axis=-1)
    case = dict(time=3, snapshots=saved, relay_masks=relays, random_masks=random,
                records=[dict(seed=s, scene_id=f'unit_{s}') for s in seeds])
    outputs, audits = {}, []
    for mode in ['policy', 'hover_all', 'freeze_relays', 'freeze_random']:
        outputs[mode], audit = run_branch(cfg, policy, case, mode, rng,
                                         tmp_path / mode, trace_count=1)
        audits.append(audit)
    np.testing.assert_array_equal(outputs['policy'][..., :len(FIELDS)], np.swapaxes(expected, 0, 1))
    assert len({a['ue_path_sha256'] for a in audits}) == 1
    assert all(torch.equal(a['torch_rng'], audits[0]['torch_rng']) for a in audits)
    with np.load(audits[1]['traces'][0]) as data:
        np.testing.assert_array_equal(data['uav_xy'], np.broadcast_to(data['uav_xy'][0], data['uav_xy'].shape))
        assert not np.array_equal(data['user_xy'][0], data['user_xy'][-1])
        assert int(data['episode_start_step']) == 3
        assert data['frozen_mask'].all()
    assert actor_digest(policy.actor) == before
    assert all(p.grad is None for p in policy.actor.parameters())
    case['relay_masks'][:] = False
    repeated, _ = run_branch(cfg, policy, case, 'freeze_relays', rng)
    np.testing.assert_array_equal(repeated, outputs['policy'])


def test_action_override_and_task_return_scale():
    original = np.ones((2, 3, 2), np.float32) * .3
    mask = np.array([[True, False, False], [False, True, False]])
    changed = apply_freeze(original, mask)
    assert (changed[mask] == 0).all() and (original == .3).all()
    data = np.zeros((3, 4, 5, len(ALL_FIELDS)))
    data[..., ALL_FIELDS.index('coverage')] = .5
    data[..., ALL_FIELDS.index('task_reward')] = 1.
    metrics = stage_metrics(data, [dict(branch_coverage=.6)] * 4, .5)
    np.testing.assert_allclose(metrics['post_minus_branch'], -.1)
    np.testing.assert_allclose(metrics['post_task_return'], 5.)
    np.testing.assert_allclose(metrics['discounted_task_return'], 1 + .5 + .25 + .125 + .0625)
    result = paired_difference(np.ones(32) * .4, np.ones(32) * .5)
    np.testing.assert_allclose([result[k] for k in ['delta', 'low', 'high']], .1)


def test_empty_intervention_precision_does_not_allow_service_changes():
    import pytest
    original = np.zeros((1, 3, len(ALL_FIELDS)))
    original[..., ALL_FIELDS.index('coverage')] = .5
    original[..., ALL_FIELDS.index('uav_nearest_neighbor_mean')] = 300.
    changed = original.copy()
    changed[..., ALL_FIELDS.index('uav_nearest_neighbor_mean')] += 2e-5
    assert_no_intervention_equivalent(changed, original)
    changed[..., ALL_FIELDS.index('coverage')] += 1 / 150
    with pytest.raises(AssertionError):
        assert_no_intervention_equivalent(changed, original)
