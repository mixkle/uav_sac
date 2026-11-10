import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.diffusion_dppo.diagnosis import (
    condition_samples, counterfactuals, event_decomposition, mixed_action,
    sample_statistics, stable_seed, sampled_backhaul_hops)
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.observations import ServiceAction
from uav_research.diffusion_dppo.policy import make_service_actor


def config(kind):
    cfg = load_config('configs/diffusion_dppo_local_smoke.yaml')
    cfg.algo.name = kind
    cfg.env.motion_scaling['enabled'] = True
    cfg.env.history_motion_source = 'post_reflection_displacement'
    cfg.diffusion.minimum_free_gib = 0
    return cfg


def test_mixed_action_uses_real_ue_identity_and_copies():
    g = ServiceAction(np.zeros((2, 2)), np.array([.2, .4]), np.array([7, 9]), np.array([0, 1, 2]))
    d = ServiceAction(np.ones((2, 2))*.2, np.array([.8, .6]), g.ue_ids.copy(), g.offsets.copy())
    mixed = mixed_action(g, d, 'G/D')
    np.testing.assert_array_equal(mixed.movement, g.movement)
    np.testing.assert_array_equal(mixed.scores, d.scores)
    mixed.scores[0] = 0
    assert d.scores[0] == .8
    d.ue_ids[:] = d.ue_ids[::-1]
    with pytest.raises(ValueError, match='identity'):
        mixed_action(g, d, 'D/G')


def test_drop_partition_handles_rescue_and_new_service():
    # Three old users: one loses backhaul but is rescued, one radio loss,
    # one valid edge omitted by Top-K. A previously unserved UE gains service.
    old = np.array([[1, 0, 0, 0], [0, 1, 1, 0]])
    after = SimpleNamespace(cfg=SimpleNamespace(num_uavs=2, capacity=2, sinr_threshold_db=0),
        B=np.array([[3], [1]]), sensed=np.ones((2, 4), bool),
        Y=np.array([[1., 1., 1., 1.], [1., -1., 1., 1.]]),
        proposals=np.array([[0, 0, 0, 0], [1, 0, 0, 1]], bool),
        C=np.array([[0, 0, 0, 0], [1, 0, 0, 1]]))
    row = event_decomposition(old, after, np.ones((2, 4), bool))
    assert row['gross_backhaul_loss'] == 1 and row['rescued_users'] == 1
    assert row['lost_backhaul'] == 0 and row['lost_radio'] == 1 and row['lost_topk'] == 1
    assert row['new_users'] == 1 and row['coverage_drop'] == .25


@pytest.mark.parametrize('kind', ['diffusion_dppo', 'gaussian_service_mappo'])
def test_frozen_samples_and_full_snapshot_counterfactual(kind):
    torch.set_num_threads(2)
    cfg = config(kind)
    env = ServiceEnvironment(cfg.env)
    obs, _, _ = env.reset(10001)
    actor = make_service_actor(cfg).eval().requires_grad_(False)
    before_rng = torch.get_rng_state().clone()
    data = condition_samples(actor, obs, 4, stable_seed(1, 2), cfg, env.uav_xy)
    assert torch.equal(before_rng, torch.get_rng_state())
    stats = sample_statistics(data, obs, cfg.env.capacity, cfg.env.num_users)
    assert len(stats) == cfg.env.num_uavs
    assert all(x['effective_cov_eigen_min'] >= -1e-10 for x in stats)
    snap = copy.deepcopy(env.snapshot())
    rows = counterfactuals(snap, cfg, data, data, repeats=2)
    for sample in [0, 1]:
        chosen = [x for x in rows if x['sample'] == sample]
        assert all(x['coverage'] == chosen[0]['coverage'] for x in chosen)
        assert all(x['reward'] == chosen[0]['reward'] for x in chosen)
    restored = ServiceEnvironment(cfg.env);restored.restore(snap)
    np.testing.assert_array_equal(restored.history, env.history)
    assert restored.snapshot()['movement_rng'] == snap['movement_rng']
    predicted=sampled_backhaul_hops(env.uav_xy,data['actual'],cfg.env)
    for sample,command in enumerate(data['actions']):
        branch=ServiceEnvironment(cfg.env);branch.restore(snap);branch.step(command)
        np.testing.assert_array_equal(predicted[sample],branch.B.min(1))
