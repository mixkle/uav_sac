"""Server optimization keeps SAC targets, gradients and checkpoint semantics."""
import copy
import types

import numpy as np
import pytest
import torch

from test_online_time_factor import assert_exact
from test_sac_history import config
from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.models import DiscountedHorizon
from uav_research.replay import JointReplay
from uav_research.runtime import configure_runtime, seed_all
from uav_research.vector import VectorEnvironment


def test_server_configuration_only_changes_device():
    local = load_config('tests/fixtures/configs/sac_coverage_time_factor_relative_central_history_1m.yaml')
    remote = load_config('tests/fixtures/configs/sac_coverage_time_factor_relative_central_history_server_1m.yaml')
    assert remote.run.device == 'cuda'
    remote.run.device = local.run.device
    assert local.as_dict() == remote.as_dict()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reused_horizon_matches_recomputed_update(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA server acceptance')
    c = config(); c.run.device = device
    d = configure_runtime(c); seed_all(17)
    model = make_learner(c, d)
    reference = copy.deepcopy(model)
    # Reference discards the optional cache and recomputes H on every Q call.
    for network in (reference.critic, reference.target):
        original = type(network).forward
        def forward(self, state, actions, *, factors=None, original=original):
            return original(self, state, actions)
        network.forward = types.MethodType(forward, network)
    e = c.env
    replay = JointReplay(32, (e.num_uavs, e.obs_dim), e.state_dim, 3)
    with VectorEnvironment(e, 2, 'batch') as env:
        obs, state, _ = env.reset([30020, 30021])
        for _ in range(4):
            actions = np.full((2, e.num_uavs, 2), .2, np.float32)
            no, ns, r, done, _, _ = env.step(actions)
            replay.add(obs, state, actions, r, no, ns, done)
            obs, state = no, ns
    batch = replay.sample(8, d)
    counts = []
    hooks = [q.horizon_factor.register_forward_hook(lambda *args: counts.append(1))
             for q in (model.critic, model.target)]
    for update in range(3):
        seed_all(123 + update); actual = model.update(batch)
        seed_all(123 + update); expected = reference.update(batch)
        assert_exact(actual, expected)
        assert_exact(model.state_dict(), reference.state_dict())
    assert len(counts) == 6  # Exactly current + next H per update.
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize('time', [float('nan'), float('inf'), -.01, 1.01, .123456])
def test_time_validation_still_rejects_invalid_values(time):
    with pytest.raises(ValueError, match='finite-horizon'):
        DiscountedHorizon(400, .995)(torch.tensor([[0., time]]))
