"""Reference-sized discrete MAPPO; continuous configurations stay unchanged."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment, initialize_users
from uav_research.vector import VectorEnvironment
from uav_research.actions import DISCRETE_DIRECTIONS


def test_reference_scene_configs_and_continuous_defaults():
    configs = [load_config(f"configs/{name}.yaml") for name in (
        "discrete_mappo", "discrete_mappo_wsl_50k", "discrete_mappo_no_robustness"
    )]
    # WSL overrides throughput reward while sharing the physical scene.
    physical = [{k:v for k,v in c.as_dict()['env'].items() if k not in ('throughput_weight','disconnection_weight')} for c in configs]
    assert all(e==physical[0] for e in physical)
    e = configs[0].env
    assert (e.width, e.height, e.num_uavs, e.num_users) == (3500, 3500, 18, 150)
    assert e.bs_positions == [[0, 0, 0], [0, 3500, 0], [3500, 0, 0]]
    assert (e.fixed_obs_dim, e.fixed_obs_dim + 33, e.state_dim) == (101, 134, 6292)
    assert (e.horizon, e.max_speed * e.dt, e.robustness_weight) == (400, 30, 0)
    assert e.user_initialization == e.uav_initialization == "uniform"
    for path in ("configs/main.yaml", "configs/wsl_50k.yaml"):
        old = load_config(path).env
        assert (old.width, old.num_uavs, old.num_users, old.num_bs) == (1600, 16, 180, 2)
        assert old.user_initialization == "gaussian_mixture"
        assert (old.horizon, old.max_speed, old.robustness_weight) == (600, 10, .5)


def test_uniform_users_ignore_inactive_cluster_settings():
    cfg = load_config("configs/discrete_mappo.yaml")
    # Inactive Gaussian fields must not constrain uniform sampling.
    cfg.env.user_initial_margin_fraction = .5
    cfg.env.user_cluster_std_fraction = 0
    cfg.env.user_cluster_centers = []
    cfg.validate()
    e = replace(cfg.env, num_users=20000)
    xy = initialize_users(e, np.random.default_rng(17)) / [e.width, e.height]
    np.testing.assert_allclose(xy.mean(0), .5, atol=.01)
    np.testing.assert_allclose(xy.var(0), 1 / 12, atol=.003)
    assert (xy.min(0) < .01).all() and (xy.max(0) > .99).all()
    np.testing.assert_array_equal(
        xy, initialize_users(e, np.random.default_rng(17)) / [e.width, e.height]
    )
    bad = copy.deepcopy(cfg)
    bad.env.user_initialization = "unknown"
    with pytest.raises(ValueError, match="UE initialization"):
        bad.validate()
    bad = copy.deepcopy(cfg)
    bad.env.bs_positions[2] = bad.env.bs_positions[0].copy()
    with pytest.raises(ValueError, match="distinct"):
        bad.validate()
    bad = copy.deepcopy(cfg)
    bad.env.uav_initialization = "bs_launch"
    with pytest.raises(ValueError, match="two left-corner"):
        bad.validate()


@pytest.mark.parametrize("backend", ["batch", "process"])
def test_three_gateway_physics_match_reference_and_vector(backend):
    cfg = load_config("configs/discrete_mappo_wsl_50k.yaml")
    e = replace(cfg.env, horizon=4)
    scalar = [UAVEnvironment(e), UAVEnvironment(replace(e, backend="numpy"))]
    for env in scalar:
        env.reset(30011)
    with VectorEnvironment(e, 2, backend) as vector:
        obs, state, _ = vector.reset([30011, 30011])
        assert obs.fixed.shape == (2, 18, 101) and state.shape == (2, 6292)
        for t in range(4):
            actions = (np.arange(18) + t) % 9
            result = vector.step(np.stack([actions, actions]))
            for i, env in enumerate(scalar):
                reference = env.step(actions)
                np.testing.assert_array_equal(result[1][i], reference[1])
                assert result[2][i] == reference[2]
                assert env.B.shape == (18, 3)
                info = reference[-1]
                assert info["reward"] == pytest.approx(
                    info["coverage"] + e.throughput_weight * info["tau"]
                    + e.robustness_weight * info["robustness"]
                    - e.disconnection_weight * info["disconnection_ratio"]
                )
                rejected = sum(info[k] for k in (
                    "sensing_rejected_users", "downlink_rejected_users",
                    "backhaul_rejected_users", "capacity_rejected_users"
                ))
                assert rejected + round(info["coverage"] * 150) == 150
        assert result[3].all()


def test_thirty_meter_actions_and_four_hundred_step_termination():
    e = load_config("configs/discrete_mappo.yaml").env
    env = UAVEnvironment(e)
    env.reset(30012)
    env.uav_xy[:] = [1750, 1750]
    actions = np.tile(np.arange(9), 2)
    env.step(actions)
    np.testing.assert_allclose(
        env.uav_xy - [1750, 1750], DISCRETE_DIRECTIONS[actions] * 30, atol=2e-6
    )
    env.t = 398
    assert not env.step(np.full(18, 8))[3]
    result = env.step(np.full(18, 8))
    assert result[3] and not result[4]
