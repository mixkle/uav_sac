from dataclasses import replace
from collections import deque
import numpy as np
import pytest
from uav_research.config import Config, EnvConfig, load_config
from uav_research.environment import (
    UAVEnvironment,
    physical_links,
    reflect,
    uniform_disk,
    C_LIGHT,
)
from uav_research.kernels import shortest_hops, associate, failure_coverage
from uav_research.vector import VectorEnvironment
from uav_research.observations import RawObservation
from uav_research.experiments import assert_observations_equal


def oracle_hops(a, direct, active):
    """Independent expanded graph BFS, with BS destinations excluded as transit."""
    u, g = direct.shape
    result = np.full((u, g), u + 1, dtype=np.int64)
    for source in range(u):
        if not active[source]:
            continue
        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            node, distance = queue.popleft()
            for gateway in range(g):
                if direct[node, gateway]:
                    result[source, gateway] = min(result[source, gateway], distance + 1)
            for other in range(u):
                if active[other] and a[node, other] and other not in visited:
                    queue.append((other, distance + 1))
                    visited.add(other)
    return result


def test_dimensions_and_power_budget():
    e = EnvConfig()
    assert e.state_dim == 6643 and e.obs_dim == 91
    assert e.subband_power_dbm == pytest.approx(16.9897000434)
    assert 10 * np.log10(e.noise_mw) == pytest.approx(-105)
    assert e.beacon_noise_dbm == pytest.approx(-112.44727494897)


def test_compiled_observation_matches_reference_at_neighbor_ties_and_empty_slots():
    cfg = replace(EnvConfig(), num_uavs=5, num_users=6, capacity=3)
    positions = np.array(
        [[100, 100], [200, 100], [100, 200], [0, 100], [800, 800]], float
    )
    users = np.array(
        [[100, 100], [100, 110], [200, 100], [100, 200], [799, 799], [0, 100]], float
    )
    reference = UAVEnvironment(replace(cfg, backend="numpy"))
    fast = UAVEnvironment(replace(cfg, backend="numba"))
    ref_obs, ref_state, _ = reference.set_positions(positions, users)
    obs, state, _ = fast.set_positions(positions, users)
    assert_observations_equal(obs, ref_obs)
    np.testing.assert_array_equal(state, ref_state)
    starts = cfg.self_dim + 2 * cfg.num_bs + np.arange(3) * cfg.neighbor_dim
    np.testing.assert_allclose(obs.fixed[0, starts], [2 / 5, 3 / 5, 4 / 5])
    assert not obs.fixed[4, cfg.self_dim + 2 * cfg.num_bs :].any()


def test_single_transmitter_physical_units():
    e = replace(EnvConfig(), num_uavs=1, num_users=1)
    a, direct, sinr, rate, loss, uplink = physical_links(
        e, np.array([[0, 0]]), np.array([[0, 0]])
    )
    theta = 90
    p = 1 / (1 + e.los_a * np.exp(-e.los_b * (theta - e.los_a)))
    expected = (
        20 * np.log10(4 * np.pi * e.frequency_hz * e.altitude / C_LIGHT)
        + p * e.los_extra_db
        + (1 - p) * e.nlos_extra_db
    )
    assert loss[0, 0] == pytest.approx(expected)
    gamma = e.subband_power_dbm - expected - 10 * np.log10(e.noise_mw)
    assert sinr[0, 0] == pytest.approx(gamma)
    assert rate[0, 0] == pytest.approx(
        e.bandwidth_hz / e.capacity * np.log2(1 + 10 ** (gamma / 10))
    )


def test_interference_excludes_self_and_uses_full_load():
    e = replace(EnvConfig(), num_uavs=2, num_users=1)
    _, _, gamma, _, loss, _ = physical_links(
        e, np.array([[0, 0], [0, 0]]), np.array([[0, 0]])
    )
    received = 10 ** ((e.subband_power_dbm - loss[0, 0]) / 10)
    assert gamma[0, 0] == pytest.approx(
        10 * np.log10(received / (received + e.noise_mw))
    )
    assert gamma.max() < 0


def test_two_independent_gateway_subnets_and_isolated_uav():
    a = np.eye(5, dtype=bool)
    a[0, 1] = a[1, 0] = True
    a[3, 4] = a[4, 3] = True
    direct = np.zeros((5, 2), bool)
    direct[0, 0] = direct[4, 1] = True
    hops = shortest_hops(a, direct, np.ones(5, bool))
    np.testing.assert_array_equal(hops, [[1, 6], [2, 6], [6, 6], [6, 2], [6, 1]])
    # No implicit core-network edge connects the two BS columns.
    assert hops[1, 1] == 6 and hops[3, 0] == 6
    retained, switched, _ = failure_coverage(
        a, direct, np.ones(5, dtype=np.int64), hops, 5
    )
    assert retained[0] == pytest.approx(0.4) and retained[4] == pytest.approx(0.4)


def test_gateway_failover_fixed_association():
    a = np.eye(3, dtype=bool)
    a[0, 1] = a[1, 0] = a[1, 2] = a[2, 1] = True
    direct = np.array([[1, 0], [0, 0], [0, 1]], bool)
    nominal = shortest_hops(a, direct, np.ones(3, bool))
    retained, switched, post = failure_coverage(
        a, direct, np.array([2, 5, 3]), nominal, 10
    )
    np.testing.assert_allclose(retained, [0.8, 0.5, 0.7])
    assert switched[0] == pytest.approx(0.5)
    np.testing.assert_array_equal(post[0, 0], [4, 4])


def test_shared_gateway_bottleneck():
    a = np.ones((4, 4), bool)
    a[1:, 1:] = np.eye(3, dtype=bool)
    direct = np.zeros((4, 2), bool)
    direct[0, :] = True
    hops = shortest_hops(a, direct, np.ones(4, bool))
    retained, _, _ = failure_coverage(a, direct, np.ones(4, dtype=np.int64), hops, 4)
    assert (hops <= 4).all() and retained[0] == 0


def test_graph_kernels_against_independent_oracle():
    rng = np.random.default_rng(0)
    for u in range(2, 10):
        for _ in range(10):
            a = rng.random((u, u)) < 0.3
            a |= a.T
            np.fill_diagonal(a, True)
            direct = rng.random((u, 2)) < 0.2
            for failed in [-1, *range(u)]:
                active = np.arange(u) != failed
                np.testing.assert_array_equal(
                    shortest_hops(a, direct, active), oracle_hops(a, direct, active)
                )


def test_association_priority_capacity_and_ties():
    hops = np.array([[1, 3], [3, 1]])
    sinr = np.array([[5, 3, -10], [5, -10, 3]], float)
    c, n, _ = associate(sinr, hops, 0, 1)
    np.testing.assert_array_equal(c, [[0, 1, 0], [0, 0, 1]])
    c, _, _ = associate(np.array([[4.0], [4.0]]), np.array([[2, 3], [1, 3]]), 0, 1)
    np.testing.assert_array_equal(c, [[0], [1]])
    c, _, _ = associate(np.array([[4.0], [4.0]]), np.array([[1, 3], [1, 3]]), 4, 1)
    np.testing.assert_array_equal(c, [[1], [0]])
    c, _, _ = associate(np.array([[4.0], [4.0]]), np.full((2, 2), 3), 0, 1)
    assert c.sum() == 0


def test_distance_thresholds_and_reflection():
    e = replace(EnvConfig(), num_uavs=2, num_users=1, uu_range=300, ug_range=500)
    a, direct, *_ = physical_links(e, np.array([[0, 0], [300, 0]]), np.array([[0, 0]]))
    assert a.all()
    np.testing.assert_allclose(
        reflect(np.array([[-1, 805], [2401, -1601]]), [800, 800]), [[1, 795], [799, 1]]
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_reset_conditions_and_reward_bound(backend):
    cfg = load_config("configs/smoke.yaml").env
    cfg.backend = backend
    env = UAVEnvironment(cfg)
    obs, state, info = env.reset(0)
    assert (env.uav_xy >= 0).all()
    assert (env.uav_xy <= [cfg.width, cfg.height]).all()
    rng = np.random.default_rng(7)
    for _ in range(cfg.horizon):
        obs, state, reward, done, truncated, info = env.step(
            uniform_disk(rng, (cfg.num_uavs,))
        )
        assert (
            0 <= info["robustness"] <= (1 - 1 / cfg.num_uavs) * info["coverage"] + 1e-12
        )
        assert np.all(env.C.sum(0) <= 1) and np.all(env.C.sum(1) <= cfg.capacity)
    assert done and not truncated and state[-1] == 1
    with pytest.raises(RuntimeError):
        env.step(np.zeros((cfg.num_uavs, 2)))


def test_observation_masks_sort_and_information_isolation():
    e = replace(
        EnvConfig(), num_uavs=4, num_users=2, capacity=1, uu_range=100, ug_range=250
    )
    env = UAVEnvironment(e)
    obs, _, _ = env.set_positions(
        [[80, 80], [600, 600], [650, 600], [600, 650]], [[80, 80], [610, 610]]
    )
    start = e.self_dim + 2 * e.num_bs
    assert np.all(obs.fixed[0, start:] == 0)
    neighbors = obs.fixed[1, start:].reshape(3, e.neighbor_dim)
    np.testing.assert_allclose(neighbors[:, 0], [3 / 4, 1, 0])
    assert obs.offsets[2] > obs.offsets[1]  # Disconnected UAV still discovers UE.
    assert not env.C[1:].any()
    assert np.all(env.C <= env.sensed)
    assert obs.users.shape[1] == 4


def test_environment_restore_and_vector_equivalence():
    e = load_config("configs/smoke.yaml").env
    rng = np.random.default_rng(33)
    with VectorEnvironment(e, 2) as vector:
        obs, state, _ = vector.reset([0, 1])
        snapshots = vector.snapshot()
        actions = uniform_disk(rng, (2, e.num_uavs))
        result = vector.step(actions)
        for i in range(2):
            reference = UAVEnvironment(replace(e, backend="numpy"))
            reference.restore(snapshots[i])
            r = reference.step(actions[i])
            for x, y in zip(r[:5], [v[i] for v in result[:5]]):
                if isinstance(x, RawObservation):
                    assert_observations_equal(x, y)
                else:
                    np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)
        vector.restore(snapshots)
        repeated = vector.step(actions)
        for x, y in zip(result[:5], repeated[:5]):
            if isinstance(x, RawObservation):
                assert_observations_equal(x, y)
            else:
                np.testing.assert_array_equal(x, y)


def test_invalid_launch_rejected():
    e = replace(EnvConfig(), uav_initialization="bs_launch", launch_radius=1000)
    with pytest.raises(ValueError, match="Launch radius"):
        Config(env=e).validate()


def test_invalid_config_rejected():
    cfg = Config()
    cfg.env.ug_range = cfg.env.altitude
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = Config()
    cfg.run.validation_seeds = cfg.run.test_seeds
    with pytest.raises(ValueError):
        cfg.validate()


def test_movement_rng_is_independent_of_initialization_draws():
    cfg = load_config("configs/smoke.yaml").env
    one, two = UAVEnvironment(cfg), UAVEnvironment(cfg)
    one.reset(0)
    two.reset(0)
    two.rng.random(1000)
    action = np.zeros((cfg.num_uavs, 2), np.float32)
    one.step(action)
    two.step(action)
    np.testing.assert_array_equal(one.user_xy, two.user_xy)
