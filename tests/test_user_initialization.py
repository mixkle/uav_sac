from dataclasses import replace
import copy
import numpy as np
import pytest
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment, initialize_users


def test_gaussian_mixture_is_interior_reproducible_and_nonuniform():
    e = replace(load_config().env, num_users=20000)
    xy = initialize_users(e, np.random.default_rng(12))
    np.testing.assert_array_equal(xy, initialize_users(e, np.random.default_rng(12)))
    scaled = xy / [e.width, e.height]
    margin = e.user_initial_margin_fraction
    assert ((scaled >= margin) & (scaled <= 1 - margin)).all()
    assert not np.isin(
        scaled, [margin, 1 - margin]
    ).any()  # No clipped boundary masses.
    centers = np.asarray(e.user_cluster_centers)
    nearest = np.argmin(((scaled[:, None] - centers) ** 2).sum(2), axis=1)
    counts = np.bincount(nearest, minlength=4) / len(xy)
    np.testing.assert_allclose(counts, 0.25, atol=0.02)
    deviations = scaled - centers[nearest]
    assert np.std(deviations) == pytest.approx(e.user_cluster_std_fraction, abs=0.006)
    # Two separated x modes: the middle strip is much less populated than its area share.
    assert np.mean((scaled[:, 0] > 0.45) & (scaled[:, 0] < 0.55)) < 0.07
    assert abs(scaled.mean() - 0.5) < 0.01


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_initial_margin_fraction", -0.1),
        ("user_initial_margin_fraction", 0.5),
        ("user_cluster_std_fraction", 0),
        ("user_cluster_centers", []),
        ("user_cluster_centers", [[0.05, 0.3]]),
        ("user_cluster_centers", [[0.3]]),
    ],
)
def test_invalid_user_initialization_rejected(field, value):
    cfg = copy.deepcopy(load_config())
    setattr(cfg.env, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_margin_is_reset_only_not_a_new_physical_boundary():
    cfg = load_config().env
    env = UAVEnvironment(cfg)
    env.reset(8)
    positions = env.user_xy.copy()
    positions[0] = [5, 5]
    env.set_positions(env.uav_xy, positions)
    _, _, _, _, _, _ = env.step(np.zeros((cfg.num_uavs, 2), np.float32))
    assert (env.user_xy[0] < cfg.user_initial_margin_fraction * cfg.width).all()


def test_uniform_interior_is_only_affine_change_to_baseline_initial_users():
    baseline = load_config('configs/discrete_mappo_ablation_c_mobile_coverage.yaml')
    variant = load_config('configs/discrete_mappo_central_ue_250k.yaml')
    expected = copy.deepcopy(baseline)
    expected.env.user_initialization = 'uniform_interior'
    expected.env.user_initial_margin_fraction = 1 / 7
    assert variant == expected
    for seed in (0, 10000, 10031):
        a, b = UAVEnvironment(baseline.env), UAVEnvironment(variant.env)
        a.reset(seed)
        b.reset(seed)
        np.testing.assert_array_equal(a.uav_xy, b.uav_xy)
        np.testing.assert_allclose(b.user_xy, 500 + a.user_xy * (5 / 7))
        assert ((b.user_xy >= 500) & (b.user_xy <= 3000)).all()
        before_a, before_b = a.user_xy.copy(), b.user_xy.copy()
        a.step(np.full(18, 8, dtype=np.int64))
        b.step(np.full(18, 8, dtype=np.int64))
        # Independent movement stream is unchanged by the initial distribution.
        delta_a = a.user_xy - before_a
        delta_b = b.user_xy - before_b
        interior = ((before_a > 3) & (before_a < 3497)).all(axis=1)
        np.testing.assert_allclose(delta_a[interior], delta_b[interior], atol=1e-12)
    e = replace(variant.env, num_users=20000)
    xy = initialize_users(e, np.random.default_rng(17))
    np.testing.assert_allclose(xy.mean(axis=0), [1750, 1750], atol=20)
    np.testing.assert_allclose(xy.var(axis=0), [2500**2 / 12] * 2, rtol=.025)


@pytest.mark.parametrize('margin', [-.1, .5, float('nan')])
def test_uniform_interior_invalid_margin(margin):
    cfg = load_config('configs/discrete_mappo_central_ue_250k.yaml')
    cfg.env.user_initial_margin_fraction = margin
    with pytest.raises(ValueError, match='margin'):
        cfg.validate()


def test_uniform_interior_motion_can_leave_initial_square():
    cfg = load_config('configs/discrete_mappo_central_ue_250k.yaml').env
    env = UAVEnvironment(cfg)
    env.reset(8)
    positions = env.user_xy.copy()
    positions[0] = [100, 100]
    env.set_positions(env.uav_xy, positions)
    env.step(np.full(cfg.num_uavs, 8, dtype=np.int64))
    assert (env.user_xy[0] < 500).all()


@pytest.mark.parametrize('backend', ['batch', 'process'])
def test_uniform_interior_vector_paths(backend):
    from uav_research.vector import VectorEnvironment
    from uav_research.runtime import configure_runtime
    cfg = load_config('configs/discrete_mappo_central_ue_250k.yaml')
    configure_runtime(cfg)
    envs = [UAVEnvironment(cfg.env) for _ in range(2)]
    for i, env in enumerate(envs):
        env.reset(10000 + i)
    with VectorEnvironment(cfg.env, 2, backend) as vector:
        vector.reset([10000, 10001])
        actions = np.arange(cfg.env.num_uavs, dtype=np.int64) % 9
        for _ in range(3):
            actual = vector.step(np.stack([actions, actions]))
            for i, env in enumerate(envs):
                expected = env.step(actions)
                np.testing.assert_array_equal(actual[1][i], expected[1])
                assert actual[2][i] == expected[2]
