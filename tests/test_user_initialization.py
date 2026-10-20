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
