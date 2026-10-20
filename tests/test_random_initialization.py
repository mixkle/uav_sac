from dataclasses import replace
import numpy as np
from uav_research.config import EnvConfig, config_from_dict
from uav_research.environment import UAVEnvironment


def test_uniform_initialization_distribution_and_reproducibility():
    env = UAVEnvironment(EnvConfig())
    positions = []
    for seed in range(128):
        env.reset(seed)
        positions.append(env.uav_xy.copy() / 1600)
    xy = np.concatenate(positions)
    assert np.all((xy >= 0) & (xy <= 1))
    np.testing.assert_allclose(xy.mean(0), 0.5, atol=0.025)
    np.testing.assert_allclose(xy.var(0), 1 / 12, atol=0.008)
    env.reset(127)
    np.testing.assert_array_equal(env.uav_xy / 1600, positions[-1])


def test_uniform_does_not_screen_for_connectivity():
    env = UAVEnvironment(replace(EnvConfig(), ug_range=100.000001, uu_range=1))
    _, _, info = env.reset(0)
    assert info["coverage"] == 0


def test_legacy_launch_configuration_is_preserved():
    assert (
        config_from_dict({"env": {"launch_radius": 100}}).env.uav_initialization
        == "bs_launch"
    )
    assert config_from_dict({}).env.uav_initialization == "uniform"
    assert (
        config_from_dict(
            {"env": {"launch_radius": 100, "uav_initialization": "uniform"}}
        ).env.uav_initialization
        == "uniform"
    )


def test_replay_packing_matches_reference_with_duplicate_and_empty_groups():
    from uav_research.observations import RawObservation, stack_observations
    from uav_research.replay import JointReplay

    replay = JointReplay(3, (2, 3), 4)
    for key in ("obs", "next_obs"):
        replay.arrays[key][:] = np.arange(18).reshape(3, 2, 3)
    replay.users = [
        np.ones((2, 4), np.float32),
        np.empty((0, 4), np.float32),
        np.zeros((3, 4), np.float32),
    ]
    replay.next_users = [a.copy() for a in replay.users]
    replay.offsets[:] = [[0, 0, 2], [0, 0, 0], [0, 1, 3]]
    replay.next_offsets[:] = replay.offsets
    for next_ in (False, True):
        for indices in ([2, 0, 2, 1], [1, 1]):
            expected = stack_observations(
                [
                    RawObservation(
                        replay.arrays["obs"][i], replay.users[i], replay.offsets[i]
                    )
                    for i in indices
                ]
            )
            actual = replay.observations(indices, next_)
            for field in ("fixed", "users", "offsets"):
                np.testing.assert_array_equal(
                    getattr(actual, field), getattr(expected, field)
                )
