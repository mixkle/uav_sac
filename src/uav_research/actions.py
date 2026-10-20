"""World-coordinate action semantics shared by environments and policies."""

import numpy as np

ACTION_MAPPING_VERSION = "world-nine-v1"
c = 1 / np.sqrt(2)
DISCRETE_DIRECTIONS = np.array(
    [[0, 1], [0, -1], [-1, 0], [1, 0], [-c, c], [-c, -c], [c, c], [c, -c], [0, 0]],
    np.float32,
)
DISCRETE_DIRECTIONS.setflags(write=False)
ACTION_NAMES = (
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "backward_left",
    "forward_right",
    "backward_right",
    "hover",
)


def decode_actions(actions, shape):
    ids = np.asarray(actions)
    if ids.shape != tuple(shape) or ids.dtype.kind not in "iu":
        raise ValueError(
            "Expected integer discrete actions with shape " + str(tuple(shape))
        )
    if np.any((ids < 0) | (ids >= 9)):
        raise ValueError("Discrete action IDs must be in [0,8]")
    return DISCRETE_DIRECTIONS[ids]


def nearest_actions(vectors):
    return (
        np.square(np.asarray(vectors)[..., None, :] - DISCRETE_DIRECTIONS)
        .sum(-1)
        .argmin(-1)
        .astype(np.int64)
    )


def action_specification(env, algorithm):
    mode = env.action_space
    return dict(
        action_space=mode,
        action_mapping_version=ACTION_MAPPING_VERSION
        if mode == "discrete9"
        else "unit-disk-v1",
        policy_distribution="categorical"
        if mode == "discrete9"
        else ("deterministic" if algorithm == "matd3" else "gaussian"),
    )


def validate_action_specification(data, env, algorithm):
    expected = action_specification(env, algorithm)
    # Historical continuous artifacts did not contain a specification.
    actual = data.get(
        "action_specification",
        action_specification(
            type("Legacy", (), {"action_space": "continuous"})(), algorithm
        ),
    )
    if actual != expected:
        raise ValueError("Checkpoint action space/mapping/distribution is incompatible")
