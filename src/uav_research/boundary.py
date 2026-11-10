"""Directional boundary costs and named, actual reward contributions."""
import numpy as np


def boundary_distances(xy, width, height):
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([xy[..., 0], width - xy[..., 0], xy[..., 1], height - xy[..., 1]], -1)


def outward_components(actions):
    a = np.asarray(actions, dtype=np.float64)
    return np.maximum(np.stack([-a[..., 0], a[..., 0], -a[..., 1], a[..., 1]], -1), 0)


def directional_boundary_cost(xy, actions, width, height, margin):
    """Per-UAV b in [0,1] for legal disk commands; positions are BEFORE movement."""
    weight = np.maximum(0., 1. - boundary_distances(xy, width, height) / margin)**2
    return (weight * outward_components(actions)**2).sum(-1)


def reward_contributions(info, cfg):
    values = [info['reward_primary'], cfg.throughput_weight * info['tau'],
              cfg.robustness_weight * info['robustness'], -info['disconnection_penalty']]
    if cfg.boundary_penalty_weight > 0:
        values.append(-info['boundary_penalty'])
    return np.asarray(values, dtype=np.float64)


BOUNDARY_METRICS = [
    'task_reward', 'boundary_cost', 'boundary_penalty', 'boundary_distance_mean',
    'boundary_within_50_fraction', 'boundary_within_100_fraction',
    'boundary_within_150_fraction', 'boundary_outward_fraction',
    'boundary_buffer_fraction', 'edge_served_users', 'edge_users',
    'interior_served_users', 'interior_users',
]
