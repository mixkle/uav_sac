"""Frozen-policy SAC diagnostics; Q(s,a) includes future entropy only."""
import numpy as np


def soft_q_returns(rewards, joint_logp, gamma, alpha):
    r, logp = np.asarray(rewards, np.float64), np.asarray(joint_logp, np.float64)
    if (r.ndim != 2 or r.shape != logp.shape or not r.shape[1]
            or not np.isfinite(r).all() or not np.isfinite(logp).all()
            or not 0 <= gamma <= 1 or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Expected finite aligned [scene,time] rewards/logp and valid gamma/alpha")
    result = np.empty_like(r)
    result[:, -1] = r[:, -1]
    for t in range(r.shape[1]-2, -1, -1):
        result[:, t] = r[:, t] + gamma*(result[:, t+1]-alpha*logp[:, t+1])
    return result


def action_moments(actor, obs, step_size, samples=64):
    """Conditional command moments using an isolated caller-owned RNG stream."""
    import torch
    from .models import disk_transform
    if samples < 2:
        raise ValueError("At least two samples are required for conditional variance")
    distribution = actor.distribution(obs)
    actions = disk_transform(distribution.rsample((samples,)))[0]
    variance = actions.var(0, unbiased=True).sum(-1)*step_size**2
    return dict(command_variance_m2=variance,
                near_hover_probability=(actions.norm(dim=-1)*step_size < 1).float().mean(0),
                latent_std=distribution.stddev.mean(-1),
                mean_command_speed=(actions.norm(dim=-1)*step_size).mean(0))
