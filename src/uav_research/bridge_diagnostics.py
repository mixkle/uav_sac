"""Frozen on-policy bridge audits; never change the rollout action RNG."""
from pathlib import Path

import numpy as np
import torch

from .boundary import BOUNDARY_METRICS, boundary_distances, reward_contributions
from .actions import DISCRETE_DIRECTIONS, decode_actions
from .critic_calibration import discounted_episode_returns
from .critic_time_factor import error_metrics, horizon_factors
from .models import disk_transform, policy_actor_modules
from .observations import as_tensor_observation
from .runtime import atomic_json, isolated_rng
from .sac_diagnostics import soft_q_returns
from .vector import VectorEnvironment


DIAGNOSTICS = [
    'coverage', 'discovery_fraction', 'disconnection_ratio', 'throughput_bps',
    'robustness', 'actual_speed_mean', 'uav_nearest_neighbor_mean',
    'uav_close_neighbor_fraction', 'boundary_reflection_fraction',
    'sensing_rejected_users', 'downlink_rejected_users', 'backhaul_rejected_users',
    'capacity_rejected_users',
] + BOUNDARY_METRICS
MOMENTS = ['command_variance_m2', 'near_hover_probability',
           'mean_command_displacement_m', 'entropy_nats_per_agent']
MOVEMENT = ['command_displacement_m', 'actual_displacement_m',
            'near_zero_command_fraction', 'near_zero_actual_fraction']


def conditional_moments(actor, obs, action_space, step_size, samples=64):
    """Categorical moments are exact; transformed-Gaussian ones are sampled."""
    distribution = actor.distribution(obs)
    if action_space == 'discrete9':
        p = distribution.probs
        directions = torch.as_tensor(DISCRETE_DIRECTIONS.copy(), device=p.device) * step_size
        mean = p @ directions
        variance = (p * directions.square().sum(-1)).sum(-1) - mean.square().sum(-1)
        near = p[..., 8]
        distance = (p * directions.norm(dim=-1)).sum(-1)
        entropy = distribution.entropy()
    else:
        if samples < 2:
            raise ValueError('At least two conditional action samples required')
        action, jacobian = disk_transform(distribution.rsample((samples,)))
        variance = action.var(0, unbiased=True).sum(-1) * step_size**2
        distance = (action.norm(dim=-1) * step_size).mean(0)
        near = (action.norm(dim=-1) * step_size < 1).float().mean(0)
        entropy = distribution.entropy().sum(-1) + jacobian.mean(0)
    return torch.stack([variance.clamp_min(0), near, distance, entropy], -1)


def audit_policy(cfg, policy, output, samples=64, *, deployment_times=(), trace_seeds=(), per_agent=False):
    """Check V against environment MC; SAC Q against future-entropy soft MC.

    Output arrays are [scene,time,...]; values refer to pre-action states, while
    service diagnostics refer to the state after executing that action.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if cfg.run.evaluation_policy != 'sample':
        raise ValueError('Bridge audit requires sampled evaluation')
    if set(cfg.run.validation_seeds) & set(cfg.run.test_seeds):
        raise ValueError('Validation and reserved test seeds overlap')
    sac = cfg.algo.name == 'sac'
    moments_at = list(range(0, cfg.env.horizon, 20))
    groups = {k: [] for k in ['values', 'rewards', 'contributions', 'logp',
                              'diagnostics', 'movement', 'moments', 'boundary_distances']}
    if deployment_times:
        groups['deployment_retention'] = []
    comparison_times = sorted({t for t in (0,150,200,cfg.env.horizon-1) if t < cfg.env.horizon})
    if per_agent:
        groups.update(agent_diagnostics=[], common_means=[], common_stds=[])
    alpha = float(policy.log_alpha.exp().detach()) if sac else 0.
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(seeds), 'batch') as vector:
                obs, state, _ = vector.reset(seeds)
                chunk = {k: [] for k in groups}
                anchors = {}
                frames = {i: [env.trace_frame()] for i, env in enumerate(vector.envs)
                          if seeds[i] in trace_seeds}
                for t in range(cfg.env.horizon):
                    if t in deployment_times:
                        edges = np.stack([e.A.copy() for e in vector.envs])
                        edges[:, np.arange(cfg.env.num_uavs), np.arange(cfg.env.num_uavs)] = False
                        connected = np.stack([(e.B <= cfg.env.num_uavs).any(-1) for e in vector.envs])
                        anchors[t] = (edges, connected)
                    tensor_obs = as_tensor_observation(obs, 'cpu')
                    action, logp, _ = policy.actor.sample(tensor_obs)
                    if per_agent and t in comparison_times:
                        # Identical actual local input for every network. No
                        # random draws, extra actor inputs, or executed actions.
                        common = tensor_obs.split_agents()[0]
                        distributions = [actor.distribution(common) for actor in policy_actor_modules(policy.actor)]
                        means = torch.cat([d.loc for d in distributions], -2)
                        stds = torch.cat([d.scale for d in distributions], -2)
                        if len(distributions) == 1:
                            means=means.expand(-1,cfg.env.num_uavs,-1)
                            stds=stds.expand(-1,cfg.env.num_uavs,-1)
                        chunk['common_means'].append(means.numpy())
                        chunk['common_stds'].append(stds.numpy())
                    if t in moments_at:
                        with isolated_rng():
                            torch.manual_seed(970001 + start * cfg.env.horizon + t)
                            moments = conditional_moments(
                                policy.actor, tensor_obs, cfg.env.action_space,
                                cfg.env.max_speed * cfg.env.dt, samples,
                            )
                            chunk['moments'].append(moments.numpy())
                    value = (torch.stack(policy.critic(torch.as_tensor(state), action), -1)
                             if sac else policy.values(state)[..., None])
                    command = (decode_actions(action.numpy(), (len(seeds), cfg.env.num_uavs))
                               if cfg.env.action_space == 'discrete9' else action.numpy())
                    command_distance = np.linalg.norm(command, axis=-1) * cfg.env.max_speed * cfg.env.dt
                    obs, state, _, done, truncated, infos = vector.step(action.numpy())
                    for i in frames:
                        frames[i].append(vector.envs[i].trace_frame())
                    if deployment_times:
                        current_edges = np.stack([e.A for e in vector.envs])
                        current_nodes = np.stack([(e.B <= cfg.env.num_uavs).any(-1) for e in vector.envs])
                        retention = []
                        for at in deployment_times:
                            if at not in anchors:
                                retention.extend([np.ones(len(seeds)), np.ones(len(seeds))])
                                continue
                            edges, nodes = anchors[at]
                            ed, nd = edges.sum((1, 2)), nodes.sum(1)
                            retention.extend([
                                np.divide((edges & current_edges).sum((1, 2)), ed,
                                          out=np.ones(len(seeds)), where=ed > 0),
                                np.divide((nodes & current_nodes).sum(1), nd,
                                          out=np.ones(len(seeds)), where=nd > 0),
                            ])
                        chunk['deployment_retention'].append(np.stack(retention, -1))
                    if not np.all(done == (t == cfg.env.horizon - 1)) or truncated.any():
                        raise ValueError('Unexpected termination during full-episode audit')
                    reward = np.array([i['reward'] for i in infos])
                    contribution = np.stack([reward_contributions(i, cfg.env) for i in infos])
                    np.testing.assert_allclose(contribution.sum(-1), reward, atol=1e-12, rtol=1e-12)
                    actual_distance = np.stack([np.linalg.norm(e.actual_displacement, axis=-1)
                                                for e in vector.envs])
                    if per_agent:
                        coverage=np.asarray([i['agent_coverage'] for i in infos])
                        np.testing.assert_allclose(coverage.sum(-1),[i['coverage'] for i in infos],atol=1e-14)
                        chunk['agent_diagnostics'].append(np.stack([
                            coverage,command_distance,actual_distance,logp.numpy()],-1))
                    chunk['boundary_distances'].append(np.stack([boundary_distances(e.uav_xy, cfg.env.width, cfg.env.height).min(-1) for e in vector.envs]))
                    chunk['values'].append(value.numpy())
                    chunk['rewards'].append(reward)
                    chunk['contributions'].append(contribution)
                    chunk['logp'].append(logp.sum(-1).numpy())
                    chunk['diagnostics'].append(np.array([[i[k] for k in DIAGNOSTICS] for i in infos]))
                    chunk['movement'].append(np.stack([command_distance.mean(-1), actual_distance.mean(-1),
                        (command_distance < 1).mean(-1), (actual_distance < 1).mean(-1)], -1))
                for key in groups:
                    groups[key].append(np.swapaxes(np.stack(chunk[key]), 0, 1))
                for i, items in frames.items():
                    np.savez_compressed(output / f'trace_{seeds[i]}.npz',
                        **{k: np.stack([f[k] for f in items]) for k in items[0]},
                        bs_positions=np.asarray(cfg.env.bs_positions), scene_seed=seeds[i],
                        width=cfg.env.width, height=cfg.env.height)
    data = {key: np.concatenate(value) for key, value in groups.items()}
    if deployment_times:
        data['deployment_times'] = np.array(deployment_times)
    if per_agent:
        data['agent_diagnostic_names']=np.array(['coverage','command_displacement_m','actual_displacement_m','logp'])
        data['common_observation_times']=np.array(comparison_times)
    env_return = discounted_episode_returns(data['rewards'], cfg.algo.discount)
    label = soft_q_returns(data['rewards'], data['logp'], cfg.algo.discount, alpha) if sac else env_return
    h = horizon_factors(np.arange(cfg.env.horizon, 0, -1), cfg.algo.discount)
    components = np.stack([discounted_episode_returns(data['contributions'][..., i], cfg.algo.discount)
                           for i in range(data['contributions'].shape[-1])], -1)
    np.testing.assert_allclose(components.sum(-1), env_return, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(label[:, -1], data['rewards'][:, -1])
    task_rewards = data['contributions'][..., :4].sum(-1)
    data.update(task_rewards=task_rewards, task_returns=discounted_episode_returns(task_rewards, cfg.algo.discount),
                contribution_names=np.array(cfg.env.reward_component_names), environment_returns=env_return, value_labels=label, horizon_factors=h,
                component_returns=components, scene_seeds=np.array(cfg.run.validation_seeds),
                diagnostic_names=np.array(DIAGNOSTICS), moment_names=np.array(MOMENTS),
                moment_times=np.array(moments_at), movement_names=np.array(MOVEMENT), alpha=np.array(alpha))
    if any(not np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fc'):
        raise ValueError('Nonfinite bridge diagnostic')
    prediction = data['values'].min(-1) if sac else data['values'][..., 0]
    metrics = {}
    for scale, pred, target in [('value', prediction, label), ('average', prediction / h, label / h)]:
        result = error_metrics(pred, target)
        slices = dict(all=slice(None), first100=slice(0, 100), middle100=slice(150, 250),
                      last100=slice(-100, None), last20=slice(-20, None), last1=slice(-1, None))
        for key, row in result['stages'].items():
            selected = target[:, slices[key]]
            std = float(selected.std())
            row['label_std'] = std
            row['normalized_rmse'] = row['rmse'] / std if std else None
        metrics[scale] = result
    metrics.update(value_semantics='soft Q with future entropy' if sac else 'environment V without entropy',
                   moment_semantics='exact categorical' if cfg.env.action_space == 'discrete9' else f'{samples}-sample Gaussian',
                   alpha=alpha, moment_mean=dict(zip(MOMENTS, data['moments'].mean((0, 1, 2)).tolist())),
                   movement_mean=dict(zip(MOVEMENT, data['movement'].mean((0, 1)).tolist())))
    np.savez_compressed(output / 'audit.npz', **data)
    atomic_json(output / 'metrics.json', metrics)
    return data, metrics
