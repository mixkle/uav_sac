"""Read-only frozen-last action-distribution audit on existing validation scenes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from uav_research.actions import DISCRETE_DIRECTIONS
from uav_research.environment import reflect
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import as_tensor_observation
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json
from uav_research.plotting import plt, style, save_figure, read_csv

ROOT = Path('analysis/mappo_full_reward_time_factor_250k_seed0')
OUT = ROOT / 'policy_randomness'


def metrics(p, displacement):
    """Exact conditional moments, not variance of arbitrary action IDs."""
    mean = (p[..., None] * displacement).sum(-2)
    var = (p[..., None] * (displacement - mean[..., None, :])**2).sum(-2)
    mode = p.argmax(-1)
    modal = np.take_along_axis(displacement, mode[..., None, None], -2).squeeze(-2)
    mse_mode = (p * ((displacement-modal[..., None, :])**2).sum(-1)).sum(-1)
    return var.sum(-1), mse_mode


def audit(mode):
    from uav_research.vector import VectorEnvironment
    run = Path(f'runs/mappo_full_reward_{mode}_250k_seed0')
    ckpt = run/'last.pt'
    before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    cfg, policy = policy_from_checkpoint(ckpt, 'cpu')
    configure_runtime(cfg)
    policy.actor.eval()
    probabilities, positions, coverage = [], [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start+cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(seeds), 'batch') as env:
                obs, _, _ = env.reset(seeds)
                ps, xy, cov = [], [], []
                for _ in range(cfg.env.horizon):
                    dist = policy.actor.distribution(as_tensor_observation(obs, 'cpu'))
                    ps.append(dist.probs.numpy())
                    xy.append(np.stack([e.uav_xy.copy() for e in env.envs]))
                    obs, _, _, _, _, infos = env.step(dist.sample().numpy())
                    cov.append([i['coverage'] for i in infos])
                probabilities.append(np.stack(ps).transpose(1, 0, 2, 3))
                positions.append(np.stack(xy).transpose(1, 0, 2, 3))
                coverage.append(np.array(cov).T)
    p = np.concatenate(probabilities).astype(np.float64)
    xy = np.concatenate(positions)
    coverage = np.concatenate(coverage)
    old = np.load(ROOT/f'{mode}_last_critic.npz')
    np.testing.assert_array_equal(coverage, old['raw'][..., 0])
    np.testing.assert_allclose(p.sum(-1), 1, atol=2e-7)
    # Renormalize float32 rounding for precise analytical moments.
    p /= p.sum(-1, keepdims=True)
    commands = np.broadcast_to(DISCRETE_DIRECTIONS.astype(float)*cfg.env.max_speed*cfg.env.dt, (*p.shape, 2))
    actual = reflect(xy[..., None, :]+commands, [cfg.env.width, cfg.env.height])-xy[..., None, :]
    var, modal_mse = metrics(p, commands)
    actual_var, actual_modal_mse = metrics(p, actual)
    entropy = -(p*np.log(np.maximum(p, 1e-300))).sum(-1)
    top = np.sort(p, axis=-1)[..., -1]
    fields = dict(entropy=entropy, max_probability=top,
                  top_two_gap=top-np.sort(p, axis=-1)[..., -2],
                  mode_indicator_variance=top*(1-top), hover_probability=p[..., 8],
                  effective_actions=np.exp(entropy), command_variance_m2=var,
                  actual_variance_m2=actual_var, modal_mse_m2=modal_mse,
                  actual_modal_mse_m2=actual_modal_mse,
                  independent_draw_disagreement=1-(p*p).sum(-1))
    summary = {}
    for stage, sl in [('all', slice(None)), ('first100', slice(0, 100)), ('last100', slice(-100, None))]:
        q = p[:, sl]
        part = {k: float(v[:, sl].mean()) for k,v in fields.items()}
        part.update(command_rms_spread_m=float(np.sqrt(part['command_variance_m2'])),
                    actual_rms_spread_m=float(np.sqrt(part['actual_variance_m2'])),
                    modal_rms_distance_m=float(np.sqrt(part['modal_mse_m2'])),
                    fraction_top_over_50=float((q.max(-1)>.5).mean()),
                    fraction_top_over_80=float((q.max(-1)>.8).mean()),
                    fraction_top_over_90=float((q.max(-1)>.9).mean()),
                    max_probability_quantiles=np.quantile(q.max(-1), [.1,.5,.9]).tolist(),
                    action_probabilities=q.mean((0,1,2)).tolist())
        summary[stage] = part
    summary['actor_parameters'] = sum(v.numel() for v in policy.actor.parameters())
    summary['encoder_parameters'] = sum(v.numel() for v in policy.actor.encoder.parameters())
    summary['critic_parameters'] = sum(v.numel() for v in policy.value.parameters())
    summary['checkpoint_sha256'] = before
    assert before == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    np.savez_compressed(OUT/f'{mode}.npz', probabilities=p.astype(np.float32),
                        coverage=coverage, scene_seeds=cfg.run.validation_seeds,
                        action_seed=cfg.run.evaluation_action_seed, **fields)
    print(mode, json.dumps(summary['all']), flush=True)
    return summary


def plots(summary):
    style()
    fig, axes = plt.subplots(2,3,figsize=(13,7))
    for mode in summary:
        data = np.load(OUT/f'{mode}.npz')
        for ax, field, scale in zip(axes.flat[:4], ['entropy','max_probability','command_variance_m2','hover_probability'], [1,100,1,100]):
            ax.plot(np.arange(400), data[field].mean((0,2))*scale, label=mode)
        axes[1,1].hist(data['max_probability'].ravel(), bins=np.linspace(0,1,41), density=True, histtype='step', label=mode)
        axes[1,2].plot(np.arange(9), summary[mode]['all']['action_probabilities'], 'o-', label=mode)
    for ax,title in zip(axes.flat, ['Conditional entropy (nats)','Largest action probability (%)','Command variance trace (m²)','Hover probability (%)','Largest-probability distribution','Mean probabilities (hover=8)']):
        ax.set_title(title); ax.legend()
    for ax in axes.flat[:4]: ax.set_xlabel('Pre-action episode step')
    save_figure(fig, OUT, 'last_policy_randomness')
    fig, axes = plt.subplots(2,3,figsize=(13,7))
    training = {}
    for mode in summary:
        rows = read_csv(Path(f'runs/mappo_full_reward_{mode}_250k_seed0/metrics.csv'))
        x = np.array([float(r['env_steps']) for r in rows])/1000
        actor = np.array([float(r['actor_loss']) for r in rows])
        entropy = np.array([float(r['categorical_entropy']) for r in rows])
        curves = [actor+.01*entropy, -.01*entropy,
                  np.array([float(r['value_loss']) for r in rows]),
                  np.array([float(r['actor_grad_norm']) for r in rows]),
                  np.array([float(r['ppo_approx_kl']) for r in rows]),
                  np.array([float(r['ppo_clip_fraction']) for r in rows])]
        for ax, y in zip(axes.flat, curves): ax.plot(x,y,label=mode)
        after = np.array([float(r['env_steps']) for r in rows])>=100000
        # All envs start at t=0 and reset on the same 400-step schedule.
        transitions = np.diff(np.r_[0, x*1000]).round().astype(int)
        vector_steps = transitions//16
        begins = np.cumsum(vector_steps)-vector_steps
        ends = begins+vector_steps
        terminal = ends//400 > begins//400
        train = dict(last={k:float(rows[-1][k]) for k in ['actor_loss','value_loss','categorical_entropy','ppo_approx_kl','ppo_clip_fraction','actor_grad_norm','q_grad_norm']},
                     previous={k:float(rows[-2][k]) for k in ['actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction']},
                     late_value_loss_quantiles=np.quantile(curves[2][after],[0,.5,1]).tolist())
        for b in [False,True]:
            mask=after & (terminal==b)
            train[f'late_value_loss_terminal_{b}']=float(curves[2][mask].mean())
        training[mode]=train
    axes[0,2].set_yscale('log')
    for ax, name in zip(axes.flat, ['Actor surrogate term','Entropy contribution to actor loss','Value loss (log scale)','Actor gradient norm (before clipping)','Approximate KL','PPO clip fraction']):
        ax.set_title(name); ax.set_xlabel('Joint transitions (k)');ax.legend()
    save_figure(fig,OUT,'loss_decomposition')
    return training


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    # Analytical invariants for exact conditional variance.
    uniform=np.ones((1,9))/9
    v,_=metrics(uniform,np.broadcast_to(DISCRETE_DIRECTIONS.astype(float)*30,(1,9,2)))
    np.testing.assert_allclose(v,800,rtol=1e-7)
    one=np.eye(9)
    v,m=metrics(one,np.broadcast_to(DISCRETE_DIRECTIONS.astype(float)*30,(9,9,2)))
    np.testing.assert_allclose(v,0);np.testing.assert_allclose(m,0)
    summary={mode:audit(mode) for mode in ['direct','time_factor']}
    atomic_json(OUT/'summary.json',dict(policies=summary,training=plots(summary),
                protocol='Frozen last; 32 original validation scenes; 400 steps; 18 UAV; batch16; action seed880001; original coverage trajectories exactly reproduced; no training.'))
