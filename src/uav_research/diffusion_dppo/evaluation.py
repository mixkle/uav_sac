"""Sampling evaluation, episode-level summaries and true terminal MC audit."""
from pathlib import Path
import csv
import time
import numpy as np
import torch

from ..runtime import isolated_rng, seed_all, atomic_json
from ..topk_diagnostic import atomic_npz
from .environment import ServiceVector
from .observations import chain_diagnostics
from .storage import Progress, ensure_budget


def mc_returns(reward, gamma):
    reward = np.asarray(reward, dtype=np.float64)
    result = np.zeros_like(reward)
    carry = np.zeros(reward.shape[:-1], np.float64)
    for t in reversed(range(reward.shape[-1])):
        carry = reward[..., t]+gamma*carry
        result[..., t] = carry
    return result


def frame(env):
    result = env.trace_frame()
    result["proposals"] = getattr(env, "proposals", np.zeros_like(env.C, bool)).copy()
    result["scores"] = getattr(env, "last_scores", np.zeros_like(env.Y)).copy()
    result["service_age"] = getattr(env, "service_age", np.zeros(env.cfg.num_users)).copy()
    return result


@torch.no_grad()
def action_moments(actor, obs, cfg, generator, repeats=64, positions=None, return_stages=False):
    total = np.zeros((*obs.fixed.shape[:-1], 2))
    total2 = np.zeros_like(total)
    effective, effective2, actual, actual2 = [np.zeros_like(total) for _ in range(4)]
    from ..motion import hop_speed_factor
    from ..environment import reflect
    ms = cfg.env.motion_scaling
    hops = np.rint(obs.fixed[..., 3]*(cfg.env.num_uavs+1)).astype(np.int64)
    scale = hop_speed_factor(hops,cfg.env.num_uavs,ms["min_ratio"],ms["hop_growth"]) if ms["enabled"] else np.ones(hops.shape)
    if positions is None:positions=obs.fixed[..., :2]*np.array([cfg.env.width,cfg.env.height])
    # Dedicated generator; these diagnostic samples never consume execution RNG.
    for _ in range(repeats):
        action, _ = actor.sample(obs, generator=generator)
        x = action.movement.astype(np.float64)*cfg.env.max_speed*cfg.env.dt
        total += x
        total2 += x*x
        y=x*scale[...,None]
        proposed=positions+y
        moved=(reflect(proposed,[cfg.env.width,cfg.env.height]) if cfg.env.uav_boundary=="reflect"
               else np.clip(proposed,[0,0],[cfg.env.width,cfg.env.height]))-positions
        effective+=y;effective2+=y*y;actual+=moved;actual2+=moved*moved
    mean = total/repeats
    variance = np.maximum(0, total2/repeats-mean*mean)
    if return_stages:
        return np.stack([mean,variance],-2),np.stack([effective/repeats,np.maximum(0,effective2/repeats-(effective/repeats)**2)],-2),np.stack([actual/repeats,np.maximum(0,actual2/repeats-(actual/repeats)**2)],-2)
    return mean, variance


def evaluate(cfg, actor, seeds, output=None, *, traces=2, critic=None, action_seed=None,
             moment_repeats=0, progress=None, greedy=False):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("Evaluation requires scenes")
    progress = progress or Progress(cfg.diffusion.heartbeat_seconds)
    started = time.perf_counter()
    all_metrics, all_values, records, all_moments = [], [], [], []
    all_effective_moments, all_actual_moments = [], []
    keys = None
    output = Path(output) if output is not None else None
    if output:
        ensure_budget(cfg, output)
        output.mkdir(parents=True, exist_ok=True)
    with isolated_rng():
        seed_all(cfg.run.evaluation_action_seed if action_seed is None else action_seed)
        for start in range(0, len(seeds), cfg.run.num_envs):
            chosen = seeds[start:start+cfg.run.num_envs]
            if greedy:
                from ..vector import VectorEnvironment
                vector = VectorEnvironment(cfg.env, len(chosen), "batch")
            else:
                vector = ServiceVector(cfg.env, len(chosen), "batch")
            try:
                obs, state, _ = vector.reset(chosen)
                frames = [[frame(e)] if start+i < traces else [] for i, e in enumerate(vector.envs)]
                values, metrics, moment_rows = [], [], []
                effective_rows, actual_rows = [], []
                probe_generator = torch.Generator(device=next(actor.parameters()).device).manual_seed(982001+start)
                for t in range(cfg.env.horizon):
                    if critic is not None:
                        values.append(critic.values(state).detach().cpu().numpy())
                    if moment_repeats and t % 20 == 0:
                        moments,effective,actual=action_moments(actor,obs,cfg,probe_generator,moment_repeats,
                            positions=np.stack([e.uav_xy for e in vector.envs]),return_stages=True)
                        moment_rows.append(moments);effective_rows.append(effective);actual_rows.append(actual)
                    action, chain = actor.sample(obs)
                    obs, state, reward, done, truncated, infos = vector.step(action)
                    for env, info in zip(vector.envs, infos):
                        info["near_hover_fraction"] = float((np.linalg.norm(env.actual_displacement, axis=1) < 1.).mean())
                    if chain is not None:
                        for info, diagnostics in zip(infos, chain_diagnostics(chain, cfg.env.num_uavs)):
                            info.update(diagnostics)
                    if keys is None:
                        keys = sorted(k for k, v in infos[0].items() if np.isscalar(v))
                    metrics.append([[info[k] for k in keys] for info in infos])
                    for i, e in enumerate(vector.envs):
                        if frames[i]:
                            frames[i].append(frame(e))
                    expected = np.asarray([r["coverage"]+cfg.env.throughput_weight*r["tau"] for r in infos])
                    if not np.allclose(reward, expected, rtol=2e-7, atol=2e-7):
                        raise AssertionError("Environment reward differs from its named contributions")
                    progress(f"evaluate scenes={chosen[0]}..{chosen[-1]} t={t+1}/{cfg.env.horizon}")
                    if t == cfg.env.horizon-1 and not done.all():
                        raise AssertionError("Full evaluation must truly terminate")
                    if truncated.any():
                        raise AssertionError("Unexpected truncation in finite task")
                metrics = np.asarray(metrics).transpose(1, 0, 2)
                all_metrics.append(metrics)
                if values:
                    all_values.append(np.asarray(values).T)
                if moment_rows:
                    all_moments.append(np.asarray(moment_rows).transpose(1, 0, 2, 3, 4))
                    all_effective_moments.append(np.asarray(effective_rows).transpose(1,0,2,3,4))
                    all_actual_moments.append(np.asarray(actual_rows).transpose(1,0,2,3,4))
                for i, seed in enumerate(chosen):
                    series = dict(zip(keys, metrics[i].T))
                    row = {"scene_seed": int(seed), "return": float(series["reward"].sum())}
                    for key in keys:
                        row[key] = float(series[key].mean())
                        row["last100_"+key] = float(series[key][-100:].mean())
                        row["final_"+key] = float(series[key][-1])
                    row["maintain_coverage"] = float(series["coverage"][-100:].mean()-series["coverage"][50:150].mean()) if cfg.env.horizon >= 150 else 0.
                    row["peak_coverage_step"] = int(np.argmax(series["coverage"])+1)
                    for k in ("direct_switches", "service_interruptions", "service_reconnections",
                              "service_reconnections_changed_uav", "completed_service_spells", "completed_service_duration_sum"):
                        if k in series:
                            row["total_"+k] = float(series[k].sum())
                    records.append(row)
                    if output and frames[i]:
                        atomic_npz(output/f"trace_{seed}.npz", **{
                            k: np.asarray([f[k] for f in frames[i]]) for k in frames[i][0]},
                            bs_positions=np.asarray(cfg.env.bs_positions), width=cfg.env.width,
                            height=cfg.env.height, scene_seed=seed)
            finally:
                vector.close()
    data = np.concatenate(all_metrics)
    summary = {k: float(np.mean([r[k] for r in records])) for k in records[0] if k != "scene_seed"}
    summary.update(scenes=len(seeds), elapsed_seconds=time.perf_counter()-started,
                   action_seed=cfg.run.evaluation_action_seed if action_seed is None else action_seed)
    result = dict(summary=summary, episodes=records)
    if output:
        with (output/"episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        atomic_npz(output/"step_metrics.npz", metrics=data, names=np.asarray(keys), scene_seeds=seeds)
        atomic_json(output/"summary.json", summary)
        if all_moments:
            atomic_npz(output/"action_moments.npz", moments=np.concatenate(all_moments),
                       effective_moments=np.concatenate(all_effective_moments),actual_moments=np.concatenate(all_actual_moments),
                       times=np.arange(0, cfg.env.horizon, 20), scene_seeds=seeds, repeats=moment_repeats)
    if all_values:
        predicted = np.concatenate(all_values)
        reward = data[:, :, keys.index("reward")]
        coverage = data[:, :, keys.index("coverage")]
        rate = cfg.env.throughput_weight*data[:, :, keys.index("tau")]
        g, gc, gr = [mc_returns(x, cfg.algo.discount) for x in (reward, coverage, rate)]
        np.testing.assert_allclose(g, gc+gr, atol=1e-5, rtol=1e-6)
        n = np.arange(cfg.env.horizon, 0, -1)
        h = n if cfg.algo.discount == 1 else -np.expm1(n*np.log(cfg.algo.discount))/(1-cfg.algo.discount)
        error = predicted-g
        summary["critic_rmse"] = float(np.sqrt(np.mean(error**2)))
        std = float(g.std())
        summary["critic_normalized_rmse"] = summary["critic_rmse"]/std if std else None
        if output:
            atomic_npz(output/"critic_predictions.npz", predicted=predicted, mc_return=g,
                       horizon_factor=h, predicted_average=predicted/h, mc_average=g/h,
                       mc_coverage=gc, mc_throughput=gr, scene_seeds=seeds)
            atomic_json(output/"summary.json", summary)
    return result
