"""Frozen best-policy topology diagnosis on the original validation protocol.

Writes only diagnostic artifacts. Does not update policies or run baselines.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from report_ue_reward_ablations import GROUPS
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


def stats(env):
    e, p = env.cfg, env.uav_xy
    hops = env.B.min(1)
    connected, direct = hops <= e.num_uavs, hops == 1
    relay = connected & ~direct
    sensed_connected = env.sensed[connected].any(0)
    sensed_direct = env.sensed[direct].any(0)
    q = p[connected]
    if len(q) > 1:
        d = np.linalg.norm(q[:, None] - q[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(1)
        nn, close = float(nearest.mean()), float((nearest < 100).mean())
    else:
        nn, close = 0.0, 0.0
    bs = np.asarray(e.bs_positions)[:, :2]
    bs_distance = np.linalg.norm(p[:, None] - bs[None, :], axis=-1).min(1)
    bounds = np.array([e.width, e.height])
    edge_distance = np.minimum(p, bounds - p).min(1)
    radius = np.sqrt(e.ug_range**2 - e.altitude**2)
    served = env.C.any(0)
    remote_users = np.linalg.norm(env.user_xy[:, None] - bs[None, :], axis=-1).min(1) > radius
    return dict(
        coverage=env.info['coverage'], sensing=env.info['discovery_fraction'], reward=env.info['reward'],
        connected_count=float(connected.sum()), direct_count=float(direct.sum()), relay_count=float(relay.sum()),
        connected_sensing=float(sensed_connected.mean()),
        relay_only_sensing=float((sensed_connected & ~sensed_direct).mean()),
        relay_served_coverage=float(env.C[relay].any(0).mean()),
        connected_nn_m=nn, connected_close100=close,
        connected_edge30=float((edge_distance[connected] < 30).mean()) if connected.any() else 0.,
        connected_edge150=float((edge_distance[connected] < 150).mean()) if connected.any() else 0.,
        connected_bs_distance_m=float(bs_distance[connected].mean()) if connected.any() else 0.,
        connected_max_extension_m=float(np.maximum(bs_distance[connected] - radius, 0).max()) if connected.any() else 0.,
        connected_mean_hops=float(hops[connected].mean()) if connected.any() else 0.,
        connected_max_hops=float(hops[connected].max()) if connected.any() else 0.,
        connected_sensing_multiplicity=float(env.sensed[connected].sum()/max(1, sensed_connected.sum())),
        connected_idle_count=float((connected & (env.counts == 0)).sum()),
        remote_service_coverage=float((served & remote_users).mean()),
        downlink_rejected_users=env.info['downlink_rejected_users'],
        backhaul_rejected_users=env.info['backhaul_rejected_users'],
        capacity_rejected_users=env.info['capacity_rejected_users'],
        user_displacement_mean=env.info['user_displacement_mean'],
    )


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(root, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries, curves, stages, episodes = {}, [], [], []
    for group in GROUPS:
        report = json.loads((root/group/'report.json').read_text())
        saved = report['evaluations']['best']
        checkpoint = Path(saved['checkpoint'])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == saved['checkpoint_sha256']
        cfg = load_config(Path(report['run'])/'config.yaml')
        device = configure_runtime(cfg)
        cfg, policy = policy_from_checkpoint(checkpoint, device)
        assert cfg.run.evaluation_policy == 'sample'
        assert not set(cfg.run.validation_seeds) & set(cfg.run.test_seeds)
        batches, fields = [], None
        verified_frames = 0
        with isolated_rng():
            torch.manual_seed(cfg.run.evaluation_action_seed)
            for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
                seeds = cfg.run.validation_seeds[start:start+cfg.run.num_envs]
                with VectorEnvironment(cfg.env, len(seeds), 'batch') as vector:
                    obs, _, _ = vector.reset(seeds)
                    traces = {i: np.load(root/group/'validation_best'/f'trace_{seed}.npz')
                              for i, seed in enumerate(seeds) if seed in cfg.run.validation_seeds[:2]}
                    logs = []
                    for t in range(cfg.env.horizon+1):
                        rows = [stats(env) for env in vector.envs]
                        fields = list(rows[0])
                        logs.append([[row[k] for k in fields] for row in rows])
                        for i, trace in traces.items():
                            np.testing.assert_array_equal(vector.envs[i].uav_xy, trace['uav_xy'][t])
                            np.testing.assert_array_equal(vector.envs[i].user_xy, trace['user_xy'][t])
                            np.testing.assert_array_equal(vector.envs[i].B, trace['B'][t])
                            verified_frames += 1
                        if t < cfg.env.horizon:
                            actions = policy.act(obs, False)
                            obs, _, _, _, _, _ = vector.step(actions)
                    for trace in traces.values():
                        trace.close()
                    batches.append(np.asarray(logs))
        data = np.concatenate(batches, axis=1)
        means = data.mean(1)
        with (root/group/'validation_best'/'episodes.csv').open() as stream:
            original_rows = {int(row['scene_seed']): row for row in csv.DictReader(stream)}
        for i, seed in enumerate(cfg.run.validation_seeds):
            for key, original in [('coverage', 'coverage'), ('sensing', 'discovery_fraction')]:
                np.testing.assert_allclose(data[1:,i,fields.index(key)].mean(), float(original_rows[seed][original]), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(data[1:,i,fields.index('reward')].sum(), float(original_rows[seed]['return']), atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(data[1:, :, fields.index('coverage')].mean(), saved['summary']['mean']['coverage'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(data[1:, :, fields.index('reward')].sum(0).mean(), saved['summary']['mean']['return'], atol=1e-10, rtol=1e-12)
        np.savez_compressed(output/f'{group}_best.npz', values=data, fields=fields,
                            scene_seeds=cfg.run.validation_seeds, checkpoint_env_steps=saved['checkpoint_env_steps'])
        for t in range(len(means)):
            curves.append(dict(group=group, step=t, **dict(zip(fields, means[t]))))
        group_stages = {}
        for stage, values in [('initial', data[0]), ('midpoint', data[200]), ('final', data[-1]),
                              ('mean', data[1:].mean(0)), ('last100', data[-100:].mean(0))]:
            values = values.mean(0)
            group_stages[stage] = dict(zip(fields, values))
            stages.append(dict(group=group, best_env_steps=saved['checkpoint_env_steps'], stage=stage, **group_stages[stage]))
        for i, seed in enumerate(cfg.run.validation_seeds):
            row = dict(group=group, scene_seed=seed)
            for stage, values in [('initial', data[0, i]), ('midpoint', data[200, i]), ('final', data[-1, i]),
                                  ('mean', data[1:, i].mean(0)), ('last100', data[-100:, i].mean(0))]:
                row.update({f'{stage}_{k}': float(v) for k,v in zip(fields,values)})
            episodes.append(row)
        summaries[group] = dict(best_env_steps=saved['checkpoint_env_steps'], checkpoint_sha256=digest,
                                matched_trace_frames=verified_frames, stages=group_stages)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        print(group, {stage: {k: round(group_stages[stage][k],4) for k in
              ['coverage','connected_count','relay_count','connected_sensing','connected_edge30']}
              for stage in ['initial','midpoint','final']}, flush=True)
    write_csv(output/'curves.csv', curves)
    write_csv(output/'stages.csv', stages)
    write_csv(output/'episodes.csv', episodes)
    atomic_json(output/'summary.json', summaries)
    figures(summaries, curves, output)
    final_topology_panels(root, output, summaries)


def figures(summaries, curves, output):
    style()
    panels = [('coverage', 'Actual service coverage', 100, '% of UE'),
              ('connected_sensing', 'UE sensed by connected UAVs', 100, '% of UE'),
              ('connected_count', 'BS-reachable UAVs', 1, 'UAV count'),
              ('relay_count', 'Reachable UAVs requiring relays', 1, 'UAV count'),
              ('connected_edge30', 'Connected UAVs within 30 m of boundary', 100, '% of connected UAVs'),
              ('connected_nn_m', 'Nearest connected-UAV distance', 1, 'm')]
    fig, axes = plt.subplots(2,3,figsize=(14,7.7))
    for i, group in enumerate(GROUPS):
        selected = [row for row in curves if row['group']==group]
        label=group[0].upper()+(' (step 0)' if summaries[group]['best_env_steps']==0 else '')
        for ax,(key,title,scale,unit) in zip(axes.flat,panels):
            ax.plot([r['step'] for r in selected],[r[key]*scale for r in selected],
                    color=COLORS[i],label=label,ls='--' if group.startswith('e_') else '-',lw=1.4,alpha=.9)
            ax.set(title=title,xlabel='Episode step',ylabel=unit)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Best checkpoint | paired 32 validation scenes | D/E are identical untrained actors',fontsize=12)
    save_figure(fig,output/'figures','connected_network_evolution')


def final_topology_panels(root, output, summaries):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    style()
    for seed in [10000, 10001]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        for ax, group in zip(axes.flat, GROUPS):
            with np.load(root/group/'validation_best'/f'trace_{seed}.npz') as trace:
                p, users, bs = trace['uav_xy'][-1], trace['user_xy'][-1], trace['bs_positions'][:, :2]
                hops = trace['B'][-1].min(1)
                direct, connected = hops == 1, hops <= len(p)
                relay = connected & ~direct
                served = trace['C'][-1].any(0)
                sensed = trace['sensed'][-1].any(0)
                cfg = load_config(Path(json.loads((root/group/'report.json').read_text())['run'])/'config.yaml')
                radius = np.sqrt(cfg.env.ug_range**2-cfg.env.altitude**2)
                for b in bs:
                    ax.add_patch(Circle(b, radius, fill=False, ls=':', color='#999999', lw=.6))
                ax.scatter(users[~sensed,0], users[~sensed,1], s=7, color='#bbbbbb', alpha=.6)
                ax.scatter(users[sensed & ~served,0], users[sensed & ~served,1], s=8, color=COLORS[3], alpha=.7)
                ax.scatter(users[served,0], users[served,1], s=10, color=COLORS[2], alpha=.8)
                for i in range(len(p)):
                    for j in range(i):
                        if trace['A'][-1,i,j]:
                            ax.plot(p[[i,j],0],p[[i,j],1],color=COLORS[0] if connected[i] else '#999999',lw=.7,alpha=.7)
                    for g in range(len(bs)):
                        if trace['B'][-1,i,g]==1:
                            ax.plot([p[i,0],bs[g,0]],[p[i,1],bs[g,1]],color=COLORS[1],ls='--',lw=.7)
                for mask,color,marker in [(direct,COLORS[0],'^'),(relay,COLORS[2],'s'),(~connected,COLORS[4],'^')]:
                    ax.scatter(p[mask,0],p[mask,1],s=38,color=color,marker=marker,edgecolors='white',lw=.4,zorder=4)
                ax.scatter(bs[:,0],bs[:,1],marker='*',s=105,color=COLORS[1],clip_on=False,zorder=5)
                step=summaries[group]['best_env_steps']
                ax.set(xlim=(0,cfg.env.width),ylim=(0,cfg.env.height),aspect='equal',xlabel='x (m)',ylabel='y (m)',
                       title=f"{group[0].upper()} | best {step:,}"+(' (untrained)' if step==0 else '')+
                             f"\nReachable {connected.sum()}/18, via relay {relay.sum()} | served {served.mean():.1%}")
        handles=[Line2D([],[],ls='none',marker=m,color=c,label=l) for m,c,l in
                 [('*',COLORS[1],'BS'),('^',COLORS[0],'Direct BS access'),('s',COLORS[2],'Requires relays'),
                  ('^',COLORS[4],'Disconnected'),('o',COLORS[2],'Served UE'),('o',COLORS[3],'Sensed, unserved UE')]]
        fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.04),frameon=False)
        fig.suptitle(f'Best policies: terminal topology | scene {seed} | dotted arcs: direct BS-access radius',fontsize=12)
        save_figure(fig,output/'figures',f'best_final_networks_{seed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('analysis/ue_reward_ablations_250k_seed0'))
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    diagnose(args.root.resolve(), (args.output or args.root/'topology_diagnosis').resolve())


if __name__ == '__main__':
    main()
