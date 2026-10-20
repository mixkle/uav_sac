"""Diagnose saved discrete MAPPO policies without updates or baseline runs.

Cross-environment evaluations are sensitivity checks, not retrained ablations.
"""
from pathlib import Path
from dataclasses import replace
import csv
import hashlib
import json
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment
from uav_research.kernels import shortest_hops, associate
from uav_research.actions import DISCRETE_DIRECTIONS
from uav_research.environment import C_LIGHT
from uav_research.plotting import plt, style, save_figure, COLORS

RUN = Path('runs/wsl_discrete_mappo_paper_scene_250k_seed0')
OUT = Path('analysis/mappo_discrete_250k_seed0/diagnosis')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def stats(env):
    p, e = env.uav_xy, env.cfg
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(1)
    seen = env.sensed.any(0)
    bsdist = np.linalg.norm(p[:, None] - np.asarray(e.bs_positions)[None, :, :2], axis=-1).min(1)
    spread = np.sqrt(np.mean(np.sum((p - p.mean(0))**2, axis=1)))
    keys = ['coverage', 'reward', 'discovery_fraction', 'disconnected_uavs',
            'downlink_rejected_users', 'backhaul_rejected_users', 'capacity_rejected_users',
            'boundary_reflection_fraction', 'component_count']
    return {**{k: float(env.info[k]) for k in keys},
            'nearest_neighbor_m': float(nearest.mean()),
            'neighbor_under100_fraction': float((nearest < 100).mean()),
            'neighbor_under200_fraction': float((nearest < 200).mean()),
            'pairs_under200': float(np.triu(d < 200, 1).sum()),
            'spread_rms_m': float(spread), 'nearest_bs_m': float(bsdist.mean()),
            'discovery_multiplicity': float(env.sensed.sum() / max(1, seen.sum())),
            'idle_uav_fraction': float((env.counts == 0).mean()),
            'edge30_fraction': float(np.any((p < 30) | (p > [e.width-30, e.height-30]), axis=1).mean())}


def frozen(env):
    e = env.cfg
    dist2 = np.sum((env.uav_xy[:,None] - np.asarray(e.bs_positions)[None,:,:2])**2, axis=-1) + e.altitude**2
    candidates = np.where(env.sensed, env.Y, -np.inf)
    results = {}
    for ug in (1000, 1500):
        hops = shortest_hops(env.A, dist2 <= ug**2, np.ones(e.num_uavs, bool))
        _, n, _ = associate(candidates, hops, e.sinr_threshold_db, e.capacity)
        results[f'coverage_ug{ug}'] = float(n.sum()/e.num_users)
    # Upper diagnostic ceiling keeping discovery, interference and capacity fixed.
    _, n, _ = associate(candidates, np.ones_like(env.B), e.sinr_threshold_db, e.capacity)
    results['coverage_ideal_backhaul'] = float(n.sum()/e.num_users)
    return results


def run_case(cfg, policy, checkpoint, mode, ug, action_seed):
    ecfg = replace(cfg.env, ug_range=float(ug))
    episode_rows, curve_rows, frozen_rows = [], [], []
    all_v, all_mc = [], []
    label = f'{checkpoint}_{mode}_ug{ug}_seed{action_seed}'
    captured = {}
    hook = policy.actor.net.register_forward_hook(lambda module, inputs, output: captured.update(logits=output.detach()))
    trace_matches = 0
    with isolated_rng():
        torch.manual_seed(action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start+cfg.run.num_envs]
            with VectorEnvironment(ecfg, len(seeds), 'batch') as vector:
                obs, state, _ = vector.reset(seeds)
                logs, probabilities, actions, values, rewards = [], [], [], [], []
                official = {}
                if mode == 'sample' and ug == 1500 and action_seed == 880001 and start == 0:
                    for i, seed in enumerate(seeds[:2]):
                        path = OUT.parent / f'validation_{checkpoint}' / f'trace_{seed}.npz'
                        if path.exists(): official[i] = np.load(path)
                for step in range(ecfg.horizon+1):
                    snap = [stats(env) for env in vector.envs]
                    logs.append(snap)
                    curve_rows.append(dict(case=label, batch=start, step=step,
                                           **{k:float(np.mean([r[k] for r in snap])) for k in snap[0]}))
                    if mode == 'sample' and ug == 1500 and action_seed == 880001 and step % 100 == 0:
                        for i,env in enumerate(vector.envs):
                            frozen_rows.append(dict(checkpoint=checkpoint, scene_seed=seeds[i], step=step, **frozen(env)))
                    for i, data in official.items():
                        np.testing.assert_array_equal(vector.envs[i].uav_xy, data['uav_xy'][step])
                        if step: np.testing.assert_array_equal(vector.envs[i].last_action_ids, data['action_ids'][step])
                        trace_matches += 1
                    if step == ecfg.horizon: break
                    a = policy.act(obs, mode == 'argmax')
                    with torch.no_grad():
                        probabilities.append(captured['logits'].softmax(-1).cpu().numpy())
                        values.append(policy.values(state).cpu().numpy())
                    actions.append(a)
                    obs, state, reward, done, truncated, _ = vector.step(a)
                    # Evaluation reports float64 info rewards; vector learner rewards are float32.
                    rewards.append(np.asarray([env.info['reward'] for env in vector.envs], dtype=np.float64))
                for data in official.values(): data.close()
                prob, acts, v, rew = map(np.stack, [probabilities, actions, values, rewards])
                mc = np.zeros_like(rew); running = np.zeros(len(seeds))
                for step in reversed(range(ecfg.horizon)):
                    running = rew[step] + cfg.algo.discount*running; mc[step] = running
                all_v.append(v); all_mc.append(mc)
                for i, seed in enumerate(seeds):
                    row = dict(case=label, checkpoint=checkpoint, mode=mode, ug_range=ug,
                               action_seed=action_seed, scene_seed=seed)
                    for k in logs[0][0]:
                        row['initial_'+k] = logs[0][i][k]
                        row['mean_'+k] = float(np.mean([t[i][k] for t in logs[1:]]))
                        row['final_'+k] = logs[-1][i][k]
                    row.update(episode_return=float(rew[:,i].sum()),
                               max_probability=float(prob[:,i].max(-1).mean()),
                               entropy=float(-(prob[:,i]*np.log(prob[:,i].clip(1e-30))).sum(-1).mean()),
                               action_differs_argmax=float((acts[:,i] != prob[:,i].argmax(-1)).mean()),
                               expected_speed_m=float(np.linalg.norm(prob[:,i] @ DISCRETE_DIRECTIONS, axis=-1).mean()*ecfg.max_speed),
                               value_mc_rmse=float(np.sqrt(np.mean((v[:,i]-mc[:,i])**2))),
                               value_mc_bias=float((v[:,i]-mc[:,i]).mean()))
                    episode_rows.append(row)
    hook.remove()
    v, mc = np.concatenate(all_v, axis=1), np.concatenate(all_mc, axis=1)
    numeric = [k for k in episode_rows[0] if k not in ('case','checkpoint','mode','ug_range','action_seed','scene_seed')]
    result = {k: float(np.mean([r[k] for r in episode_rows])) for k in numeric}
    result.update(value_mc_explained_variance=float(1-np.var(mc-v)/np.var(mc)),
                  official_trace_frames_matched=trace_matches,
                  value_by_time={str(t): {'predicted':float(v[t].mean()), 'realized_discounted_return':float(mc[t].mean())}
                                 for t in (0,100,200,300,399)})
    if mode == 'sample' and ug == 1500 and action_seed == 880001:
        official_summary = json.loads((OUT.parent/f'validation_{checkpoint}'/'summary.json').read_text())['mean']
        np.testing.assert_allclose(result['mean_coverage'], official_summary['coverage'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result['episode_return'], official_summary['return'], atol=1e-10, rtol=1e-10)
    print(label, {k:round(result[k],4) for k in ['mean_coverage','final_coverage','initial_nearest_neighbor_m',
         'final_nearest_neighbor_m','final_neighbor_under200_fraction','action_differs_argmax','value_mc_explained_variance']},flush=True)
    return result, episode_rows, curve_rows, frozen_rows


def figures(summary, curves, frozen_rows):
    style()
    fig, ax = plt.subplots(figsize=(8,3.8))
    labels, means = [], []
    for ckpt in ('best','last'):
        for mode,ug in [('sample',1500),('argmax',1500),('sample',1000)]:
            key=f'{ckpt}_{mode}_ug{ug}_seed880001'
            labels.append(f'{ckpt}\n{mode}, UG{ug}');means.append(summary[key]['mean_coverage']*100)
    bars=ax.bar(range(len(labels)),means,color=[COLORS[i%3] for i in range(len(labels))])
    ax.bar_label(bars, fmt='%.2f');ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel='Mean service coverage (%)', title='Same saved weights: action selection and backhaul sensitivity',ylim=(0,max(means)*1.2))
    save_figure(fig,OUT,'controlled_evaluation')
    fig, axs = plt.subplots(2,2,figsize=(10,7))
    for ckpt,color in [('best',COLORS[0]),('last',COLORS[1])]:
        rr=[r for r in curves if r['case']==f'{ckpt}_sample_ug1500_seed880001']
        for ax,key,title in zip(axs.flat,['coverage','discovery_fraction','nearest_neighbor_m','neighbor_under200_fraction'],
                               ['Service coverage','Instantaneous discovery','Mean nearest UAV distance (m)','UAV fraction with a neighbor <200 m']):
            ys=[np.mean([r[key] for r in rr if r['step']==t]) for t in range(401)]
            ax.plot(range(401),ys,color=color,label=ckpt);ax.set(xlabel='Episode step',ylabel=title);ax.legend()
    fig.suptitle('32 fixed validation scenes | sampled policy | action seed 880001')
    save_figure(fig,OUT,'clustering_curves')
    with (RUN/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    x=np.array([float(r['env_steps']) for r in rows])/1000
    actor=np.array([float(r['actor_loss']) for r in rows]);entropy=np.array([float(r['categorical_entropy']) for r in rows])
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    axs[0,0].plot(x,actor,label='Total actor loss');axs[0,0].plot(x,actor+.01*entropy,label='Clipped surrogate loss');axs[0,0].plot(x,-.01*entropy,label='Entropy term');axs[0,0].legend(fontsize=8)
    axs[0,0].set_ylabel('Actor objective components')
    axs[0,1].semilogy(x,[float(r['value_loss']) for r in rows]);axs[0,1].set_ylabel('Normalized value loss (log scale)')
    axs[1,0].plot(x,entropy,label='Categorical entropy');axs[1,0].axhline(np.log(9),ls='--',color='grey',label='Maximum log(9)');axs[1,0].legend();axs[1,0].set_ylabel('Entropy (nats per agent)')
    axs[1,1].plot(x,[float(r['ppo_clip_fraction']) for r in rows],label='Clip fraction');axs[1,1].plot(x,[float(r['ppo_approx_kl']) for r in rows],label='Logged approximate KL');axs[1,1].legend();axs[1,1].set_ylabel('PPO update diagnostics')
    for ax in axs.flat: ax.set_xlabel('Joint environment transitions (k)')
    save_figure(fig,OUT,'loss_diagnostics')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,ckpt in zip(axs,('best','last')):
        points=summary[f'{ckpt}_sample_ug1500_seed880001']['value_by_time']
        ts=[int(t) for t in points]
        ax.plot(ts,[points[str(t)]['predicted'] for t in ts],'-o',label='Critic V(s)')
        ax.plot(ts,[points[str(t)]['realized_discounted_return'] for t in ts],'-o',label='Realized discounted return')
        ax.set(xlabel='State time (last action at t=399)',ylabel='Remaining discounted team return',title=f'{ckpt} | 32 validation scenes')
        ax.legend(fontsize=8)
    save_figure(fig,OUT,'critic_horizon')


def critic_time_probe(cfg, device):
    e=cfg.env
    min_loss=20*np.log10(4*np.pi*e.frequency_hz*e.altitude/C_LIGHT)+min(e.los_extra_db,e.nlos_extra_db)
    snr=10**((e.subband_power_dbm-min_loss)/10)/e.noise_mw
    max_rate=e.bandwidth_hz/e.capacity*np.log2(1+snr)
    result={'single_step_reward_upper_bound':1+e.throughput_weight*max_rate/e.rate_reference_bps,
            'max_single_ue_rate_bps':max_rate,'checkpoints':{}}
    assert e.robustness_weight == 0
    for name in ('best','last'):
        c,p=policy_from_checkpoint(RUN/f'{name}.pt',device)
        with VectorEnvironment(c.env,len(c.run.validation_seeds),'batch') as v:
            _,state,_=v.reset(c.run.validation_seeds)
        values={}
        for t in (0,100,200,300,399,400):
            probe=state.copy();probe[:,-1]=t/c.env.horizon
            with torch.no_grad(): values[str(t)]=float(p.values(probe).mean())
        result['checkpoints'][name]={'fixed_initial_geometry_change_only_time':values,
                                   'popart_std':p.normalizer.std.item(),'popart_mean':p.normalizer.mean.item()}
    (OUT/'critic_time_probe.json').write_text(json.dumps(result,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=load_config(RUN/'config.yaml');device=configure_runtime(cfg)
    summary={};episodes=[];curves=[];frozen_rows=[]
    for ckpt in ('best','last'):
        pcfg,policy=policy_from_checkpoint(RUN/f'{ckpt}.pt',device)
        for mode,ug,seed in [('sample',1500,880001),('argmax',1500,880001),('sample',1000,880001),('sample',1500,880002)]:
            result,ep,cs,fr=run_case(pcfg,policy,ckpt,mode,ug,seed)
            summary[f'{ckpt}_{mode}_ug{ug}_seed{seed}']=result
            episodes.extend(ep);curves.extend(cs);frozen_rows.extend(fr)
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
            write_csv(OUT/'episodes.csv',episodes);write_csv(OUT/'curves.csv',curves);write_csv(OUT/'frozen_backhaul.csv',frozen_rows)
    provenance={'run':str(RUN.resolve()),'config':cfg.as_dict(),
                'checkpoints_sha256':{k:hashlib.sha256((RUN/f'{k}.pt').read_bytes()).hexdigest() for k in ('best','last')},
                'scope':'No training or baselines. UG1000 is a same-weight environment sensitivity check. Action seeds are not independent training seeds.',
                'cluster_thresholds_m':[100,200], 'value_mc':'Single realized stochastic rollout per scene, not a noise-free ground truth value.'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    figures(summary,curves,frozen_rows)
    critic_time_probe(cfg,device)

if __name__=='__main__': main()
