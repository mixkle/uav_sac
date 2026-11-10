"""Reproducible frozen-policy diagnosis of the completed hop-scaling experiment."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from uav_research.runtime import atomic_json, seed_all
from uav_research.topk_diagnostic import atomic_npz, file_hash
from uav_research.diffusion_dppo.diagnosis import (
    FIXED_TIMES, COMBINATIONS, load_actor, candidate_graph, condition_samples,
    counterfactuals, event_decomposition, mixed_action, sample_statistics,
    stable_seed, read_csv, write_csv)
from uav_research.diffusion_dppo.environment import ServiceEnvironment, ServiceVector, step_batch
from uav_research.diffusion_dppo.observations import stack, offsets
from uav_research.diffusion_dppo.storage import configure


def setup(args):
    cfg, _, _ = load_actor(args.inputs, 'gaussian', 'best', args.device)
    cfg.run.cpu_threads = args.threads
    cfg.run.numba_threads = args.threads
    configure(cfg)
    return cfg


def collect(args):
    cfg, actor, saved = load_actor(args.inputs, args.group, args.kind, args.device)
    cfg.run.cpu_threads = cfg.run.numba_threads = args.threads
    configure(cfg)
    out = args.output/'states'
    out.mkdir(parents=True, exist_ok=True)
    target = out/f'{args.kind}_{args.group}.pt'
    if target.exists():
        raise FileExistsError(target)
    with np.load(args.inputs/args.kind/args.group/'step_metrics.npz') as z:
        reference = z['metrics'][:, :, z['names'].tolist().index('coverage')]
    generator = torch.Generator(device=args.device).manual_seed(880001)
    states, events, actual = [], [], []
    times = [t for t in FIXED_TIMES if t < args.steps]
    max_difference = 0.
    for start in range(0, args.scene_count, 16):
        seeds = list(range(10000+start, 10000+min(start+16, args.scene_count)))
        vector = ServiceVector(cfg.env, len(seeds), 'batch')
        obs, _, _ = vector.reset(seeds)
        worst = [None]*len(seeds)
        fixed = [{} for _ in seeds]
        series = [[] for _ in seeds]
        last_event = [-2]*len(seeds)
        merged_event = [-1]*len(seeds)
        try:
            for t in range(args.steps):
                snapshots = vector.snapshot()
                old_c = [e.C.copy() for e in vector.envs]
                old_a = [e.A.copy() for e in vector.envs]
                old_h = [e.B.min(1).copy() for e in vector.envs]
                old_coverage = [e.info['coverage'] for e in vector.envs]
                old_graph = [candidate_graph(e.obs, cfg.env.num_uavs, cfg.env.num_users) for e in vector.envs]
                action, _ = actor.sample(obs, generator=generator)
                obs, _, _, _, _, infos = vector.step(action)
                for i, (scene, env, info) in enumerate(zip(seeds, vector.envs, infos)):
                    coverage = info['coverage']
                    series[i].append(coverage)
                    item = dict(kind=args.kind, source=args.group, scene=scene, t=t,
                        state_id=f'{args.kind}:{args.group}:{scene}:{t}', is_fixed=int(t in times),
                        is_event=0, coverage_before=old_coverage[i], snapshot=snapshots[i])
                    if t in times:
                        fixed[i][t] = item
                    drop = old_coverage[i]-coverage
                    if t > 0 and drop >= .10-1e-12:
                        row = event_decomposition(old_c[i], env, old_graph[i])
                        if last_event[i] != t-1:
                            merged_event[i] += 1
                        last_event[i] = t
                        row.update(kind=args.kind, source=args.group, scene=scene, t_before=t,
                            t_after=t+1, merged_event=merged_event[i],
                            newly_disconnected=int(((old_h[i] <= cfg.env.num_uavs) &
                                                   (env.B.min(1) > cfg.env.num_uavs)).sum()),
                            uav_lost=json.dumps(np.flatnonzero((old_h[i] <= cfg.env.num_uavs) &
                                                   (env.B.min(1) > cfg.env.num_uavs)).tolist()),
                            uu_edges_lost=int(np.triu(old_a[i].astype(bool) & ~env.A.astype(bool), 1).sum()))
                        events.append(row)
                        item.update(is_event=1, event_drop=drop)
                        if worst[i] is None or drop > worst[i]['event_drop']:
                            worst[i] = item
                if t % 100 == 99:
                    print(f'collect {args.kind}/{args.group} scenes={seeds[0]}..{seeds[-1]} t={t+1}', flush=True)
            for i, scene in enumerate(seeds):
                if worst[i] is not None:
                    if worst[i]['t'] in fixed[i]:
                        fixed[i][worst[i]['t']].update(is_event=1, event_drop=worst[i]['event_drop'])
                    else:
                        fixed[i][worst[i]['t']] = worst[i]
                states.extend(fixed[i][t] for t in sorted(fixed[i]))
                actual.append(series[i])
                if args.scene_count == 32 and args.steps == 400:
                    max_difference = max(max_difference, float(np.max(np.abs(
                        np.asarray(series[i])-reference[scene-10000]))))
        finally:
            vector.close()
    if args.scene_count == 32 and args.steps == 400 and max_difference > 1e-6:
        raise AssertionError(f'Frozen replay differs from original validation: {max_difference}')
    torch.save(dict(version='diagnosis-full-state-v1', states=states,
                    config=cfg.as_dict(), action_seed=880001, checkpoint_steps=saved['env_steps']), target)
    write_csv(args.output/'exp1_motion_vs_scoring'/f'drop_attribution_{args.kind}_{args.group}.csv', events)
    atomic_npz(out/f'replay_{args.kind}_{args.group}.npz', coverage=np.asarray(actual))
    atomic_json(out/f'{args.kind}_{args.group}.json', dict(states=len(states), events=len(events),
        checkpoint_steps=saved['env_steps'], state_sha256=file_hash(target),
        replay_max_coverage_difference=max_difference,
        reference_check=args.scene_count == 32 and args.steps == 400,
        scene_count=args.scene_count, steps=args.steps))
    print('COLLECT_COMPLETE', args.kind, args.group, len(states), len(events), max_difference, flush=True)


@torch.no_grad()
def likelihood_parts(actor, obs, chain):
    device = next(actor.parameters()).device
    enc = actor.encode(obs.tensor(device))
    movement = torch.as_tensor(chain.movement, device=device)
    scores = torch.as_tensor(chain.scores, device=device)
    if chain.kind == 'gaussian':
        mm, ml, sm, sl = actor.distribution(enc)
        lm = (-.5*((movement[:, -1]-mm)*(-ml).exp()).square()-ml-.5*math.log(2*math.pi)).sum(-1)
        ls = -.5*((scores[:, -1]-sm)*(-sl).exp()).square()-sl-.5*math.log(2*math.pi)
        from uav_research.diffusion_dppo.policy import sum_segments
        return lm[:, None].cpu().numpy(), sum_segments(ls[:, None], enc.lengths).cpu().numpy()
    from uav_research.diffusion_dppo.policy import sum_segments
    mov, score = [], []
    for j, step in enumerate(reversed(range(actor.steps))):
        mm, sm = actor.transition(enc, movement[:, j], scores[:, j], step)
        std = actor.transition_std[step]
        lm = (-.5*((movement[:, j+1]-mm)/std).square()-std.log()-.5*math.log(2*math.pi)).sum(-1)
        ls = -.5*((scores[:, j+1]-sm)/std).square()-std.log()-.5*math.log(2*math.pi)
        mov.append(lm);score.append(sum_segments(ls[:, None], enc.lengths).squeeze(-1))
    return torch.stack(mov, 1).cpu().numpy(), torch.stack(score, 1).cpu().numpy()


def probe(args):
    setup(args)
    policies = {}
    for name, group, kind in [('D0', 'diffusion', 'step0'), ('Db', 'diffusion', 'best'),
                              ('Dl', 'diffusion', 'last'), ('Gb', 'gaussian', 'best'),
                              ('Gl', 'gaussian', 'last')]:
        cfg, actor, _ = load_actor(args.inputs, group, kind, args.device)
        policies[name] = (cfg, actor)
    states = []
    for group in ['diffusion', 'gaussian']:
        data = torch.load(args.output/'states'/f'{args.kind}_{group}.pt', weights_only=False)
        states.extend(data['states'])
    raw = {p: [] for p in policies}
    raw_ids = {p: [] for p in policies}
    routing = []
    statistics, interventions, densities = [], [], []
    cfg = policies['Db'][0]
    main_d, main_g = ('Db', 'Gb') if args.kind == 'best' else ('Dl', 'Gl')
    begun = time.monotonic()
    for index, item in enumerate(states):
        env = ServiceEnvironment(cfg.env)
        env.restore(item['snapshot'])
        obs = env.obs.copy()
        metadata = {k: item[k] for k in ['state_id', 'kind', 'source', 'scene', 't', 'is_fixed', 'is_event', 'coverage_before']}
        selected = {}
        for pi, (policy, (pcfg, actor)) in enumerate(policies.items()):
            seed = stable_seed(0 if args.kind == 'best' else 1,
                0 if item['source'] == 'diffusion' else 1, item['scene'], item['t'], pi)
            data = condition_samples(actor, obs, args.repeats, seed, pcfg, env.uav_xy)
            for row in sample_statistics(data, obs, cfg.env.capacity, cfg.env.num_users):
                statistics.append(dict(metadata, policy=policy, sampling_seed=seed, **row))
            raw[policy].append({k: data[k] for k in ['command', 'effective', 'actual', 'latent_movement']})
            raw_ids[policy].append(dict(scores=data['scores'].T, latent_scores=data['latent_scores'].T))
            if policy in (main_d, main_g):
                selected[policy] = data
            # Fixed, predeclared likelihood audit cohort; no optimization.
            if item['is_fixed'] and item['scene'] % 4 == 0 and item['t'] in (100, 300):
                repeated = stack([obs]*args.repeats)
                lm, ls = likelihood_parts(actor, repeated, data['chain'])
                old = data['chain'].logp
                with torch.no_grad():
                    recomputed = actor.evaluate_chain(repeated, data['chain']).cpu().numpy()
                difference = recomputed-old
                assert np.isfinite(difference).all()
                count = np.tile(obs.candidate_lengths, args.repeats)
                for j in range(actor.steps):
                    for lo, hi, bucket in [(0, 0, 'empty'), (1, cfg.env.capacity, 'within_capacity'),
                                            (cfg.env.capacity+1, 10**9, 'over_capacity')]:
                        mask = (count >= lo) & (count <= hi)
                        if not mask.any():
                            continue
                        x = difference[mask, j]
                        densities.append(dict(metadata, policy=policy, bucket=bucket,
                            denoise_step=actor.steps-j, candidate_count=float(count[mask].mean()),
                            movement_logp_mean=float(lm[mask, j].mean()), movement_logp_std=float(lm[mask, j].std()),
                            score_logp_mean=float(ls[mask, j].mean()), score_logp_std=float(ls[mask, j].std()),
                            sum_parts_error=float(np.max(np.abs((lm+ls-old)[mask, j]))),
                            recompute_logratio_p05=float(np.quantile(x, .05)),
                            recompute_logratio_p50=float(np.median(x)), recompute_logratio_p95=float(np.quantile(x, .95)),
                            recompute_max_error=float(np.max(np.abs(x))),
                            recompute_clip_fraction=float((np.abs(np.expm1(x)) > .1).mean())))
        routing.append(dict(state_id=item['state_id'], ids=obs.candidate_ids,
                            offsets=offsets(obs.candidate_lengths)))
        for row in counterfactuals(item['snapshot'], cfg, selected[main_g], selected[main_d], args.counter_repeats):
            interventions.append(dict(metadata, **row))
        if index % 20 == 19 or index == len(states)-1:
            print(f'probe {args.kind} states={index+1}/{len(states)} seconds={time.monotonic()-begun:.1f}', flush=True)
    destination = args.output/'exp2_action_distribution'
    destination.mkdir(parents=True, exist_ok=True)
    route_lengths = [len(x['ids']) for x in routing]
    for policy in policies:
        arrays = {k: np.stack([r[k] for r in raw[policy]]) for k in raw[policy][0]}
        arrays.update({k: np.concatenate([r[k] for r in raw_ids[policy]]) for k in raw_ids[policy][0]})
        arrays.update(state_ids=np.asarray([x['state_id'] for x in routing]),
                      ue_ids=np.concatenate([x['ids'] for x in routing]),
                      state_score_offsets=offsets(route_lengths),
                      candidate_uav_offsets=np.stack([x['offsets'] for x in routing]), repeats=args.repeats)
        atomic_npz(destination/f'conditional_samples_{args.kind}_{policy}.npz', **arrays)
    write_csv(destination/f'conditional_uav_{args.kind}.csv', statistics)
    write_csv(args.output/'exp1_motion_vs_scoring'/f'paired_one_step_{args.kind}.csv', interventions)
    write_csv(args.output/'exp3_dppo_update'/f'likelihood_audit_{args.kind}.csv', densities)
    atomic_json(destination/f'complete_{args.kind}.json', dict(states=len(states), repeats=args.repeats,
        intervention_rows=len(interventions), statistic_rows=len(statistics), policies=list(policies)))
    print('PROBE_COMPLETE', args.kind, flush=True)


def forks(args):
    setup(args)
    cfg, diffusion, _ = load_actor(args.inputs, 'diffusion', args.kind, args.device)
    _, gaussian, _ = load_actor(args.inputs, 'gaussian', args.kind, args.device)
    states = []
    for group in ['diffusion', 'gaussian']:
        states += [x for x in torch.load(args.output/'states'/f'{args.kind}_{group}.pt', weights_only=False)['states']
                   if x['is_fixed'] and x['t'] in (100, 200)]
    values, identities, frames = [], [], {}
    keys = ['coverage', 'tau', 'reward', 'disconnection_ratio', 'actual_speed_mean', 'invalidated_proposals']
    from uav_research.diffusion_dppo.evaluation import frame
    for start in range(0, len(states), 16):
        chosen = states[start:start+16]
        branches = []
        for item in chosen:
            env = ServiceEnvironment(cfg.env);env.restore(item['snapshot'])
            branches += [copy.deepcopy(env) for _ in COMBINATIONS]
            identities.append({k:item[k] for k in ['state_id', 'scene', 'source', 't', 'kind']})
        g_rng = torch.Generator(device=args.device).manual_seed(stable_seed(81, start, args.kind == 'last'))
        d_rng = torch.Generator(device=args.device).manual_seed(stable_seed(82, start, args.kind == 'last'))
        block = []
        for t in range(50):
            obs = stack([e.obs for e in branches])
            g = gaussian.sample(obs, generator=g_rng)[0].split(cfg.env.num_uavs)
            d = diffusion.sample(obs, generator=d_rng)[0].split(cfg.env.num_uavs)
            results = step_batch(branches, [mixed_action(ga, da, COMBINATIONS[i % 4]) for i, (ga, da) in enumerate(zip(g, d))])
            for i in range(0, len(branches), 4):
                for j in range(1, 4):
                    np.testing.assert_array_equal(branches[i].user_xy, branches[i+j].user_xy)
            block.append([[r[-1][k] for k in keys] for r in results])
            if t+1 in (10, 25, 50):
                for i, item in enumerate(chosen):
                    if item['scene'] in (10000, 10001):
                        for j, combo in enumerate(COMBINATIONS):
                            key=f"{item['state_id']}:{combo}"
                            frames.setdefault(key, []).append(frame(branches[i*4+j]))
        values.append(np.asarray(block).reshape(50, len(chosen), 4, len(keys)).transpose(1, 2, 0, 3))
        print(f'fork {args.kind} states={start+len(chosen)}/{len(states)}', flush=True)
    out = args.output/'exp1_motion_vs_scoring'
    atomic_npz(out/f'forks_{args.kind}.npz', values=np.concatenate(values), names=np.asarray(keys),
               identities_json=np.asarray([json.dumps(x) for x in identities]), combinations=np.asarray(COMBINATIONS))
    torch.save(frames, out/f'fork_frames_{args.kind}.pt')


def cases(args):
    """Export actual actions and batch policy RNG, then hold one UAV at a time."""
    cfg, actor, _ = load_actor(args.inputs, args.group, args.kind, args.device)
    cfg.run.cpu_threads=cfg.run.numba_threads=args.threads;configure(cfg)
    selected=torch.load(args.output/'states'/f'{args.kind}_{args.group}.pt',weights_only=False)['states']
    lookup={(x['scene'],x['t']):x for x in selected}
    source_rng=torch.Generator(device=args.device).manual_seed(880001)
    from uav_research.diffusion_dppo.evaluation import frame
    records={};ablations=[]
    for start in range(0,args.scene_count,16):
        seeds=list(range(10000+start,10000+min(start+16,args.scene_count)))
        vector=ServiceVector(cfg.env,len(seeds),'batch');obs,_,_=vector.reset(seeds)
        try:
            for t in range(args.steps):
                chosen=[(i,lookup[(scene,t)]) for i,scene in enumerate(seeds) if (scene,t) in lookup]
                before={i:frame(vector.envs[i]) for i,_ in chosen}
                policy_rng=source_rng.get_state().clone() if chosen else None
                action,_=actor.sample(obs,generator=source_rng)
                actions=action.split(cfg.env.num_uavs)
                obs,_,_,_,_,infos=vector.step(action)
                for i,item in chosen:
                    check=ServiceEnvironment(cfg.env);check.restore(item['snapshot'])
                    check.step(actions[i])
                    np.testing.assert_array_equal(check.C,vector.envs[i].C)
                    np.testing.assert_array_equal(check.uav_xy,vector.envs[i].uav_xy)
                    np.testing.assert_array_equal(check.user_xy,vector.envs[i].user_xy)
                    records[item['state_id']]=dict(before=before[i],after=frame(vector.envs[i]),
                        factual_action=actions[i],policy_rng_before_vector_call=policy_rng,
                        source_batch_scenes=seeds)
                    if item['is_event']:
                        original=ServiceEnvironment(cfg.env);original.restore(item['snapshot'])
                        branches=[];commands=[]
                        for held in range(cfg.env.num_uavs):
                            branches.append(copy.deepcopy(original))
                            command=copy.deepcopy(actions[i]);command.movement[held]=0
                            commands.append(command)
                        rows=step_batch(branches,commands)
                        for held,(branch,result) in enumerate(zip(branches,rows)):
                            np.testing.assert_array_equal(branch.user_xy,check.user_xy)
                            ablations.append(dict(state_id=item['state_id'],kind=args.kind,source=args.group,
                                scene=item['scene'],t=item['t'],held_uav=held,
                                factual_coverage=check.info['coverage'],hold_coverage=result[-1]['coverage'],
                                rescued_users=int(branch.C.sum()-check.C.sum()),
                                factual_disconnected=int((check.B.min(1)>cfg.env.num_uavs).sum()),
                                hold_disconnected=int((branch.B.min(1)>cfg.env.num_uavs).sum()),
                                source_hop=int(original.B.min(1)[held]),
                                source_motion_norm=float(np.linalg.norm(actions[i].movement[held])),
                                source_speed_factor=float(check.speed_factor[held])))
                if t%100==99:print(f'cases {args.kind}/{args.group} batch={start} t={t+1}',flush=True)
        finally:vector.close()
    assert len(records)==len(selected)
    torch.save(records,args.output/'states'/f'factual_{args.kind}_{args.group}.pt')
    write_csv(args.output/'exp1_motion_vs_scoring'/f'hold_one_uav_{args.kind}_{args.group}.csv',ablations)
    print('CASES_COMPLETE',args.kind,args.group,len(records),flush=True)


def noise(args):
    """Untrained inference intervention, not a new on-policy baseline."""
    cfg,actor,_=load_actor(args.inputs,args.group,args.kind,args.device)
    cfg.run.cpu_threads=cfg.run.numba_threads=args.threads;configure(cfg)
    if not 0 < args.noise_ratio <= 1:
        raise ValueError('Diagnostic noise ratio must be in (0,1]')
    if args.group=='diffusion':
        with torch.no_grad():actor.transition_std.mul_(args.noise_ratio)
        interpretation='All reverse-transition std multiplied; initial z8 standard normal unchanged'
    else:
        original=actor.distribution
        def distribution(enc):
            mm,ml,sm,sl=original(enc)
            return mm,ml+math.log(args.noise_ratio),sm,sl+math.log(args.noise_ratio)
        actor.distribution=distribution
        interpretation='Movement and score std multiplied after original log-std clipping'
    from uav_research.diffusion_dppo.evaluation import evaluate
    destination=args.output/'exp2_action_distribution'/f'noise_{args.kind}_{args.group}'
    evaluate(cfg,actor,list(range(10000,10000+args.scene_count)),destination,
             moment_repeats=0,traces=2)
    atomic_json(destination/'intervention.json',dict(noise_ratio=args.noise_ratio,
        checkpoint_sha256=file_hash(args.inputs/'snapshots'/args.group/(args.kind+'.pt')),
        semantics=interpretation,action_seed=880001,
        qualification='Changes frozen policy distribution; no inference about unique training cause'))
    print('NOISE_COMPLETE',args.kind,args.group,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['collect', 'probe', 'fork', 'cases', 'noise'])
    parser.add_argument('--inputs', type=Path, default=Path('analysis/hop_scaling_joint_service_10m_seed0'))
    parser.add_argument('--output', type=Path, default=Path('analysis/diffusion_gaussian_diagnosis_20261110'))
    parser.add_argument('--group', choices=['diffusion', 'gaussian'], default='diffusion')
    parser.add_argument('--kind', choices=['best', 'last'], default='best')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--scene-count', type=int, default=32)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--repeats', type=int, default=64)
    parser.add_argument('--counter-repeats', type=int, default=16)
    parser.add_argument('--noise-ratio', type=float, default=.8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[{'collect':'collect', 'probe':'probe', 'fork':'forks', 'cases':'cases', 'noise':'noise'}[args.stage]](args)


if __name__ == '__main__':
    main()
