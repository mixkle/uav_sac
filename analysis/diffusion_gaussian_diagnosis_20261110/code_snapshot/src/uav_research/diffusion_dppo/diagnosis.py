"""Frozen policy interventions. No production environment or policy semantics change."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..environment import reflect
from ..runtime import atomic_json
from ..topk_diagnostic import maximum_assignment
from .config import from_dict
from .environment import ServiceEnvironment, resolve_topk, step_batch
from .observations import ServiceAction, stack, offsets
from .policy import make_service_actor
from .storage import load


FIXED_TIMES = (25, 100, 200, 300, 375)
COMBINATIONS = ('G/G', 'D/D', 'G/D', 'D/G')


def stable_seed(*values):
    return int(np.random.SeedSequence([991001, *map(int, values)]).generate_state(1)[0])


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_actor(root, group, kind, device):
    path = Path(root)/'snapshots'/group/(kind+'.pt')
    saved = load(path, kinds={'dppo_model'})
    cfg = from_dict(saved['config'])
    cfg.run.device = device
    cfg.diffusion.actor_update_devices = []
    actor = make_service_actor(cfg).to(device)
    actor.load_state_dict(saved['learner']['actor'])
    actor.eval().requires_grad_(False)
    return cfg, actor, saved


def check_same_routing(left, right):
    if not (np.array_equal(left.ue_ids, right.ue_ids) and
            np.array_equal(left.offsets, right.offsets)):
        raise ValueError('UE identity/offset mismatch in mixed action')


def mixed_action(gaussian, diffusion, combination):
    check_same_routing(gaussian, diffusion)
    movement = gaussian if combination[0] == 'G' else diffusion
    scores = gaussian if combination[-1] == 'G' else diffusion
    return ServiceAction(movement.movement.copy(), scores.scores.copy(),
                         scores.ue_ids.copy(), scores.offsets.copy())


def candidate_graph(obs, u, m):
    graph = np.zeros((u, m), bool)
    route = offsets(obs.candidate_lengths)
    for i in range(u):
        graph[i, obs.candidate_ids[route[i]:route[i+1]]] = True
    return graph


def event_decomposition(previous_c, after, before_graph, threshold=0.10):
    """Fixed-order filtering of incumbent edges, with rescued/new service separate.

    Gross losses can be rescued by a different UAV. Actual lost-user attribution
    uses the same backhaul -> radio -> candidate -> Top-K priority.
    """
    previous_c = previous_c.astype(bool)
    m = previous_c.shape[1]
    old_served = previous_c.any(0)
    served = after.C.any(0)
    backhaul = np.broadcast_to((after.B.min(1) <= after.cfg.num_uavs)[:, None], previous_c.shape)
    radio = after.sensed & (after.Y >= after.cfg.sinr_threshold_db)
    stages = [previous_c, previous_c & backhaul]
    stages += [stages[-1] & radio]
    stages += [stages[-1] & before_graph]
    stages += [stages[-1] & after.proposals]
    masks = [x.any(0) for x in stages]
    counts = [int(x.sum()) for x in masks]
    names = ('backhaul', 'radio', 'candidate_timing', 'topk')
    row = {f'gross_{name}_loss': counts[i]-counts[i+1] for i, name in enumerate(names)}
    lost = old_served & ~served
    for i, name in enumerate(names):
        row[f'lost_{name}'] = int((lost & masks[i] & ~masks[i+1]).sum())
    if sum(row['lost_'+name] for name in names) != int(lost.sum()):
        raise AssertionError('Lost-user decomposition is incomplete')
    new = served & ~old_served
    rescued = served & old_served & ~masks[-1]
    row.update(old_served=int(old_served.sum()), served=int(served.sum()),
               lost_users=int(lost.sum()), new_users=int(new.sum()), rescued_users=int(rescued.sum()),
               coverage_drop=float((old_served.sum()-served.sum())/m))
    assert row['old_served']-row['served'] == row['lost_users']-row['new_users']
    assert row['old_served']-row['served'] == sum(row['gross_'+n+'_loss'] for n in names)-row['rescued_users']-row['new_users']
    row['is_drop_event'] = int(row['coverage_drop'] >= threshold-1e-12)
    row['reachable_after'] = int((after.B.min(1) <= after.cfg.num_uavs).sum())
    edges = backhaul & radio
    row['physical_max_served'] = int((maximum_assignment(edges, after.cfg.capacity) >= 0).sum())
    row['protocol_gap_users'] = row['physical_max_served']-row['served']
    row['feasibility_gap_users'] = m-row['physical_max_served']
    return row


def condition_samples(actor, obs, repeats, seed, cfg, positions):
    """One batched call, complete raw actions retained; no execution RNG consumed."""
    repeated = stack([obs]*repeats)
    generator = torch.Generator(device=next(actor.parameters()).device).manual_seed(seed)
    action, chain = actor.sample(repeated, generator=generator)
    u = cfg.env.num_uavs
    r = len(obs.candidate_ids)
    command = action.movement.reshape(repeats, u, 2).astype(np.float64)*cfg.env.max_speed*cfg.env.dt
    hops = np.rint(obs.fixed.reshape(u, -1)[:, 3]*(u+1)).astype(np.int64)
    from ..motion import hop_speed_factor
    ms = cfg.env.motion_scaling
    scale = hop_speed_factor(hops, u, ms['min_ratio'], ms['hop_growth']) if ms['enabled'] else np.ones(u)
    effective = command*scale[None, :, None]
    actual = reflect(positions[None]+effective, [cfg.env.width, cfg.env.height])-positions[None]
    scores = action.scores.reshape(repeats, r)
    latent_m = chain.movement[:, -1].reshape(repeats, u, 2)
    latent_s = chain.scores[:, -1].reshape(repeats, r)
    actions = action.split(u)
    return dict(command=command, effective=effective, actual=actual, scores=scores,
                latent_movement=latent_m, latent_scores=latent_s, hops=hops, scale=scale,
                actions=actions, chain=chain)


def sample_statistics(data, obs, capacity, m):
    command, effective, actual = [data[k] for k in ('command', 'effective', 'actual')]
    k, u, _ = command.shape
    route = offsets(obs.candidate_lengths)
    ids = obs.candidate_ids
    before = candidate_graph(obs, u, m)
    owners, choices = [], []
    for sample in range(k):
        values = np.zeros((u, m))
        for i in range(u):
            lo, hi = route[i:i+2]
            values[i, ids[lo:hi]] = data['scores'][sample, lo:hi]
        c, _, _ = resolve_topk(before, before, values, capacity)
        owners.append(np.where(c.any(0), c.argmax(0), -1))
        choices.append(np.where(before.any(0), np.where(before, values, -1.).argmax(0), -1))
    owners, choices = np.asarray(owners), np.asarray(choices)
    observed = before.any(0)
    pair_a, pair_b = np.arange(0, k-1, 2), np.arange(1, k, 2)
    owner_consistency = float((owners[pair_a][:, observed] == owners[pair_b][:, observed]).mean()) if observed.any() else np.nan
    choice_consistency = float((choices[pair_a][:, observed] == choices[pair_b][:, observed]).mean()) if observed.any() else np.nan
    rows = []
    for i in range(u):
        row = dict(uav=i, hop=int(data['hops'][i]), scale=float(data['scale'][i]),
                   candidate_count=int(route[i+1]-route[i]), sample_count=k,
                   ue_owner_consistency=owner_consistency, ue_preference_consistency=choice_consistency)
        for name, value in [('command', command), ('effective', effective), ('actual', actual)]:
            x = value[:, i]
            mean = x.mean(0)
            cov = (x-mean).T@(x-mean)/k
            eig = np.linalg.eigvalsh(cov)
            norm = np.linalg.norm(x, axis=1)
            unit = x/np.maximum(norm[:, None], 1e-12)
            row.update({name+'_mean_norm': float(np.linalg.norm(mean)),
                        name+'_magnitude_mean': float(norm.mean()),
                        name+'_variance_trace': float(np.trace(cov)),
                        name+'_cov_eigen_min': float(eig[0]), name+'_cov_eigen_max': float(eig[1]),
                        name+'_near_hover': float((norm < 1).mean()),
                        name+'_direction_resultant': float(np.linalg.norm(unit.mean(0)))})
        lo, hi = route[i:i+2]
        if hi > lo:
            q = data['scores'][:, lo:hi]
            z = data['latent_scores'][:, lo:hi]
            top = np.zeros_like(q, bool)
            for draw in range(k):
                selected = np.lexsort((ids[lo:hi], -q[draw]))[:capacity]
                top[draw, selected] = True
            union = (top[pair_a] | top[pair_b]).sum(1)
            row.update(score_mean=float(q.mean()), score_variance=float(q.var(0).mean()),
                       score_latent_mean=float(z.mean()), score_latent_variance=float(z.var(0).mean()),
                       topk_jaccard=float(((top[pair_a] & top[pair_b]).sum(1)/union).mean()))
        else:
            row.update(score_mean=np.nan, score_variance=np.nan, score_latent_mean=np.nan,
                       score_latent_variance=np.nan, topk_jaccard=np.nan)
        rows.append(row)
    return rows


def counterfactuals(snapshot, cfg, gaussian, diffusion, repeats=16):
    env = ServiceEnvironment(cfg.env)
    env.restore(snapshot)
    branches, labels = [], []
    for sample in range(repeats):
        g, d = gaussian['actions'][sample], diffusion['actions'][sample]
        for combo in COMBINATIONS:
            branches.append(copy.deepcopy(env))
            labels.append((sample, combo, mixed_action(g, d, combo)))
    initial = env.snapshot()
    results = step_batch(branches, [x[2] for x in labels])
    reference_users = branches[0].user_xy
    rows = []
    for branch, (sample, combo, _), result in zip(branches, labels, results):
        np.testing.assert_array_equal(branch.user_xy, reference_users)
        assert branch.t == snapshot['t']+1
        info = result[-1]
        rows.append(dict(sample=sample, combination=combo,
            **{k: float(info[k]) for k in ('coverage', 'tau', 'reward', 'proposal_count',
                'invalidated_proposals', 'proposal_acceptance', 'disconnection_ratio', 'actual_speed_mean')}))
    assert env.snapshot()['movement_rng'] == initial['movement_rng']
    return rows


def scene_bootstrap(differences, seed=950001):
    values = np.asarray(differences, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=None, median=None, low=None, high=None, scenes=0)
    indices = np.random.default_rng(seed).integers(len(values), size=(2000, len(values)))
    low, high = np.quantile(values[indices].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                low=float(low), high=float(high), scenes=len(values))


def sampled_backhaul_hops(positions, actual_displacements, cfg):
    """Exact geometry-only backhaul result; UE motion/radio cannot affect it."""
    from ..kernels import shortest_hops
    moved = np.asarray(positions)[None]+np.asarray(actual_displacements)
    distance2 = ((moved[:, :, None]-moved[:, None, :])**2).sum(-1)
    adjacency = distance2 <= cfg.uu_range**2
    adjacency[:, np.arange(cfg.num_uavs), np.arange(cfg.num_uavs)] = False
    bs = np.asarray(cfg.bs_positions)
    direct2 = ((moved[:, :, None]-bs[None, None, :, :2])**2).sum(-1)
    direct2 += (cfg.altitude-bs[:, 2])**2
    direct = direct2 <= cfg.ug_range**2
    return np.stack([shortest_hops(a, b, np.ones(cfg.num_uavs, bool)).min(1)
                     for a, b in zip(adjacency, direct)])
