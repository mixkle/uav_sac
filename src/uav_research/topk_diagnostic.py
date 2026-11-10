"""Read-only, fixed-transition association experiments; no environment hooks."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
from numba import njit

SCORE_NAMES = ('teacher', 'constant', 'random')
SCHEMES = ('P0', 'P1')
DEFAULT_REWARD_WEIGHTS = (1., .1, .5, -.1)
METRICS = (
    'served', 'coverage', 'throughput_bps', 'tau', 'robustness', 'disconnection',
    'reward_coverage', 'reward_throughput', 'reward_robustness',
    'reward_disconnection', 'task_reward', 'protocol_loss', 'fill_gain',
    'reassignment_loss', 'timing_loss', 'feasibility_loss', 'duplicate_proposals',
    'invalidated_proposals', 'rejected_proposals', 'fill_rounds', 'fill_messages',
    'fill_rejections', 'idle_with_unserved_fraction',
)
AUX_NAMES = ('pre_edges', 'known_edges', 'full_edges', 'over_capacity_fraction',
             'multi_candidate_fraction', 'max_candidate_uavs', 'source_served',
             'one_candidate_fraction', 'no_candidate_fraction')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@njit(cache=True)
def maximum_assignment(edges, capacity):
    """Exact capacitated bipartite matching by deterministic augmenting paths."""
    u, m = edges.shape
    owner = np.full(m, -1, np.int16)
    counts = np.zeros(u, np.int64)
    for root in range(m):
        if owner[root] >= 0:
            continue
        seen_u = np.zeros(u, np.bool_)
        seen_m = np.zeros(m, np.bool_)
        parent_u = np.full(u, -1, np.int64)
        queue = np.empty(m, np.int64)
        queue[0] = root
        seen_m[root] = True
        head, tail, end = 0, 1, -1
        while head < tail and end < 0:
            current = queue[head]
            head += 1
            for i in range(u):
                if not edges[i, current] or seen_u[i]:
                    continue
                seen_u[i] = True
                parent_u[i] = current
                if counts[i] < capacity:
                    end = i
                    break
                for child in range(m):
                    if owner[child] == i and not seen_m[child]:
                        seen_m[child] = True
                        queue[tail] = child
                        tail += 1
        while end >= 0:
            child = parent_u[end]
            previous = owner[child]
            owner[child] = end
            counts[end] += 1
            if previous < 0:
                break
            counts[previous] -= 1
            end = previous
    return owner


@njit(cache=True)
def _ranked_candidates(before, scores):
    u, m = before.shape
    rank = np.full((u, m), -1, np.int64)
    lengths = np.zeros(u, np.int64)
    for i in range(u):
        # Stable insertion sort: equal scores keep ascending UE IDs.
        for j in range(m):
            if before[i, j]:
                n = lengths[i]
                at = n
                while at > 0 and scores[i, rank[i, at - 1]] < scores[i, j]:
                    rank[i, at] = rank[i, at - 1]
                    at -= 1
                rank[i, at] = j
                lengths[i] += 1
    return rank, lengths


@njit(cache=True)
def _topk_and_fill(before, after, scores, capacity):
    u, m = before.shape
    ranks, lengths = _ranked_candidates(before, scores)
    proposal = np.zeros((u, m), np.bool_)
    valid = np.zeros((u, m), np.bool_)
    attempted = np.zeros((u, m), np.bool_)
    p0 = np.full(m, -1, np.int16)
    invalid, rejected, duplicates = 0, 0, 0
    for i in range(u):
        for k in range(min(capacity, lengths[i])):
            j = ranks[i, k]
            proposal[i, j] = True
            attempted[i, j] = True
            if after[i, j]:
                valid[i, j] = True
            else:
                invalid += 1
    for j in range(m):
        offers = 0
        for i in range(u):
            if valid[i, j]:
                offers += 1
                if p0[j] < 0 or scores[i, j] > scores[p0[j], j]:
                    p0[j] = i
        if offers > 0:
            rejected += offers - 1
            duplicates += offers - 1
    p1 = p0.copy()
    counts = np.zeros(u, np.int64)
    for j in range(m):
        if p1[j] >= 0:
            counts[p1[j]] += 1
    rounds, messages, fill_reject = 0, 0, 0
    while True:
        requests = np.zeros((u, m), np.bool_)
        sent = 0
        for i in range(u):
            remaining = capacity - counts[i]
            for k in range(lengths[i]):
                if remaining == 0:
                    break
                j = ranks[i, k]
                if after[i, j] and not attempted[i, j]:
                    requests[i, j] = True
                    attempted[i, j] = True
                    remaining -= 1
                    sent += 1
        if sent == 0:
            break
        rounds += 1
        messages += sent
        for j in range(m):
            chosen, offers = -1, 0
            for i in range(u):
                if requests[i, j]:
                    offers += 1
                    if chosen < 0 or scores[i, j] > scores[chosen, j]:
                        chosen = i
            if offers and p1[j] < 0:
                p1[j] = chosen
                counts[chosen] += 1
                fill_reject += offers - 1
            else:
                fill_reject += offers
    return p0, p1, proposal, valid, np.array([
        duplicates, invalid, rejected, rounds, messages, fill_reject], np.int64)


def topk_and_fill(before, after, scores, capacity):
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    scores = np.asarray(scores, np.float64)
    if before.ndim != 2 or before.shape != after.shape or before.shape != scores.shape:
        raise ValueError('Candidate masks and scores must have identical [U,M] shape')
    if capacity < 1 or not np.isfinite(scores[before]).all():
        raise ValueError('Positive capacity and finite candidate scores required')
    return _topk_and_fill(before, after, scores, capacity)


def feasible(env):
    return (env.sensed & (env.Y >= env.cfg.sinr_threshold_db)
            & (env.B.min(-1)[:, None] <= env.cfg.num_uavs)).copy()


def score_array(kind, before, sinr, threshold, scene, step, repeat=0, seed=970001):
    result = np.zeros(before.shape, np.float64)
    if kind == 'teacher':
        x = np.clip((sinr[before] - threshold) / 10., -700., 700.)
        result[before] = 1. / (1. + np.exp(-x))
    elif kind == 'constant':
        result[before] = .5
    elif kind == 'random':
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [seed, int(scene), int(step), int(repeat)])))
        result[before] = rng.uniform(np.nextafter(0., 1.), 1., before.sum())
    else:
        raise ValueError(kind)
    return result


@njit(cache=True)
def _assignment_metrics(owner, rates, retained_mask, reachable, capacity, edges, vref, weights):
    u, m = rates.shape
    counts = np.zeros(u, np.int64)
    throughput, served = 0., 0
    for j in range(m):
        if owner[j] >= 0:
            i = owner[j]
            counts[i] += 1
            served += 1
            throughput += rates[i, j]
    robust = 0.
    for failed in range(u):
        for i in range(u):
            if retained_mask[failed, i]:
                robust += counts[i] / m / u
    disconnected = 1. - reachable.sum() / u
    coverage = served / m
    tau = throughput / (m * vref)
    idle = 0
    for i in range(u):
        if counts[i] < capacity:
            for j in range(m):
                if owner[j] < 0 and edges[i, j]:
                    idle += 1
                    break
    reward_coverage = weights[0] * coverage
    reward_throughput = weights[1] * tau
    reward_robustness = weights[2] * robust
    reward_disconnection = weights[3] * disconnected
    return np.array([served, coverage, throughput, tau, robust, disconnected,
                     reward_coverage, reward_throughput, reward_robustness, reward_disconnection,
                     reward_coverage + reward_throughput + reward_robustness + reward_disconnection,
                     idle / u], np.float64)


def assignment_metrics(owner, rates, retained_mask, reachable, capacity, edges, vref,
                       reward_weights=DEFAULT_REWARD_WEIGHTS):
    weights = np.asarray(reward_weights, dtype=np.float64)
    if weights.shape != (4,) or not np.isfinite(weights).all():
        raise ValueError('Expected four finite signed reward weights')
    return _assignment_metrics(owner, rates, retained_mask, reachable,
                               capacity, edges, vref, weights)


def validate_assignment(owner, edges, capacity):
    assigned = np.flatnonzero(owner >= 0)
    if np.any(owner < -1) or np.any(owner >= edges.shape[0]):
        raise AssertionError('Invalid owner ID')
    assert edges[owner[assigned], assigned].all()
    assert np.bincount(owner[assigned], minlength=edges.shape[0]).max(initial=0) <= capacity


def audit_transition(before, after, scores, capacity, rates, retained_mask,
                     reachable, vref, known_owner=None, full_owner=None,
                     *, reward_weights=DEFAULT_REWARD_WEIGHTS):
    known = before & after
    if known_owner is None:
        known_owner = maximum_assignment(known, capacity)
    if full_owner is None:
        full_owner = maximum_assignment(after, capacity)
    p0, p1, proposal, valid, extra = topk_and_fill(before, after, scores, capacity)
    n0, n1, nk, nf = [(x >= 0).sum() for x in (p0, p1, known_owner, full_owner)]
    assert n0 == valid.any(0).sum()
    assert 0 <= n0 <= n1 <= nk <= nf <= before.shape[1]
    assert np.array_equal(p0[p0 >= 0], p1[p0 >= 0])
    rows = []
    for owner in (p0, p1):
        validate_assignment(owner, known, capacity)
        base = assignment_metrics(owner, rates, retained_mask, reachable, capacity, known, vref,
                                  reward_weights)
        n = int(base[0])
        gap = [nk - n, n1 - n0, nk - n1, nf - nk, before.shape[1] - nf]
        if owner is p1:
            # protocol_loss for P1 refers to remaining avoidable deficit.
            gap[0] = nk - n1
        assert before.shape[1] - n == gap[0] + gap[3] + gap[4]
        assert abs(base[6:10].sum() - base[10]) < 1e-12
        assert 0 <= base[4] <= (1 - 1 / before.shape[0]) * base[1] + 1e-12
        rows.append(np.r_[base[:11], gap, extra, base[-1]])
    return np.stack([p0, p1]), proposal, valid, np.stack(rows)


def collect(cfg, policy, output, seeds, steps, *, progress=print, observer=None):
    """Original greedy environment evolves once; optional observers see copies only."""
    import torch
    from .observations import as_tensor_observation
    from .runtime import isolated_rng
    from .vector import VectorEnvironment
    seeds = list(map(int, seeds))
    if not 1 <= steps <= cfg.env.horizon or set(seeds) & set(cfg.run.test_seeds):
        raise ValueError('Invalid diagnostic horizon or reserved test scenes')
    out = Path(output) / 'trajectories'
    out.mkdir(parents=True, exist_ok=True)
    fields = ('uav_xy', 'user_xy', 'A', 'B', 'direct', 'C', 'Y', 'rates',
              'uplink', 'sensed', 'history', 'post_hops')
    start = time.perf_counter()
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for at in range(0, len(seeds), cfg.run.num_envs):
            chosen = seeds[at:at + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(chosen), 'batch') as vec:
                obs, _, _ = vec.reset(chosen)
                frames = [{key: [getattr(e, key).copy()] for key in fields}
                          for e in vec.envs]
                actions, infos = [[] for _ in chosen], [[] for _ in chosen]
                movement = [[json.dumps(e.movement_rng.bit_generator.state, sort_keys=True)]
                            for e in vec.envs]
                for t in range(steps):
                    action = policy.act(obs, False)
                    before = [feasible(e) for e in vec.envs]
                    before_sinr = [e.Y.copy() for e in vec.envs]
                    obs, _, _, done, truncated, data = vec.step(action)
                    if truncated.any() or np.any(done != (t + 1 == cfg.env.horizon)):
                        raise AssertionError('Unexpected termination')
                    for i, e in enumerate(vec.envs):
                        for key in fields:
                            frames[i][key].append(getattr(e, key).copy())
                        actions[i].append(action[i].copy())
                        infos[i].append([data[i]['coverage'], data[i]['reward'],
                                         data[i]['discovery_fraction'], data[i]['disconnection_ratio']])
                        movement[i].append(json.dumps(e.movement_rng.bit_generator.state, sort_keys=True))
                        if observer is not None:
                            observer(chosen[i], t, before[i], before_sinr[i],
                                     feasible(e), e)
                    if (t + 1) % 100 == 0:
                        progress(f'collect scenes {chosen[0]}–{chosen[-1]}: {t+1}/{steps}, '
                                 f'{time.perf_counter()-start:.1f}s', flush=True)
                for i, scene in enumerate(chosen):
                    arrays = {k: np.asarray(v) for k, v in frames[i].items()}
                    # Compact exact topology / ownership storage, no float cast of radio quantities.
                    for key in ('A', 'direct', 'C', 'sensed'):
                        arrays[key] = arrays[key].astype(bool)
                    for key in ('B', 'post_hops'):
                        arrays[key] = arrays[key].astype(np.int16)
                    arrays.update(actions=np.asarray(actions[i]), source_metrics=np.asarray(infos[i]),
                                  movement_rng=np.array(movement[i]), scene_seed=scene,
                                  pre_action_times=np.arange(steps), schema_version=1)
                    atomic_npz(out / f'scene_{scene}.npz', **arrays)
                progress(f'saved {len(chosen)} scenes; collection {time.perf_counter()-start:.1f}s', flush=True)


def audit_scene(trajectory, output, cfg, repeats=20, random_seed=970001,
                *, trajectory_capacity=None, reward_weights=DEFAULT_REWARD_WEIGHTS):
    """Audit a saved scene, without loading a network or generating a trajectory."""
    if repeats < 1:
        raise ValueError('At least one random repeat required')
    path = Path(trajectory)
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    # With fixed total power/bandwidth and full-load interference, changing N
    # scales signal, interference and noise equally: SINR stays identical.
    # Link bandwidth/rate scales as N_source/N_audit. Original arrays remain
    # read-only; this is a counterfactual on recorded positions, not a rollout.
    trajectory_capacity = cfg.env.capacity if trajectory_capacity is None else trajectory_capacity
    if trajectory_capacity < 1:
        raise ValueError('Positive trajectory capacity required')
    rate_scale = trajectory_capacity / cfg.env.capacity
    scene = int(d['scene_seed'])
    steps, u, m = len(d['actions']), cfg.env.num_uavs, cfg.env.num_users
    kinds = ['teacher', 'constant'] + ['random'] * repeats
    nscore = len(kinds)
    owners = np.full((steps, nscore, 2, m), -1, np.int16)
    proposals = np.zeros((steps, nscore, u, m), bool)
    values = np.empty((steps, nscore, 2, len(METRICS)), np.float64)
    optima = np.full((steps, 2, m), -1, np.int16)
    auxiliaries = np.empty((steps, len(AUX_NAMES)))
    counts = np.empty((steps, nscore, 2, u), np.int16)
    pre_counts = np.empty((steps, u), np.int16)
    known_counts = np.empty((steps, u), np.int16)
    ue_counts = np.empty((steps, m), np.int16)
    teacher_scores = np.empty((steps, u, m), np.float64)
    solver_seconds = np.zeros((steps, 3))
    for t in range(steps):
        before = (d['sensed'][t] & (d['Y'][t] >= cfg.env.sinr_threshold_db)
                  & (d['B'][t].min(-1)[:, None] <= u))
        after = (d['sensed'][t+1] & (d['Y'][t+1] >= cfg.env.sinr_threshold_db)
                 & (d['B'][t+1].min(-1)[:, None] <= u))
        known = before & after
        stamp = time.perf_counter()
        optima[t, 0] = maximum_assignment(known, cfg.env.capacity)
        solver_seconds[t, 0] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        optima[t, 1] = maximum_assignment(after, cfg.env.capacity)
        solver_seconds[t, 1] = time.perf_counter() - stamp
        for own, edges in zip(optima[t], (known, after)):
            validate_assignment(own, edges, cfg.env.capacity)
        retained_mask = (d['post_hops'][t+1].min(-1) <= u)
        reachable = d['B'][t+1].min(-1) <= u
        pre_counts[t], known_counts[t] = before.sum(-1), known.sum(-1)
        ue_counts[t] = known.sum(0)
        auxiliaries[t] = [before.sum(), known.sum(), after.sum(),
                           (before.sum(-1) > cfg.env.capacity).mean(),
                           (known.sum(0) >= 2).mean(), known.sum(0).max(initial=0),
                           d['C'][t+1].sum(), (known.sum(0) == 1).mean(),
                           (known.sum(0) == 0).mean()]
        stamp = time.perf_counter()
        for k, kind in enumerate(kinds):
            scores = score_array(kind, before, d['Y'][t], cfg.env.sinr_threshold_db,
                                 scene, t, max(0, k-2), random_seed)
            if k == 0:
                teacher_scores[t] = scores
            own, proposal, _, row = audit_transition(
                before, after, scores, cfg.env.capacity, d['rates'][t+1] * rate_scale,
                retained_mask, reachable, cfg.env.rate_reference_bps, *optima[t],
                reward_weights=reward_weights)
            owners[t, k], proposals[t, k], values[t, k] = own, proposal, row
            for scheme in range(2):
                counts[t, k, scheme] = np.bincount(own[scheme][own[scheme] >= 0], minlength=u)
        solver_seconds[t, 2] = time.perf_counter() - stamp
    atomic_npz(output, scene_seed=scene, metric_names=np.array(METRICS),
               score_names=np.array(kinds), schemes=np.array(SCHEMES), owners=owners,
               proposals=proposals, metrics=values, optima=optima,
               aux=auxiliaries, aux_names=np.array(AUX_NAMES),
               pre_counts=pre_counts, known_counts=known_counts, ue_counts=ue_counts,
               counts=counts, teacher_scores=teacher_scores, solver_seconds=solver_seconds,
               random_seed=random_seed, random_repeats=repeats,
               audit_capacity=cfg.env.capacity, trajectory_capacity=trajectory_capacity,
               link_rate_scale=rate_scale,
               reward_weights=np.asarray(reward_weights, dtype=np.float64),
               trajectory_sha256=file_hash(path), schema_version=1)
