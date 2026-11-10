"""Paired interventions from identical M1-C deployment states; no training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import FIELDS, actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from report_mappo_training import topology
from uav_research.config import config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import stack_observations
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


SOURCE = Path('analysis/mappo_central_best_generalization_100k_seed0')
OUTPUT = Path('analysis/mappo_central_best_deployment_branches_100k_seed0')
MODES = ['policy', 'hover_all', 'freeze_relays', 'freeze_random']
LABELS = dict(policy='Continue policy', hover_all='Hover all UAVs',
              freeze_relays='Freeze initial critical relays',
              freeze_random='Freeze matched random connected UAVs')
EXTRA_FIELDS = ['initial_link_retention', 'initial_connected_retention']
ALL_FIELDS = FIELDS + EXTRA_FIELDS


def critical_relays(env):
    """Remove each node on the CURRENT graph; exclude its own service loss.

    Protect every currently reachable node whose removal disconnects another
    currently reachable UAV. This is an oracle intervention, not a local actor.
    The mask is fixed for the rest of the branch, never chosen from outcomes.
    """
    u = env.cfg.num_uavs
    connected = (env.B <= u).any(-1)
    reachable_after = (env.post_hops <= u).any(-1)
    dependent = connected[None, :] & ~reachable_after
    np.fill_diagonal(dependent, False)
    dependent &= connected[:, None]
    return dependent.any(-1), dependent.sum(-1), dependent @ env.counts


def freeze_masks(mode, relay_masks, random_masks):
    if mode == 'policy':
        return np.zeros_like(relay_masks)
    if mode == 'hover_all':
        return np.ones_like(relay_masks)
    if mode == 'freeze_relays':
        return relay_masks.copy()
    if mode == 'freeze_random':
        return random_masks.copy()
    raise ValueError(mode)


def apply_freeze(actions, mask):
    action = np.array(actions, copy=True)
    if action.shape[:-1] != mask.shape or action.shape[-1] != 2:
        raise ValueError('Freeze mask and joint continuous actions disagree')
    action[mask] = 0
    return action


def capture_states(cfg, policy, scenes, times, action_seed, reference):
    """Reproduce complete original rollouts, retaining exact branch RNG states."""
    cases, baseline = [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(action_seed)
        for start in range(0, len(scenes), cfg.run.num_envs):
            batch = scenes[start:start + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(batch), 'batch') as vector:
                vector.restore([s['snapshot'] for s in batch])
                obs = stack_observations([e.obs for e in vector.envs])
                initial = np.array([e.info['coverage'] for e in vector.envs])
                trajectory = []
                for t in range(cfg.env.horizon):
                    if t in times:
                        records, relays, randoms = [], [], []
                        for i, env in enumerate(vector.envs):
                            mask, dependent_uavs, dependent_users = critical_relays(env)
                            candidates = np.flatnonzero((env.B <= cfg.env.num_uavs).any(-1))
                            rng = np.random.default_rng(np.random.SeedSequence([960001, batch[i]['seed'], t]))
                            random = np.zeros(cfg.env.num_uavs, bool)
                            random[rng.choice(candidates, int(mask.sum()), replace=False)] = True
                            assert random.sum() == mask.sum()
                            coverage = env.info['coverage']
                            records.append(dict(scene_id=batch[i]['scene_id'], seed=batch[i]['seed'],
                                                branch_time=t, initial_coverage=float(initial[i]),
                                                branch_coverage=coverage,
                                                branch_sensing=env.info['discovery_fraction'],
                                                branch_disconnection=env.info['disconnection_ratio'],
                                                good=bool(coverage >= .4 and coverage >= initial[i] + .05),
                                                relay_count=int(mask.sum()),
                                                relay_uav_ids=(np.flatnonzero(mask) + 1).tolist(),
                                                random_uav_ids=(np.flatnonzero(random) + 1).tolist(),
                                                dependent_uav_counts=dependent_uavs.tolist(),
                                                dependent_served_users=dependent_users.tolist()))
                            relays.append(mask)
                            randoms.append(random)
                        cases.append(dict(start=start, time=t, records=records,
                                          snapshots=vector.snapshot(), torch_rng=torch.get_rng_state().clone(),
                                          relay_masks=np.stack(relays), random_masks=np.stack(randoms)))
                    action = policy.act(obs, deterministic=False)
                    obs, _, _, done, truncated, infos = vector.step(action)
                    assert np.all(done == (t == cfg.env.horizon - 1)) and not truncated.any()
                    trajectory.append([[info[k] for k in FIELDS] for info in infos])
                baseline.append(np.swapaxes(np.array(trajectory), 0, 1))
    baseline = np.concatenate(baseline)
    np.testing.assert_array_equal(baseline, reference[:len(scenes)])
    return sorted(cases, key=lambda c: (c['time'], c['start'])), baseline


def run_branch(cfg, policy, case, mode, rng_state, output=None, trace_count=0):
    """Every mode draws all actor actions, then overrides the frozen nodes.

    Drawing for all nodes, including all-hover, keeps the Gaussian noise stream
    identical across branches. Each mode restores the complete environment.
    """
    frozen = freeze_masks(mode, case['relay_masks'], case['random_masks'])
    user_path = hashlib.sha256()
    with isolated_rng(), torch.inference_mode():
        torch.set_rng_state(rng_state.clone())
        with VectorEnvironment(cfg.env, len(case['snapshots']), 'batch') as vector:
            vector.restore(case['snapshots'])
            obs = stack_observations([e.obs for e in vector.envs])
            initial_xy = np.stack([e.uav_xy for e in vector.envs])
            initial_edges = np.stack([e.A.copy() for e in vector.envs])
            diagonal = np.arange(cfg.env.num_uavs)
            initial_edges[:, diagonal, diagonal] = False
            initial_connected = np.stack([(e.B <= cfg.env.num_uavs).any(-1) for e in vector.envs])
            frames = {i: [vector.envs[i].trace_frame()] for i in range(min(trace_count, len(vector.envs)))}
            data = []
            for t in range(case['time'], cfg.env.horizon):
                original = policy.act(obs, deterministic=False)
                actions = apply_freeze(original, frozen)
                obs, _, _, done, truncated, infos = vector.step(actions)
                assert np.all(done == (t == cfg.env.horizon - 1)) and not truncated.any()
                actual_xy = np.stack([e.uav_xy for e in vector.envs])
                if not np.array_equal(actual_xy[frozen], initial_xy[frozen]):
                    raise AssertionError('A frozen UAV moved')
                user_path.update(np.stack([e.user_xy for e in vector.envs]).tobytes())
                extra = []
                for i, env in enumerate(vector.envs):
                    edges = initial_edges[i]
                    retained_edges = (env.A & edges).sum() / edges.sum() if edges.any() else 1.
                    connected = initial_connected[i]
                    retained_nodes = (((env.B <= cfg.env.num_uavs).any(-1) & connected).sum()
                                      / connected.sum()) if connected.any() else 1.
                    extra.append([retained_edges, retained_nodes])
                values = np.array([[info[k] for k in FIELDS] for info in infos])
                data.append(np.column_stack([values, extra]))
                for i in frames:
                    frames[i].append(vector.envs[i].trace_frame())
            final_rng = torch.get_rng_state().clone()
            paths = []
            if output:
                output = Path(output)
                output.mkdir(parents=True, exist_ok=True)
                for i, items in frames.items():
                    record = case['records'][i]
                    path = output / f"trace_{record['seed']}.npz"
                    np.savez_compressed(path, **{k: np.stack([f[k] for f in items]) for k in items[0]},
                                        bs_positions=np.array(cfg.env.bs_positions), width=cfg.env.width,
                                        height=cfg.env.height, scene_seed=record['seed'],
                                        scene_id=record['scene_id'], episode_start_step=case['time'],
                                        action_description=LABELS[mode], frozen_mask=frozen[i])
                    paths.append(str(path))
    result = np.swapaxes(np.asarray(data), 0, 1)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite branch diagnostic')
    return result, dict(ue_path_sha256=user_path.hexdigest(), torch_rng=final_rng, traces=paths)


def stage_metrics(data, records, discount):
    """data=[repeat,scene,post-branch time,field]; return one mean per scene."""
    cov = data[..., ALL_FIELDS.index('coverage')].mean(0)
    branch = np.array([r['branch_coverage'] for r in records])
    rewards = data[..., ALL_FIELDS.index('task_reward')].mean(0)
    result = dict(post_coverage=cov.mean(-1), last100_coverage=cov[:, -100:].mean(-1),
                  final_coverage=cov[:, -1], post_minus_branch=cov.mean(-1) - branch,
                  final_minus_branch=cov[:, -1] - branch,
                  post_task_return=rewards.sum(-1),
                  discounted_task_return=(rewards * discount ** np.arange(rewards.shape[-1])).sum(-1))
    for key in ALL_FIELDS:
        if key != 'coverage':
            result[key] = data[..., ALL_FIELDS.index(key)].mean((0, 2))
    return result


def paired_difference(left, right):
    result = interval(np.asarray(right) - np.asarray(left))
    return dict(delta=result['mean'], low=result['low'], high=result['high'])


def assert_no_intervention_equivalent(left, right):
    # Other scenes in the same inference batch can have different set lengths.
    # CPU GEMM may then round continuous outputs slightly differently, even for
    # a scene with no intervened nodes. Do not relax integer/combinatorial data.
    np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-6)
    exact = ['coverage', 'discovery_fraction', 'disconnection_ratio', 'robustness',
             'uav_close_neighbor_fraction', 'boundary_within_150_fraction',
             'boundary_reflection_fraction', 'sensing_rejected_users',
             'downlink_rejected_users', 'backhaul_rejected_users',
             'capacity_rejected_users'] + EXTRA_FIELDS
    indices = [ALL_FIELDS.index(k) for k in exact]
    np.testing.assert_array_equal(left[..., indices], right[..., indices])


def publish(out):
    out = Path(out)
    experiment = json.loads((out / 'experiment.json').read_text())
    cfg = config_from_dict(experiment['config'])
    summaries, differences, scene_rows = [], [], []
    all_data = {}
    for branch_time in experiment['branch_times']:
        records = experiment['records'][str(branch_time)]
        with np.load(out / f't{branch_time}' / 'curves.npz') as saved:
            assert list(saved['fields']) == ALL_FIELDS
            arrays = {mode: saved[mode].copy() for mode in MODES}
        all_data[branch_time] = arrays
        masks = dict(all=np.ones(len(records), bool), good=np.array([r['good'] for r in records]),
                     has_relays=np.array([r['relay_count'] > 0 for r in records]))
        metrics = {mode: stage_metrics(a, records, cfg.algo.discount) for mode, a in arrays.items()}
        for subset, selected in masks.items():
            if not selected.any():
                continue
            for mode in MODES:
                for key, value in metrics[mode].items():
                    summaries.append(dict(time=branch_time, subset=subset, mode=mode, metric=key,
                                          scenes=int(selected.sum()), **interval(value[selected])))
            for left, right in [('policy', 'hover_all'), ('policy', 'freeze_relays'),
                                ('policy', 'freeze_random'), ('freeze_random', 'freeze_relays')]:
                for key in metrics[left]:
                    differences.append(dict(time=branch_time, subset=subset, comparison=f'{right}-{left}',
                                            metric=key, scenes=int(selected.sum()),
                                            **paired_difference(metrics[left][key][selected], metrics[right][key][selected])))
        for mode in MODES:
            for i, record in enumerate(records):
                scene_rows.append(dict(time=branch_time, mode=mode, **record,
                                       **{key: float(v[i]) for key, v in metrics[mode].items()}))
    atomic_json(out / 'summary.json', dict(summaries=summaries, differences=differences))
    write_csv(out / 'summary.csv', summaries)
    write_csv(out / 'paired_differences.csv', differences)
    write_csv(out / 'scene_metrics.csv', scene_rows)
    style()
    for t, arrays in all_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        keys = ['coverage', 'discovery_fraction', 'disconnection_ratio',
                'uav_nearest_neighbor_mean', 'initial_link_retention', 'actual_speed_mean']
        for color, mode in zip(COLORS, MODES):
            for ax, key in zip(axes.flat, keys):
                y = arrays[mode][..., ALL_FIELDS.index(key)].mean((0, 1))
                ax.plot(np.arange(t + 1, cfg.env.horizon + 1), y, label=LABELS[mode], color=color)
        for ax, key in zip(axes.flat, keys):
            ax.set(xlabel='Episode step', title=key)
            ax.legend(fontsize=6)
        fig.suptitle(f'Identical states at t={t}; 32 scenes; 3 future action streams')
        save_figure(fig, out / 'figures', f'branches_t{t}')
    fig, axes = plt.subplots(1, len(all_data), figsize=(12, 4), squeeze=False)
    for ax, (t, arrays) in zip(axes.flat, all_data.items()):
        records = experiment['records'][str(t)]
        ax.scatter(t, np.mean([r['branch_coverage'] for r in records]) * 100, color='black', s=30, zorder=5)
        for color, mode in zip(COLORS, MODES):
            y = arrays[mode][..., ALL_FIELDS.index('coverage')].mean((0, 1)) * 100
            ax.plot(np.r_[t, np.arange(t + 1, cfg.env.horizon + 1)],
                    np.r_[np.mean([r['branch_coverage'] for r in records]) * 100, y],
                    label=LABELS[mode], color=color)
        ax.set(xlabel='Episode step', ylabel='Coverage (%)', title=f'Branch at t={t}')
        ax.legend(fontsize=6)
    save_figure(fig, out / 'figures', 'coverage_branches')
    for t in all_data:
        for mode in MODES:
            folder = out / f't{t}' / mode
            for trace in sorted(folder.glob('trace_*.npz')):
                topology(trace, folder / 'figures', 'best', experiment['checkpoint_step'], cfg,
                         boundary_overlay=True)
    lines = ['# M1-C best：从同一部署状态分支执行', '',
             '冻结同一个100k best，不训练、不更新网络。使用上一轮独立新场景的前32个（610000–610031），'
             '原策略种子880001产生前缀，固定在t=150和t=200保存完整环境与动作随机状态。',
             '每个状态使用3套未来动作噪声，四分支共享同一噪声流和完全相同的UE移动序列。'
             '第1套延续原随机流，并逐时刻精确复现上一轮基线；其余两套使用独立固定种子。', '',
             '分支：原策略继续；全部悬停；固定分支时关键中继；随机固定同数量联网UAV。'
             '后两组的节点集合在分支时固定，之后不重新挑选。悬停组仍正常推进UE与回合时间。', '',
             '关键中继指：在分支时拓扑移除该UAV，会让至少另一架当前联网UAV无法到达任一BS。'
             '排除它自身服务损失，记录下游受影响UAV数和服务UE数。使用所有此类节点，不按未来效果挑选。'
             '这是使用全局拓扑的诊断干预，不是可部署的局部策略。', '',
             '较好状态预先定义为：分支覆盖≥40%，且比该布局t=0覆盖高至少5个百分点。'
             '主表保留全部32场景；good子集与有关键中继子集另报，不按未来表现筛选。', '',
             '| 分支时刻 | 较好状态数 | 含关键中继状态数 | 平均固定中继数 | 分支覆盖 |',
             '|---|---:|---:|---:|---:|']
    for t in all_data:
        records = experiment['records'][str(t)]
        lines.append(f"|{t}|{sum(r['good'] for r in records)}/32|{sum(r['relay_count']>0 for r in records)}/32|"
                     f"{np.mean([r['relay_count'] for r in records]):.2f}|{np.mean([r['branch_coverage'] for r in records]):.2%}|")
    lines += ['', '| 时刻 | 分支 | 后续平均覆盖 | 最后100步覆盖 | 终态覆盖 | 后续原任务回报 |',
              '|---|---|---:|---:|---:|---:|']
    for t in all_data:
        for mode in MODES:
            values = {r['metric']: r['mean'] for r in summaries
                      if r['time'] == t and r['subset'] == 'all' and r['mode'] == mode}
            lines.append(f"|{t}|{LABELS[mode]}|{values['post_coverage']:.2%}|{values['last100_coverage']:.2%}|"
                         f"{values['final_coverage']:.2%}|{values['post_task_return']:.3f}|")
    lines += ['', '| 时刻 | 对比 | 后续平均覆盖差 | 95%场景区间 |', '|---|---|---:|---|']
    for row in differences:
        if row['subset'] == 'all' and row['metric'] == 'post_coverage':
            lines.append(f"|{row['time']}|{row['comparison']}|{100*row['delta']:+.2f} pp|"
                         f"[{100*row['low']:+.2f}, {100*row['high']:+.2f}] pp|")
    for t in all_data:
        lines += ['', f'![t={t}分支](figures/branches_t{t}.png)']
    lines += ['', '![覆盖](figures/coverage_branches.png)', '',
              '配对bootstrap 2000次、种子950001；先平均每场景的3套未来动作噪声，再按32个场景重采样。'
              't150/t200同属一批场景，分别报告，不当作64个独立场景；区间不是训练种子置信区间。', '',
              't150和t200之后的长度分别为250和200步，不能直接比较两表回报总量；使用各自对照差值。'
              '同时保存折扣任务回报，折扣0.995。所有分支都在原t400真正终止。', '',
              '全部悬停优于继续策略，支持后续移动在破坏当前布局，但不能单独定位奖励、探索或优势估计。'
              '固定中继优于随机固定同数量节点，才更支持初始中继位置具有额外作用。'
              '固定初始中继也不能保证移动的另一端不脱链，失败不能排除动态中继协作问题。'
              '随机控制只使用每个状态一套固定随机节点集合，相关结论条件于这次节点抽样。', '',
              '全部逐时刻数据见t150/t200的curves.npz；逐场景与good子集结果见scene_metrics.csv、summary.csv；'
              '完整初始分支快照见branch_states.pt；UE路径与随机流校验、源哈希见experiment.json。', '',
              '重建图表：`.venv/bin/python scripts/check_mappo_deployment_branches.py --rebuild`。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return summaries, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.rebuild:
        publish(args.output)
        return
    if args.output.exists():
        raise FileExistsError('Use a new output directory, or --rebuild saved results')
    start = time.perf_counter()
    parent = json.loads((args.source / 'experiment.json').read_text())
    run = Path(parent['run'])
    cfg, policy = policy_from_checkpoint(run / 'best.pt', 'cpu')
    configure_runtime(cfg)
    assert cfg.as_dict() == parent['config']
    assert policy.provenance['checkpoint_env_steps'] == parent['checkpoint_step'] == 100000
    for source, expected in parent['source_hashes'].items():
        if digest(Path(source)) != expected:
            raise ValueError(f'Original source changed: {source}')
    sources = [run / 'best.pt', args.source / 'initial_scenes.pt', args.source / 'experiment.json',
               args.source / 'unseen/curves.npz', Path(__file__), Path('src/uav_research/environment.py'),
               Path('src/uav_research/vector.py'), Path('src/uav_research/models.py')]
    hashes = {str(p): digest(p) for p in sources}
    initial = torch.load(args.source / 'initial_scenes.pt', weights_only=False)
    scenes = initial['unseen'][:32]
    if len(scenes) != 32 or [s['seed'] for s in scenes] != list(range(610000, 610032)):
        raise ValueError('Expected first 32 fresh scenes from the generalization audit')
    policy.actor.eval()
    policy.actor.requires_grad_(False)
    before = actor_digest(policy.actor)
    assert before == parent['actor_hash_after']
    times = [150, 200]
    with np.load(args.source / 'unseen/curves.npz') as saved:
        assert list(saved['fields']) == FIELDS and saved['action_seeds'][0] == 880001
        reference = saved['diagnostics'][0, :32].copy()
    cases, original = capture_states(cfg, policy, scenes, times, 880001, reference)
    args.output.mkdir(parents=True)
    torch.save(cases, args.output / 'branch_states.pt')
    print('Original prefixes/full rollouts exactly reproduced; all branch states captured.', flush=True)
    all_results = {t: {mode: np.empty((3, 32, cfg.env.horizon - t, len(ALL_FIELDS)))
                       for mode in MODES} for t in times}
    records, checks = {str(t): [] for t in times}, []
    for case in cases:
        t, begin = case['time'], case['start']
        end = begin + len(case['records'])
        records[str(t)].extend(case['records'])
        rng_description = []
        for repeat in range(3):
            if repeat == 0:
                rng_state = case['torch_rng']
                rng_description.append('original continuation')
            else:
                seed = int(np.random.SeedSequence([971001, t, begin, repeat]).generate_state(1)[0])
                with isolated_rng():
                    torch.manual_seed(seed)
                    rng_state = torch.get_rng_state().clone()
                rng_description.append(seed)
            diagnostics = []
            for mode in MODES:
                path = args.output / f't{t}' / mode
                data, audit = run_branch(cfg, policy, case, mode, rng_state, path,
                                         trace_count=2 if repeat == 0 and begin == 0 else 0)
                all_results[t][mode][repeat, begin:end] = data
                diagnostics.append(audit)
                if repeat == 0 and mode == 'policy':
                    np.testing.assert_array_equal(data[..., :len(FIELDS)], original[begin:end, t:])
            assert len({d['ue_path_sha256'] for d in diagnostics}) == 1
            assert all(torch.equal(d['torch_rng'], diagnostics[0]['torch_rng']) for d in diagnostics)
            # When no critical relay exists, both partial-freeze branches must
            # be exactly identical to the unchanged policy under paired noise.
            empty = ~case['relay_masks'].any(-1)
            for mode in ['freeze_relays', 'freeze_random']:
                assert_no_intervention_equivalent(all_results[t][mode][repeat, begin:end][empty],
                                                  all_results[t]['policy'][repeat, begin:end][empty])
            checks.append(dict(time=t, start=begin, repeat=repeat,
                               future_action_rng=rng_description[-1],
                               identical_ue_path_sha256=diagnostics[0]['ue_path_sha256'],
                               identical_final_torch_rng_sha256=hashlib.sha256(diagnostics[0]['torch_rng'].numpy().tobytes()).hexdigest(),
                               no_intervention_continuous_tolerance=dict(rtol=2e-6, atol=2e-6),
                               no_intervention_discrete_metrics_exact=True,
                               frozen_positions_unchanged=True))
            print(f't={t}, scenes {begin}-{end-1}, future stream {repeat+1}/3: four branches complete.', flush=True)
        folder = args.output / f't{t}'
        np.savez_compressed(folder / f'chunk_{begin}.npz', **{k: v[:, begin:end] for k, v in all_results[t].items()})
    for t in times:
        np.savez_compressed(args.output / f't{t}' / 'curves.npz', **all_results[t],
                            fields=np.array(ALL_FIELDS), episode_steps=np.arange(t + 1, cfg.env.horizon + 1))
    assert actor_digest(policy.actor) == before
    assert all(p.grad is None for p in policy.actor.parameters())
    assert hashes == {str(p): digest(p) for p in sources}
    record = dict(config=cfg.as_dict(), run=str(run), checkpoint_step=parent['checkpoint_step'],
                  parent=str(args.source), branch_times=times, prefix_action_seed=880001,
                  future_repeats=3, records=records, checks=checks, source_hashes=hashes,
                  actor_hash_before=before, actor_hash_after=actor_digest(policy.actor),
                  elapsed_seconds=time.perf_counter() - start)
    atomic_json(args.output / 'experiment.json', record)
    _, differences = publish(args.output)
    print(json.dumps([r for r in differences if r['subset'] == 'all' and r['metric'] == 'post_coverage'], indent=2))


if __name__ == '__main__':
    main()
