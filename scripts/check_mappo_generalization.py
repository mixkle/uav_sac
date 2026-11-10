"""Frozen-policy, full-episode comparison of reconstructed seen and unseen scenes.

No learner update is performed. Training scenes are reconstructed from the
initialization/motion RNG streams, then checked against the full checkpoint.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from uav_research.boundary import reward_contributions
from uav_research.config import config_from_dict
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import stack_observations
from uav_research.plotting import COLORS, plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng, load_checkpoint
from uav_research.vector import VectorEnvironment
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from report_mappo_training import topology


DEFAULT_RUN = 'runs/continuous_mappo_full_reward_time_factor_relative_central_100k_seed0'
DEFAULT_OUT = 'analysis/mappo_central_best_generalization_100k_seed0'
FIELDS = ['coverage', 'discovery_fraction', 'disconnection_ratio', 'task_reward',
          'reward', 'robustness', 'tau', 'actual_speed_mean',
          'uav_nearest_neighbor_mean', 'uav_close_neighbor_fraction',
          'boundary_within_150_fraction', 'boundary_reflection_fraction',
          'sensing_rejected_users', 'downlink_rejected_users',
          'backhaul_rejected_users', 'capacity_rejected_users']
STAGES = {'all': slice(None), 'first100': slice(0, 100),
          'middle100': slice(150, 250), 'steps51_150': slice(50, 150),
          'last100': slice(-100, None), 'last20': slice(-20, None),
          'last1': slice(-1, None)}
LABELS = {'seen': 'Seen training scenes', 'unseen': 'Fresh unseen scenes',
          'validation': 'Model-selection validation scenes'}


def actor_digest(actor):
    h = hashlib.sha256()
    for name, value in actor.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def reconstruct_seen(cfg, full, worker_seeds, selected_episodes):
    """Rebuild only exogenous state; UAV actions do not affect either RNG.

    _move advances precisely the same UE RNG calls as training; zero commands
    avoid an expensive radio/graph evaluation at every reconstruction step.
    Checks include the final UE positions, not merely RNG states.
    """
    if full['env_steps'] % cfg.run.num_envs:
        raise ValueError('Checkpoint is not at a complete vector step')
    ticks = full['env_steps'] // cfg.run.num_envs
    complete, remainder = divmod(ticks, cfg.env.horizon)
    if len(worker_seeds) != cfg.run.num_envs:
        raise ValueError('Training worker count differs from saved seed list')
    if not selected_episodes or min(selected_episodes) < 0 or max(selected_episodes) >= complete:
        raise ValueError('Choose completed training episodes only')
    scenes, checks = [], []
    zero = np.zeros((cfg.env.num_uavs, 2), np.float32)
    for worker, seed in enumerate(worker_seeds):
        env = UAVEnvironment(cfg.env)
        env.reset(int(seed))
        for episode in range(complete + 1):
            if episode in selected_episodes:
                scenes.append(dict(scene_id=f'seen_w{worker:02d}_e{episode:02d}',
                                   worker=worker, episode=episode, seed=int(seed),
                                   snapshot=env.snapshot()))
            count = cfg.env.horizon if episode < complete else remainder
            for _ in range(count):
                env._move(zero)
            if episode < complete:
                env.reset()
        saved = full['environments'][worker]
        for key in ['initial_user_xy', 'initial_uav_xy', 'user_xy']:
            np.testing.assert_array_equal(getattr(env, key), saved[key])
        assert env.rng.bit_generator.state == saved['rng']
        assert env.movement_rng.bit_generator.state == saved['movement_rng']
        assert env.initial_seed == saved['initial_seed'] and env.t == saved['t'] == remainder
        checks.append(dict(worker=worker, seed=int(seed), completed_episodes=complete,
                           checkpoint_episode=complete, checkpoint_time=remainder,
                           exact_initial_layout_and_ue_rng_match=True))
    scenes.sort(key=lambda s: (s['episode'], s['worker']))
    return scenes, checks


def new_scenes(cfg, seeds, prefix):
    scenes = []
    for seed in seeds:
        env = UAVEnvironment(cfg.env)
        env.reset(int(seed))
        scenes.append(dict(scene_id=f'{prefix}_{seed}', seed=int(seed),
                           worker=-1, episode=-1, snapshot=env.snapshot()))
    return scenes


def evaluate_scenes(cfg, policy, scenes, action_seed, out=None, trace_count=0):
    """Identical sampled actor, vector batch size, and horizon for every cohort."""
    chunks, initial, traces = [], [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(action_seed)
        for start in range(0, len(scenes), cfg.run.num_envs):
            batch = scenes[start:start + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(batch), 'batch') as vector:
                vector.restore([s['snapshot'] for s in batch])
                if any(e.t != 0 for e in vector.envs):
                    raise ValueError('Full-episode evaluation must start at time zero')
                initial.extend([[e.info[k] for k in FIELDS] for e in vector.envs])
                obs = stack_observations([e.obs for e in vector.envs])
                frames = {i: [e.trace_frame()] for i, e in enumerate(vector.envs)
                          if start + i < trace_count}
                steps = []
                for t in range(cfg.env.horizon):
                    action = policy.act(obs, deterministic=False)
                    obs, _, _, done, truncated, infos = vector.step(action)
                    np.testing.assert_array_equal(done, np.full(len(batch), t == cfg.env.horizon - 1))
                    assert not truncated.any()
                    for info in infos:
                        np.testing.assert_allclose(reward_contributions(info, cfg.env).sum(),
                                                   info['reward'], atol=1e-12, rtol=1e-12)
                    steps.append([[info[k] for k in FIELDS] for info in infos])
                    for i in frames:
                        frames[i].append(vector.envs[i].trace_frame())
                chunks.append(np.swapaxes(np.asarray(steps), 0, 1))
                if out:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    for i, items in frames.items():
                        record = batch[i]
                        path = Path(out) / f"trace_{record['scene_id']}.npz"
                        np.savez_compressed(path, **{k: np.stack([f[k] for f in items]) for k in items[0]},
                                            bs_positions=np.array(cfg.env.bs_positions),
                                            scene_seed=start + i, scene_id=record['scene_id'],
                                            width=cfg.env.width, height=cfg.env.height)
                        traces.append(str(path))
    data = np.concatenate(chunks)
    if not np.isfinite(data).all():
        raise FloatingPointError('Nonfinite evaluation metric')
    return data, np.asarray(initial), traces


def metric_values(data):
    """Return one value per scene, averaging action seeds before inference."""
    cov = data[..., FIELDS.index('coverage')].mean(0)
    result = {'coverage': cov.mean(-1), 'last100_coverage': cov[:, -100:].mean(-1),
              'final_coverage': cov[:, -1], 'steps51_150_coverage': cov[:, 50:150].mean(-1),
              'last100_minus_steps51_150': cov[:, -100:].mean(-1) - cov[:, 50:150].mean(-1)}
    for key in ['discovery_fraction', 'disconnection_ratio', 'boundary_within_150_fraction',
                'boundary_reflection_fraction', 'actual_speed_mean']:
        result[key] = data[..., FIELDS.index(key)].mean((0, 2))
    result['task_return'] = data[..., FIELDS.index('task_reward')].sum(-1).mean(0)
    return result


def cluster_values(values, records, seen):
    # Seen scenes share one continuing UE random stream per training worker.
    # Preserve this dependence instead of treating 64 seen scenes as independent.
    if seen:
        workers = sorted({s['worker'] for s in records})
        return np.array([values[[i for i, s in enumerate(records) if s['worker'] == w]].mean()
                         for w in workers])
    return np.asarray(values)


def interval(values, seed=950001):
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(2000, len(values)))
    lo, hi = np.quantile(values[draws].mean(-1), [.025, .975])
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi))


def independent_difference(left, right, seed=950001):
    """Scene sets are independent; matching action seeds is not scene pairing."""
    rng = np.random.default_rng(seed)
    a = left[rng.integers(len(left), size=(2000, len(left)))].mean(-1)
    b = right[rng.integers(len(right), size=(2000, len(right)))].mean(-1)
    lo, hi = np.quantile(b - a, [.025, .975])
    return dict(delta=float(right.mean() - left.mean()), low=float(lo), high=float(hi))


def publish(out):
    out = Path(out)
    info = json.loads((out / 'experiment.json').read_text())
    datasets, summaries, rows, sample_rows = {}, {}, [], []
    for group in LABELS:
        with np.load(out / group / 'curves.npz') as saved:
            assert list(saved['fields']) == FIELDS
            datasets[group] = saved['diagnostics'].copy()
            initial = saved['initial'].copy()
        records = info['scenes'][group]
        data = datasets[group]
        metrics = metric_values(data)
        summaries[group] = {key: interval(cluster_values(value, records, group == 'seen'))
                            for key, value in metrics.items()}
        summaries[group]['initial_coverage'] = dict(mean=float(initial[..., FIELDS.index('coverage')].mean()))
        for repeat, action_seed in enumerate(info['action_seeds']):
            sample_rows.append(dict(group=group, action_seed=action_seed,
                                    **{k: float(v.mean()) for k, v in metric_values(data[repeat:repeat + 1]).items()}))
            for i, record in enumerate(records):
                for stage, sl in STAGES.items():
                    row = dict(group=group, action_seed=action_seed, scene_id=record['scene_id'],
                               worker=record['worker'], episode=record['episode'], stage=stage)
                    row.update({key: float(data[repeat, i, sl, j].mean()) for j, key in enumerate(FIELDS)})
                    rows.append(row)
    differences = []
    for group in ['unseen', 'validation']:
        a, b = metric_values(datasets['seen']), metric_values(datasets[group])
        for key in a:
            left = cluster_values(a[key], info['scenes']['seen'], True)
            right = cluster_values(b[key], info['scenes'][group], False)
            differences.append(dict(comparison=f'{group}-seen', metric=key,
                                    **independent_difference(left, right)))
    atomic_json(out / 'summary.json', dict(groups=summaries, differences=differences,
                                          interval_unit='training-worker clusters / independent new scenes; action-seed-averaged'))
    write_csv(out / 'episodes_by_stage.csv', rows)
    write_csv(out / 'action_seed_summary.csv', sample_rows)
    write_csv(out / 'differences.csv', differences)
    style()
    plot_fields = ['coverage', 'discovery_fraction', 'disconnection_ratio',
                   'boundary_within_150_fraction', 'uav_nearest_neighbor_mean', 'actual_speed_mean']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for color, (group, data) in zip(COLORS, datasets.items()):
        for ax, key in zip(axes.flat, plot_fields):
            y = data[..., FIELDS.index(key)].mean(1)
            for curve in y:
                ax.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=.15, linewidth=.7)
            ax.plot(np.arange(1, y.shape[-1] + 1), y.mean(0), color=color, label=LABELS[group], linewidth=1.8)
    for ax, key in zip(axes.flat, plot_fields):
        ax.set(xlabel='Episode step', title=key)
        ax.legend(fontsize=6)
    save_figure(fig, out / 'figures', 'full_episode_comparison')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, key in zip(axes, ['coverage', 'last100_coverage', 'final_coverage']):
        for i, (group, table) in enumerate(summaries.items()):
            m = table[key]
            ax.bar(i, 100 * m['mean'], color=COLORS[i], alpha=.8)
            ax.errorbar(i, 100 * m['mean'], yerr=[[100 * (m['mean'] - m['low'])],
                        [100 * (m['high'] - m['mean'])]], color='black', capsize=3)
        ax.set(xticks=range(3), xticklabels=['Seen', 'Fresh unseen', 'Validation'],
               ylabel='Coverage (%)', title=key)
    save_figure(fig, out / 'figures', 'coverage_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for group, data in datasets.items():
        for j, ax in enumerate(axes):
            cov = data[..., FIELDS.index('coverage')].mean(0)
            ax.plot(np.arange(1, cov.shape[1] + 1), cov.mean(0), label=LABELS[group])
            ax.set(xlabel='Episode step', ylabel='Coverage', title='Mean frozen-policy coverage')
            if j:
                ax.set(xlim=(200, 400))
            ax.legend(fontsize=6)
    save_figure(fig, out / 'figures', 'coverage_maintenance')
    cfg = config_from_dict(info['config'])
    for group in LABELS:
        for trace in info['traces'][group]:
            topology(Path(trace), out / group / 'figures', 'best', info['checkpoint_step'], cfg,
                     boundary_overlay=True)
    lines = ['# M1-C best：统一口径的泛化与部署维持检查', '',
             f"来源：`{info['run']}/best.pt`，{info['checkpoint_step']}联合转移。actor冻结，不更新任何网络。",
             '所有组均使用中心2500×2500 m UE初始化、移动UE、采样策略、16环境批量和完整400步。', '',
             f"动作采样种子：{info['action_seeds']}。每个场景重复3次，先按场景平均，再统计组均值。",
             f"训练场景：16个训练worker各抽取已完成回合{info['selected_episodes']}（回合编号从0开始），共64个；选择不使用评估结果。",
             '训练场景重建与最终完整检查点的初始UAV/UE位置、当前UE位置及两套环境随机流逐一精确相符。',
             '新场景：64个独立新种子，与训练、验证和保留测试集隔离。',
             '原验证集：32个场景，参与过best选模，单独报告，不作为完全未见数据。保留测试集未使用。', '',
             '| 场景组 | 初始覆盖 | 全回合覆盖 | 第51–150步 | 后100步覆盖 | 终态覆盖 | 任务回报 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group, table in summaries.items():
        lines.append(f"|{LABELS[group]}|" + '|'.join(f"{table[k]['mean']:.2%}" for k in
                     ['initial_coverage', 'coverage', 'steps51_150_coverage', 'last100_coverage', 'final_coverage']) +
                     f"|{table['task_return']['mean']:.3f}|")
    lines += ['', '| 比较 | 指标 | 差值 | 95%场景区间 |', '|---|---|---:|---|']
    for row in differences:
        if row['metric'] in ['coverage', 'last100_coverage', 'final_coverage']:
            lines.append(f"|{row['comparison']}|{row['metric']}|{100*row['delta']:+.2f} pp|"
                         f"[{100*row['low']:+.2f}, {100*row['high']:+.2f}] pp|")
    lines += ['', '区间使用2000次bootstrap、种子950001。训练组按16个worker聚类重采样（同worker的4个场景保留在一起）；'
              '新场景组按场景重采样。两组布局不同，差值使用独立重采样，不能称为配对场景差值。'
              '3个动作种子先取平均，区间不是训练种子置信区间，也不涵盖全部动作采样不确定性。', '',
              '| 场景组 | 后100步减第51–150步覆盖 | 95%场景区间 |', '|---|---:|---|']
    for group, table in summaries.items():
        m = table['last100_minus_steps51_150']
        lines.append(f"|{LABELS[group]}|{100*m['mean']:+.2f} pp|[{100*m['low']:+.2f}, {100*m['high']:+.2f}] pp|")
    lines += ['', '上述回合内差值使用预先固定的两个时间窗口；不以每回合最大覆盖作比较，避免选最大值产生偏差。', '',
              '![完整回合](figures/full_episode_comparison.png)', '',
              '![覆盖比较](figures/coverage_comparison.png)', '',
              '![后期维持](figures/coverage_maintenance.png)', '',
              '训练已见的是初始布局与外生UE运动序列；最终冻结策略产生的新UAV轨迹通常没有在训练中原样出现。'
              '本实验检查场景泛化和固定策略的部署维持，不把训练期间混合多版策略产生的轨迹作为冻结模型表现。', '',
              '原训练coverage日志为采集末时刻的16环境均值，与本实验完整回合均值不同。',
              '若已见与新场景都退化，更支持部署维持问题；若新场景明显低于已见，支持场景泛化差距。'
              '差值区间跨零只表示本次未检出明确差距，不能证明两组等价。', '',
              '完整来源、哈希、重建校验见experiment.json；逐时刻数据见各组curves.npz；'
              '逐场景/时段数据见episodes_by_stage.csv；各动作种子汇总见action_seed_summary.csv。', '',
              '重建图表：`.venv/bin/python scripts/check_mappo_generalization.py --output ' + str(out) + ' --rebuild`。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return summaries, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.rebuild:
        publish(args.output)
        return
    if (args.output / 'experiment.json').exists():
        raise FileExistsError('Existing experiment; use --rebuild or a new output directory')
    start = time.perf_counter()
    run = args.run
    manifest = json.loads((run / 'manifest.json').read_text())
    sessions = json.loads((run / 'sessions.json').read_text())
    if len(sessions) != 1 or sessions[0].get('resume_from') is not None:
        raise ValueError('This reconstruction requires one uninterrupted training session')
    full_path = run / Path(manifest['resume_checkpoint']).name
    sources = [run / 'best.pt', run / 'config.yaml', run / 'manifest.json', run / 'sessions.json',
               run / 'validation.csv', full_path / 'metadata.pt']
    source_hashes = {str(p): digest(p) for p in sources}
    cfg, policy = policy_from_checkpoint(run / 'best.pt', 'cpu')
    configure_runtime(cfg)
    if not (cfg.algo.name == 'mappo' and cfg.env.action_space == 'continuous'
            and cfg.env.uav_initialization == 'uniform' and cfg.env.user_initialization == 'uniform_interior'
            and cfg.env.horizon == 400 and cfg.run.num_envs == 16):
        raise ValueError('Expected the M1-C continuous MAPPO scene and evaluation protocol')
    full = load_checkpoint(full_path, mmap=True)
    if full['env_steps'] != cfg.run.total_steps or policy.provenance['checkpoint_env_steps'] != full['env_steps']:
        raise ValueError('This audit expects M1-C best at the completed training budget')
    if full['config'] != cfg.as_dict():
        raise ValueError('Checkpoint configuration mismatch')
    seeds = sessions[0]['random_streams']['worker_initialization_seeds']
    expected = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg.run.seed, 101]).spawn(cfg.run.num_envs)]
    assert seeds == expected
    completed = full['env_steps'] // cfg.run.num_envs // cfg.env.horizon
    selected = np.linspace(0, completed - 1, 4, dtype=int).tolist()
    seen, checks = reconstruct_seen(cfg, full, seeds, selected)
    unseen_seeds = list(range(610000, 610000 + len(seen)))
    if set(unseen_seeds) & set(seeds + cfg.run.validation_seeds + cfg.run.test_seeds):
        raise ValueError('Scene seed overlap')
    cohorts = dict(seen=seen, unseen=new_scenes(cfg, unseen_seeds, 'unseen'),
                   validation=new_scenes(cfg, cfg.run.validation_seeds, 'validation'))
    # Verify no fresh initial layout accidentally matches any seen layout.
    layouts = {hashlib.sha256(s['snapshot']['initial_user_xy'].tobytes() +
                             s['snapshot']['initial_uav_xy'].tobytes()).hexdigest() for s in seen}
    assert all(hashlib.sha256(s['snapshot']['initial_user_xy'].tobytes() +
                             s['snapshot']['initial_uav_xy'].tobytes()).hexdigest() not in layouts
               for group in ['unseen', 'validation'] for s in cohorts[group])
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(cohorts, args.output / 'initial_scenes.pt')
    policy.actor.eval()
    policy.actor.requires_grad_(False)
    before = actor_digest(policy.actor)
    action_seeds = [cfg.run.evaluation_action_seed + i for i in range(3)]
    traces = {}
    for group, scenes in cohorts.items():
        data, initials = [], []
        for repeat, action_seed in enumerate(action_seeds):
            values, initial, trace_paths = evaluate_scenes(cfg, policy, scenes, action_seed,
                                                        args.output / group, trace_count=2 if repeat == 0 else 0)
            data.append(values)
            initials.append(initial)
            if repeat == 0:
                traces[group] = trace_paths
            print(f"{group}: action seed {action_seed}, {len(scenes)} full episodes, "
                  f"coverage {values[..., FIELDS.index('coverage')].mean():.4%}", flush=True)
        np.savez_compressed(args.output / group / 'curves.npz', diagnostics=np.stack(data),
                            initial=np.stack(initials), fields=np.array(FIELDS),
                            scene_ids=np.array([s['scene_id'] for s in scenes]),
                            action_seeds=np.array(action_seeds), steps=np.arange(1, cfg.env.horizon + 1))
    with np.load(args.output / 'validation/curves.npz') as saved:
        coverage = saved['diagnostics'][0, ..., FIELDS.index('coverage')].mean()
        task_return = saved['diagnostics'][0, ..., FIELDS.index('task_reward')].sum(-1).mean()
    expected = next(r for r in reversed(read_csv(run / 'validation.csv'))
                    if int(r['env_steps']) == full['env_steps'])
    np.testing.assert_allclose(coverage, float(expected['coverage']), rtol=0, atol=1e-12)
    np.testing.assert_allclose(task_return, float(expected['task_return']), rtol=0, atol=1e-10)
    after = actor_digest(policy.actor)
    assert before == after and all(p.grad is None for p in policy.actor.parameters())
    assert source_hashes == {str(p): digest(p) for p in sources}
    record = dict(run=str(run), checkpoint_step=full['env_steps'], config=cfg.as_dict(),
                  selected_episodes=selected, action_seeds=action_seeds,
                  scenes={k: [{x: y for x, y in s.items() if x != 'snapshot'} for s in v]
                          for k, v in cohorts.items()}, reconstruction_checks=checks,
                  traces=traces, source_hashes=source_hashes,
                  actor_hash_before=before, actor_hash_after=after,
                  validation_original_seed_reproduced=dict(coverage=float(coverage), task_return=float(task_return)),
                  script_hash=digest(Path(__file__)), elapsed_seconds=time.perf_counter() - start)
    atomic_json(args.output / 'experiment.json', record)
    summaries, differences = publish(args.output)
    print(json.dumps(dict(coverage={k: v['coverage'] for k, v in summaries.items()},
                          differences=[r for r in differences if r['metric'] == 'coverage']), indent=2), flush=True)


if __name__ == '__main__':
    main()
