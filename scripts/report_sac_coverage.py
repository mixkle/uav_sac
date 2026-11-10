"""Read-only pure-coverage SAC assessment, including cached shared PPO reference.

CPU audits use 32 scenes and three paired action seeds. Native CUDA reevaluation
is kept separate: CPU and CUDA random streams are not identical. No replay is
loaded and no learner update is performed. --rebuild uses saved arrays only.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_history import per_scene
from report_mappo_training import topology
from report_sac_time_factor import digest, training_plots
from uav_research.algorithms import make_learner
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng, seed_all, source_digest

ACTION_SEEDS = [880001, 880002, 880003]


def scene_statistics(files, sac):
    samples = []
    scenes = None
    for path in files:
        with np.load(path) as raw:
            data = {k: raw[k] for k in raw.files}
        if scenes is None:
            scenes = data['scene_seeds']
        np.testing.assert_array_equal(scenes, data['scene_seeds'])
        if sac:
            # per_scene is shared with V audits: explicitly use min twin-Q.
            data['values'] = data['values'].min(-1)[..., None]
        stats = per_scene(data)
        cov = data['diagnostics'][..., list(data['diagnostic_names']).index('coverage')]
        rolling = np.stack([np.convolve(x, np.ones(20)/20, mode='valid') for x in cov])
        stats['peak_step_rolling20'] = rolling.argmax(1)+20
        stats['peak_coverage_rolling20'] = rolling.max(1)
        stats['soft_value_correlation'] = np.array([
            np.corrcoef(a, b)[0, 1] if a.std() and b.std() else 0.
            for a, b in zip(data['values'][..., 0], data['value_labels'])])
        samples.append(stats)
    averaged = {key: np.stack([x[key] for x in samples]).mean(0) for key in samples[0]}
    return [dict(scene_seed=int(seed), **{k: float(v[i]) for k, v in averaged.items()})
            for i, seed in enumerate(scenes)]


def mean_curves(files):
    curves = []
    for f in files:
        with np.load(f) as d:
            names = list(d['diagnostic_names'])
            row = {k: d['diagnostics'][..., names.index(k)].mean(0) for k in
                   ['coverage', 'discovery_fraction', 'disconnection_ratio',
                    'actual_speed_mean', 'boundary_within_150_fraction',
                    'uav_close_neighbor_fraction']}
            row['q'] = d['values'].min(-1).mean(0)
            row['mc'] = d['value_labels'].mean(0)
            row['env_mc'] = d['environment_returns'].mean(0)
            row['future_entropy'] = (d['value_labels']-d['environment_returns']).mean(0)
            row['h'] = d['horizon_factors'].copy()
            row['rmse2'] = ((d['values'].min(-1)-d['value_labels'])**2).mean(0)
            row['variance'] = d['moments'][..., list(d['moment_names']).index('command_variance_m2')].mean((0, 2))
            row['moment_times'] = d['moment_times'].copy()
        curves.append(row)
    return {k: np.mean([c[k] for c in curves], axis=0) for k in curves[0]}


def publish(out, record):
    out = Path(out)
    style()
    figdir = out/'figures'
    run = Path(record['run'])
    cfg = load_config(run/'config.yaml')
    training_plots({'SAC': run}, figdir)
    all_rows, summary = [], []
    stats_by_ck = {}
    for ck, item in record['checkpoints'].items():
        rows = scene_statistics(item['files'], True)
        stats_by_ck[ck] = rows
        all_rows.extend([dict(group='sac', checkpoint=ck, checkpoint_step=item['step'], **r) for r in rows])
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'scene_seed'}
        mean.update(group='sac', checkpoint=ck, checkpoint_step=item['step'])
        for stage in ['all', 'first100', 'middle100', 'last100', 'last20', 'last1']:
            mean[f'{stage}_value_rmse'] = float(np.sqrt(mean[f'{stage}_value_mse']))
        summary.append(mean)
        if ck != 'initial':
            for scene in [10000, 10001]:
                trace = Path(item['files'][0]).parent/f'trace_{scene}.npz'
                topology(trace, figdir, ck, item['step'], cfg,
                         boundary_overlay=True, frame_steps=[0, 150, 200, 400])
    write_csv(out/'summary.csv', summary)
    write_csv(out/'per_scene.csv', all_rows)
    curves = {ck: mean_curves(item['files']) for ck, item in record['checkpoints'].items()}
    for fields, filename in [
        (['coverage', 'discovery_fraction', 'disconnection_ratio', 'actual_speed_mean',
          'boundary_within_150_fraction', 'uav_close_neighbor_fraction'], 'deployment_comparison'),
        (['q', 'mc', 'env_mc', 'future_entropy'], 'soft_q_comparison'),
    ]:
        fig, axes = plt.subplots(2, 3 if len(fields) == 6 else 2, figsize=(13, 7))
        for ck, c in curves.items():
            for ax, key in zip(axes.flat, fields):
                ax.plot(np.arange(1, len(c[key])+1), c[key], label=ck)
                ax.set(title=key, xlabel='Episode step')
                ax.legend(fontsize=7)
        save_figure(fig, figdir, filename)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ck in ['best', 'last']:
        c = curves[ck]
        axes[0, 0].plot(c['q'], label=ck+' min-Q')
        axes[0, 0].plot(c['mc'], '--', label=ck+' soft MC')
        axes[0, 1].plot(c['q']/c['h'], label=ck+' Q/H')
        axes[0, 1].plot(c['mc']/c['h'], '--', label=ck+' MC/H')
        axes[0, 2].plot(np.sqrt(c['rmse2']), label=ck)
        axes[1, 0].plot(c['moment_times'], c['variance'], label=ck)
        axes[1, 1].plot(c['env_mc'], label=ck+' coverage MC')
        axes[1, 1].plot(c['future_entropy'], '--', label=ck+' future entropy')
        axes[1, 2].plot(c['q'], label=ck+' Q')
        axes[1, 2].plot(c['mc'], '--', label=ck+' MC')
    titles = ['Cumulative soft Q', 'Average soft Q', 'Soft-Q RMSE',
              'Conditional displacement variance (m²)', 'MC reward decomposition', 'Final 20 states']
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Pre-action time')
        ax.legend(fontsize=7)
    axes[1, 2].set_xlim(380, 399)
    axes[1, 2].set_ylim(0, max(float(np.max(curves[k][name][-20:]))
                              for k in ['best', 'last'] for name in ['q', 'mc'])*1.08)
    save_figure(fig, figdir, 'critic_and_randomness')
    differences = []
    for a, b, label in [('initial', 'best', 'best-initial'),
                        ('initial', 'last', 'last-initial'), ('best', 'last', 'last-best')]:
        for key in ['coverage', 'last100_coverage', 'final_coverage', 'maintenance',
                    'all_discovery_fraction', 'all_disconnection_ratio']:
            delta = np.array([r[key] for r in stats_by_ck[b]])-np.array([r[key] for r in stats_by_ck[a]])
            differences.append(dict(comparison=label, metric=key, **interval(delta)))
    # Same physical environment; reuse immutable CPU PPO validation, no PPO training.
    ppo = record['ppo_reference']
    ppo_rows = scene_statistics(ppo['files'], False)
    sac_rows = stats_by_ck['last']
    assert [r['scene_seed'] for r in ppo_rows] == [r['scene_seed'] for r in sac_rows]
    for key in ['coverage', 'last100_coverage', 'final_coverage', 'maintenance']:
        delta = np.array([r[key] for r in sac_rows])-np.array([r[key] for r in ppo_rows])
        differences.append(dict(comparison='SAC-last minus shared-PPO-last', metric=key, **interval(delta)))
    write_csv(out/'paired_differences.csv', differences)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ppo_curves = mean_curves(ppo['files'])
    for ax, key in zip(axes, ['coverage', 'discovery_fraction', 'disconnection_ratio']):
        ax.plot(curves['initial'][key], label='Initial actor, CPU')
        ax.plot(curves['last'][key], label='SAC last, 1M')
        ax.plot(ppo_curves[key], label='Shared PPO last, 1M')
        ax.set(title=key, xlabel='Episode step'); ax.legend(fontsize=7)
    save_figure(fig, figdir, 'sac_vs_shared_ppo_last')
    record['summary'] = summary
    record['paired_differences'] = differences
    atomic_json(out/'comparison.json', record)
    lines = ['# SAC纯覆盖1M：冻结策略验证', '',
             '环境：中心UE、移动UE、一步历史、连续圆盘动作、时间因子双Q。没有修改训练算法或继续训练。', '',
             '## 统一CPU评估', '',
             '32场景10000–10031；best/last各用880001/880002/880003三个动作种子，先按场景平均重复。初始actor仅880001；与历史共享PPO相同CPU协议。', '',
             '|模型|步数|平均覆盖|51–150步|后100步|终态|断网UAV|150m靠边|实际速度m/s|min-Q soft-MC RMSE|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['checkpoint']}|{r['checkpoint_step']}|{r['coverage']:.3%}|{r['steps51_150_coverage']:.3%}|{r['last100_coverage']:.3%}|{r['final_coverage']:.3%}|{r['all_disconnection_ratio']:.3%}|{r['all_boundary_within_150_fraction']:.3%}|{r['all_actual_speed_mean']:.2f}|{r['all_value_rmse']:.3f}|")
    lines += ['', '初始随机actor的Q没有经过学习，其误差仅作参照。best按训练期间CUDA验证任务回报选择；CPU复验与CUDA训练日志的随机流不同，不能要求数值完全一致。', '',
              '## 训练期间CUDA验证', '', '|步数|覆盖率|回报|感知率|断网UAV比例|', '|---|---:|---:|---:|---:|']
    vals = read_csv(run/'validation.csv')
    for step in [0, 100000, 250000, record['selection']['env_steps'], 500000, 750000, 1000000]:
        row = min(vals, key=lambda r: abs(int(r['env_steps'])-step))
        lines.append(f"|{row['env_steps']}|{float(row['coverage']):.3%}|{float(row['return']):.3f}|{float(row['discovery_fraction']):.3%}|{float(row['disconnection_ratio']):.3%}|")
    lines += ['', '训练metrics.csv的coverage是最近一次采集时刻的16环境均值，不是完整训练回合均值；判断改善优先看固定验证集曲线。', '',
              '## 图表', '',
              '![训练验证](figures/validation_comparison.png)', '![Loss](figures/sac_loss_diagnostics.png)',
              '![部署](figures/deployment_comparison.png)', '![Q与随机性](figures/critic_and_randomness.png)',
              '![PPO对照](figures/sac_vs_shared_ppo_last.png)']
    for ck in ['best', 'last']:
        for scene in [10000, 10001]:
            lines += ['', f'![{ck} {scene}](figures/topology_{ck}_{scene}.png)']
    lines += ['', '## 结论边界', '',
              'Q标签为冻结当前actor与最终温度下的未来熵soft-Q MC，排除当前动作熵；不是单纯覆盖率MC。低TD loss或Q均值接近target均值，不证明同一状态下动作排序正确。',
              '每个模型使用自身on-policy轨迹；Q误差含MC采样噪声。场景bootstrap不代表训练种子置信区间，best还包含验证选模效应。',
              'SAC与PPO的环境相同，但学习率、熵机制、更新方式、回放、Q/V和归一化不同；不是仅改变算法名字的严格单因素对照。',
              '详细解释见 interpretation.md；所有图可使用 --rebuild 从保存数组重建。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def build(run, out, ppo_reference):
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'comparison.json').exists():
        raise FileExistsError('Use --rebuild for a completed assessment')
    cfg = load_config(run/'config.yaml')
    manifest = json.loads((run/'manifest.json').read_text())
    metadata = json.loads((run/'metadata.json').read_text())
    assert manifest['status'] == 'completed' and manifest['env_steps'] == 1000000
    assert manifest['updates'] == 247500
    assert metadata['source_hash'] == source_digest(), 'Training/evaluation core sources differ'
    assert cfg.algo.name == 'sac' and cfg.env.reward_primary == 'coverage'
    assert all(getattr(cfg.env, k) == 0 for k in ['throughput_weight', 'robustness_weight',
                                                'disconnection_weight', 'boundary_penalty_weight'])
    files = [run/n for n in ['config.yaml', 'manifest.json', 'metadata.json', 'best_selection.json',
                            'metrics.csv', 'validation.csv', 'best.pt', 'last.pt']]
    ppo = json.loads(Path(ppo_reference).read_text())['groups']['shared']
    assert cfg.env == load_config(Path(ppo['run'])/'config.yaml').env
    assert ppo['checkpoints']['last']['step'] == 1000000
    files += [Path(f) for f in ppo['checkpoints']['last']['files']]
    hashes = {str(f): digest(f) for f in files}
    cpu_cfg = copy.deepcopy(cfg); cpu_cfg.run.device = 'cpu'
    configure_runtime(cpu_cfg)
    record = dict(run=str(run), manifest=manifest, metadata=metadata,
                  selection=json.loads((run/'best_selection.json').read_text()),
                  source_sha256=hashes, training_seeds=1, action_seeds=ACTION_SEEDS,
                  core_source_hash=source_digest(), checkpoints={},
                  ppo_reference=dict(run=ppo['run'], files=ppo['checkpoints']['last']['files']))
    started = time.perf_counter()
    for ck in ['initial', 'best', 'last']:
        if ck == 'initial':
            with isolated_rng():
                seed_all(cfg.run.seed); policy = make_learner(cpu_cfg, 'cpu')
            step = 0
        else:
            _, policy = policy_from_checkpoint(run/f'{ck}.pt', 'cpu')
            step = policy.provenance['checkpoint_env_steps']
        policy.actor.eval().requires_grad_(False)
        policy.critic.eval().requires_grad_(False)
        before = actor_digest(policy.actor)
        audit_files = []
        for action_seed in ACTION_SEEDS[:1] if ck == 'initial' else ACTION_SEEDS:
            ec = copy.deepcopy(cpu_cfg); ec.run.evaluation_action_seed = action_seed
            target = out/ck/f'seed_{action_seed}'
            data, metrics = audit_policy(ec, policy, target, 64,
                deployment_times=(150, 200), trace_seeds=(10000, 10001) if action_seed == 880001 else ())
            np.testing.assert_allclose(data['rewards'], data['diagnostics'][..., 0], atol=1e-12)
            if ck == 'initial':
                np.testing.assert_allclose(data['rewards'].mean(), .27228541666666667, atol=1e-10)
            audit_files.append(str(target/'audit.npz'))
            print(f'{ck}/{action_seed} step={step}, coverage={data["rewards"].mean():.6%}, '
                  f'soft-Q RMSE={metrics["value"]["stages"]["all"]["rmse"]:.4f}', flush=True)
        assert actor_digest(policy.actor) == before
        record['checkpoints'][ck] = dict(step=step, files=audit_files, actor_sha256=before,
                                        alpha=float(policy.log_alpha.exp().detach()))
    assert hashes == {str(f): digest(f) for f in files}
    record['elapsed_seconds'] = time.perf_counter()-started
    atomic_json(out/'comparison.json', record)
    publish(out, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/sac_coverage_time_factor_relative_central_history_server_1m_seed0')
    parser.add_argument('--output', default='analysis/sac_coverage_central_history_1m_seed0')
    parser.add_argument('--ppo-reference', default='analysis/mappo_actor_sharing_coverage_1m_seed0/comparison.json')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.rebuild:
        publish(Path(args.output), json.loads((Path(args.output)/'comparison.json').read_text()))
    else:
        build(args.run, args.output, args.ppo_reference)
