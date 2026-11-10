"""Read-only capacity experiment validation and MC audit; --rebuild resamples nothing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_online_time_factor import bootstrap_difference, write_csv
from report_mappo_bridge import digest, log_plots
from report_mappo_history import per_scene
from report_mappo_training import build_report
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import COLORS, plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime


DEFAULT_RUN = 'runs/mappo_coverage_throughput_capacity10_1m_seed0'
DEFAULT_OUTPUT = 'analysis/mappo_coverage_throughput_capacity10_1m_seed0'


def publish(run, out):
    cfg = load_config(run / 'config.yaml')
    base = json.loads((out / 'report.json').read_text())
    for filename, expected in base['source_file_sha256'].items():
        if digest(run / filename) != expected:
            raise ValueError(f'Source changed since validation: {filename}')
    manifest = base['manifest']
    history = read_csv(run / 'validation.csv')
    logs = {name: read_csv(run / name) for name in ['metrics.csv', 'validation.csv']}
    assert manifest['status'] == 'completed' and manifest['env_steps'] == cfg.run.total_steps
    selection = json.loads((run / 'best_selection.json').read_text())
    selected = max(history, key=lambda r: float(r[cfg.run.best_metric]))
    assert int(selected['env_steps']) == selection['env_steps']
    np.testing.assert_allclose(float(selected[cfg.run.best_metric]), selection['selection_score'])
    for key in ['actor_loss', 'value_loss', 'ppo_approx_kl', 'ppo_clip_fraction', 'actor_grad_norm', 'q_grad_norm']:
        assert np.isfinite([float(r[key]) for r in logs['metrics.csv']]).all(), key
    group = dict(config=cfg.as_dict(), logs=logs)
    style()
    log_plots({'Capacity10': group}, out / 'figures')
    arrays, stats, critic, summaries = {}, {}, {}, {}
    rows = []
    for name in ['best', 'last']:
        with np.load(out / f'critic_{name}' / 'audit.npz') as saved:
            data = {k: saved[k] for k in saved.files}
        arrays[name] = data
        critic[name] = json.loads((out / f'critic_{name}' / 'metrics.json').read_text())
        stats[name] = per_scene(data)
        served = data['agent_diagnostics'][..., 0] * cfg.env.num_users
        np.testing.assert_allclose(served, np.round(served), atol=1e-10)
        assert served.min() >= 0 and served.max() <= cfg.env.capacity + 1e-10
        np.testing.assert_allclose(data['contributions'][..., 2:], 0, atol=0)
        assert data['contribution_names'].tolist() == ['coverage', 'throughput', 'robustness', 'disconnection']
        diag = data['diagnostics']; names = list(data['diagnostic_names'])
        cov = diag[..., names.index('coverage')]
        smoothed = np.stack([np.convolve(c, np.ones(20) / 20, mode='valid') for c in cov])
        extras = dict(
            capacity_utilization=served.sum(-1).mean(1) / (cfg.env.capacity * cfg.env.num_uavs),
            full_uav_fraction=(served >= cfg.env.capacity - 1e-10).mean((1, 2)),
            empty_uav_fraction=(served < .5).mean((1, 2)),
            peak_rolling20_step=smoothed.argmax(1) + 20,
        )
        stats[name].update(extras)
        summaries[name] = {k: float(v.mean()) for k, v in stats[name].items()}
        for i, seed in enumerate(data['scene_seeds']):
            rows.append(dict(checkpoint=name, scene_seed=int(seed),
                             **{k: float(v[i]) for k, v in stats[name].items()}))
        summary = base['evaluations'][name]['summary']['mean']
        np.testing.assert_allclose(cov.mean(), summary['coverage'], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(data['rewards'].sum(1).mean(), summary['return'], rtol=1e-9, atol=1e-9)
        with np.load(out / f'validation_{name}' / 'reward_curves.npz') as curves:
            np.testing.assert_array_equal(data['rewards'], curves['reward'])
        # Store mean curves separately for inspection without rerunning the policy.
        series = {'episode_step': np.arange(1, cfg.env.horizon + 1),
                  **{str(k): diag[..., j].mean(0) for j, k in enumerate(names)}}
        write_csv(out / f'episode_diagnostics_{name}.csv',
                  [dict(zip(series, v)) for v in zip(*series.values())])

    write_csv(out / 'per_scene_diagnostics.csv', rows)
    np.testing.assert_array_equal(arrays['best']['scene_seeds'], arrays['last']['scene_seeds'])
    comparisons = [dict(metric=k, **bootstrap_difference(stats['best'][k], stats['last'][k], k))
                   for k in ['coverage', 'last100_coverage', 'final_coverage', 'maintenance',
                             'train_return', 'all_throughput_bps', 'all_disconnection_ratio']]
    write_csv(out / 'last_minus_best_paired.csv', comparisons)
    results = dict(means=summaries, critic=critic, last_minus_best=comparisons,
                   sampling_seed=cfg.run.evaluation_action_seed,
                   scene_seeds=cfg.run.validation_seeds, training_seeds=[cfg.run.seed],
                   bootstrap=dict(replicates=2000, seed=950001, unit='validation scene'),
                   source_file_sha256=base['source_file_sha256'])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for n, (key, title) in enumerate([
        ('coverage', 'Actual service coverage'), ('discovery_fraction', 'Instantaneous sensing'),
        ('disconnection_ratio', 'Disconnected UAV fraction'),
        ('capacity_rejected_users', 'UE rejected by capacity'),
        ('boundary_within_150_fraction', 'UAV within 150 m of boundary'),
        ('actual_speed_mean', 'Actual UAV speed (m/s)'),
    ]):
        for i, (name, data) in enumerate(arrays.items()):
            values = data['diagnostics'][..., list(data['diagnostic_names']).index(key)]
            axes.flat[n].plot(np.arange(1, cfg.env.horizon + 1), values.mean(0), color=COLORS[i], label=name)
        axes.flat[n].set(title=title, xlabel='Episode step')
        axes.flat[n].legend()
    save_figure(fig, out / 'figures', 'deployment_and_capacity')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (name, data) in enumerate(arrays.items()):
        pred, target = data['values'][..., 0], data['value_labels']
        t = np.arange(cfg.env.horizon)
        for ax, sl in [(axes[0], slice(None)), (axes[1], slice(-20, None))]:
            ax.plot(t[sl], target[:, sl].mean(0), color=COLORS[i], label=f'{name} MC')
            ax.plot(t[sl], pred[:, sl].mean(0), '--', color=COLORS[i], label=f'{name} V')
        axes[2].plot(t, (pred / data['horizon_factors']).mean(0), '--', color=COLORS[i], label=f'{name} V/H')
        axes[2].plot(t, (target / data['horizon_factors']).mean(0), color=COLORS[i], label=f'{name} MC/H')
    for ax, title in zip(axes, ['Cumulative task return', 'Final 20 pre-action states', 'Discounted average task reward']):
        ax.set(title=title, xlabel='Pre-action time t'); ax.legend(fontsize=7)
    save_figure(fig, out / 'figures', 'critic_mc_accuracy')

    initial = history[0]
    text = '# 容量10：覆盖率＋数据率，1M MAPPO 验证\n\n'
    text += (f"实际完成{manifest['env_steps']:,}条联合转移、{manifest['updates']}轮PPO，"
             f"耗时{manifest['elapsed_seconds']/60:.2f}分钟，峰值RSS {manifest['peak_rss_mib']/1024:.3f} GiB。"
             '共享连续actor、一步历史、时间因子critic；奖励严格为 $r=\\rho+0.1\\tau$。\n\n'
             '本次重新执行best、last各32个完整验证回合；固定场景10000–10031、采样动作种子880001、'
             '批次16、400步。覆盖、回报与训练期记录逐项核对一致；没有训练新模型或使用保留测试集。'
             'MC审计的奖励序列与常规验证逐元素相同。源检查点、配置、训练日志哈希在验证前后不变。\n\n')
    text += '| 模型 | 训练步数 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回报 | 断联率 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    text += f"| 初始 | 0 | {float(initial['coverage']):.2%} | — | — | {float(initial['throughput_bps'])/1e6:.2f} | {float(initial['return']):.3f} | {float(initial['disconnection_ratio']):.2%} |\n"
    for name in ['best', 'last']:
        item = base['evaluations'][name]; m = item['summary']['mean']
        text += f"| {name} | {item['checkpoint_env_steps']:,} | {m['coverage']:.2%} | {m['last100_coverage']:.2%} | {m['final_coverage']:.2%} | {m['throughput_bps']/1e6:.2f} | {m['return']:.3f} | {m['disconnection_ratio']:.2%} |\n"
    best, last = summaries['best'], summaries['last']
    text += (f"\n本轮已经学到有效服务：best和last相对初始平均覆盖分别提升"
             f"{100*(best['coverage']-float(initial['coverage'])):.2f}、"
             f"{100*(last['coverage']-float(initial['coverage'])):.2f}个百分点。"
             f"但1M的last较best下降{100*(best['coverage']-last['coverage']):.2f}个百分点；"
             '后期主要是波动和部分退化，不能据此判断延长训练一定继续改善。\n')
    text += '\n## 部署维持与容量\n\n'
    for name, m in summaries.items():
        text += (f"- {name}：第51–150步覆盖{m['steps51_150_coverage']:.2%}，后100步{m['last100_coverage']:.2%}，"
                 f"维持差{100*m['maintenance']:+.2f}个百分点。150m内靠边比例{m['all_boundary_within_150_fraction']:.2%}，"
                 f"反射比例{m['all_boundary_reflection_fraction']:.2%}，近邻小于100m比例{m['all_uav_close_neighbor_fraction']:.2%}。"
                 f"平均实际速度{m['all_actual_speed_mean']:.2f}m/s，近悬停概率{m['near_hover_probability']:.2%}，"
                 f"条件位移方差（两个轴方差之和）{m['command_variance_m2']:.2f}m²。"
                 f"总容量利用率{m['capacity_utilization']:.2%}，满载UAV比例{m['full_uav_fraction']:.2%}，"
                 f"零服务UAV比例{m['empty_uav_fraction']:.2%}。\n")
    text += ('\n容量总数180大于150，空闲本身不等于浪费。阶段性淘汰人数见逐回合CSV与图，'
             '并非通过改变单一条件就可全部恢复的因果收益。局部容量不足与其他UAV空闲可以同时发生。\n\n'
             '固定场景10000中，best形成较分散的多跳拓扑，终态18架UAV均联网、覆盖70%；'
             '场景10001中，best仍有局部聚集和断联，终态覆盖46.67%。'
             '这两个预指定示例支持“已能展开，但不同场景仍存在明显差异”，不能用单图代替32场景统计。'
             '平均速度仍然较高，因此覆盖保持不等于学会了悬停或固定拓扑。\n\n'
             '## 训练后期变化\n\n'
             'best由完整回合覆盖＋数据率回报选择，不按终态覆盖选。以下是last减best的配对场景bootstrap区间；'
             '它是已用于选模的32个场景上的描述性诊断，不是独立测试，也不是训练种子置信区间。\n\n')
    for row in comparisons:
        text += f"- {row['metric']}：{row['delta']:+.6g}，95%场景区间[{row['low']:+.6g}, {row['high']:+.6g}]。\n"
    text += '\n## Critic审计\n\n| 模型 | 全程累计价值RMSE | RMSE/MC标准差 | 最后20步RMSE | 最后一步RMSE | 同时刻跨场景残差相关性 |\n|---|---:|---:|---:|---:|---:|\n'
    for name, c in critic.items():
        v = c['value']; s = v['stages']
        text += f"| {name} | {s['all']['rmse']:.4f} | {s['all']['normalized_rmse']:.4f} | {s['last20']['rmse']:.4f} | {s['last1']['rmse']:.4f} | {v['time_centered_correlation']:.4f} |\n"
    text += ('\nMC使用各模型自身策略轨迹的两项环境奖励，不含actor熵项；一次轨迹回报含策略与UE运动噪声，'
             '不能当成精确期望价值。平滑的平均预测曲线不等于能判断相近部署的动作优劣。\n\n'
             '两组累计价值平均曲线均能跟随剩余时长下降，但同时刻跨场景相关性只有约0.43–0.47，'
             '平均奖励尺度的RMSE约0.081–0.095，说明仍有明显的场景区分误差，不能称critic已完全正确。'
             'last的MC累计误差更小而策略覆盖更低，也说明不能仅凭价值loss或MC误差选择更好的部署策略。\n\n'
             '## 与旧容量20结果的比较边界\n\n'
             '本轮既减小容量又新增数据率奖励；容量20→10还将单用户带宽由1MHz改为2MHz，'
             '同一布局下合法链路速率翻倍，SINR不变，并改变服务人数与感知人数的观测归一化。'
             '因此不能把与旧纯覆盖模型的差异单独归因于容量或数据率，也不能宣称吞吐提升完全来自学到的部署。\n\n'
             '## 图表与复现\n\n')
    for label, file in [('验证随训练变化', 'validation_progress'), ('Loss与策略诊断', 'Capacity10_training_diagnostics'),
                        ('奖励分量', 'reward_components_progress'), ('部署维持和容量', 'deployment_and_capacity'),
                        ('Critic与MC', 'critic_mc_accuracy'), ('best场景10000拓扑', 'topology_best_10000'),
                        ('last场景10000拓扑', 'topology_last_10000'), ('best场景10001拓扑', 'topology_best_10001'),
                        ('last场景10001拓扑', 'topology_last_10001')]:
        text += f'![{label}](figures/{file}.png)\n\n'
    text += ('图表均有PDF/SVG/PNG；逐场景CSV、逐步奖励、拓扑与critic预测已保存。'
             '训练日志奖励是最后采集步的16环境均值，不能当成完整回合平均。\n\n'
             '重建（不重新采样）：\n\n```bash\n.venv/bin/python scripts/report_mappo_capacity.py --rebuild\n```\n')
    (out / 'analysis_report.md').write_text(text)
    atomic_json(out / 'diagnostics.json', results)
    print(json.dumps({k: {m: v for m, v in s.items() if m in ['coverage', 'maintenance', 'last100_coverage', 'final_coverage', 'all_actual_speed_mean']} for k, s in summaries.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if not args.rebuild:
        if not (args.output / 'report.json').exists():
            build_report(args.run, args.output)
        source = [args.run / f for f in ['config.yaml', 'best.pt', 'last.pt', 'metrics.csv', 'validation.csv']]
        hashes = {str(p): digest(p) for p in source}
        for name in ['best', 'last']:
            cfg, policy = policy_from_checkpoint(args.run / f'{name}.pt', 'cpu')
            configure_runtime(cfg)
            policy.actor.eval().requires_grad_(False)
            policy.value.eval().requires_grad_(False)
            audit_policy(cfg, policy, args.output / f'critic_{name}', 64,
                         deployment_times=(150, 200), per_agent=True)
            print(f'{name} MC/action/capacity audit complete', flush=True)
        assert hashes == {str(p): digest(p) for p in source}
        atomic_json(args.output / 'audit_provenance.json', dict(source_sha256=hashes,
                    script_sha256=digest(__file__), actor_updates=0, reserved_test_scenes_used=False))
    publish(args.run, args.output)


if __name__ == '__main__':
    main()
