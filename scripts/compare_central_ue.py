"""Compare central-UE initialization against C using saved validation results."""
import csv
import json
from pathlib import Path
import numpy as np
from uav_research.config import load_config
from uav_research.plotting import plt, style, save_figure, read_csv


def main():
    out = Path('analysis/mappo_central_ue_250k_seed0')
    sources = {
        'Full-world UE': (Path('runs/mappo_ablation_c_mobile_coverage_250k_seed0'),
                          Path('analysis/ue_reward_ablations_250k_seed0/c_mobile_coverage')),
        'Central UE': (Path('runs/mappo_central_ue_250k_seed0'), out),
    }
    configs = [load_config(run/'config.yaml') for run, _ in sources.values()]
    configs[0].env.user_initialization = 'uniform_interior'
    configs[0].env.user_initial_margin_fraction = 1/7
    assert configs[0] == configs[1], 'Unexpected additional configuration change'
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    rows, initial = [], {}
    for label, (run, reportdir) in sources.items():
        history = read_csv(run/'validation.csv')
        initial[label] = float(history[0]['coverage'])
        x = np.array([int(r['env_steps']) for r in history]) / 1000
        y = np.array([float(r['coverage']) for r in history]) * 100
        axes[0].plot(x, y, label=label)
        axes[1].plot(x, y - y[0], label=label)
        report = json.loads((reportdir/'report.json').read_text())
        for ckpt in ['best', 'last']:
            evaluation = report['evaluations'][ckpt]
            m = evaluation['summary']['mean']
            row = dict(method=label, checkpoint=ckpt, step=evaluation['checkpoint_env_steps'],
                       untrained_coverage=initial[label], gain=m['coverage']-initial[label])
            for key in ['coverage','last100_coverage','final_coverage','discovery_fraction',
                        'disconnection_ratio','final_disconnection_ratio','hover_fraction',
                        'uav_nearest_neighbor_mean','final_uav_nearest_neighbor_mean',
                        'uav_close_neighbor_fraction','final_uav_close_neighbor_fraction']:
                row[key] = m[key]
            rows.append(row)
    for ax in axes:
        ax.set(xlabel='Joint transitions (k)')
        ax.legend()
    axes[0].set(ylabel='Validation coverage (%)', title='32 scenarios; sampled policy')
    axes[1].set(ylabel='Gain over own untrained policy (pp)', title='Learning gain within each scenario distribution')
    save_figure(fig, out/'figures', 'initialization_training_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ckpt in zip(axes, ['best', 'last']):
        for label, (_, reportdir) in sources.items():
            with np.load(reportdir/f'validation_{ckpt}'/'reward_curves.npz') as data:
                ax.plot(data['steps'], data['reward'].mean(axis=0)*100, label=label)
        ax.set(xlabel='Episode step', ylabel='Coverage (%)', title=ckpt)
        ax.legend()
    save_figure(fig, out/'figures', 'initialization_coverage_horizon')
    with (out/'initialization_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 中央 UE 初始化与原 C 组对照', '',
             '两组均从头训练250k，训练种子0；32个固定验证种子，采样评估。仅UE初始分布改变。', '',
             'UE在中央2500×2500 m内独立均匀初始化，之后仍按原移动模型运动。UAV在3500×3500 m全区域初始化。', '',
             '| 场景 | 模型 | 步数 | 未训练覆盖 | 全回合覆盖 | 相对未训练增益 | 最后100步 | 终态覆盖 | 平均断联 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['checkpoint']} | {r['step']} | " +
                     ' | '.join(f"{100*r[k]:.2f}%" for k in ['untrained_coverage','coverage']) +
                     f" | {100*r['gain']:+.2f} pp | " +
                     ' | '.join(f"{100*r[k]:.2f}%" for k in ['last100_coverage','final_coverage','disconnection_ratio'])+' |')
    lines += ['', 'best按全回合覆盖选择，可能早于250k；last用于同训练预算比较。', '',
              '用户密度从12.245增至24人/km²，同时远离角落BS。绝对覆盖变化同时包含场景和策略变化；相对未训练增益也只是单训练种子的描述，不能证明泛化优势。', '',
              '![训练对比](figures/initialization_training_comparison.png)', '',
              '![回合内覆盖](figures/initialization_coverage_horizon.png)', '',
              '![Best固定场景10000拓扑](figures/topology_best_10000.png)', '',
              '![Last固定场景10000拓扑](figures/topology_last_10000.png)', '',
              '完整损失、奖励、PPO诊断及场景10001拓扑见[独立评估报告](report.md)。', '',
              '复现比较：`.venv/bin/python scripts/compare_central_ue.py`；配置差异在脚本中断言。']
    with np.load(out/'validation_best'/'reward_curves.npz') as data:
        curve = data['reward'].mean(axis=0)
    quarters = [curve[i:i+100].mean() for i in range(0, 400, 100)]
    lines += ['', '## 回合内表现与拓扑解读', '',
              '32个验证场景的平均覆盖率，按回合四个100步阶段统计为：' +
              ' → '.join(f'{x:.2%}' for x in quarters) + '。', '',
              f'逐步平均曲线在第{curve.argmax()+1}步达到{curve.max():.2%}；'
              '单步峰值容易受波动影响，应结合上述100步均值判断部署维持。', '',
              '本次回合中期覆盖提高，但后期又回落，因此仍不能认为形成了稳定的覆盖部署。'
              '训练进度中的性能改善与一个400步回合内部的后期退化，是两个不同的时间尺度。', '']
    for seed in [10000, 10001]:
        with np.load(out/'validation_best'/f'trace_{seed}.npz') as data:
            cov = data['coverage'][[0, 200, 400]]
            reachable = np.rint(18*(1-data['disconnection_ratio'][[0, 200, 400]])).astype(int)
        lines.append(f'- 固定场景{seed}：初始/200步/终态覆盖分别为' +
                     '/'.join(f'{x:.2%}' for x in cov) + '，联网UAV分别为' +
                     '/'.join(map(str, reachable)) + '架。')
    lines += ['', '场景10000终态有多架UAV聚在右下边界，另一些集中于左上方向；联网数量增加却未保持用户覆盖。'
              '场景10001中期形成面向区域内部的多跳连接，终态则有局部聚集和右侧断联子网。'
              '这两个固定场景用于展示具体失效方式，不能替代32场景总体统计。', '',
              '本次只检验初始用户分布，不重做critic校准，因此不能凭value loss下降断言价值预测问题已经解决。'
              '吞吐、鲁棒性、断联权重均为0；学习到的目标仍是实际联网服务覆盖率。']
    (out/'initialization_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
