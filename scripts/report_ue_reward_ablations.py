"""Validate A-D or A-F ablation runs and compare physical metrics on paired scenes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from report_mappo_training import build_report
from uav_research.config import load_config
from uav_research.plotting import COLORS, plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json


GROUPS = {
    'a_mobile_connected': ('coverage', 1.0, .5),
    'b_static_connected': ('coverage', 0.0, .5),
    'c_mobile_coverage': ('coverage', 1.0, 0.0),
    'd_mobile_sensing': ('sensing', 1.0, 0.0),
    'e_mobile_sensing_connected': ('sensing', 1.0, .5),
    'f_static_coverage': ('coverage', 0.0, 0.0),
}
METRICS = ['coverage', 'discovery_fraction', 'disconnection_ratio']


def selected_groups(runs):
    if len(runs) not in (4, 6):
        raise ValueError('Provide runs A, B, C, D or A, B, C, D, E, F in that order')
    return dict(list(GROUPS.items())[:len(runs)])


def check_runs(runs):
    groups = selected_groups(runs)
    configs = [load_config(Path(run) / 'config.yaml') for run in runs]
    reference = None
    for (name, expected), run, cfg in zip(groups.items(), runs, configs):
        actual = (cfg.env.reward_primary, cfg.env.user_speed_std, cfg.env.disconnection_weight)
        if actual != expected or cfg.env.throughput_weight != 0 or cfg.env.robustness_weight != 0:
            raise ValueError(f'{name}: expected objective/motion/mu {expected} and w=lambda=0')
        if cfg.algo.name != 'mappo' or cfg.env.action_space != 'discrete9':
            raise ValueError('This comparison requires discrete MAPPO')
        if set(cfg.run.validation_seeds) & set(cfg.run.test_seeds):
            raise ValueError('Reserved test scenes must not enter this comparison')
        frozen = cfg.as_dict()
        for key in ['reward_primary', 'user_speed_std', 'disconnection_weight']:
            frozen['env'].pop(key)
        # Configurations must differ only in the three experimental factors.
        if reference is not None and frozen != reference:
            raise ValueError('Uncontrolled scenario, learner, or run settings differ')
        reference = frozen
        manifest = json.loads((Path(run) / 'manifest.json').read_text())
        if manifest['status'] != 'completed' or manifest['env_steps'] != cfg.run.total_steps:
            raise ValueError(f'{name}: final checkpoint must complete its configured budget')
    return configs


def compare(runs, output, reports=None):
    groups = selected_groups(runs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = check_runs(runs)
    if reports is None:
        reports = {name: build_report(run, output / name) for name, run in zip(groups, runs)}
    if set(reports) != set(groups):
        raise ValueError('Reports must match exactly the selected experiment groups')
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    curve_rows = []
    for i, (name, run) in enumerate(zip(groups, runs)):
        history = read_csv(Path(run) / 'validation.csv')
        points = []
        for row in history:
            step = int(row['env_steps'])
            means = json.loads((Path(run) / 'validation_history' / f'{step:012d}.json').read_text())['summary']['mean']
            point = dict(group=name, env_steps=step, **{key: means[key] for key in METRICS})
            points.append(point)
            curve_rows.append(point)
        for ax, key in zip(axes, METRICS):
            ax.plot([p['env_steps'] / 1000 for p in points], [p[key] for p in points],
                    label=name[0].upper(), color=COLORS[i], linewidth=1.4)
            ax.set(xlabel='Joint environment transitions (k)', ylabel='Fraction', title=key.replace('_', ' '))
            ax.legend(title='Group')
    fig.suptitle('Matched validation scenes; compare physical metrics, not different reward definitions', fontsize=10)
    save_figure(fig, output / 'figures', 'ablation_validation_curves')

    summaries, paired = [], []
    comparisons = [('b_static_connected', 'a_mobile_connected'),
                   ('c_mobile_coverage', 'a_mobile_connected'),
                   ('d_mobile_sensing', 'c_mobile_coverage')]
    if len(groups) == 6:
        comparisons += [('e_mobile_sensing_connected', 'd_mobile_sensing'),
                        ('e_mobile_sensing_connected', 'a_mobile_connected'),
                        ('f_static_coverage', 'c_mobile_coverage'),
                        ('f_static_coverage', 'b_static_connected')]
    per_scene = {}
    for name, report in reports.items():
        for checkpoint in ['last', 'best']:
            item = report['evaluations'][checkpoint]
            if checkpoint == 'last' and item['checkpoint_env_steps'] != configs[0].run.total_steps:
                raise ValueError('Last checkpoint is not at the common final budget')
            fields = METRICS + ['final_' + k for k in METRICS] + ['last100_' + k for k in METRICS]
            summaries.append(dict(group=name, checkpoint=checkpoint, env_steps=item['checkpoint_env_steps'],
                                  **{k: item['summary']['mean'][k] for k in fields}))
            episodes = read_csv(Path(item['reward_curve']).parent / 'episodes.csv')
            per_scene[name, checkpoint] = {int(row['scene_seed']): row for row in episodes}
    for second, first in comparisons:
        a, b = per_scene[first, 'last'], per_scene[second, 'last']
        if set(a) != set(b):
            raise ValueError('Final evaluations must use exactly paired scenes')
        for seed in sorted(a):
            paired.append(dict(comparison=f'{second[0].upper()}-{first[0].upper()}', scene_seed=seed,
                               **{k: float(b[seed][k]) - float(a[seed][k]) for k in METRICS}))
    for filename, rows in [('validation_curves.csv', curve_rows), ('summary.csv', summaries), ('paired_final_differences.csv', paired)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(output / 'comparison.json', dict(
        runs={name: str(Path(run).resolve()) for name, run in zip(groups, runs)},
        groups=groups, summary=summaries, paired_final_differences=paired,
        training_seed=configs[0].run.seed, common_budget=configs[0].run.total_steps,
        note='Final checkpoint is primary. Best is selected by each own reward. Single-seed diagnosis; no training-seed confidence interval.',
    ))
    text = '# UE 运动与奖励目标对照\n\n'
    text += f'{len(groups)}组共同预算 {configs[0].run.total_steps:,} 条联合转移，训练种子 {configs[0].run.seed}。主要比较最后检查点，最佳模型另列。\n\n'
    text += '| 组别 | 模型 | 步数 | 平均覆盖 | 平均感知 | 平均断联 | 末100步覆盖 | 末100步感知 | 终态覆盖 | 终态感知 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in summaries:
        text += f"| {row['group']} | {row['checkpoint']} | {row['env_steps']} | "
        text += ' | '.join(f"{row[k]:.2%}" for k in METRICS + ['last100_coverage', 'last100_discovery_fraction', 'final_coverage', 'final_discovery_fraction']) + ' |\n'
    text += '\n![验证曲线](figures/ablation_validation_curves.png)\n\n'
    text += 'B−A检查UE运动，C−A检查断联惩罚，D−C检查纯覆盖与纯感知目标。配对差值见CSV；跨奖励组不按总回报排名。单种子差异不能证明唯一根因。\n\n'
    if len(groups) == 6:
        text += 'E−D检查感知目标下的断联惩罚，E−A在相同惩罚下比较感知与覆盖目标；F−C检查纯覆盖下的UE运动，F−B检查静止UE下的断联惩罚。\n\n'
    for name in groups:
        text += f'- [{name} 的Loss、奖励、拓扑与逐场景结果]({name}/report.md)\n'
    (output / 'report.md').write_text(text)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, metavar='RUN',
                        help='A B C D, or A B C D E F; order is fixed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.runs, args.output)


if __name__ == '__main__':
    main()
