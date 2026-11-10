"""Read-only M0 discrete MAPPO / M1 Gaussian MAPPO / S0 SAC bridge report."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import config_from_dict, load_config
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.plotting import plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime
from compare_online_time_factor import bootstrap_difference, write_csv
from report_mappo_training import topology


DEFAULT_RUNS = [
    'runs/discrete_mappo_full_reward_time_factor_relative_100k_seed0',
    'runs/continuous_mappo_full_reward_time_factor_relative_100k_seed0',
    'runs/sac_full_reward_time_factor_relative_100k_seed0',
]
LABELS = ['M0', 'M1', 'S0']
SOURCE_FILES = ['config.yaml', 'manifest.json', 'metrics.csv', 'validation.csv',
                'best_selection.json', 'best.pt', 'last.pt']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_configs(configs):
    reference = configs['S0']
    for label, cfg in configs.items():
        env = cfg.as_dict()['env']; expected = reference.as_dict()['env']
        env.pop('action_space'); expected.pop('action_space')
        if env != expected or cfg.env.reward_primary != 'coverage':
            raise ValueError('Bridge requires matching physical environment, reward and coordinates')
        if cfg.algo.critic_parameterization != 'time_factor':
            raise ValueError('All bridge critics must use time_factor')
        for key in ['seed', 'total_steps', 'num_envs', 'validation_seeds', 'test_seeds',
                    'evaluation_policy', 'evaluation_action_seed', 'cpu_threads', 'numba_threads']:
            if getattr(cfg.run, key) != getattr(reference.run, key):
                raise ValueError(f'Bridge changed run.{key}')
        for key in ['hidden', 'ue_hidden', 'log_std_min', 'log_std_max', 'discount']:
            if getattr(cfg.algo, key) != getattr(reference.algo, key):
                raise ValueError(f'Bridge changed shared algo.{key}')
        expected_name, space = ('sac', 'continuous') if label == 'S0' else (
            'mappo', 'discrete9' if label == 'M0' else 'continuous')
        if (cfg.algo.name, cfg.env.action_space) != (expected_name, space):
            raise ValueError(f'{label} has wrong algorithm/action space')
    m0, m1 = [configs[k].as_dict() for k in ['M0', 'M1']]
    m0['env']['action_space'] = m1['env']['action_space']
    m0['algo']['mappo_state_dependent_std'] = m1['algo']['mappo_state_dependent_std']
    if m0 != m1 or not configs['M1'].algo.mappo_state_dependent_std:
        raise ValueError('MAPPO configs may differ only in action space and std type')


def audit_run(label, run, out, samples):
    cfg = load_config(run / 'config.yaml')
    configure_runtime(cfg)
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['env_steps'] != cfg.run.total_steps:
        raise ValueError(f'{run} is not a complete configured run')
    logs = {name: read_csv(run / name) for name in ['metrics.csv', 'validation.csv']}
    initial = next(r for r in logs['validation.csv'] if int(r['env_steps']) == 0)
    result = dict(config=cfg.as_dict(), manifest=manifest, logs=logs, initial=initial, checkpoints={})
    for checkpoint in ['best', 'last']:
        pcfg, policy = policy_from_checkpoint(run / f'{checkpoint}.pt', 'cpu')
        if pcfg.env != cfg.env or pcfg.algo != cfg.algo:
            raise ValueError('Saved checkpoint differs from run config')
        step = policy.provenance['checkpoint_env_steps']
        if checkpoint == 'last' and step != cfg.run.total_steps:
            raise ValueError('Last checkpoint does not match completed budget')
        selection = json.loads((run / 'best_selection.json').read_text())
        if checkpoint == 'best' and step != selection['env_steps']:
            raise ValueError('Best selection step does not match checkpoint')
        target = out / label / f'validation_{checkpoint}'
        rows, summary = evaluate(pcfg, policy, cfg.run.validation_seeds, target,
                                  trace_count=min(2, len(cfg.run.validation_seeds)), reward_curves=True)
        match = [r for r in logs['validation.csv'] if int(r['env_steps']) == step]
        if not match:
            raise ValueError(f'No logged validation at checkpoint step {step}')
        for key in ['return', 'coverage', 'discovery_fraction', 'disconnection_ratio']:
            np.testing.assert_allclose(summary['mean'][key], float(match[-1][key]), atol=1e-9, rtol=1e-9)
        if checkpoint == 'best':
            np.testing.assert_allclose(summary['mean']['return'], selection['validation_return'], atol=1e-9, rtol=1e-9)
            metric = cfg.run.best_metric
            key = metric if metric in match[-1] else 'return'
            score = selection.get('selection_score', selection['validation_return'])
            np.testing.assert_allclose(summary['mean'][metric], score, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(float(match[-1][key]), max(float(r[key]) for r in logs['validation.csv']), atol=1e-9)
        data, critic = audit_policy(pcfg, policy, out / label / checkpoint, samples)
        with np.load(target / 'reward_curves.npz') as curves:
            np.testing.assert_array_equal(data['rewards'], curves['reward'])
        result['checkpoints'][checkpoint] = dict(step=step, summary=summary, episodes=rows, critic=critic)
        if cfg.env.action_space == 'continuous':
            auxiliary = copy.deepcopy(pcfg)
            auxiliary.run.evaluation_policy = 'deterministic'
            _, aux_summary = evaluate(auxiliary, policy, cfg.run.validation_seeds,
                                      out / label / f'deterministic_{checkpoint}')
            result['checkpoints'][checkpoint]['deterministic_auxiliary'] = aux_summary
        print(f"{label}/{checkpoint}: step {step}, coverage {summary['mean']['coverage']:.4%}", flush=True)
    return result


def paired_results(groups):
    results = []
    for checkpoint in ['best', 'last']:
        for left, right in itertools.combinations(LABELS, 2):
            a = groups[left]['checkpoints'][checkpoint]['episodes']
            b = groups[right]['checkpoints'][checkpoint]['episodes']
            if [r['scene_seed'] for r in a] != [r['scene_seed'] for r in b]:
                raise ValueError('Paired scene order differs')
            for key in ['return', 'coverage', 'last100_coverage', 'final_coverage',
                        'discovery_fraction', 'robustness', 'disconnection_ratio']:
                x, y = [np.array([r[key] for r in rows]) for rows in [a, b]]
                results.append(dict(checkpoint=checkpoint, comparison=f'{right}-{left}', metric=key,
                                    **bootstrap_difference(x, y, key)))
    return results


def log_plots(groups, figures):
    for time_key, filename, divisor in [('env_steps', 'validation_progress', 1000),
                                       ('elapsed_seconds', 'validation_wall_time', 60)]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fields = ['return', 'coverage', 'discovery_fraction', 'disconnection_ratio', 'robustness', 'throughput_bps']
        for label, g in groups.items():
            rows = g['logs']['validation.csv']
            x = np.array([float(r[time_key]) for r in rows]) / divisor
            for ax, field in zip(axes.flat, fields):
                ax.plot(x, [float(r[field]) for r in rows], label=label)
        for ax, field in zip(axes.flat, fields):
            ax.set(title=field, xlabel='Joint transitions (k)' if time_key == 'env_steps' else 'Wall time (min)')
            ax.legend(fontsize=7)
        save_figure(fig, figures, filename)
    # Use saved per-step contributions for training and reconstruct validation
    # contributions from the full validation-history metrics, not coverage alone.
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    names = ['Coverage', 'Throughput', 'Robustness', 'Disconnection']
    for label, g in groups.items():
        e = g['config']['env']
        weights = np.array([1., e['throughput_weight'], e['robustness_weight'], -e['disconnection_weight']])
        for row_index, table in enumerate(['metrics.csv', 'validation.csv']):
            rows = list({r['env_steps']: r for r in g['logs'][table]}.values())
            x = np.array([int(r['env_steps']) for r in rows]) / 1000
            terms = np.array([[float(r['coverage']), float(r['throughput_bps']) /
                              (e['num_users'] * e['rate_reference_bps']), float(r['robustness']),
                              float(r['disconnection_ratio'])] for r in rows]) * weights
            for i, ax in enumerate(axes[row_index]):
                ax.plot(x, terms[:, i], label=label)
                ax.set(title=('Training step snapshot: ' if row_index == 0 else 'Validation episode mean: ') + names[i],
                       xlabel='Joint transitions (k)')
                ax.legend(fontsize=7)
    save_figure(fig, figures, 'reward_components_progress')
    for label, g in groups.items():
        rows = list({r['env_steps']: r for r in g['logs']['metrics.csv']}.values())
        if not rows:
            continue
        fields = (['actor_loss', 'q_loss', 'alpha', 'joint_entropy', 'actor_grad_norm', 'q_grad_norm']
                  if label == 'S0' else ['actor_loss', 'value_loss', 'joint_entropy', 'ppo_approx_kl',
                    'ppo_clip_fraction', 'encoder_grad_norm', 'actor_grad_norm', 'q_grad_norm',
                    'critic_value_clip_fraction', 'critic_return_normalizer_mean',
                    'critic_return_normalizer_std', 'critic_average_normalizer_std'])
        columns = 3
        fig, axes = plt.subplots((len(fields) + 2) // columns, columns,
                                 figsize=(13, 3 * ((len(fields) + 2) // columns)), squeeze=False)
        x = np.array([int(r['env_steps']) for r in rows]) / 1000
        for ax, field in zip(axes.flat, fields):
            ax.plot(x, [float(r[field]) if r.get(field) else np.nan for r in rows])
            ax.set(title=field, xlabel='Joint transitions (k)')
        fig.suptitle(f'{label}: separate loss and entropy scales')
        save_figure(fig, figures, f'{label}_training_diagnostics')


def publish(out, comparison):
    style()
    groups = comparison['groups']; figures = out / 'figures'
    log_plots(groups, figures)
    policy_rows, critic_rows, moment_rows = [], [], []
    for checkpoint in ['best', 'last']:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        reward_fig, reward_axes = plt.subplots(1, 4, figsize=(15, 3.7))
        for label in LABELS:
            g = groups[label]; cfg = config_from_dict(g['config'])
            with np.load(out / label / checkpoint / 'audit.npz') as data:
                diagnostic_names = list(data['diagnostic_names'])
                for ax, field in zip(axes.flat, ['coverage', 'discovery_fraction', 'disconnection_ratio',
                        'actual_speed_mean', 'uav_nearest_neighbor_mean', 'uav_close_neighbor_fraction']):
                    ax.plot(data['diagnostics'][..., diagnostic_names.index(field)].mean(0), label=label)
                    ax.set(title=field, xlabel='Post-action step')
                for i, ax in enumerate(reward_axes):
                    ax.plot(data['contributions'][..., i].mean(0), label=label)
                    ax.set(title=['Coverage', 'Throughput', 'Robustness', 'Disconnection'][i], xlabel='Post-action step')
                # Keep V and soft-Q diagnostics separate: their labels are different.
                pred = data['values'].min(-1) if label == 'S0' else data['values'][..., 0]
                target = data['value_labels']; h = data['horizon_factors']
                for stage, sl in [('all', slice(None)), ('last100', slice(-100, None)), ('last1', slice(-1, None))]:
                    row = dict(group=label, checkpoint=checkpoint, stage=stage)
                    for i, field in enumerate(diagnostic_names):
                        row[str(field)] = float(data['diagnostics'][:, sl, i].mean())
                    for i, term in enumerate(['coverage', 'throughput', 'robustness', 'disconnection']):
                        row['reward_' + term] = float(data['contributions'][:, sl, i].mean())
                    row['reward_total'] = float(data['rewards'][:, sl].mean())
                    row['retention_ratio'] = row['robustness'] / row['coverage'] if row['coverage'] else None
                    for i, field in enumerate(data['movement_names']):
                        row[str(field)] = float(data['movement'][:, sl, i].mean())
                    policy_rows.append(row)
                critic = g['checkpoints'][checkpoint]['critic']
                for scale in ['value', 'average']:
                    for stage, values in critic[scale]['stages'].items():
                        critic_rows.append(dict(group=label, checkpoint=checkpoint, scale=scale,
                                                stage=stage, **values))
                vf, va = plt.subplots(2, 3, figsize=(13, 7))
                for ax in va[0, :2]:
                    ax.plot(target.mean(0), label='Sampled soft MC' if label == 'S0' else 'Environment MC')
                    ax.plot(pred.mean(0), label='Prediction')
                va[0, 1].set_xlim(max(0, len(h) - 20), len(h) - 1)
                va[0, 2].plot((target / h).mean(0), label='MC / H')
                va[0, 2].plot((pred / h).mean(0), label='Prediction / H')
                va[1, 0].plot(np.sqrt(((pred - target)**2).mean(0)), label='Value RMSE')
                va[1, 1].plot(np.sqrt((((pred - target) / h)**2).mean(0)),
                              label='Average soft-Q RMSE' if label == 'S0' else 'Average reward RMSE')
                times = data['moment_times']; moments = data['moments'].mean((0, 2))
                for t, values in zip(times, moments):
                    moment_rows.append(dict(group=label, checkpoint=checkpoint, time=int(t),
                                            **dict(zip(map(str, data['moment_names']), map(float, values)))))
                va[1, 2].plot(times, moments[:, 0], label='Command variance (m²)')
                for ax in va.flat:
                    ax.set(xlabel='Pre-action time'); ax.legend(fontsize=7)
                vf.suptitle(f'{label} {checkpoint}: {g["checkpoints"][checkpoint]["critic"]["value_semantics"]}')
                save_figure(vf, out / label / 'figures', f'critic_{checkpoint}')
                mf, ma = plt.subplots(1, 4, figsize=(14, 3.6))
                for i, ax in enumerate(ma):
                    ax.plot(times, moments[:, i]); ax.set(title=str(data['moment_names'][i]), xlabel='Pre-action time')
                mf.suptitle(f'{label} {checkpoint}: entropy units depend on distribution')
                save_figure(mf, out / label / 'figures', f'action_moments_{checkpoint}')
            for seed in cfg.run.validation_seeds[:2]:
                topology(out / label / f'validation_{checkpoint}' / f'trace_{seed}.npz',
                         out / label / 'figures', checkpoint, g['checkpoints'][checkpoint]['step'], cfg)
        for ax in axes.flat:
            ax.legend(fontsize=7)
        for ax in reward_axes:
            ax.legend(fontsize=7)
        save_figure(fig, figures, f'deployment_{checkpoint}')
        save_figure(reward_fig, figures, f'reward_components_{checkpoint}')
    lines = ['# MAPPO动作桥接：M0 / M1 / S0', '',
        '主要比较同预算last；best按完整环境奖励选择。所有模型使用自身on-policy轨迹。', '',
        '| 组别 | 模型 | 步数 | 初始覆盖 | 平均覆盖 | 后100步 | 终态覆盖 | 回合奖励 | 价值RMSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, g in groups.items():
        for ck, item in g['checkpoints'].items():
            m = item['summary']['mean']; rmse = item['critic']['value']['stages']['all']['rmse']
            lines.append(f"| {label} | {ck} | {item['step']} | {float(g['initial']['coverage']):.2%} | "
                         f"{m['coverage']:.2%} | {m['last100_coverage']:.2%} | {m['final_coverage']:.2%} | {m['return']:.4f} | {rmse:.3f} |")
    lines += ['', 'M0为九动作MAPPO，M1为状态相关高斯连续MAPPO，S0为连续SAC。MAPPO的V标签为环境MC，SAC的Q标签含未来熵；不将两类RMSE作为同一预测任务排名。', '',
        '分类熵和连续微分熵不能直接按数值大小比较；空间条件方差与近悬停概率使用同一物理单位。训练日志奖励是最近采集步的环境均值，不是整回合均值。', '',
        '![验证进度](figures/validation_progress.png)', '![墙钟](figures/validation_wall_time.png)',
        '![奖励分量随训练变化](figures/reward_components_progress.png)',
        '![Last部署](figures/deployment_last.png)', '![Last奖励](figures/reward_components_last.png)', '',
        '## 场景配对差值', '',
        '2000次场景bootstrap，seed950001；不是训练种子置信区间。best使用同一验证集选模，区间含选择效应。', '',
        '| 模型 | 比较 | 指标 | 差值 | 95%场景区间 |', '|---|---|---|---:|---|']
    for row in comparison['paired']:
        lines.append(f"| {row['checkpoint']} | {row['comparison']} | {row['metric']} | {row['delta']:+.5f} | [{row['low']:+.5f}, {row['high']:+.5f}] |")
    for label, g in groups.items():
        lines += ['', f'## {label}', '', f'![训练诊断](figures/{label}_training_diagnostics.png)']
        for ck in ['best', 'last']:
            lines += [f'![{ck}价值]({label}/figures/critic_{ck}.png)',
                      f'![{ck}动作]({label}/figures/action_moments_{ck}.png)']
            for seed in g['config']['run']['validation_seeds'][:2]:
                lines.append(f'![{ck}场景{seed}]({label}/figures/topology_{ck}_{seed}.png)')
    lines += ['', 'M1改善而S0退化，支持继续检查SAC学习机制；M0改善而M1退化，提示动作及分布表达或其优化适配值得检查。M0/M1都不改善时，应结合100k预算及坐标变化判断。价值预测改善不足以证明部署维持已解决。', '',
              '来源配置、日志快照、检查点/源码哈希、峰值内存与运行耗时见comparison.json；逐状态预测、标签与动作矩见各组best/last/audit.npz。保留测试集未参与。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    write_csv(out / 'policy_metrics.csv', policy_rows)
    write_csv(out / 'critic_metrics.csv', critic_rows)
    write_csv(out / 'action_moments.csv', moment_rows)
    write_csv(out / 'paired_differences.csv', comparison['paired'])


def build(runs, output, samples=64):
    out = Path(output)
    if (out / 'comparison.json').exists():
        raise ValueError('Report already exists; use --rebuild-only or choose a new output')
    runs = dict(zip(LABELS, map(Path, runs)))
    configs = {k: load_config(v / 'config.yaml') for k, v in runs.items()}
    validate_configs(configs)
    files = [run / name for run in runs.values() for name in SOURCE_FILES]
    files += list(Path('src/uav_research').glob('*.py')) + [Path(__file__)]
    hashes = {str(p): digest(p) for p in files}
    groups = {k: audit_run(k, run, out, samples) for k, run in runs.items()}
    comparison = dict(groups=groups, paired=paired_results(groups), source_sha256=hashes,
                      training_seed_count=1, moment_samples=samples, bootstrap_repeats=2000,
                      bootstrap_seed=950001)
    assert all(digest(p) == value for p, value in hashes.items())
    atomic_json(out / 'comparison.json', comparison)
    publish(out, comparison)
    return comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs=3, default=DEFAULT_RUNS, metavar=('M0', 'M1', 'S0'))
    p.add_argument('--output', type=Path, default=Path('analysis/mappo_continuous_bridge_100k_seed0'))
    p.add_argument('--moment-samples', type=int, default=64)
    p.add_argument('--rebuild-only', action='store_true')
    args = p.parse_args()
    if args.rebuild_only:
        publish(args.output, json.loads((args.output / 'comparison.json').read_text()))
    else:
        build(args.runs, args.output, args.moment_samples)


if __name__ == '__main__':
    main()
