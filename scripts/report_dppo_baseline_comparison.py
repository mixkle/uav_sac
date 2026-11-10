"""Rebuild a baseline/DPPO comparison from frozen evaluation files only.

No training or trajectory sampling. Green/blue colours follow the user preference; line styles and marker shapes
provide additional visual distinction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from uav_research.plotting import style, save_figure
from uav_research.diffusion_dppo.report import error_metrics


LABELS = ['baseline_best', 'baseline_last', 'best', 'last']
NAMES = dict(baseline_best='Baseline best', baseline_last='Baseline last',
             best='DPPO best', last='DPPO last')
GREEN, BLUE, GOLD, GRAY = '#009E73', '#0072B2', '#E69F00', '#666666'


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def line_kwargs(name):
    baseline = name.startswith('baseline')
    best = name.endswith('best')
    return dict(color=GREEN if baseline else BLUE,
                ls='--' if best else '-', lw=1.5 if best else 2,
                marker='o' if baseline else 's', markevery=50, markersize=3,
                label=NAMES[name])


def smooth(y, n):
    return np.convolve(y, np.ones(n) / n, mode='valid')


def topology(out, series):
    handles = [Line2D([], [], ls='none', marker=m, color=c, label=l)
               for m, c, l in [('o', GREEN, 'Served UE'), ('x', GOLD, 'Sensed only UE'),
                                ('.', GRAY, 'Unseen UE'), ('^', GREEN, 'Connected UAV'),
                                ('s', BLUE, 'Disconnected UAV'), ('*', 'black', 'BS')]]
    for kind in ['best', 'last']:
        for scene in [10000, 10001]:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))
            for row, label in enumerate(['baseline_' + kind, kind]):
                with np.load(out / label / f'trace_{scene}.npz') as z:
                    for ax, t in zip(axes[row], [0, 150, 200, 400]):
                        xy, users, c, graph, hops = [z[k][t] for k in ['uav_xy', 'user_xy', 'C', 'A', 'B']]
                        bs, w, h = z['bs_positions'], float(z['width']), float(z['height'])
                        seen, served = z['sensed'][t].any(0), c.any(0)
                        for mask, color, marker in [(~seen, GRAY, '.'), (seen & ~served, GOLD, 'x'), (served, GREEN, 'o')]:
                            ax.scatter(users[mask, 0], users[mask, 1], c=color, marker=marker, s=10, linewidths=.7)
                        for i in range(len(xy)):
                            for j in range(i):
                                if graph[i, j]:
                                    ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=GRAY, alpha=.5, lw=.65)
                            for g in range(len(bs)):
                                if hops[i, g] == 1:
                                    ax.plot([xy[i, 0], bs[g, 0]], [xy[i, 1], bs[g, 1]], '--', color=BLUE, lw=.75)
                            ids = np.flatnonzero(c[i])
                            for m in ids:
                                ax.plot([xy[i, 0], users[m, 0]], [xy[i, 1], users[m, 1]], color=GREEN, alpha=.18, lw=.45)
                        linked = hops.min(1) <= len(xy)
                        for mask, color, marker in [(linked, GREEN, '^'), (~linked, BLUE, 's')]:
                            ax.scatter(xy[mask, 0], xy[mask, 1], s=35, c=color, marker=marker, edgecolors='black', linewidths=.3)
                        ax.scatter(bs[:, 0], bs[:, 1], marker='*', color='black', s=90, clip_on=False, zorder=5)
                        ax.add_patch(Rectangle((150, 150), w - 300, h - 300, fill=False, ls=':', ec=GRAY, lw=.6))
                        ax.set(xlim=(0, w), ylim=(0, h), aspect='equal',
                               title=f'{NAMES[label]} | t={t} | {served.mean():.1%}', xlabel='x (m)', ylabel='y (m)')
            fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(.5, -.055), fontsize=8)
            fig.suptitle(f'Fixed scene {scene}; dotted rectangle: 150 m from boundary', y=1.01)
            save_figure(fig, out / 'figures', f'topology_{kind}_{scene}')


def main(out, run, baseline_run):
    out, run, baseline_run = map(Path, [out, run, baseline_run])
    sources = json.loads((out / 'report_sources.json').read_text())
    assert sources['labels'] == LABELS, 'Only baseline and current DPPO checkpoints are allowed'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    budget = int(manifest['env_steps'])
    assert budget > 0 and manifest['optimizer_steps'] == manifest['updates'] * 40
    budget_label = f'{budget / 1e6:g}M'
    NAMES.update(best=f'DPPO {budget_label} best', last=f'DPPO {budget_label} last')
    # A continuation is one experiment; historical milestones are not additional groups.
    for name in LABELS:
        source = sources['sources'][name]
        assert digest(source['path']) == source['sha256'], f'Checkpoint changed: {name}'
    for name in ('training_metrics.csv', 'training_validation.csv'):
        source = sources['sources'][name]
        assert digest(source['path']) == source['sha256'], f'Training log changed: {name}'
    models, series, episodes, critic, moments = {}, {}, {}, {}, {}
    for name in LABELS:
        p = out / name
        models[name] = json.loads((p / 'summary.json').read_text())
        episodes[name] = sorted(read_csv(p / 'episodes.csv'), key=lambda r: int(r['scene_seed']))
        assert [int(r['scene_seed']) for r in episodes[name]] == list(range(10000, 10032))
        with np.load(p / 'step_metrics.npz') as z:
            a, names = z['metrics'], z['names'].tolist()
            assert a.shape[:2] == (32, 400) and z['scene_seeds'].tolist() == list(range(10000, 10032))
            series[name] = {k: a[:, :, j].copy() for j, k in enumerate(names)}
        s = series[name]
        np.testing.assert_allclose(s['reward'], s['coverage'] + .1 * s['tau'], atol=2e-7)
        np.testing.assert_allclose(s['tau'], s['throughput_bps'] / 750e6, atol=1e-10)
        for key in ['coverage', 'throughput_bps', 'reward', 'discovery_fraction']:
            np.testing.assert_allclose(s[key].mean(), models[name][key], atol=1e-7)
        with np.load(p / 'critic_predictions.npz') as z:
            np.testing.assert_allclose(z['mc_return'], z['mc_coverage'] + z['mc_throughput'], atol=1e-5)
            critic[name] = dict(cumulative=error_metrics(z['predicted'], z['mc_return']),
                                average=error_metrics(z['predicted_average'], z['mc_average']))
        if (p / 'action_moments.npz').exists():
            with np.load(p / 'action_moments.npz') as z:
                moments[name] = dict(mean_command_displacement_m=float(np.linalg.norm(z['moments'][:, :, :, 0], axis=-1).mean()),
                                     covariance_trace_m2=float(z['moments'][:, :, :, 1].sum(-1).mean()))
    train, valid = read_csv(run / 'metrics.csv'), read_csv(run / 'validation.csv')
    assert len(train) == manifest['updates']
    assert all(np.isfinite(float(r[k])) for r in train for k in ['actor_loss', 'value_loss', 'actor_grad_norm', 'critic_grad_norm'])
    reproduction = {}
    for name, t in [('best', manifest['best_step']), ('last', manifest['env_steps'])]:
        record = next(r for r in valid if int(r['env_steps']) == t)
        reproduction[name] = {k: abs(models[name][k] - float(record[k])) for k in ['return', 'coverage', 'throughput_bps', 'last100_coverage', 'final_coverage']}
        assert max(reproduction[name].values()) < 1e-5
    rng, paired = np.random.default_rng(950001), []
    for left, right in [('last', 'baseline_last'), ('best', 'baseline_best'), ('last', 'best')]:
        a, b = episodes[left], episodes[right]
        draws = rng.integers(len(a), size=(2000, len(a)))
        for key in ['return', 'coverage', 'last100_coverage', 'final_coverage', 'maintain_coverage',
                    'throughput_bps', 'discovery_fraction', 'disconnection_ratio', 'boundary_within_150_fraction',
                    'boundary_reflection_fraction', 'robustness']:
            d = np.array([float(x[key]) - float(y[key]) for x, y in zip(a, b)])
            low, high = np.quantile(d[draws].mean(1), [.025, .975])
            paired.append(dict(comparison=f'{left} - {right}', metric=key,
                               difference=float(d.mean()), low=float(low), high=float(high)))
    with (out / 'paired_scene_differences.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    windows = []
    for lower in range(0, budget, 500_000):
        rows = [r for r in valid if lower < int(r['env_steps']) <= lower + 500_000]
        windows.append(dict(lower=lower, upper=lower + 500_000, evaluations=len(rows),
            **{k: float(np.mean([float(r[k]) for r in rows])) for k in ['return', 'coverage', 'last100_coverage', 'final_coverage']}))
    peaks = {name: dict(step=int(s['coverage'].mean(0).argmax() + 1), coverage=float(s['coverage'].mean(0).max())) for name, s in series.items()}
    tail = [r for r in train if int(r['env_steps']) > budget - 500_000]
    optimization = {k: float(np.mean([float(r[k]) for r in tail])) for k in
                    ['actor_loss', 'value_loss', 'ppo_approx_kl', 'ppo_clip_fraction', 'actor_grad_norm',
                     'critic_value_clip_fraction', 'denoise_8_kl', 'denoise_8_clip', 'movement_head_grad_norm', 'score_head_grad_norm']}
    scene_distribution = {name: dict(final_median=float(np.median([float(r['final_coverage']) for r in rows])),
                                    final_below_30=sum(float(r['final_coverage']) < .3 for r in rows),
                                    final_above_70=sum(float(r['final_coverage']) >= .7 for r in rows)) for name, rows in episodes.items()}
    fixed_cases = {}
    for name in LABELS:
        fixed_cases[name] = {}
        for scene in (10000, 10001):
            with np.load(out / name / f'trace_{scene}.npz') as z:
                eligible = z['sensed'] & (z['Y'] >= -3) & (z['B'].min(2)[:, :, None] <= 18)
                c = z['C']
                assert np.all(c.sum(1) <= 1) and np.all(c.sum(2) <= 10)
                assert not np.any(c.astype(bool) & ~eligible)
                np.testing.assert_allclose(c.sum((1, 2)) / 150, z['coverage'])
                states = []
                for t in (150, 200, 399, 400):
                    states.append(dict(time=t, served_users=int(c[t].sum()),
                                       sensed_users=int(z['sensed'][t].any(0).sum()),
                                       connected_uavs=int((z['B'][t].min(1) <= 18).sum()),
                                       eligible_users=int(eligible[t].any(0).sum()),
                                       proposal_count=int(z['proposals'][t].sum()) if 'proposals' in z else None))
                fixed_cases[name][str(scene)] = dict(states=states,
                    final_new_eligible_users=int((eligible[-1] & ~eligible[-2]).any(0).sum()),
                    final_new_eligible_edges=int((eligible[-1] & ~eligible[-2]).sum()))
    result = dict(models=models, critic=critic, moments=moments, peaks=peaks, paired=paired,
                  manifest=manifest, validation_reproduction=reproduction, optimization_last500k=optimization,
                  validation_windows=windows, final_distribution=scene_distribution,
                  fixed_scene_audit=fixed_cases,
                  confidence='paired validation-scene bootstrap, 2000 draws, seed 950001; not training-seed confidence',
                  comparison_groups=['baseline', f'{budget_label} DPPO'], figures_palette=dict(baseline=GREEN, dppo=BLUE, auxiliary=GOLD))
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    style()
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[GREEN, BLUE, GOLD, GRAY, '#56B4E9', '#222222', '#999999', '#9A7D00'])
    figdir = out / 'figures'
    for filename, keys in [('task_curves', [('coverage', 100, 'Coverage (%)'), ('throughput_bps', 1e-6, 'Total throughput (Mbps)'), ('reward', 1, 'Reward per step'), ('discovery_fraction', 100, 'Instantaneous sensing (%)')]),
                           ('deployment', [('boundary_within_150_fraction', 100, 'Within 150 m of edge (%)'), ('boundary_reflection_fraction', 100, 'Boundary reflection (%)'), ('uav_close_neighbor_fraction', 100, 'Close neighbour (%)'), ('disconnection_ratio', 100, 'Disconnected UAV (%)')]),
                           ('reward_components', [('coverage', 1, 'Coverage contribution'), ('tau', .1, 'Throughput contribution (0.1 tau)'), ('reward', 1, 'Total reward'), ('near_hover_fraction', 100, 'Actual displacement <1 m (%)')])]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for ax, (key, scale, ylabel) in zip(axes.flat, keys):
            for name in LABELS:
                ax.plot(np.arange(1, 401), series[name][key].mean(0) * scale, **line_kwargs(name))
            ax.set(xlabel='Episode step', ylabel=ylabel)
        axes.flat[0].legend(fontsize=7)
        save_figure(fig, figdir, filename)
    for filename, keys in [('protocol', [('proposal_acceptance', 100, 'Proposal acceptance (%)'), ('over_capacity_fraction', 100, 'Over-capacity candidate fraction (%)'), ('invalidated_proposals', 1, 'Invalidated proposals per step'), ('unscored_waiting_edges', 1, 'Newly eligible, unscored edges')]),
                           ('score_and_motion_distribution', [('score_latent_mean', 1, 'Score latent mean'), ('score_latent_std', 1, 'Score latent standard deviation'), ('score_saturation_fraction', 100, 'Saturated scores (%)'), ('actual_speed_mean', 1, 'Actual displacement per step (m)')])]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for ax, (key, scale, ylabel) in zip(axes.flat, keys):
            for name in LABELS:
                if key in series[name]:
                    ax.plot(np.arange(1, 401), series[name][key].mean(0) * scale, **line_kwargs(name))
            ax.set(xlabel='Episode step', ylabel=ylabel)
        axes.flat[0].legend(fontsize=7)
        save_figure(fig, figdir, filename)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in LABELS:
        path = out / name / 'action_moments.npz'
        if not path.exists():
            continue
        with np.load(path) as z:
            axes[0].plot(z['times'], np.linalg.norm(z['moments'][:, :, :, 0], axis=-1).mean((0, 2)), **line_kwargs(name))
            axes[1].plot(z['times'], z['moments'][:, :, :, 1].sum(-1).mean((0, 2)), **line_kwargs(name))
    axes[0].set(xlabel='Episode step', ylabel='Conditional mean command displacement (m)')
    axes[1].set(xlabel='Episode step', ylabel='Trace of conditional displacement covariance (m²)')
    axes[0].legend(fontsize=7)
    save_figure(fig, figdir, 'action_randomness')
    for kind in ['best', 'last']:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
        for name in ['baseline_' + kind, kind]:
            s = models[name]
            for ax, metric in zip(axes, ['coverage', 'throughput_bps', 'return']):
                scale = 100 if metric == 'coverage' else 1e-6 if metric == 'throughput_bps' else 1
                ix = 0 if name.startswith('baseline') else 1
                bars = ax.bar(ix, s[metric] * scale, color=GREEN if ix == 0 else BLUE, hatch='//' if ix else '', alpha=.85)
                ax.bar_label(bars, fmt='%.2f', padding=3)
                ax.set_xticks([0, 1], ['Baseline', f'DPPO {budget_label}']); ax.set_ylim(0, max(models['baseline_' + kind][metric], models[kind][metric]) * scale * 1.17)
                ax.set_ylabel({'coverage': 'Mean coverage (%)', 'throughput_bps': 'Mean throughput (Mbps)', 'return': 'Episode return'}[metric])
        fig.suptitle(f'{kind} checkpoints; budgets differ: baseline 1M / DPPO {budget_label} + pretraining')
        save_figure(fig, figdir, f'performance_{kind}')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, metric in zip(axes.flat, ['return', 'coverage', 'throughput_bps', 'last100_coverage']):
        y = np.array([float(r[metric]) for r in valid]); x = np.array([int(r['env_steps']) / 1e6 for r in valid])
        scale = 100 if 'coverage' in metric else 1e-6 if metric == 'throughput_bps' else 1
        ax.plot(x, y * scale, color=BLUE, alpha=.25, lw=.7)
        ax.plot(x[9:], smooth(y, 10) * scale, color=BLUE, label='DPPO: trailing 10 evaluations')
        ax.axhline(models['baseline_last'][metric] * scale, color=GREEN, ls='--', label='Baseline last, GPU re-evaluation')
        ax.scatter([manifest['best_step'] / 1e6], [models['best'][metric] * scale], marker='s', color=BLUE, s=35)
        ax.set(xlabel='Cumulative online joint transitions (million)', ylabel=metric + (' (%)' if 'coverage' in metric else ' (Mbps)' if metric == 'throughput_bps' else ''))
    axes.flat[0].legend(fontsize=7); save_figure(fig, figdir, 'validation_learning')
    base_train = read_csv(baseline_run / 'metrics.csv')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, ['actor_loss', 'value_loss', 'ppo_approx_kl', 'ppo_clip_fraction', 'coverage', 'reward']):
        for rows, color, lab, n in [(base_train, GREEN, 'Baseline', 15), (train, BLUE, f'DPPO {budget_label}', 50)]:
            y = np.array([float(r[metric]) for r in rows]); x = np.array([int(r['env_steps']) / 1e6 for r in rows])
            ax.plot(x, y, color=color, lw=.5, alpha=.15)
            ax.plot(x[n - 1:], smooth(y, n), color=color, label=lab, ls='--' if color == GREEN else '-')
        ax.set(xlabel='Cumulative joint transitions (million)', ylabel=metric)
    axes.flat[0].legend(fontsize=8); save_figure(fig, figdir, 'loss_and_training')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, suffix in zip(axes, ['kl', 'clip']):
        for step in [1, 4, 8]:
            key = f'denoise_{step}_{suffix}'; vals = np.array([float(r[key]) for r in train])
            ax.plot(np.array([int(r['env_steps']) / 1e6 for r in train])[49:], smooth(vals, 50), label=f'k={step}', ls={1: '-', 4: '--', 8: ':'}[step])
        ax.set(xlabel='Cumulative joint transitions (million)', ylabel=f'DPPO per-denoising-step {suffix}'); ax.legend()
    save_figure(fig, figdir, 'denoising_diagnostics')
    for kind in ['best', 'last']:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
        for name in ['baseline_' + kind, kind]:
            with np.load(out / name / 'critic_predictions.npz') as z:
                color = GREEN if name.startswith('baseline') else BLUE
                for row, (pred, target, ylabel) in enumerate([('predicted', 'mc_return', 'Cumulative task value'), ('predicted_average', 'mc_average', 'Discounted average task reward')]):
                    for col, sl in enumerate([slice(None), slice(-20, None)]):
                        x = np.arange(400)[sl]
                        axes[row, col].plot(x, z[pred].mean(0)[sl], color=color, label=NAMES[name] + ' prediction')
                        axes[row, col].plot(x, z[target].mean(0)[sl], color=color, ls='--', label=NAMES[name] + ' MC')
                        axes[row, col].set(xlabel='State time before action', ylabel=ylabel)
        axes[0, 0].legend(fontsize=6); save_figure(fig, figdir, f'critic_mc_{kind}')
    topology(out, series)
    lines = [f'# {budget_label}扩散DPPO与baseline验证分析', '',
             f'本报告只比较baseline和{budget_label}组；best/last是各组的选模与最终检查点，不新增其他训练组。', '',
             f"{budget_label}运行完整结束：{manifest['env_steps']:,}条在线联合转移，{manifest['updates']:,}轮PPO，{manifest['optimizer_steps']:,}次actor/critic优化；best来自{manifest['best_step']:,}步。", '',
             '## 验证口径', '',
             '32场景10000–10031、每场景400步、采样种子880001、批次16、CUDA FP32。baseline复用已保存的同口径GPU验证，模型哈希重新核对。正式保留测试集未使用。',
             '奖励为覆盖率＋0.1×归一化吞吐。吞吐归一化分母750 Mbps；单位Mbps表示全系统总吞吐，回合回报为400步未折扣奖励之和。',
             f'baseline在CPU训练、GPU重新验证，浮点及采样路径与其历史CPU报告不同，不能混用两种口径的数值。{budget_label} best/last重新验证与训练内验证数值一致。', '',
             '## 核心结果', '',
             '| 模型 | 全回合覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 |', '|---|---:|---:|---:|---:|---:|']
    for name in LABELS:
        s = models[name]
        lines.append(f"| {NAMES[name]} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_bps']/1e6:.3f} | {s['return']:.4f} |")
    base, last = [models[n] for n in ['baseline_last', 'last']]
    delta_cov = last['coverage'] - base['coverage']
    delta_rate = last['throughput_bps'] - base['throughput_bps']
    best, baseline_best = models['best'], models['baseline_best']
    def interval_statement(metric):
        row = next(r for r in paired if r['comparison'] == 'last - baseline_last' and r['metric'] == metric)
        return '在零以上' if row['low'] > 0 else '在零以下' if row['high'] < 0 else '包含零'
    scene_lines = []
    for scene in (10000, 10001):
        lookup = {name: next(r for r in episodes[name] if int(r['scene_seed']) == scene) for name in LABELS}
        scene_lines.append(f"场景{scene}：baseline best/last终态覆盖为{float(lookup['baseline_best']['final_coverage']):.2%}/{float(lookup['baseline_last']['final_coverage']):.2%}，"
                           f"{budget_label} best/last为{float(lookup['best']['final_coverage']):.2%}/{float(lookup['last']['final_coverage']):.2%}。")
    lines += ['', '## 结果解读', '',
              f"**Best模型的任务收益。** {budget_label} best位于{manifest['best_step']:,}步，相对baseline best平均覆盖变化{(best['coverage']-baseline_best['coverage'])*100:+.3f}个百分点，总吞吐变化{(best['throughput_bps']-baseline_best['throughput_bps'])/1e6:+.3f}Mbps（{best['throughput_bps']/baseline_best['throughput_bps']-1:+.2%}），任务回报变化{best['return']-baseline_best['return']:+.4f}。后100步覆盖为{best['last100_coverage']:.3%}，相对第51–150步变化{best['maintain_coverage']*100:+.3f}个百分点。跨场景平均覆盖峰值为第{peaks['best']['step']}步的{peaks['best']['coverage']:.3%}，终态{best['final_coverage']:.3%}。末段平均能否维持与峰值是否回落是不同判断。", '',
              f"**各组训练终点模型的表现。** 以各组last为主，{budget_label}相对baseline的平均覆盖差值为{delta_cov*100:+.3f}个百分点、总吞吐差值为{delta_rate/1e6:+.3f}Mbps（{delta_rate/base['throughput_bps']:+.2%}）、回报差值为{last['return']-base['return']:+.4f}。两组训练预算不同。覆盖和吞吐的回合奖励贡献分别变化{400*delta_cov:+.4f}与{400*.1*delta_rate/750e6:+.4f}，相加等于总回报差。", '',
              f"last配对场景区间：平均覆盖{interval_statement('coverage')}，总吞吐{interval_statement('throughput_bps')}，总回报{interval_statement('return')}，后100步覆盖{interval_statement('last100_coverage')}。best参与了验证集选模，存在选择偏差。", '',
              f"**回合内展开过程。** 前100步平均覆盖：baseline {series['baseline_last']['coverage'][:, :100].mean():.3%}，{budget_label} {series['last']['coverage'][:, :100].mean():.3%}。{budget_label} last的跨场景平均覆盖在第{peaks['last']['step']}步达到{peaks['last']['coverage']:.3%}，终态为{last['final_coverage']:.3%}；baseline峰值在第{peaks['baseline_last']['step']}步。峰值到终态变化为{(last['final_coverage']-peaks['last']['coverage'])*100:+.3f}个百分点。", '',
              f"**部署维持。** 最后100步覆盖相对baseline变化{(last['last100_coverage']-base['last100_coverage'])*100:+.3f}个百分点；{budget_label}第51–150步覆盖为{series['last']['coverage'][:, 50:150].mean():.3%}，后100步为{last['last100_coverage']:.3%}，维持差值{last['maintain_coverage']*100:+.3f}个百分点。baseline对应两段为{series['baseline_last']['coverage'][:, 50:150].mean():.3%}与{base['last100_coverage']:.3%}。绝对覆盖与回合内退化幅度需分别判断。", '',
              f"**联网、发现与边界。** {budget_label}断联UAV比例{last['disconnection_ratio']:.2%}，baseline {base['disconnection_ratio']:.2%}；瞬时感知率分别为{last['discovery_fraction']:.2%}与{base['discovery_fraction']:.2%}；靠边比例分别为{last['boundary_within_150_fraction']:.2%}与{base['boundary_within_150_fraction']:.2%}，反射比例分别为{last['boundary_reflection_fraction']:.2%}与{base['boundary_reflection_fraction']:.2%}。这些指标用于区分回传、空间展开及边界运动，不能单凭相关性判定唯一根因。", '',
              '固定场景10000与10001的拓扑均按预定编号展示。' + ' '.join(scene_lines) + '某个场景提高不能代表全体。', '',
              f"**Critic与MC。** {budget_label} last累计价值RMSE为{critic['last']['cumulative']['all']['rmse']:.3f}，baseline为{critic['baseline_last']['cumulative']['all']['rmse']:.3f}；归一化RMSE分别为{critic['last']['cumulative']['normalized_rmse']:.3f}与{critic['baseline_last']['cumulative']['normalized_rmse']:.3f}。平均误差不能单独验证对保持动作的长期价值判断。", '',
              f"**Best到last的波动。** last相对best平均覆盖变化{(last['coverage']-best['coverage'])*100:+.3f}个百分点，吞吐变化{(last['throughput_bps']-best['throughput_bps'])/1e6:+.3f}Mbps，回报变化{last['return']-best['return']:+.4f}。仅凭best位于训练后期不能证明尚未收敛，也不能证明训练已经稳定。", '',
              '## 场景配对差值', '', '差值方向均为左减右。95%区间由场景级配对bootstrap得到，2000次、种子950001；不是训练种子置信区间。best使用同一验证集选模，存在选择偏差。', '',
              '| 比较 | 指标 | 均值差 | 95%场景区间 |', '|---|---|---:|---|']
    for r in paired:
        if r['comparison'] == 'last - best' or r['metric'] not in ['return', 'coverage', 'last100_coverage', 'final_coverage', 'throughput_bps', 'maintain_coverage']:
            continue
        key = r['metric']; scale = 1e-6 if key == 'throughput_bps' else 100 if 'coverage' in key else 1
        unit = ' Mbps' if key == 'throughput_bps' else ' 个百分点' if 'coverage' in key else ''
        lines.append(f"| {r['comparison']} | {key} | {r['difference']*scale:+.3f}{unit} | [{r['low']*scale:+.3f}, {r['high']*scale:+.3f}] |")
    lines += ['', '## 部署与诊断', '', '| 模型 | 瞬时感知 | 断联UAV | 距边界150m内 | 反射 | 聚集比例 | 后期−前期覆盖 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        s = models[name]
        lines.append(f"| {NAMES[name]} | {s['discovery_fraction']:.2%} | {s['disconnection_ratio']:.2%} | {s['boundary_within_150_fraction']:.2%} | {s['boundary_reflection_fraction']:.2%} | {s['uav_close_neighbor_fraction']:.2%} | {s['maintain_coverage']*100:+.3f}个百分点 |")
    lines += ['', '### 固定场景末步审计', '',
              '以下记录固定场景的最后两个状态。候选资格含发现、下行及回传条件；实际服务另受移动前提议约束。', '',
              '| 模型 | 场景 | 状态时刻 | 联网UAV | 感知UE | 可服务候选UE | 实际服务UE | 提议边数 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('best', 'last'):
        for scene, case in fixed_cases[name].items():
            for row in case['states'][-2:]:
                lines.append(f"| {NAMES[name]} | {scene} | {row['time']} | {row['connected_uavs']} | {row['sensed_users']} | {row['eligible_users']} | {row['served_users']} | {row['proposal_count']} |")
            a, b = case['states'][-2:]
            if b['eligible_users'] - b['served_users'] >= 50:
                lines += ['', f"场景{scene}的{NAMES[name]}：末步联网UAV从{a['connected_uavs']}变为{b['connected_uavs']}，当前可服务候选UE从{a['eligible_users']}变为{b['eligible_users']}；但只有{b['proposal_count']}条移动前提议，最终服务{b['served_users']}人。新增资格涉及{case['final_new_eligible_users']}个UE、{case['final_new_eligible_edges']}条边。该帧的未服务不能全部解释为下行不足或容量已满；协议要求新候选等待下一次决策，而此时回合已结束。前一状态的回传变化仍需结合完整轨迹判断。", '']
    lines += ['', '后期−前期覆盖固定定义为第301–400步均值减第51–150步均值。反射比例和靠边比例是不同指标。', '', '### 服务阶段未获服务人数（互斥的顺序淘汰统计）', '',
              '| 模型 | 未感知 | 下行不足 | 回传不足 | 容量/协议后未服务 |', '|---|---:|---:|---:|---:|']
    for name in LABELS:
        s = models[name]
        lines.append('| ' + NAMES[name] + ' | ' + ' | '.join(f'{s[k]:.2f}' for k in ['sensing_rejected_users', 'downlink_rejected_users', 'backhaul_rejected_users', 'capacity_rejected_users']) + ' |')
    lines += ['', '最后一列在Top-K路径包含协议竞争及评分时序后的未服务，不能全部归因于容量10；baseline与新方法的候选图和关联方式不同。', '',
              '### Critic与MC', '', '| 模型 | 累计价值RMSE | 按MC标准差归一化RMSE | 去除时刻均值后的相关性 |', '|---|---:|---:|---:|']
    for name in LABELS:
        c = critic[name]['cumulative']
        lines.append(f"| {NAMES[name]} | {c['all']['rmse']:.4f} | {c['normalized_rmse']:.4f} | {c['time_centered_scene_correlation']:.4f} |")
    lines += ['', 'MC使用完整覆盖＋数据率环境奖励，真正终止处为零。各模型使用自身on-policy轨迹，并非相同状态分布的critic对比；平均MC误差不等于对维持动作的反事实价值判断准确。', '',
              '## 训练与成本', '', f"累计在线墙钟{manifest['elapsed_seconds']/3600:.2f}小时，包含验证{manifest['evaluation_seconds']/3600:.2f}小时；示范和预训练成本另计。峰值RSS约{manifest['peak_rss_mib']/1024:.2f}GiB、显存约{manifest['peak_gpu_mib']/1024:.2f}GiB。", '',
              f'完整训练历史作为一条0–{budget_label}曲线展示，未拆成额外训练组。历史baseline训练曲线使用其原CPU日志，跨设备与本次GPU验证有区别；主要性能结论以冻结模型的统一GPU验证为准。',
              'actor loss的绝对值不能直接比较Gaussian PPO与DPPO是否收敛；DPPO的去噪步概率比与Gaussian动作概率比不同。最近训练趋势和loss不能保证继续增加预算必然改善。', '',
              f"最后50万步actor loss均值{optimization['actor_loss']:.6f}，value loss {optimization['value_loss']:.6f}，近似KL {optimization['ppo_approx_kl']:.6f}，PPO裁剪比例{optimization['ppo_clip_fraction']:.2%}；k=8去噪步裁剪比例为{optimization['denoise_8_clip']:.2%}。存在实际策略更新，loss接近零不代表无学习。", '',
              '## 最近训练趋势', '',
              f'以下是同一条{budget_label}训练历史的末端时间窗口，不视为独立实验组。', '',
              '| 累计在线步数窗口 | 平均验证覆盖 | 平均验证回报 |', '|---|---:|---:|---:|']
    for w in windows[-4:]:
        lines.append(f"| {w['lower']/1e6:.1f}–{w['upper']/1e6:.1f}M | {w['coverage']:.3%} | {w['return']:.3f} |")
    lines += ['', '末端窗口用于检查训练趋势，不以best出现时刻或单次loss判断收敛；不能从末端斜率外推追加训练收益。本次没有启动追加训练。', '',
              '## 结论边界', '',
              f'baseline为1M在线样本，DPPO为{budget_label}在线样本并有64k示范和50轮预训练。关联协议、actor、优化方法均有改变。此结果比较最终表现，不能据此宣称相同成本下DPPO优于baseline，也不能把差异单独归因于服务评分。',
              '仅训练种子0，32个验证场景参与选模。场景区间不替代多训练种子验证。', '',
              '## 图表与重建', '',
              '绿色为baseline，蓝色为DPPO；best为虚线，last为实线，同时使用圆/方形标记。拓扑用绿色圆点表示已服务UE、金色叉表示仅感知UE、灰点表示未感知UE；联网/断联UAV使用不同形状。全部图表提供PDF/SVG/PNG。', '',
              '- [最终性能](figures/performance_last.png)', '- [回合覆盖与吞吐](figures/task_curves.png)', '- [部署与边界](figures/deployment.png)',
              '- [奖励分量](figures/reward_components.png)', '- [完整验证学习曲线](figures/validation_learning.png)', '- [Loss与训练](figures/loss_and_training.png)',
              '- [分步KL和裁剪](figures/denoising_diagnostics.png)', '- [Last critic](figures/critic_mc_last.png)',
              '- [关联协议](figures/protocol.png)', '- [评分与运动分布](figures/score_and_motion_distribution.png)', '- [动作条件方差](figures/action_randomness.png)',
              '- [场景10000 last拓扑](figures/topology_last_10000.png)', '- [场景10001 last拓扑](figures/topology_last_10001.png)', '',
              '重建命令（只读保存数据，不重新采样）：', '', '```bash',
              f'.venv-dppo-compat/bin/python scripts/report_dppo_baseline_comparison.py --run {run} --baseline-run {baseline_run} --output {out}', '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out), best_step=manifest['best_step'], models={k: {m: v[m] for m in ['coverage', 'throughput_bps', 'return', 'last100_coverage', 'maintain_coverage']} for k, v in models.items()}), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--baseline-run', required=True)
    a = p.parse_args()
    main(a.output, a.run, a.baseline_run)
