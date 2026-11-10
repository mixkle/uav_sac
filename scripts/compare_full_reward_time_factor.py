"""Validate complete-reward MAPPO critics and rebuild a paired comparison report.

No training occurs here. Each critic is evaluated on its own frozen actor's
on-policy rollouts; MC labels contain all four environment reward components.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from compare_online_time_factor import bootstrap_difference, write_csv
from uav_research.config import load_config
from uav_research.critic_calibration import discounted_episode_returns
from uav_research.critic_time_factor import error_metrics, horizon_factors
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import plt, read_csv, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


OUT = Path('analysis/mappo_full_reward_time_factor_250k_seed0')
RUNS = {mode: Path(f'runs/mappo_full_reward_{mode}_250k_seed0')
        for mode in ['direct', 'time_factor']}
STAGES = {'all': slice(None), 'first100': slice(0, 100),
          'middle100': slice(150, 250), 'last100': slice(-100, None),
          'last20': slice(-20, None), 'last1': slice(-1, None)}
DIAGNOSTICS = ['coverage', 'tau', 'throughput_bps', 'robustness',
               'disconnection_ratio', 'discovery_fraction', 'hover_fraction',
               'actual_speed_mean', 'uav_nearest_neighbor_mean',
               'uav_close_neighbor_fraction', 'boundary_reflection_fraction',
               'sensing_rejected_users', 'downlink_rejected_users',
               'backhaul_rejected_users', 'capacity_rejected_users']


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit_full_reward(run, checkpoint, output):
    cfg, policy = policy_from_checkpoint(run / f'{checkpoint}.pt', 'cpu')
    configure_runtime(cfg)
    weights = np.array([1., cfg.env.throughput_weight, cfg.env.robustness_weight,
                        -cfg.env.disconnection_weight])
    all_values, all_raw, all_diagnostics = [], [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(seeds), 'batch') as vector:
                obs, state, _ = vector.reset(seeds)
                values, raw, diagnostics = [], [], []
                for t in range(cfg.env.horizon):
                    values.append(policy.values(state).cpu().numpy())
                    obs, state, _, done, truncated, infos = vector.step(policy.act(obs, False))
                    terms = np.array([[i[k] for k in ['reward_primary', 'tau', 'robustness',
                                                      'disconnection_ratio']] for i in infos])
                    actual = np.array([i['reward'] for i in infos])
                    np.testing.assert_allclose((terms * weights).sum(-1), actual, rtol=1e-12, atol=1e-12)
                    assert np.all(done == (t == cfg.env.horizon - 1)) and not truncated.any()
                    raw.append(terms)
                    diagnostics.append(np.array([[i[k] for k in DIAGNOSTICS] for i in infos]))
                all_values.append(np.stack(values).T)
                all_raw.append(np.stack(raw).transpose(1, 0, 2))
                all_diagnostics.append(np.stack(diagnostics).transpose(1, 0, 2))
    values = np.concatenate(all_values)
    raw = np.concatenate(all_raw)
    contributions = raw * weights
    rewards = contributions.sum(-1)
    returns = discounted_episode_returns(rewards, cfg.algo.discount)
    components = np.stack([discounted_episode_returns(contributions[..., i], cfg.algo.discount)
                           for i in range(4)], -1)
    np.testing.assert_allclose(components.sum(-1), returns, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(returns[:, -1], rewards[:, -1])
    np.testing.assert_allclose(returns[:, :-1], rewards[:, :-1] + cfg.algo.discount * returns[:, 1:])
    h = horizon_factors(np.arange(cfg.env.horizon, 0, -1), cfg.algo.discount)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite critic predictions')
    np.savez_compressed(output, values=values, returns=returns, rewards=rewards,
                        average_predictions=values / h, average_labels=returns / h,
                        horizon_factors=h, raw=raw, contributions=contributions,
                        component_returns=components, diagnostics=np.concatenate(all_diagnostics),
                        diagnostic_names=np.array(DIAGNOSTICS),
                        contribution_names=np.array(['coverage', 'weighted_throughput',
                                                     'weighted_robustness', 'signed_disconnection']),
                        scene_seeds=cfg.run.validation_seeds, state_times=np.arange(cfg.env.horizon),
                        action_seed=cfg.run.evaluation_action_seed)
    return cfg


def critic_scores(values, returns, h):
    result = {}
    for name, pred, label in [('value', values, returns),
                              ('average_reward', values / h, returns / h)]:
        metrics = error_metrics(pred, label)
        for stage, item in metrics['stages'].items():
            std = float(label[:, STAGES[stage]].std(ddof=0))
            item['label_std'] = std
            item['normalized_rmse'] = item['rmse'] / std if std > 0 else None
        result[name] = metrics
    result['average_prediction_outside_unit_interval'] = float(((values / h < 0) | (values / h > 1)).mean())
    result['average_label_outside_unit_interval'] = float(((returns / h < 0) | (returns / h > 1)).mean())
    return result


def policy_scores(data):
    d = data['diagnostics']
    names = list(data['diagnostic_names'])
    result = {}
    for stage in ['all', 'last100', 'last1']:
        sl = STAGES[stage]
        metric = {k: float(d[:, sl, names.index(k)].mean()) for k in names}
        cov = d[:, sl, names.index('coverage')]
        rob = d[:, sl, names.index('robustness')]
        metric.update(mean_reward=float(data['rewards'][:, sl].mean()),
                      retention_ratio=float(rob.sum() / cov.sum()) if cov.sum() else None,
                      zero_coverage_episodes=int((cov.sum(1) == 0).sum()))
        for i, key in enumerate(data['contribution_names']):
            metric[str(key) + '_contribution'] = float(data['contributions'][:, sl, i].mean())
        result[stage] = metric
    return result


def plot(out, summary):
    style()
    figs = out / 'figures'
    figs.mkdir(exist_ok=True)
    for xkey, name, xlabel in [('env_steps', 'training_comparison', 'Joint transitions (k)'),
                               ('elapsed_seconds', 'wall_time_comparison', 'Wall time (min)')]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for mode, run in RUNS.items():
            rows = read_csv(run / 'validation.csv')
            x = np.array([float(r[xkey]) for r in rows]) / (1000 if xkey == 'env_steps' else 60)
            for ax, field in zip(axes.flat, ['return', 'coverage', 'throughput_bps', 'robustness',
                                            'disconnection_ratio', 'discovery_fraction']):
                ax.plot(x, [float(r[field]) / (1e6 if field == 'throughput_bps' else 1) for r in rows], label=mode)
        for ax, title in zip(axes.flat, ['Full episode return', 'Service coverage', 'Throughput (Mbps)',
                                        'Fault-retained coverage', 'Disconnected fraction', 'Sensing fraction']):
            ax.set(xlabel=xlabel, title=title)
            ax.legend(fontsize=7)
        save_figure(fig, figs, name)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = ['actor_loss', 'value_loss', 'categorical_entropy', 'ppo_approx_kl',
              'ppo_clip_fraction', 'max_action_probability']
    for mode, run in RUNS.items():
        rows = list({r['env_steps']: r for r in read_csv(run / 'metrics.csv')}.values())
        for ax, field in zip(axes.flat, fields):
            ax.plot([int(r['env_steps']) / 1000 for r in rows], [float(r[field]) for r in rows], label=mode)
    for ax, field in zip(axes.flat, fields):
        ax.set(title=field, xlabel='Joint transitions (k)'); ax.legend(fontsize=7)
    save_figure(fig, figs, 'loss_and_policy_diagnostics')
    rows = list({r['env_steps']: r for r in read_csv(RUNS['time_factor'] / 'metrics.csv')}.values())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    groups = [['critic_value_clip_fraction', 'critic_clipped_loss_fraction'],
              ['critic_average_prediction_mean', 'critic_average_target_mean'],
              ['critic_average_prediction_outside', 'critic_average_target_outside'],
              ['critic_return_normalizer_mean', 'critic_return_normalizer_std'],
              ['critic_average_normalizer_mean', 'critic_average_normalizer_std'],
              ['value_return_mean', 'q_grad_norm']]
    for ax, keys in zip(axes.flat, groups):
        for key in keys:
            ax.plot([int(r['env_steps']) / 1000 for r in rows], [float(r[key]) for r in rows], label=key)
        ax.set(xlabel='Joint transitions (k)'); ax.legend(fontsize=6)
    axes[0, 1].set_title('Discounted average TASK reward')
    axes[0, 2].set_title('Outside [0,1]: distribution, not invalidity')
    save_figure(fig, figs, 'online_critic_diagnostics')
    for ck in ['best', 'last']:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for mode in RUNS:
            with np.load(out / f'{mode}_{ck}_critic.npz') as data:
                v, g, h = data['values'], data['returns'], data['horizon_factors']
                axes[0, 0].plot(v.mean(0), label=mode + ' V')
                axes[0, 0].plot(g.mean(0), '--', label=mode + ' MC')
                axes[0, 1].plot(v.mean(0), label=mode + ' V')
                axes[0, 1].plot(g.mean(0), '--', label=mode + ' MC')
                axes[0, 2].plot((v / h).mean(0), label=mode + ' V/H')
                axes[0, 2].plot((g / h).mean(0), '--', label=mode + ' G/H')
                axes[1, 0].plot(np.sqrt(((v - g)**2).mean(0)), label=mode)
                axes[1, 1].plot((v - g).mean(0), label=mode)
                axes[1, 2].scatter((g / h).ravel()[::4], (v / h).ravel()[::4], s=2, alpha=.15, label=mode)
        for ax, title in zip(axes.flat, ['Cumulative task value', 'Last 20 pre-action states',
                                        'Discounted average task reward', 'Value RMSE', 'Value bias',
                                        'Average reward: MC vs prediction']):
            ax.set(title=title, xlabel='State time before action'); ax.legend(fontsize=6)
        axes[0, 1].set_xlim(380, 399)
        axes[1, 2].set(xlabel='MC G/H', ylabel='Predicted V/H')
        save_figure(fig, figs, f'critic_horizon_{ck}')
        # The direct critic's terminal errors can dwarf the reward scale. Show
        # the factor model separately as well, without imposing a [0,1] range.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        with np.load(out / f'time_factor_{ck}_critic.npz') as data:
            c, target = data['average_predictions'], data['average_labels']
            axes[0].plot(c.mean(0), label='Predicted discounted average task reward')
            axes[0].plot(target.mean(0), label='MC discounted average task reward')
            axes[0].set(xlabel='State time before action', ylabel='Average task reward', title=f'time_factor / {ck}')
            axes[1].scatter(target.ravel()[::4], c.ravel()[::4], s=3, alpha=.15)
            low = min(float(c.min()), float(target.min()))
            high = max(float(c.max()), float(target.max()))
            axes[1].plot([low, high], [low, high], '--', color='gray')
            axes[1].set(xlabel='MC G/H', ylabel='Prediction V/H', title='Every fourth state, fixed order')
            axes[0].legend(fontsize=7)
        save_figure(fig, figs, f'average_reward_detail_{ck}')
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fields = ['coverage', 'discovery_fraction', 'disconnection_ratio', 'hover_fraction',
                  'uav_nearest_neighbor_mean', 'uav_close_neighbor_fraction']
        for mode in RUNS:
            with np.load(out / f'{mode}_{ck}_critic.npz') as data:
                names = list(data['diagnostic_names'])
                for ax, field in zip(axes.flat, fields):
                    ax.plot(np.arange(1, 401), data['diagnostics'][..., names.index(field)].mean(0), label=mode)
        for ax, field in zip(axes.flat, fields):
            ax.set(title=field, xlabel='Episode step after action'); ax.legend(fontsize=7)
        save_figure(fig, figs, f'deployment_maintenance_{ck}')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, mode in zip(axes, RUNS):
            with np.load(out / f'{mode}_{ck}_critic.npz') as data:
                for i, field in enumerate(data['contribution_names']):
                    ax.plot(data['contributions'][..., i].mean(0), label=str(field))
                ax.plot(data['rewards'].mean(0), color='black', label='Total task reward')
            ax.set(title=mode, xlabel='Episode step after action', ylabel='Actual reward contribution')
            ax.legend(fontsize=7)
        save_figure(fig, figs, f'reward_components_{ck}')


def publish(out, summary):
    plot(out, summary)
    lines = ['# 完整奖励 MAPPO：direct 与时间因子 critic', '',
             '两组从头训练种子0、250000联合转移、123轮PPO；配置仅critic参数化不同。'
             '奖励为覆盖+0.1吞吐+0.5故障保留覆盖−0.1断联率。'
             '保留GAE、Value Clipping及原C物理场景。best按全回合完整奖励选模，主要比较last。', '',
             '## 策略结果', '',
             '| 方法 | 模型 | 步数 | 平均奖励 | 覆盖 | 后100步覆盖 | 终态覆盖 | 故障保留覆盖 | 保留比例q | 断联率 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in RUNS:
        for ck in ['best', 'last']:
            item = summary['policies'][f'{mode}_{ck}']; a, tail, end = [item['stages'][k] for k in ['all', 'last100', 'last1']]
            q = f"{a['retention_ratio']:.2%}" if a['retention_ratio'] is not None else '不适用'
            lines.append(f"| {mode} | {ck} | {item['step']} | {a['mean_reward']:.4f} | {a['coverage']:.2%} | {tail['coverage']:.2%} | {end['coverage']:.2%} | {a['robustness']:.2%} | {q} | {a['disconnection_ratio']:.2%} |")
    lines += ['', '| 方法 | 模型 | 覆盖贡献 | 吞吐贡献 | 鲁棒性贡献 | 断联贡献 |',
              '|---|---|---:|---:|---:|---:|']
    for mode in RUNS:
        for ck in ['best', 'last']:
            a = summary['policies'][f'{mode}_{ck}']['stages']['all']
            keys = ['coverage_contribution', 'weighted_throughput_contribution',
                    'weighted_robustness_contribution', 'signed_disconnection_contribution']
            lines.append(f'| {mode} | {ck} | ' + ' | '.join(f'{a[k]:.5f}' for k in keys) + ' |')
    lines += ['', '以上贡献为每步均值，带符号相加等于平均任务奖励。'
              '各阶段全部服务、吞吐、感知、移动、聚集、奖励贡献和零覆盖计数见policy_metrics.csv。'
              'q按本阶段全部场景的J总和除以覆盖总和计算；零覆盖标记单独保存。', '',
              '![训练验证](figures/training_comparison.png)', '', '![墙钟](figures/wall_time_comparison.png)', '',
              '![Loss与策略](figures/loss_and_policy_diagnostics.png)', '',
              '![在线critic](figures/online_critic_diagnostics.png)', '',
              '## Critic MC诊断', '',
              '| 方法 | 模型 | 累计RMSE | 累计NRMSE | 平均奖励RMSE | 末20步RMSE | 末步V | 末步MC | 去时间均值相关性 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, item in summary['critics'].items():
        mode, ck = key.rsplit('_', 1); v = item['value']['stages']; corr = item['value']['time_centered_correlation']
        corrtext = '未定义' if corr is None else f'{corr:.3f}'
        nrmse = v['all']['normalized_rmse']; nt = '不适用' if nrmse is None else f'{nrmse:.3f}'
        lines.append(f"| {mode} | {ck} | {v['all']['rmse']:.3f} | {nt} | {item['average_reward']['stages']['all']['rmse']:.4f} | {v['last20']['rmse']:.3f} | {v['last1']['prediction_mean']:.3f} | {v['last1']['label_mean']:.3f} | {corrtext} |")
    lines += ['', 'MC取完整环境奖励，终止后为0，不包含actor熵正则。'
              '保存的四项折扣分量之和已核对等于总MC回报。各网络使用自身on-policy轨迹，'
              '不是同状态网络比较。MC含采样噪声。NRMSE为RMSE除以对应阶段MC标签总体标准差；'
              '该标准差仅用于诊断，不参与训练。完整奖励及其平均值可以小于0或大于1，'
              '超出[0,1]不定义为价值范围违规。', '',
              '![Best预测](figures/critic_horizon_best.png)', '', '![Last预测](figures/critic_horizon_last.png)', '',
              '![平均任务奖励预测放大](figures/average_reward_detail_last.png)', '',
              '![Best奖励](figures/reward_components_best.png)', '', '![Last奖励](figures/reward_components_last.png)', '',
              '![部署维持](figures/deployment_maintenance_last.png)', '',
              '## 配对差值', '',
              'time_factor减direct，2000次场景bootstrap、seed950001。仅一个训练种子，区间不是训练种子置信区间。'
              '这些场景也用于best选模，best区间包含选模影响，不作为保留测试集的无偏估计。'
              '奖励、覆盖、吞吐、鲁棒性正值有利；断联率和误差负值有利。', '',
              '| 模型 | 指标 | 差值（原单位） | 95%场景区间 |', '|---|---|---:|---:|']
    for r in summary['paired']:
        lines.append(f"| {r['checkpoint']} | {r['metric']} | {r['delta']:+.5f} | [{r['low']:+.5f}, {r['high']:+.5f}] |")
    direct = summary['policies']['direct_last']['stages']; factor = summary['policies']['time_factor_last']['stages']
    lines += ['', '## 结果解读', '',
              f"last全程覆盖变化{100*(factor['all']['coverage']-direct['all']['coverage']):+.2f}个百分点，"
              f"平均任务奖励变化{factor['all']['mean_reward']-direct['all']['mean_reward']:+.4f}。"
              f"后100步覆盖变化{100*(factor['last100']['coverage']-direct['last100']['coverage']):+.2f}个百分点，"
              f"终态变化{100*(factor['last1']['coverage']-direct['last1']['coverage']):+.2f}个百分点。", '',
              '| last指标 | direct | time_factor |', '|---|---:|---:|']
    for field in ['throughput_bps', 'robustness', 'discovery_fraction', 'hover_fraction', 'actual_speed_mean',
                  'sensing_rejected_users', 'downlink_rejected_users', 'backhaul_rejected_users', 'capacity_rejected_users']:
        lines.append(f"| {field} | {direct['all'][field]:.5f} | {factor['all'][field]:.5f} |")
    for field in ['uav_nearest_neighbor_mean', 'uav_close_neighbor_fraction', 'boundary_reflection_fraction']:
        lines.append(f"| 终态 {field} | {direct['last1'][field]:.5f} | {factor['last1'][field]:.5f} |")
    for mode in RUNS:
        for ck in ['best', 'last']:
            with np.load(out / f'{mode}_{ck}_critic.npz') as d:
                coverage = d['diagnostics'][..., 0].mean(0)
            lines += ['', f"{mode} {ck} 四个100步阶段覆盖：" + ' → '.join(f'{coverage[i:i+100].mean():.2%}' for i in range(0, 400, 100)) + '。']
    old_error = summary['critics']['direct_last']['value']['stages']['all']['rmse']
    new_error = summary['critics']['time_factor_last']['value']['stages']['all']['rmse']
    coverage_interval = next(r for r in summary['paired'] if r['checkpoint'] == 'last' and r['metric'] == 'coverage')
    coverage_evidence = ('本次32验证场景的覆盖差区间全部大于零。' if coverage_interval['low'] > 0 else
                         '本次32验证场景的覆盖差区间全部小于零。' if coverage_interval['high'] < 0 else
                         '本次32验证场景的覆盖差区间包含零，不能据此确认稳定的覆盖增益。')
    lines += ['', f'last累计价值RMSE从{old_error:.3f}变为{new_error:.3f}。' + coverage_evidence]
    for mode in RUNS:
        best = summary['policies'][f'{mode}_best']['stages']['all']
        last = summary['policies'][f'{mode}_last']['stages']['all']
        lines += ['', f"{mode}从best到last：平均任务奖励变化{last['mean_reward']-best['mean_reward']:+.4f}，"
                  f"覆盖变化{100*(last['coverage']-best['coverage']):+.2f}个百分点。"]
    old_score = summary['critics']['direct_last']
    new_score = summary['critics']['time_factor_last']
    peak_old = summary['policies']['direct_best']['stages']['all']
    peak_new = summary['policies']['time_factor_best']['stages']['all']
    lines += ['', '**价值校准改善主要发生在中后期。** '
              f"累计NRMSE从{old_score['value']['stages']['all']['normalized_rmse']:.3f}降至"
              f"{new_score['value']['stages']['all']['normalized_rmse']:.3f}，支持误差改善不只是回报尺度变化。"
              f"但前100步RMSE仍为{new_score['value']['stages']['first100']['rmse']:.2f}"
              f"（direct为{old_score['value']['stages']['first100']['rmse']:.2f}），不能称为所有时段均已准确。"
              '末端巨大高估修正和全程条件预测能力需要分开判断。', '',
              '**本次优势更接近保留训练后期表现，而不是突破历史峰值。** '
              f"两组best覆盖仅相差{100*(peak_new['coverage']-peak_old['coverage']):+.2f}个百分点；"
              '对应场景区间见上表。时间因子best恰为last，说明本轮没有出现direct那样的best到last回退，'
              '但一个种子和一个终点不能证明后续训练必然稳定，也不能预测延长预算的收益。', '',
              '**故障保留覆盖提高，不等于条件故障保留比例提高。** '
              f"last的J从{direct['all']['robustness']:.2%}变为{factor['all']['robustness']:.2%}，"
              f"q却从{direct['all']['retention_ratio']:.2%}变为{factor['all']['retention_ratio']:.2%}。"
              '本轮绝对故障服务增益主要伴随正常覆盖增长，未显示保留比例改善。', '',
              '**服务增益主要来自回传资格，尚未形成更大的感知覆盖。** '
              f"回传淘汰减少{direct['all']['backhaul_rejected_users']-factor['all']['backhaul_rejected_users']:.2f}人／步，"
              f"但感知淘汰增加{factor['all']['sensing_rejected_users']-direct['all']['sensing_rejected_users']:.2f}人／步，"
              f"下行淘汰增加{factor['all']['downlink_rejected_users']-direct['all']['downlink_rejected_users']:.2f}人／步。"
              '容量淘汰为0。说明新策略在本次数据中让更多已发现用户获得联网服务，不能据此声称网络探索范围扩大。', '',
              '**回合内部的部署维持仍未解决。** '
              f"时间因子组平均实际速度{factor['all']['actual_speed_mean']:.2f}m/s，"
              f"悬停率{factor['all']['hover_fraction']:.2%}，仍持续移动。"
              '其四个100步阶段的覆盖仍逐段下降。最近邻距离与100m聚集比例分别反映整体间距和极近距离聚集，'
              '两者不能互相替代；固定拓扑仍需结合空间分布判断。']
    for seed in [10000, 10001]:
        path = out / 'time_factor' / 'validation_last' / f'trace_{seed}.npz'
        with np.load(path) as trace:
            cov = trace['coverage'][[0, 200, 400]]
            sensed = trace['discovery_fraction'][[0, 200, 400]]
            # B stores full per-BS hop counts; a UAV is reachable iff one is <=U.
            uavs = trace['uav_xy'].shape[1]
            reachable = (trace['B'][[0, 200, 400]].min(-1) <= uavs).sum(-1)
        lines += ['', f"固定场景{seed}：初始／200步／终态覆盖" + '／'.join(f'{v:.1%}' for v in cov)
                  + '，感知' + '／'.join(f'{v:.1%}' for v in sensed)
                  + '，联网UAV ' + '／'.join(str(int(v)) for v in reachable) + '。']
    lines += ['', '场景10000终态仍见左右两片密集UAV及局部贴边，上方中部存在较大未感知区域；'
              '场景10001右侧形成多跳链路，但左侧部署稀少，终态覆盖仍低于初始。'
              '因此“能形成部分多跳拓扑”与“能持续铺开、均衡服务整个场景”是两个不同的结论。']
    lines += ['', '总奖励收益须结合各分量解释；critic误差下降不等于已经学会展开。'
              'best与last分别报告，不因last退化追加训练或更换选模标准。', '',
              '## 固定拓扑与运行来源', '', '固定场景10000、10001；每图含初始、200步和终态。']
    for ck in ['best', 'last']:
        for seed in [10000, 10001]:
            for mode in RUNS:
                lines += ['', f'### {mode} / {ck} / 场景{seed}', '',
                          f'![拓扑]({mode}/figures/topology_{ck}_{seed}.png)']
    lines += ['', '| 方法 | 训练耗时min | 峰值RSS MiB |', '|---|---:|---:|']
    for mode, m in summary['manifests'].items():
        lines.append(f"| {mode} | {m['elapsed_seconds']/60:.2f} | {m['peak_rss_mib']:.2f} |")
    lines += ['', '两组初始actor哈希、训练前验证和短流程见acceptance.json；'
              '配置、模型、日志哈希见comparison.json；逐状态预测、MC及四项分量见*_critic.npz。', '',
              '重建图表：`.venv/bin/python scripts/compare_full_reward_time_factor.py --report-only`。', '',
              '[实验说明](../../docs/full_reward_time_factor_experiment.md)。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def build(out, report_only=False):
    if report_only:
        publish(out, json.loads((out / 'comparison.json').read_text()))
        return
    out.mkdir(parents=True, exist_ok=True)
    configs = {mode: load_config(run / 'config.yaml') for mode, run in RUNS.items()}
    change = configs['time_factor'].as_dict(); change['algo']['critic_parameterization'] = 'direct'
    assert change == configs['direct'].as_dict()
    assert [configs['direct'].env.throughput_weight, configs['direct'].env.robustness_weight,
            configs['direct'].env.disconnection_weight] == [.1, .5, .1]
    source = [p / n for p in RUNS.values() for n in ['config.yaml', 'best.pt', 'last.pt',
                                                    'metrics.csv', 'validation.csv', 'manifest.json']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    reports = {mode: json.loads((out / mode / 'report.json').read_text()) for mode in RUNS}
    critics, policies, paired, critic_rows, policy_rows, manifests = {}, {}, [], [], [], {}
    initial = []
    for mode, run in RUNS.items():
        m = json.loads((run / 'manifest.json').read_text()); manifests[mode] = m
        assert m['status'] == 'completed' and m['env_steps'] == 250000 and m['updates'] == 123
        first = read_csv(run / 'validation.csv')[0]; initial.append(first)
        np.testing.assert_allclose(float(first['coverage']), .3482265625, atol=1e-12, rtol=0)
        for ck in ['best', 'last']:
            entry = reports[mode]['evaluations'][ck]
            assert hashes[str(run / f'{ck}.pt')] == entry['checkpoint_sha256']
            path = out / f'{mode}_{ck}_critic.npz'
            audit_full_reward(run, ck, path)
            with np.load(path) as data:
                with np.load(out / mode / f'validation_{ck}' / 'reward_curves.npz') as reference:
                    np.testing.assert_allclose(data['rewards'], reference['reward'], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(data['contributions'], reference['contributions'])
                    np.testing.assert_array_equal(data['scene_seeds'], reference['scene_seeds'])
                np.testing.assert_allclose(data['rewards'].sum(1).mean(), entry['summary']['mean']['return'], rtol=1e-12)
                metrics = critic_scores(data['values'], data['returns'], data['horizon_factors'])
                stages = policy_scores(data)
            critics[f'{mode}_{ck}'] = metrics
            policies[f'{mode}_{ck}'] = dict(step=entry['checkpoint_env_steps'], stages=stages)
            for scale in ['value', 'average_reward']:
                for stage, row in metrics[scale]['stages'].items():
                    critic_rows.append(dict(method=mode, checkpoint=ck, scale=scale, stage=stage, **row))
            for stage, row in stages.items():
                policy_rows.append(dict(method=mode, checkpoint=ck, stage=stage, **row))
            print(mode, ck, 'reward', stages['all']['mean_reward'], 'coverage', stages['all']['coverage'],
                  'critic RMSE', metrics['value']['stages']['all']['rmse'], flush=True)
    for key in ['return', 'coverage', 'throughput_bps', 'robustness', 'disconnection_ratio']:
        assert initial[0][key] == initial[1][key]
    for ck in ['best', 'last']:
        with np.load(out / f'direct_{ck}_critic.npz') as a, np.load(out / f'time_factor_{ck}_critic.npz') as b:
            assert np.array_equal(a['scene_seeds'], b['scene_seeds'])
            names = list(a['diagnostic_names'])
            arrays = {'mean_reward': (a['rewards'].mean(1), b['rewards'].mean(1))}
            for key in ['coverage', 'robustness', 'disconnection_ratio', 'throughput_bps']:
                arrays[key] = tuple(d['diagnostics'][..., names.index(key)].mean(1) for d in [a, b])
            for key, sl in [('last100_coverage', slice(-100, None)), ('final_coverage', slice(-1, None))]:
                arrays[key] = tuple(d['diagnostics'][:, sl, 0].mean(1) for d in [a, b])
            for key, (left, right) in arrays.items():
                paired.append(dict(checkpoint=ck, metric=key, **bootstrap_difference(left, right, key)))
            paired.append(dict(checkpoint=ck, metric='value_rmse',
                               **bootstrap_difference(a['values'] - a['returns'], b['values'] - b['returns'], 'rmse')))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in source)
    summary = dict(critics=critics, policies=policies, paired=paired, manifests=manifests,
                   source_sha256=hashes, config_difference={'algo.critic_parameterization': ['direct', 'time_factor']},
                   validation_scenes=32, evaluation_action_seed=880001, training_seeds=[0],
                   output_semantics='cumulative_task_return_and_discounted_average_task_reward')
    atomic_json(out / 'comparison.json', summary)
    write_csv(out / 'critic_metrics.csv', critic_rows)
    write_csv(out / 'policy_metrics.csv', policy_rows)
    write_csv(out / 'paired_differences.csv', paired)
    publish(out, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    build(OUT, args.report_only)
