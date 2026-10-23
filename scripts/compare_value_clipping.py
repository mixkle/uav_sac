"""Read-only comparison of C and its value-clipping-only ablation.

Run report_mappo_training.py for each run first. No fitting or training occurs here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import torch

from uav_research.config import load_config
from uav_research.critic_calibration import discounted_episode_returns
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import plt, style, save_figure, read_csv
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json
from uav_research.vector import VectorEnvironment


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out, summaries, audits, rows, paired):
    lines = [
        '# 纯覆盖 MAPPO：仅关闭 Critic Value Clipping',
        '',
        '训练种子0，移动UE，奖励仅为联网服务覆盖率。两组均完成250000条联合转移、123轮PPO；'
        '配置仅 `algo.value_clipping` 不同，训练前验证覆盖率均为34.8227%。'
        '基准复用历史C组，对照从头训练，不继承权重或Adam。',
        '',
        '## 验证覆盖率',
        '',
        '同一组32个验证场景、动作采样种子880001、16环境批次；未使用测试集。',
        '',
        '| 模型 | 步数 | 全回合覆盖 | 最后100步覆盖 | 终态覆盖 | 平均断联率 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['method']} {row['checkpoint']} | {row['step']} | "
                     + ' | '.join(f"{100*row[k]:.2f}%" for k in
                                  ['coverage','last100_coverage','final_coverage','disconnection_ratio'])+' |')
    lines.extend(['', '主要比较同预算的last，best作为辅助。best按全回合覆盖选择，不按终态选择。', ''])
    for name in ['last','best']:
        delta=100*(summaries[f'unclipped_{name}']['coverage']-summaries[f'clipped_{name}']['coverage'])
        wins=sum(row['coverage']>0 for row in paired if row['checkpoint']==name)
        lines.append(f'- {name}：关闭clipping的全回合覆盖差为 **{delta:+.2f}个百分点**；32个配对场景中{wins}个改善。')
    lines.extend([
        '', '![训练对照](figures/training_comparison.png)',
        '', '![回合内覆盖变化](figures/coverage_horizon.png)',
        '', '## Critic 原始尺度校准',
        '', '冻结各自actor与critic，完整采集400步回合，再用终止回报0反向计算实际折扣回报。'
        '下表的末步是执行最后一次动作之前的状态t=399，不是终止状态t=400。',
        '',
        '| 模型 | 全程RMSE | 最后20步RMSE | 末步预测V | 末步实际回报 |',
        '|---|---:|---:|---:|---:|',
    ])
    for key, stages in audits.items():
        lines.append(f"| {key} | {stages['all']['rmse']:.3f} | {stages['last20']['rmse']:.3f} | "
                     f"{stages['last1']['prediction_mean']:.3f} | {stages['last1']['return_mean']:.3f} |")
    lines.extend([
        '', '![Last价值校准](figures/critic_horizon_last.png)',
        '', '![Best价值校准](figures/critic_horizon_best.png)',
        '', '末步真实价值上界为1。当前网络输出无显式有限时域约束；去掉Value Clipping没有同时增加时间编码、'
        '改变GAE目标或采用完整回报监督。这些指标可判断末端校准是否改善，但不能用两种定义不同的训练loss直接比较拟合质量。',
        '', 'Monte Carlo回报具有采样噪声；这里是各自策略、各自访问状态上的on-policy误差，'
        '不是把两个critic放在同一状态分布上的比较，也不是精确条件期望。'
        '实际奖励与覆盖逐步一致，诊断回合平均值与正式验证结果一致。',
        '', '## 固定场景拓扑',
        '',
        '固定展示验证场景10000（10001同样保存在各组figures），没有按表现筛选。',
        '',
        '原C组best：[拓扑](../ue_reward_ablations_250k_seed0/c_mobile_coverage/figures/topology_best_10000.png)。',
        '',
        '![无价值裁剪best拓扑](no_value_clip/figures/topology_best_10000.png)',
        '',
        '原C组last：[拓扑](../ue_reward_ablations_250k_seed0/c_mobile_coverage/figures/topology_last_10000.png)。',
        '',
        '![无价值裁剪last拓扑](no_value_clip/figures/topology_last_10000.png)',
        '', '## 结论边界与产物',
        '',
        '本次单种子对照不支持“只去掉Value Clipping就能改善覆盖”的假设。'
        '它检验的是完整在线MAPPO训练中这一开关的作用；不能据此断言critic没有问题，'
        '也不能把结果直接归因于某个尚未独立检验的优化或观察因素。原C组继续作为基准。',
        '',
        '关闭裁剪后全程RMSE和末端高估降低，但前100步的低估更明显，价值预测仍未呈现正确的有限回合衰减。'
        '全程平均偏差接近零可能来自前期低估与后期高估相互抵消。'
        '同时新策略覆盖下降、真实回报尺度也下降，因此RMSE下降不能单独解释为critic结构性问题已修复。',
        '',
        '后续若继续定位，建议在冻结的同一actor、同一批状态与完整回报目标上比较时间特征利用或有限时域价值参数化，'
        '避免把策略变化、状态分布变化与critic拟合能力混在一起。本次没有自动启动这些新实验。',
        '',
        '- `summary.csv`：四个检查点的服务、发现、断联、间距与聚集指标。',
        '- `paired_differences.csv`：每个场景的无裁剪减基准覆盖差。',
        '- `critic_errors.csv`：按回合阶段统计的预测误差。',
        '- `*_critic.npz`：完整逐时预测、折扣回报、奖励与种子。',
        '- `comparison.json`：配置差异、汇总、源文件SHA256；源运行保持不变。',
        '- `no_value_clip/report.md`：新组的独立验证、loss、奖励、拓扑报告。',
        '',
        '复现方法见 [实验说明](../../docs/critic_value_clipping_ablation.md)。',
    ])
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def audit(run, checkpoint, output, expected):
    configure_runtime(load_config(run / 'config.yaml'))
    cfg, policy = policy_from_checkpoint(run / f'{checkpoint}.pt', 'cpu')
    configure_runtime(cfg)
    values, rewards = [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start + cfg.run.num_envs]
            with VectorEnvironment(cfg.env, len(seeds), 'batch') as env:
                obs, state, _ = env.reset(seeds)
                vs, rs = [], []
                for t in range(cfg.env.horizon):
                    vs.append(policy.values(state).cpu().numpy())
                    obs, state, _, done, truncated, infos = env.step(policy.act(obs, False))
                    reward = np.array([i['reward'] for i in infos])
                    np.testing.assert_array_equal(reward, [i['coverage'] for i in infos])
                    assert np.all(done == (t == cfg.env.horizon - 1)) and not truncated.any()
                    rs.append(reward)
                values.append(np.stack(vs).T)
                rewards.append(np.stack(rs).T)
    v, r = np.concatenate(values), np.concatenate(rewards)
    g = discounted_episode_returns(r, cfg.algo.discount)
    np.testing.assert_allclose(r.mean(), expected['coverage'], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(g[:, -1], r[:, -1])
    np.testing.assert_allclose(g[:, :-1], r[:, :-1] + cfg.algo.discount*g[:, 1:])
    np.savez_compressed(output, values=v, returns=g, rewards=r, scene_seeds=cfg.run.validation_seeds,
                        action_seed=cfg.run.evaluation_action_seed)
    stages = {}
    for name, sl in [('all', slice(None)), ('first100', slice(0,100)),
                     ('middle100', slice(150,250)), ('last100', slice(-100,None)),
                     ('last20', slice(-20,None)), ('last1', slice(-1,None))]:
        error = v[:,sl] - g[:,sl]
        stages[name] = dict(prediction_mean=float(v[:,sl].mean()), return_mean=float(g[:,sl].mean()),
                            bias=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                            mae=float(np.abs(error).mean()))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/mappo_ablation_c_mobile_coverage_250k_seed0'))
    parser.add_argument('--variant', type=Path, default=Path('runs/mappo_critic_no_value_clip_250k_seed0'))
    parser.add_argument('--baseline-report', type=Path, default=Path('analysis/ue_reward_ablations_250k_seed0/c_mobile_coverage'))
    parser.add_argument('--output', type=Path, default=Path('analysis/critic_value_clipping_250k_seed0'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    figs = out/'figures'
    figs.mkdir(exist_ok=True)
    runs = {'clipped': args.baseline, 'unclipped': args.variant}
    reports = {'clipped': args.baseline_report, 'unclipped': out/'no_value_clip'}
    configs = {k: load_config(v/'config.yaml') for k,v in runs.items()}
    assert configs['clipped'].algo.value_clipping is True
    assert configs['unclipped'].algo.value_clipping is False
    variant = configs['unclipped'].as_dict()
    variant['algo']['value_clipping'] = True
    assert variant == configs['clipped'].as_dict(), 'Additional experimental differences'
    assert configs['clipped'].env.reward_primary == 'coverage'
    assert all(getattr(configs['clipped'].env,k)==0 for k in ('throughput_weight','robustness_weight','disconnection_weight'))
    cfg = configs['clipped']
    assert not set(cfg.run.validation_seeds) & set(cfg.run.test_seeds)
    source_files = [p/n for p in runs.values() for n in ['best.pt','last.pt','config.yaml','metrics.csv','validation.csv']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    summaries, audits, rows = {}, {}, []
    for label, run in runs.items():
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['status']=='completed' and manifest['env_steps']==250000
        report = json.loads((reports[label]/'report.json').read_text())
        for name in ['best','last']:
            entry = report['evaluations'][name]
            assert entry['checkpoint_sha256']==hashes[str(run/f'{name}.pt')]
            mean = entry['summary']['mean']
            key = label+'_'+name
            summaries[key] = mean
            rows.append(dict(method=label,checkpoint=name,step=entry['checkpoint_env_steps'],
                             **{k:mean[k] for k in ['coverage','last100_coverage','final_coverage',
                                  'discovery_fraction','disconnection_ratio','hover_fraction',
                                  'uav_nearest_neighbor_mean','uav_close_neighbor_fraction']}))
            audits[key] = audit(run,name,out/f'{key}_critic.npz',mean)
            print(key, audits[key]['last1'], flush=True)
    write_csv(out/'summary.csv', rows)
    error_rows = [dict(method=k, stage=s, **v) for k, stages in audits.items() for s,v in stages.items()]
    write_csv(out/'critic_errors.csv', error_rows)
    paired = []
    for name in ['best','last']:
        episodes = {k: read_csv(reports[k]/f'validation_{name}'/'episodes.csv') for k in runs}
        for left, right in zip(episodes['clipped'], episodes['unclipped']):
            assert left['scene_seed']==right['scene_seed']
            paired.append(dict(checkpoint=name, scene_seed=left['scene_seed'],
                               **{k:float(right[k])-float(left[k]) for k in ['coverage','last100_coverage','final_coverage']}))
    write_csv(out/'paired_differences.csv',paired)
    style()
    fig, axes = plt.subplots(1,3,figsize=(14,3.8))
    validations = {}
    for label,run in runs.items():
        val = read_csv(run/'validation.csv'); validations[label]=val
        axes[0].plot([float(x['env_steps']) for x in val], [float(x['coverage']) for x in val], label=label)
        train = list({int(x['env_steps']):x for x in read_csv(run/'metrics.csv')}.values())
        for ax, key in zip(axes[1:], ['value_loss','actor_loss']):
            ax.plot([float(x['env_steps']) for x in train], [float(x[key]) for x in train],label=label,alpha=.8)
    assert validations['clipped'][0]['coverage']==validations['unclipped'][0]['coverage']
    for ax,title in zip(axes,['Validation mean coverage','Value loss (different objectives)','Actor loss including entropy']):
        ax.set(xlabel='Joint transitions',title=title); ax.legend()
    save_figure(fig,figs,'training_comparison')
    for name in ['best','last']:
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        for ax,label in zip(axes,runs):
            data=np.load(out/f'{label}_{name}_critic.npz')
            ax.plot(data['values'].mean(0),label='Predicted V(s)')
            ax.plot(data['returns'].mean(0),label='Sampled discounted return')
            ax.set(title=f'{label} / {name}',xlabel='Time before action',ylabel='Raw return units'); ax.legend()
        save_figure(fig,figs,f'critic_horizon_{name}')
    fig, axes = plt.subplots(1, 2, figsize=(10,3.7))
    for ax, name in zip(axes, ['best','last']):
        for label in runs:
            data=np.load(out/f'{label}_{name}_critic.npz')
            ax.plot(np.arange(1,401),data['rewards'].mean(0),label=label)
        ax.set(title=f'{name}: mean coverage over 32 scenes',xlabel='Step after action',ylabel='Coverage')
        ax.legend()
    save_figure(fig,figs,'coverage_horizon')
    for p in source_files:
        assert hashes[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(out/'comparison.json', dict(summaries=summaries,critic=audits,source_sha256=hashes,
        config_difference={'algo.value_clipping':[True,False]},training_seed=0,
        evaluation_action_seed=cfg.run.evaluation_action_seed,scene_seeds=cfg.run.validation_seeds,
        caveat='On-policy sampled returns per checkpoint; not exact conditional V or same-state critic comparison.'))
    write_report(out,summaries,audits,rows,paired)
    print('Saved',out,flush=True)


if __name__=='__main__':
    main()
