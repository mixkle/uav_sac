"""Read-only critic audit of the central-UE 250k checkpoint; no fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_value_clipping import audit
from uav_research.config import load_config
from uav_research.plotting import plt, style, save_figure


def main():
    run = Path('runs/mappo_central_ue_250k_seed0')
    parent = Path('analysis/mappo_central_ue_250k_seed0')
    out = parent/'critic_audit'
    out.mkdir(parents=True, exist_ok=True)
    figs = out/'figures'
    figs.mkdir(exist_ok=True)
    cfg = load_config(run/'config.yaml')
    report = json.loads((parent/'report.json').read_text())
    expected = report['evaluations']['last']
    assert expected['checkpoint_env_steps'] == 250000
    assert cfg.env.reward_primary == 'coverage'
    assert all(getattr(cfg.env, k) == 0 for k in ['throughput_weight','robustness_weight','disconnection_weight'])
    assert not set(cfg.run.validation_seeds) & set(cfg.run.test_seeds)
    files = [run/n for n in ['last.pt','best.pt','config.yaml','metrics.csv','validation.csv']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert hashes[str(run/'last.pt')] == expected['checkpoint_sha256']
    stages = audit(run, 'last', out/'critic_predictions.npz', expected['summary']['mean'])
    with np.load(out/'critic_predictions.npz') as data:
        v, g, r = data['values'], data['returns'], data['rewards']
    horizon, gamma = cfg.env.horizon, cfg.algo.discount
    t = np.arange(horizon)
    bound = (1-gamma**(horizon-t))/(1-gamma)
    violation = v > bound[None] + 1e-6
    rows = [dict(stage=k, **m) for k,m in stages.items()]
    with (out/'errors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, start in zip(axes, [0, 380]):
        ax.plot(t[start:], v.mean(0)[start:], label='Predicted V(s), raw scale')
        ax.plot(t[start:], g.mean(0)[start:], label='Sampled MC return')
        ax.plot(t[start:], bound[start:], '--', color='gray', label='Maximum possible return (r <= 1)')
        ax.set(xlabel='State time before action', ylabel='Discounted return', title='Full episode' if start == 0 else 'Last 20 actions')
        ax.legend(fontsize=8)
    save_figure(fig, figs, 'critic_horizon')
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].plot(t, np.sqrt(((v-g)**2).mean(0)), label='Per-state RMSE')
    axes[0].plot(t, (v-g).mean(0), label='Mean signed error')
    axes[0].axhline(0, color='gray', lw=.7)
    axes[0].set(xlabel='State time before action', ylabel='Raw return units'); axes[0].legend()
    axes[1].scatter(g.ravel(), v.ravel(), s=3, alpha=.12)
    lim = max(g.max(),v.max())
    axes[1].plot([0,lim],[0,lim], '--', color='gray')
    axes[1].set(xlabel='Sampled MC return', ylabel='Predicted V(s)', title='32 scenes × 400 states')
    save_figure(fig, figs, 'critic_errors')
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, i in zip(axes, [0,1]):
        ax.plot(t,v[i],label='Predicted V(s)');ax.plot(t,g[i],label='Sampled MC return')
        ax.set(title=f'Fixed scene {cfg.run.validation_seeds[i]}',xlabel='State time before action',ylabel='Return');ax.legend()
    save_figure(fig, figs, 'critic_scenes')
    # Existing C audit is a contextual comparison, on its own state distribution.
    baseline = Path('analysis/critic_value_clipping_250k_seed0/clipped_last_critic.npz')
    baseline_stats = None
    if baseline.exists():
        with np.load(baseline) as data:
            bv,bg = data['values'],data['returns']
        baseline_stats = dict(rmse=float(np.sqrt(((bv-bg)**2).mean())),last_prediction=float(bv[:,-1].mean()),last_return=float(bg[:,-1].mean()))
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for ax,values,returns,title in zip(axes,[bv,v],[bg,g],['C: full-world UE','Central UE']):
            ax.plot(t,values.mean(0),label='Predicted V(s)');ax.plot(t,returns.mean(0),label='Sampled MC return')
            ax.set(title=title,xlabel='State time before action',ylabel='Return');ax.legend()
        save_figure(fig,figs,'critic_initialization_comparison')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files)
    results = dict(stages=stages, upper_bound_violation_fraction=float(violation.mean()),
                   last_upper_bound_violation_scenes=int(violation[:,-1].sum()), baseline=baseline_stats,
                   source_sha256=hashes, checkpoint_step=250000, scene_seeds=cfg.run.validation_seeds,
                   action_seed=cfg.run.evaluation_action_seed, discount=gamma, horizon=horizon,
                   training=False, matches_validation_coverage=True)
    (out/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
    lines = ['# 中央UE初始化：冻结策略的critic预测检验','',
             '固定250k last的actor与critic；不训练、不改权重、不更新价值归一化统计。best同为250k。'
             '使用原32个验证场景、16环境批次、动作采样种子880001；未使用保留测试集。','',
             '每个动作前记录反归一化V(s_t)，采集完整400步回合，以G_400=0反向计算G_t=r_t+0.995G_(t+1)。'
             '不使用GAE或critic bootstrap。每步奖励严格等于联网覆盖率；回放平均覆盖与正式验证一致。','',
             '| 阶段 | 平均V预测 | 平均MC回报 | 偏差V−G | RMSE | MAE |','|---|---:|---:|---:|---:|---:|']
    for name,m in stages.items():
        lines.append(f'| {name} | '+' | '.join(f'{m[k]:.3f}' for k in ['prediction_mean','return_mean','bias','rmse','mae'])+' |')
    lines += ['', '阶段：first100为t=0–99；middle100为t=150–249；last100为t=300–399；last20为t=380–399；last1为t=399（执行末次动作前），不是终止状态。','',
              f'由于0≤r≤1，每个状态的确定上界为(1−γ^(400−t))/(1−γ)。末步上界为1，'
              f'有{violation[:,-1].sum()}/32个末步预测超过此上界；所有状态中超上界比例为{violation.mean():.2%}。','',
              '![预测与MC回报](figures/critic_horizon.png)','',
              '![误差与散点](figures/critic_errors.png)','',
              '![固定场景](figures/critic_scenes.png)','',
              '曲线是32场景平均，RMSE先按每个状态计算平方误差再汇总，不能用两条均值曲线的差代替。'
              'MC回报是采样轨迹的实现值，不是每个状态的精确条件期望；单轨迹误差包含策略与UE运动随机性。'
              '但超出理论回报上界不能用MC噪声解释。','']
    if baseline_stats:
        lines += [f"原C last的RMSE为{baseline_stats['rmse']:.3f}，末步平均预测{baseline_stats['last_prediction']:.3f}，"
                  f"末步MC回报{baseline_stats['last_return']:.3f}。两组各用自己的on-policy状态，不是同状态网络优劣测试。",'',
                  '![原C与中央初始化](figures/critic_initialization_comparison.png)','']
    lines += ['## 本次结论','',
              '中央初始化后覆盖收益提高，但critic仍表现为前期低估、后期严重高估。'
              '平均预测只缓慢下降，没有跟随剩余回合长度收缩。尤其末步超过物理回报上限，'
              '说明有限时域价值校准问题仍然存在，并非只是均值曲线形状不同。','',
              '原C与中央初始化的全程RMSE接近，改变UE分布没有消除该症状。'
              '这不等于证明网络结构无法拟合；现有MLP的监督拟合与泛化能力需要与在线GAE目标质量分开检验。','',
              '本实验检查现有critic的预测准确性，不能单凭误差将策略退化唯一归因于critic或GAE；'
              '它也不检验给定MC标签后能否拟合，那需要另行训练critic。','',
              '复现：`.venv/bin/python scripts/audit_central_ue_critic.py`。原训练文件SHA256前后保持一致；'
              '逐场景逐时预测、回报、奖励保存在critic_predictions.npz。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
