"""Rebuild the M1/M2 10M diagnostic from immutable evaluation arrays only."""
from pathlib import Path
import csv
import json
import numpy as np
from uav_research.plotting import style, save_figure
from uav_research.system_ablation.report import write_csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COLORS = {'M1': '#009E73', 'M2': '#0072B2'}
LABELS = {'M1': 'M1 Gaussian PPO', 'M2': 'M2 Diffusion DPPO'}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def bootstrap(x):
    x = np.asarray(x, dtype=float)
    draws = np.random.default_rng(950001).integers(len(x), size=(2000, len(x)))
    lo, hi = np.quantile(x[draws].mean(1), [.025, .975])
    return float(x.mean()), float(lo), float(hi)


def main():
    style()
    records = read(ROOT/'metrics/unified_best_last.csv')
    scenes = read(ROOT/'metrics/per_scene_action.csv')
    lookup = {(r['run_id'], r['checkpoint']): r for r in records}
    curves, curve_rows = {}, []
    for g in COLORS:
        for kind in ('step0', 'best', 'last'):
            hits = sorted((ROOT/g/'validation'/kind).glob('*/step_metrics.npz'))
            assert len(hits) == 3, (g, kind, hits)
            arrays = []
            for path in hits:
                with np.load(path, allow_pickle=False) as z:
                    names = z['names'].tolist()
                    assert z['metrics'].shape[:2] == (32, 400)
                    arrays.append(z['metrics'].copy())
            data = np.stack(arrays).mean(0)  # average actions within each scene
            curves[g, kind] = {name: data[:, :, i] for i, name in enumerate(names)}
            for t in range(400):
                curve_rows.append(dict(group=g, checkpoint=kind, step=t+1,
                    **{name: float(data[:, t, i].mean()) for i, name in enumerate(names)}))
    write_csv(ROOT/'metrics/episode_curve_means.csv', curve_rows)

    # Episode curves: separate best and last so best does not conceal last regression.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, kind in enumerate(('best', 'last')):
        for g in COLORS:
            for ax, key, factor, ylabel in zip(axes[row],
                ('coverage', 'throughput_bps', 'disconnection_ratio'),
                (100, 1e-6, 100), ('Coverage (%)', 'Throughput (Mbps)', 'Disconnected UAV (%)')):
                y = curves[g, kind][key].mean(0)*factor
                ax.plot(np.arange(1, 401), y, color=COLORS[g],
                        ls='-' if g == 'M1' else '--', label=LABELS[g])
                ax.axvspan(301, 400, color='#bbbbbb', alpha=.12)
                ax.set(ylabel=ylabel, title=kind, xlabel='Episode step')
        axes[row, 0].legend(fontsize=8)
    save_figure(fig, ROOT/'figures', 'best_last_episode_comparison')

    bar_metrics = [('last100_coverage', 100, 'Last100 coverage (%)'),
        ('final_coverage', 100, 'Final coverage (%)'),
        ('last100_throughput_mbps', 1, 'Last100 throughput (Mbps)'),
        ('last100_coverage_std_pp', 1, 'Mean within-episode coverage SD (pp)'),
        ('last100_max_drop_pp', 1, 'Mean maximum one-step drop (pp)'),
        ('last100_connected_fraction', 100, 'Last100 connected UAV (%)')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, scale, title) in zip(axes.flat, bar_metrics):
        for j, g in enumerate(COLORS):
            y = [float(lookup[g, k][key])*scale for k in ('best', 'last')]
            lo = [float(lookup[g, k][key+'_ci_low'])*scale for k in ('best', 'last')]
            hi = [float(lookup[g, k][key+'_ci_high'])*scale for k in ('best', 'last')]
            ax.bar(np.arange(2)+(j-.5)*.34, y, width=.32, color=COLORS[g],
                label=LABELS[g], yerr=np.array([np.array(y)-lo, np.array(hi)-y]),
                capsize=3, hatch='' if g == 'M1' else '//')
        ax.set(xticks=[0, 1], xticklabels=['best', 'last'], title=title)
    axes.flat[0].legend(fontsize=7)
    save_figure(fig, ROOT/'figures', 'late_metrics_comparison')

    # Full-scale training curves, no smoothing, plus a clearly labeled moving mean.
    phases = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for g in COLORS:
        rows = read(ROOT/g/'training_data/metrics.csv')
        val = read(ROOT/g/'training_data/validation.csv')
        for ax, data, key, scale, title in (
            (axes[0, 0], val, 'last100_coverage', 100, 'Validation last100 coverage (%)'),
            (axes[0, 1], rows, 'coverage', 100, 'Training rollout coverage (%)'),
            (axes[0, 2], rows, 'action_norm_mean', 1, 'Raw movement command norm'),
            (axes[1, 0], rows, 'actual_speed_mean', 1, 'Training actual speed (m/s)'),
            (axes[1, 1], rows, 'value_loss', 1, 'Critic training loss'),
            (axes[1, 2], rows, 'ppo_clip_fraction', 100, 'Actor PPO clipped ratios (%)')):
            x = np.array([float(r['env_steps']) for r in data])/1e6
            y = np.array([float(r[key]) for r in data])*scale
            ax.plot(x, y, color=COLORS[g], alpha=.23, lw=.6)
            window = 5 if data is val else 50
            smooth = np.array([y[max(0, i-window+1):i+1].mean() for i in range(len(y))])
            ax.plot(x, smooth, color=COLORS[g], ls='-' if g == 'M1' else '--', label=LABELS[g])
            ax.set(xlabel='Cumulative joint transitions (millions)', title=title)
            if key == 'value_loss':
                ax.set_yscale('log')
        for a, b in ((0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 10)):
            vv = [r for r in val if a*1e6 < float(r['env_steps']) <= b*1e6]
            rr = [r for r in rows if a*1e6 < float(r['env_steps']) <= b*1e6]
            phases.append(dict(group=g, start_m=a, end_m=b, evaluations=len(vv),
                validation_last100_coverage=np.mean([float(r['last100_coverage']) for r in vv]),
                **{key: np.mean([float(r[key]) for r in rr]) for key in
                   ('coverage','action_norm_mean','actual_speed_mean','ppo_clip_fraction','value_loss')}))
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, ROOT/'figures', 'training_progress_and_motion')
    write_csv(ROOT/'metrics/training_phases.csv', phases)

    # Paired best -> last regression, with the action repetitions nested in scene.
    changes = []
    metric_keys = [m[0] for m in bar_metrics] + ['last50_coverage', 'last100_drop_events_10pp',
        'episode_return', 'maintain_coverage', 'critic_rmse_all', 'critic_rmse_last100']
    for g in COLORS:
        for key in metric_keys:
            pair = {}
            for kind in ('best', 'last'):
                rows = [r for r in scenes if r['run_id'] == g and r['checkpoint'] == kind]
                pair[kind] = np.array([np.mean([float(r[key]) for r in rows if int(r['scene_seed']) == s])
                    for s in range(10000, 10032)])
            avg, lo, hi = bootstrap(pair['last']-pair['best'])
            changes.append(dict(group=g, comparison='last - best', metric=key,
                difference=avg, low=lo, high=hi))
    write_csv(ROOT/'metrics/within_group_best_last_changes.csv', changes)
    seed_rows = []
    for g in COLORS:
        for kind in ('step0', 'best', 'last'):
            for seed in (880001, 880002, 880003):
                rows = [r for r in scenes if r['run_id'] == g and r['checkpoint'] == kind and int(r['action_seed']) == seed]
                seed_rows.append(dict(group=g, checkpoint=kind, action_seed=seed,
                    **{k: np.mean([float(r[k]) for r in rows]) for k in metric_keys}))
    write_csv(ROOT/'metrics/per_action_seed_summary.csv', seed_rows)
    topology_comparisons()
    for g in COLORS:
        # Match reevaluation seed 880001 to the original selection log without reselection.
        for kind in ('best', 'last'):
            row = lookup[g, kind]
            path = ROOT/g/'validation'/kind/'880001/provenance.json'
            steps = json.loads(path.read_text())['env_steps']
            val = read(ROOT/g/'training_data/validation.csv')
            expected = next(r for r in val if int(r['env_steps']) == steps)
            actual = next(r for r in seed_rows if r['group'] == g and r['checkpoint'] == kind and r['action_seed'] == 880001)
            assert abs(actual['last100_coverage']-float(expected['last100_coverage'])) < 1e-10
    (ROOT/'rebuild_audit.json').write_text(json.dumps(dict(
        trajectories=576, groups=2, checkpoints_per_group=3, scenes=32,
        action_seeds=[880001,880002,880003], horizon=400,
        selection_log_reproduced=True, bootstrap_scenes=2000, seed=950001), indent=2))
    report(lookup, changes)
    print('Supplementary figures, paired changes, and selection reproduction checks completed.')


def topology_comparisons():
    for kind in ('best', 'last'):
        for scene in (10000, 10001):
            fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))
            for row, g in enumerate(COLORS):
                path=ROOT/g/'validation'/kind/'880001'/f'trace_{scene}.npz'
                with np.load(path, allow_pickle=False) as z:
                    for ax, t in zip(axes[row], (0,150,200,400)):
                        u, m, bs = z['uav_xy'][t], z['user_xy'][t], z['bs_positions']
                        served = z['C'][t].any(0)
                        connected = z['B'][t].min(1) <= len(u)
                        ax.scatter(m[~served,0], m[~served,1], s=7, c='#cccccc', label='Unserved UE')
                        ax.scatter(m[served,0], m[served,1], s=8, c=COLORS[g], alpha=.65, label='Served UE')
                        ax.scatter(u[connected,0], u[connected,1], s=35, c=COLORS[g], marker='^', label='Connected UAV')
                        ax.scatter(u[~connected,0], u[~connected,1], s=35, c='#222222', marker='x', label='Disconnected UAV')
                        ax.scatter(bs[:,0],bs[:,1],s=55,c='#111111',marker='s',label='BS',clip_on=False)
                        for i,j in zip(*np.where(np.triu(z['A'][t],1))):
                            ax.plot(u[[i,j],0],u[[i,j],1],c=COLORS[g],lw=.65,alpha=.5)
                        for i,b in zip(*np.where(z['B'][t]==1)):
                            ax.plot([u[i,0],bs[b,0]],[u[i,1],bs[b,1]],c=COLORS[g],lw=.65,ls='--',alpha=.5)
                        for i,xy in enumerate(u):
                            ax.annotate(str(i),xy,xytext=(2,2),textcoords='offset points',fontsize=5)
                        ax.set(xlim=(0,float(z['width'])),ylim=(0,float(z['height'])),aspect='equal',
                            title=f't={t} | C={served.mean():.1%} | {connected.sum()}/18 online')
                        ax.tick_params(labelsize=7)
                    axes[row,0].set_ylabel(LABELS[g]+'\ny (m)')
                    axes[row,0].legend(fontsize=5.7,loc='lower right')
            fig.suptitle(f'{kind} | fixed scene {scene} | action seed 880001',fontsize=12)
            save_figure(fig,ROOT/'figures',f'topology_{kind}_{scene}_comparison')


def report(lookup, changes):
    def f(g, kind, key):
        return float(lookup[g, kind][key])
    def table(metrics):
        lines = ['| 指标 | M1 best | M2 best | M1 last | M2 last |',
                 '|---|---:|---:|---:|---:|']
        for title, key, scale in metrics:
            lines.append('| '+title+' | '+' | '.join(
                f'{f(g,k,key)*scale:.3f}' for k,g in
                [('best','M1'),('best','M2'),('last','M1'),('last','M2')])+' |')
        return lines
    paired = read(ROOT/'metrics/paired_differences.csv')
    def delta(kind, key, scale=100):
        row = next(r for r in paired if r['checkpoint'] == kind and r['metric'] == key)
        return f"{float(row['difference'])*scale:+.3f} [{float(row['low'])*scale:+.3f}, {float(row['high'])*scale:+.3f}]"
    def change(g, key, scale=100):
        row = next(r for r in changes if r['group'] == g and r['metric'] == key)
        return f"{row['difference']*scale:+.3f} [{row['low']*scale:+.3f}, {row['high']*scale:+.3f}]"
    lines = ['# M1 / M2：仅运动策略，10M 完成后的冻结验证与对比', '',
        '结论：M2 在训练中学到了明显更好的部署，但后期发生退化；不能把 10M last 等同于历史 best，也不能将这次结果解释为两组都没有学到东西。', '',
        '## 1. 验证口径与来源', '',
        '- M1：Gaussian PPO，仅输出运动；M2：8步 Diffusion DPPO，仅输出运动。两组均使用固定动作前 SINR 评分、一轮 Top10、动作后过滤、UE 选择，无补位。',
        '- 场景3500×3500 m、18 UAV、150 UE、容量10、3个BS；中心UE初始化后移动，跳数速度缩放与真实位移历史；奖励仅为覆盖率＋0.1归一化数据率。',
        '- 均从零开始1M，再从完整状态续训至累计10M，无示范、BC或预训练；各4884轮PPO、195360次actor/critic优化。两组actor参数分别159172、150978。',
        '- 冻结step0/best/last，32个验证场景10000–10031，每场景400步，动作种子880001、880002、880003，共576条回合。未使用保留测试集。',
        '- best沿用训练期间选择：后100步覆盖优先、相等时后100步吞吐；不根据三个动作种子的复评重新选模。原选模种子880001已复现，覆盖偏差小于1e-10。',
        '- 每条轨迹先计算指标，场景内平均三个动作重复，再平均32场景。标准差为各轨迹窗口内总体标准差的平均，不是合并全部场景后的标准差。',
        '- 95%区间来自2000次配对场景bootstrap，种子950001；不是训练种子置信区间，也不消除在同一验证集反复选best的影响。', '',
        '| 组 | best实际步数 | last实际步数 | 累计训练耗时，含在线验证/保存 |',
        '|---|---:|---:|---:|']
    for g in COLORS:
        m=json.loads((ROOT/g/'training_data/manifest.json').read_text())
        lines.append(f"| {g} | {m['best_step']:,} | {m['env_steps']:,} | {m['elapsed_seconds']/3600:.2f} h |")
    lines += ['', '## 2. 覆盖、吞吐与回报', '']
    lines += table([
        ('后100步覆盖率 %','last100_coverage',100),
        ('后50步覆盖率 %','last50_coverage',100),
        ('终态覆盖率 %','final_coverage',100),
        ('后100步吞吐 Mbps','last100_throughput_mbps',1),
        ('后50步吞吐 Mbps','last50_throughput_mbps',1),
        ('终态吞吐 Mbps','final_throughput_mbps',1),
        ('全回合覆盖率 %，辅助指标','mean_coverage',100),
        ('场景平均未折扣回合回报','episode_return',1)])
    lines += ['', 'M2 − M1 的配对差值及95%场景区间：', '',
        '| 模型 | 后100覆盖，百分点 | 终态覆盖，百分点 | 后100吞吐，Mbps |',
        '|---|---:|---:|---:|']
    for k in ('best','last'):
        lines.append(f"| {k} | {delta(k,'last100_coverage')} | {delta(k,'final_coverage')} | {delta(k,'last100_throughput_mbps',1)} |")
    lines += ['', '## 3. 部署维持、断联与行为', '']
    lines += table([
        ('第51–150步覆盖 %','early51_150_coverage',100),
        ('后100减51–150覆盖，百分点','maintain_coverage',100),
        ('后100覆盖标准差，百分点','last100_coverage_std_pp',1),
        ('后50覆盖标准差，百分点','last50_coverage_std_pp',1),
        ('后100覆盖5%分位数 %','last100_coverage_p05',100),
        ('后100平均≥10个百分点骤降次数','last100_drop_events_10pp',1),
        ('后50平均≥10个百分点骤降次数','last50_drop_events_10pp',1),
        ('后100最大单步跌幅均值，百分点','last100_max_drop_pp',1),
        ('后50最大单步跌幅均值，百分点','last50_max_drop_pp',1),
        ('后100联网UAV比例 %','last100_connected_fraction',100),
        ('后50联网UAV比例 %','last50_connected_fraction',100),
        ('终态联网UAV比例 %','final_connected_fraction',100),
        ('首次达到自身后100覆盖均值，步','formation_step',1),
        ('后100实际速度 m/s','last100_actual_speed_mean',1),
        ('后100近悬停比例 %，实际位移<1 m','last100_near_hover_fraction',100),
        ('后100距边界150m内比例 %','last100_boundary_within_150_fraction',100),
        ('后100反射比例 %','last100_boundary_reflection_fraction',100),
        ('后100最近邻距离 m','last100_uav_nearest_neighbor_mean',1)])
    lines += ['', '窗口骤降统计包括进入该窗口的第一条转移；指标按每个场景及动作重复计算，未用平均曲线替代单场景风险。形成时间是首次越过自身后100步均值，并不保证之后一直维持。', '',
        '同组 last − best 的变化：', '',
        '| 组 | 后100覆盖，百分点 | 后100吞吐 Mbps | 后100标准差，百分点 |',
        '|---|---:|---:|---:|']
    for g in COLORS:
        lines.append(f"| {g} | {change(g,'last100_coverage')} | {change(g,'last100_throughput_mbps',1)} | {change(g,'last100_coverage_std_pp',1)} |")
    lines += ['', '## 4. Critic与训练过程', '']
    lines += table([('全程critic RMSE，逐回合后平均','critic_rmse_all',1),
        ('后100 critic RMSE','critic_rmse_last100',1),
        ('最后20步critic RMSE','critic_rmse_last20',1),
        ('最后一步critic RMSE','critic_rmse_last1',1)])
    lines += ['',
        'MC由各模型自身采样策略的完整环境奖励递推得到，不含熵。两组访问的状态分布不同，误差比较不能视为同状态的网络能力对照；平均价值准确也不保证能正确区分“维持”与“继续运动”的价值差。', '',
        '训练记录显示：M2在3–7M区间持续达到较高覆盖，7M后训练采集覆盖与固定验证覆盖都回落，同时运动指令幅度增大。这支持存在训练后期退化，而不只是验证动作种子偶然偏差。训练阶段汇总见 `metrics/training_phases.csv`。',
        '回合内的平均覆盖曲线进一步区分了两个模型：M2 best约在100–150步达到接近80%后，整体维持到400步；M2 last同样在前期达到较高覆盖，但之后持续下滑，终态已经与M1 last接近。平均曲线较平不能代替逐场景风险：M2 best后100步仍平均存在约1.22次≥10个百分点的单步骤降。',
        '固定场景10000中，M2 best在150/200/400步覆盖分别为76.0%/83.3%/87.3%，后两者均18/18联网，能看到网络实际展开。固定场景10001中，M2 last从200步的84.0%、18/18联网退化到400步的53.3%、13/18联网，说明回落确实可能涉及断链；这两个预先固定的场景只是示例，不能取代32场景汇总。',
        'M1后期整体变化较小。M2已有较好的历史best，所以当前更值得排查后期更新稳定性与部署维持，而不能只依据“尚未达到更高覆盖”继续归因为步数不足。运动增强与退化是相关证据，尚未证明其单独构成根因。',
        'Actor loss和归一化value loss的绝对数值不能作为部署质量判据；两类actor概率比定义也不同，不直接用KL/clip大小排序算法优劣。', '',
        '## 5. 图表与可复现数据', '',
        '- [best/last回合内覆盖、吞吐与断联曲线](figures/best_last_episode_comparison.png)',
        '- [后期指标与场景区间](figures/late_metrics_comparison.png)',
        '- [训练进度、运动与loss](figures/training_progress_and_motion.png)',
        '- [同场景best拓扑对比，10000](figures/topology_best_10000_comparison.png)、[10001](figures/topology_best_10001_comparison.png)',
        '- [同场景last拓扑对比，10000](figures/topology_last_10000_comparison.png)、[10001](figures/topology_last_10001_comparison.png)',
        '- [M1场景10000 best拓扑](figures/M1/trace_10000_best.png)、[M2场景10000 best拓扑](figures/M2/trace_10000_best.png)',
        '- [M1场景10001 best拓扑](figures/M1/trace_10001_best.png)、[M2场景10001 best拓扑](figures/M2/trace_10001_best.png)',
        '- [汇总指标及场景区间](metrics/unified_best_last.csv)、[逐场景/动作种子指标](metrics/per_scene_action.csv)、[两组配对差值](metrics/paired_differences.csv)',
        '- [best到last配对变化](metrics/within_group_best_last_changes.csv)、[分动作种子结果](metrics/per_action_seed_summary.csv)、[回合内曲线数据](metrics/episode_curve_means.csv)',
        '- 固定展示场景10000/10001在0、150、200、400步拓扑；不是按效果挑选。绿色实线M1、蓝色虚线M2，图表同时保存PNG/PDF/SVG。',
        '- 训练进度图的浅色为原始记录，深色为后向移动均值：验证5次评估、训练50个rollout；value loss使用对数轴。',
        '- `sources.json`、两组`evaluation_audit.json`及各次`provenance.json`记录来源与哈希；验证前后step0/best/last哈希均未改变。',
        '- 重建：先运行 `scripts/motion_only_comparison.py report --total-steps 10000000 --rebuild`，再运行本目录 `rebuild_comparison.py`；均不重新采样或训练。', '',
        '本轮结果支持扩散运动策略在这些验证场景中能达到更高水平，但不支持“训练越久就必然更好”或“退化问题已经解决”。两组训练种子均只有0；对照改变了策略表达与PPO更新定义，不能将差异唯一归因于扩散分布。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
