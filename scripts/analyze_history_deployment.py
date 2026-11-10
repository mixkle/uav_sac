"""Describe existing frozen-policy audits; no rollout, training or model changes."""
import csv
import json
from pathlib import Path
import numpy as np
from uav_research.plotting import plt, style, save_figure

ROOT = Path('analysis/mappo_central_history_100k_seed0')
OUT = ROOT/'diagnosis'


def smooth(x):
    return np.convolve(x, np.ones(21)/21, 'valid')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {k: Path('analysis/mappo_continuous_bridge_100k_seed0')/k/'last/audit.npz'
             for k in ['M0', 'M1', 'S0']}
    paths.update({k: Path('analysis/mappo_boundary_central_100k_seed0')/k/'last/audit.npz'
                  for k in ['M1-C', 'M1-B', 'M1-CB']})
    paths['M1-C-H'] = ROOT/'M1-C-H/validation/last/880001/audit.npz'
    peaks, all_peaks = [], []
    style(); fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, path in paths.items():
        with np.load(path) as data:
            np.testing.assert_array_equal(data['scene_seeds'], np.arange(10000, 10032))
            cov = data['diagnostics'][..., list(data['diagnostic_names']).index('coverage')]
        mean = cov.mean(0); curve = smooth(mean)
        individual = np.array([smooth(x).argmax()+11 for x in cov]); all_peaks.append(individual)
        peaks.append(dict(model=name, raw_mean_peak=int(mean.argmax()+1),
                          smooth_mean_peak=int(curve.argmax()+11), peak_coverage=float(curve.max()),
                          scene_peak_q25=float(np.quantile(individual,.25)),
                          scene_peak_median=float(np.median(individual)),
                          scene_peak_q75=float(np.quantile(individual,.75)),
                          fraction_peak_100_150=float(((individual>=100)&(individual<=150)).mean()),
                          source=str(path)))
        axes[0].plot(np.arange(11,391), curve*100, label=name)
    axes[0].axvspan(100,150,color='grey',alpha=.12)
    axes[0].axvline(128,color='grey',ls=':',lw=1)
    axes[0].set(xlabel='Episode step', ylabel='Coverage (%)', title='32 validation scenes; action seed 880001; 21-step mean')
    axes[0].legend(ncol=2,fontsize=8)
    axes[1].boxplot(all_peaks, tick_labels=list(paths), showfliers=True)
    axes[1].axhspan(100,150,color='grey',alpha=.12)
    axes[1].set(ylabel='Individual-scene peak step',title='Peaks are dispersed across scenes')
    save_figure(fig, OUT, 'peak_timing_comparison')
    stages, accounting, cohorts = [], [], {}
    keys=['coverage','discovery_fraction','disconnection_ratio','uav_nearest_neighbor_mean',
          'boundary_within_150_fraction','actual_speed_mean']
    fig, axes=plt.subplots(2,3,figsize=(13,7))
    for group in ['M1-C','M1-C-H']:
        arrays=[]
        for seed in [880001,880002,880003]:
            with np.load(ROOT/group/f'diagnostic/last/{seed}/audit.npz') as data:
                np.testing.assert_array_equal(data['scene_seeds'],np.arange(610000,610064))
                arrays.append(data['diagnostics'].copy()); names=list(data['diagnostic_names'])
        a=np.stack(arrays); cohorts[group]=a
        def field(key):return a[...,names.index(key)]
        # Sequential rejection diagnostics partition all UE; verify at every state.
        np.testing.assert_allclose(field('coverage'),field('discovery_fraction')-
            sum(field(k) for k in ['downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users'])/150,
            atol=1e-12,rtol=1e-12)
        for ax,key in zip(axes.flat,keys):
            y=field(key).mean((0,1));ax.plot(np.arange(1,401),y,label=group)
            ax.axvspan(100,150,color='grey',alpha=.12);ax.set(title=key,xlabel='Episode step');ax.legend(fontsize=8)
        for start,stop in [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400)]:
            stages.append(dict(group=group,start=start+1,stop=stop,
                               **{k:float(a[:,:,start:stop,j].mean()) for j,k in enumerate(names)}))
        def change(key):return float(field(key)[:,:,300:400].mean()-field(key)[:,:,100:150].mean())
        accounting.append(dict(group=group, coverage_change_pp=100*change('coverage'),
                               sensing_change_pp=100*change('discovery_fraction'),
                               downlink_change_pp=-100*change('downlink_rejected_users')/150,
                               backhaul_change_pp=-100*change('backhaul_rejected_users')/150,
                               capacity_change_pp=-100*change('capacity_rejected_users')/150))
    fig.suptitle('64 diagnostic scenes x 3 action streams; frozen last policies')
    save_figure(fig,OUT,'deployment_decline_diagnostics')
    fig,ax=plt.subplots(figsize=(10,4))
    terms=['sensing_change_pp','downlink_change_pp','backhaul_change_pp','capacity_change_pp','coverage_change_pp']
    for j,row in enumerate(accounting):
        ax.bar(np.arange(5)+(j-.5)*.35,[row[k] for k in terms],width=.35,label=row['group'])
    ax.set_xticks(np.arange(5),['Sensing','Downlink rejection','Backhaul rejection','Capacity rejection','Net coverage'])
    ax.axhline(0,color='grey',lw=.8);ax.set(ylabel='Coverage contribution change (pp)',title='Last 100 minus steps 101–150: exact accounting, not independent causal effects');ax.legend()
    save_figure(fig,OUT,'coverage_loss_accounting')
    for filename,rows in [('peak_timing.csv',peaks),('stage_metrics.csv',stages),('coverage_loss_accounting.csv',accounting)]:
        with (OUT/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(dict(peaks=peaks,accounting=accounting),indent=2))
    lines=['# 历史观测实验：覆盖退化与峰值时间复盘','',
           '来源均为已有last评估文件，本脚本不执行新训练或采样。跨实验峰值使用相同32个验证种子及动作种子880001；不同UE初始化仍是不同物理场景。21步居中移动平均只用于描述峰值，单场景峰值不构成因果证据。','',
           '|模型|平均曲线平滑峰值步|单场景峰值中位数|峰值处于100–150步的场景比例|','|---|---:|---:|---:|']
    for r in peaks:lines.append(f'|{r["model"]}|{r["smooth_mean_peak"]}|{r["scene_peak_median"]:.1f}|{r["fraction_peak_100_150"]:.1%}|')
    lines+=['','![峰值](peak_timing_comparison.png)','','![阶段变化](deployment_decline_diagnostics.png)','','![覆盖损失分解](coverage_loss_accounting.png)','',
            '64个诊断场景×3次动作重复显示：历史组后期断联继续减少，感知、覆盖却下降，近邻距离收缩、靠边增加，速度维持约20 m/步。支持“保持联网但空间服务范围收缩”，不是在统一时刻突然全部断链。',
            '阶段淘汰诊断满足覆盖=瞬时感知−下行淘汰/M−回传淘汰/M−容量淘汰/M。分解是当前顺序定义下的精确人数核算，不能把它当作独立改变SINR、回传等因素的因果效应。',
            '平均曲线在100–150附近见顶并非每个场景的共同事件。冻结验证不会在128步更新网络；actor也没有显式时间输入，critic时间因子不直接参与执行动作。128与峰值接近不足以认定rollout长度为根因。',
            '下一步可固定checkpoint与场景，仅改变UAV最大位移15/30/45 m，观察峰值是否随运动尺度移动；这是离策略动力学干预，只用于定位。另可在同一良好部署上比较动作的配对未来回报与critic优势排序，以检查保持动作是否得到正确更新信号。']
    (OUT/'notes.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(accounting,indent=2))


if __name__=='__main__':main()
