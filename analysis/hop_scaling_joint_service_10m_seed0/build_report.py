"""Standalone final report; no dependency on intermediate evaluation folders."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
from uav_research.runtime import atomic_json
from uav_research.topk_diagnostic import file_hash
from uav_research.diffusion_dppo.report import error_metrics
from uav_research.plotting import style,save_figure

root=Path('analysis/hop_scaling_joint_service_10m_seed0')
spec=importlib.util.spec_from_file_location('hop_report','scripts/report_hop_scaling_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for group in ('diffusion','gaussian'):
    shutil.copytree(root/'snapshots'/group,root/'training_data'/group,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.pt'))
for kind in ('step0','best','last'):
    sources={g:json.loads((root/kind/g/'provenance.json').read_text()) for g in ('diffusion','gaussian')}
    atomic_json(root/kind/'report_sources.json',dict(labels=['diffusion','gaussian'],sources=sources))
m.rebuild(root)
comparison=json.loads((root/'comparison.json').read_text())
all_summary={kind:{g:json.loads((root/kind/g/'summary.json').read_text())
             for g in ('diffusion','gaussian')} for kind in ('step0','best','last')}
lines=['# 跳数缩放联合运动与UE评分：两组10M完整训练验证','',
       '验证日期：2026-11-10。两组均已完成10,000,000条联合转移、4,883轮PPO、195,320次actor/critic优化器更新。保留原训练目录，使用已校验的独立模型副本进行冻结策略验证。','',
       '两组均使用Top-K评分关联、跳数速度缩放、真实位移历史、时间因子critic和覆盖＋0.1归一化吞吐奖励。扩散预先采集64,000条示范并预训练50轮，Gaussian从零初始化；本轮在线样本预算相同，但总数据与计算成本不同。该对照同时包含策略表达、预训练和PPO比率定义等差异，不是扩散分布的单因素消融。','',
       '## 评估协议','',
       '冻结step0、best、last，32个验证场景10000–10031，400步真实终止，动作采样种子880001，16环境批次；64次条件动作采样使用独立诊断随机流。各模型使用自己的on-policy轨迹。保留测试集未参与本轮验证。best按完整回合的覆盖＋0.1归一化吞吐回报选择，包含step0；主要比较同预算last。绿色实线为Diffusion、蓝色虚线为Gaussian。','',
       '## 任务表现','',
       '| 组别 | 模型 | 保存步数 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐 Mbps | 回合回报 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for kind in ('last','best','step0'):
    for group in ('diffusion','gaussian'):
        s=all_summary[kind][group];p=json.loads((root/kind/group/'provenance.json').read_text())
        lines.append(f"| {group} | {kind} | {p['env_steps']:,} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_bps']/1e6:.3f} | {s['return']:.3f} |")

gains=[]
for group in ('diffusion','gaussian'):
    rows={k:m.read(root/k/group/'episodes.csv') for k in ('step0','best','last')}
    for left,right in [('last','step0'),('best','step0'),('last','best')]:
        assert [r['scene_seed'] for r in rows[left]]==[r['scene_seed'] for r in rows[right]]
        draws=np.random.default_rng(950001).integers(32,size=(2000,32))
        for metric in ('coverage','throughput_bps','return','last100_coverage','final_coverage','maintain_coverage'):
            diff=np.array([float(a[metric])-float(b[metric]) for a,b in zip(rows[left],rows[right])])
            low,high=np.quantile(diff[draws].mean(1),[.025,.975])
            gains.append(dict(group=group,comparison=f'{left}-{right}',metric=metric,
                              difference=float(diff.mean()),low=float(low),high=float(high)))
atomic_json(root/'training_gains.json',gains)
lines+=['','## 训练收益与回退','']
for group in ('diffusion','gaussian'):
    a,b,c=[all_summary[k][group] for k in ('step0','best','last')]
    lines.append(f"- {group}：best相对step0平均覆盖{100*(b['coverage']-a['coverage']):+.3f}个百分点；last相对step0{100*(c['coverage']-a['coverage']):+.3f}个百分点；last相对best{100*(c['coverage']-b['coverage']):+.3f}个百分点。")
lines+=['','配对场景bootstrap重复2,000次、种子950001。跨组差值见`comparison.json`，各组训练增益见`training_gains.json`；这些是场景区间，不是训练种子置信区间。','',
        '## 部署维持与网络稳定性','',
        '| 组别 | 模型 | 51–150步覆盖 | 301–400步覆盖 | 维持变化 pp | 最大单步跌幅 pp | 最大5步跌幅 pp | ≥10pp事件/回合 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for kind in ('best','last'):
    for group in ('diffusion','gaussian'):
        s=all_summary[kind][group];v=comparison['metrics'][kind][group]
        lines.append(f"| {group} | {kind} | {s['last100_coverage']-s['maintain_coverage']:.3%} | {s['last100_coverage']:.3%} | {100*s['maintain_coverage']:+.3f} | {100*v['max_drop_1']:.3f} | {100*v['max_drop_5']:.3f} | {v['drop_events_10pp']:.3f} |")
lines+=['','最大跌幅先逐场景计算再平均；不把平均曲线平滑当成逐场景稳定。维持差值必须和前后两段绝对覆盖一起判断。','',
        '| 组别 | 模型 | 断联UAV比例 | 边界150m停留 | 反射比例 | 实际速度 m/s | 小于1m位移比例 | 最近邻距离 m |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for kind in ('best','last'):
    for group in ('diffusion','gaussian'):
        s=all_summary[kind][group]
        lines.append(f"| {group} | {kind} | {s['disconnection_ratio']:.3%} | {s['boundary_within_150_fraction']:.3%} | {s['boundary_reflection_fraction']:.3%} | {s['actual_speed_mean']:.3f} | {s['near_hover_fraction']:.3%} | {s['uav_nearest_neighbor_mean']:.2f} |")
lines+=['','当前last的跨场景平均曲线峰值：','']
for group in ('diffusion','gaussian'):
    with np.load(root/'last'/group/'step_metrics.npz') as z:
        curve=z['metrics'][:,:,z['names'].tolist().index('coverage')].mean(0)
    lines.append(f"- {group}：第{curve.argmax()+1}步为{curve.max():.3%}，400步为{curve[-1]:.3%}。这是先平均曲线再求峰值，不是逐场景峰值时刻的平均。")

lines+=['','## Critic准确性','',
        '以完整环境奖励计算真实终止MC回报，不加入actor熵。累计价值与平均奖励两个尺度均保存；每步MC标签严格等于覆盖与加权吞吐两项折扣回报之和。不同策略的轨迹分布不同，平均RMSE不能代替同一起点下的动作价值排序检验。','',
        '| 组别 | 模型 | 累计价值RMSE | 标签标准差归一化RMSE | 最后20步RMSE | 最后一步RMSE |',
        '|---|---|---:|---:|---:|---:|']
for kind in ('best','last'):
    for group in ('diffusion','gaussian'):
        with np.load(root/kind/group/'critic_predictions.npz') as z:
            e=error_metrics(z['predicted'],z['mc_return'])
        lines.append(f"| {group} | {kind} | {e['all']['rmse']:.4f} | {e['normalized_rmse']:.4f} | {e['last20']['rmse']:.4f} | {e['last1']['rmse']:.4f} |")

lines+=['','## 训练趋势与成本','', '| 组别 | 在线耗时 h | 验证耗时 h（已包含） | 最后10次验证覆盖均值 | 最后10次验证回报均值 |','|---|---:|---:|---:|---:|']
import matplotlib.pyplot as plt
style();fig,axes=plt.subplots(1,2,figsize=(10,3.7))
for group,color,ls in [('diffusion','#009E73','-'),('gaussian','#0072B2','--')]:
    rows=m.read(root/'training_data'/group/'validation.csv');manifest=json.loads((root/'training_data'/group/'manifest.json').read_text())
    recent=rows[-10:]
    lines.append(f"| {group} | {manifest['elapsed_seconds']/3600:.2f} | {manifest['evaluation_seconds']/3600:.2f} | {np.mean([float(x['coverage']) for x in recent]):.3%} | {np.mean([float(x['return']) for x in recent]):.3f} |")
    for ax,key in zip(axes,['coverage','return']):
        x=np.array([int(r['env_steps']) for r in rows]);y=np.array([float(r[key]) for r in rows])
        smooth=np.array([y[max(0,i-9):i+1].mean() for i in range(len(y))])
        ax.plot(x,y,color=color,ls=ls,alpha=.35,lw=.7)
        ax.plot(x,smooth,color=color,ls=ls,label=group+' (10-eval mean)');ax.set(xlabel='Joint transitions',ylabel=key)
axes[0].legend();save_figure(fig,root/'figures','validation_trend')
lines+=['','上述在线时间包含周期验证与保存，不能直接视为纯训练计算时间。扩散的示范采集与预训练成本另见`comparison.json`和`training_data/diffusion/pretrain/summary.json`。','',
        '## 文件与复现','',
        '- `snapshots/`：六份模型、完整训练日志和配置副本；哈希来源见`snapshot_manifest.json`。',
        '- `best/figures/`、`last/figures/`：任务曲线、五种行为/协议诊断、MC误差、固定场景10000/10001拓扑、动作方差。',
        '- `figures/validation_trend.png`：完整学习过程；`figures/metrics.png`：loss、KL与clipping。',
        '- 轨迹、逐场景指标、MC预测、64次条件动作统计均保存，可以从数据重建图表，无需再次采样。',
        '- 中途评估目录仅在最终报告、原始数据和模型副本校验完成后清理；具体删除记录见`cleanup_manifest.json`。',
        '', '本轮仅有种子0，验证集在训练期间参与选模，不能把结果称为未见最终测试集上的论文结论。']
interpretation = root/'interpretation.md'
if interpretation.exists():
    lines += ['', interpretation.read_text()]
if (root/'rebuild_notes.md').exists():
    lines += ['', (root/'rebuild_notes.md').read_text()]
(root/'report.md').write_text('\n'.join(lines)+'\n')
print((root/'report.md').read_text())
