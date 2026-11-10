"""Explain saved actor-sharing validation: bottlenecks, geometry and randomness.

No training or new trajectories. The optional common-initial-observation probe
only samples distributions at reset states; its RNG never executes environment
actions. Report figures can be rebuilt from the saved probe and audit arrays.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from report_mappo_actor_sharing import MODES, DEFAULT_OUTPUT
from report_mappo_bridge import digest
from compare_online_time_factor import write_csv
from uav_research.config import config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.models import disk_transform
from uav_research.plotting import plt, style, save_figure, COLORS
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json
from uav_research.vector import VectorEnvironment


def hull_area(points):
    points=sorted(set(map(tuple,points)))
    if len(points)<3:return 0.
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for seq,chain in [(points,lower),(points[::-1],upper)]:
        for point in seq:
            while len(chain)>1 and cross(chain[-2],chain[-1],point)<=0:chain.pop()
            chain.append(point)
    polygon=np.array(lower[:-1]+upper[:-1])
    return float(abs(np.dot(polygon[:,0],np.roll(polygon[:,1],1))-
                     np.dot(polygon[:,1],np.roll(polygon[:,0],1)))/2)


def scene_band(curves):
    """Input [scene,time], pointwise scene-bootstrap interval, bounded memory."""
    draws=np.random.default_rng(950001).integers(len(curves),size=(2000,len(curves)))
    sampled=np.concatenate([curves[chunk].mean(1) for chunk in np.array_split(draws,20)])
    return curves.mean(0),np.quantile(sampled,.025,axis=0),np.quantile(sampled,.975,axis=0)


def initial_probe(out,comparison):
    target=out/'initial_distribution_probe.npz'
    if target.exists():
        previous=json.loads((out/'initial_probe_sources.json').read_text())
        if any(digest(Path(path))!=value for path,value in previous['checkpoint_sha256'].items()):
            raise ValueError('Cached distribution probe checkpoints changed')
        return target
    results={};sources={}
    for mode in MODES:
        group=comparison['groups'][mode];cfg=config_from_dict(group['config']);configure_runtime(cfg)
        checkpoint=Path(group['run'])/'last.pt';sources[str(checkpoint)]=digest(checkpoint)
        _,policy=policy_from_checkpoint(checkpoint,'cpu');policy.actor.eval().requires_grad_(False)
        values={key:[] for key in ['mean_norm','std_mean','command_m','variance_m2','outer90','entropy']}
        hist=np.zeros(40,dtype=np.int64)
        with isolated_rng(),torch.inference_mode():
            for start in range(0,len(cfg.run.validation_seeds),cfg.run.num_envs):
                seeds=cfg.run.validation_seeds[start:start+cfg.run.num_envs]
                with VectorEnvironment(cfg.env,len(seeds)) as vector:
                    obs,_,_=vector.reset(seeds)
                    distribution=policy.actor.distribution(obs.to_torch('cpu'))
                    torch.manual_seed(979001+start)
                    action,jacobian=disk_transform(distribution.rsample((256,)))
                    radius=action.norm(dim=-1)
                    sample=dict(mean_norm=distribution.loc.norm(dim=-1),std_mean=distribution.scale.mean(-1),
                        command_m=radius.mean(0)*cfg.env.max_speed*cfg.env.dt,
                        variance_m2=action.var(0,unbiased=True).sum(-1)*(cfg.env.max_speed*cfg.env.dt)**2,
                        outer90=(radius>.9).float().mean(0),entropy=distribution.entropy().sum(-1)+jacobian.mean(0))
                    for key,value in sample.items():values[key].append(value.numpy())
                    hist+=np.histogram(radius.numpy(),bins=np.linspace(0,1,41))[0]
        for key,value in values.items():results[f'{mode}_{key}']=np.concatenate(value)
        results[f'{mode}_radius_histogram']=hist
        assert sources[str(checkpoint)]==digest(checkpoint)
    results['radius_bins']=np.linspace(0,1,41);np.savez_compressed(target,**results)
    atomic_json(out/'initial_probe_sources.json',dict(checkpoint_sha256=sources,samples=256,
        actor_updates=0,environment_steps=0,semantics='Same 32 initial scenes, sampled disk actions only'))
    return target


def analyze(out):
    comparison=json.loads((out/'comparison.json').read_text());stages=[];accounting=[];critic=[];topologies=[];peaks=[]
    with (out/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (out/'paired_differences.csv').open() as f:paired=list(csv.DictReader(f))
    sources={};style();fig,axes=plt.subplots(1,3,figsize=(14,4));datasets={}
    for mi,mode in enumerate(MODES):
        paths=comparison['groups'][mode]['checkpoints']['last']['files'];data=[]
        for path in paths:
            sources[path]=digest(Path(path))
            with np.load(path) as d:data.append({key:d[key].copy() for key in d.files})
        datasets[mode]=data;names=list(data[0]['diagnostic_names'])
        a=np.stack([d['diagnostics'] for d in data]);m=len(a[0]);population=comparison['groups'][mode]['config']['env']['num_users']
        field=lambda key:a[...,names.index(key)]
        mean_curve=field('coverage').mean((0,1));smoothed=np.convolve(mean_curve,np.ones(20)/20,mode='valid')
        peaks.append(dict(group=mode,raw_mean_peak=int(mean_curve.argmax()+1),
            smooth_window_end=int(smoothed.argmax()+20),smooth_window_center=float(smoothed.argmax()+10.5),
            smooth_peak_coverage=float(smoothed.max())))
        np.testing.assert_allclose(field('coverage'),field('discovery_fraction')-
            sum(field(k) for k in ['downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users'])/population,atol=1e-12)
        for stage,sl in [('all',slice(None)),('early51_150',slice(50,150)),('late301_400',slice(300,400)),('terminal',slice(-1,None))]:
            stages.append(dict(group=mode,stage=stage,**{key:float(field(key)[:,:,sl].mean()) for key in names}))
        for key in ['discovery_fraction','downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users','coverage']:
            scale=100 if key in ['discovery_fraction','coverage'] else -100/population
            change=scale*(field(key)[:,:,300:400].mean()-field(key)[:,:,50:150].mean())
            accounting.append(dict(group=mode,metric=key,late_minus_early_coverage_contribution_pp=float(change)))
        v=np.stack([d['values'][...,0] for d in data]);g=np.stack([d['value_labels'] for d in data]);h=data[0]['horizon_factors']
        e=v-g
        for stage,sl in [('all',slice(None)),('first100',slice(0,100)),('middle100',slice(150,250)),
                         ('last100',slice(-100,None)),('last20',slice(-20,None)),('last1',slice(-1,None))]:
            ee=e[:,:,sl];target=g[:,:,sl];std=float(target.std())
            critic.append(dict(group=mode,stage=stage,rmse=float(np.sqrt(np.mean(ee**2))),
                mae=float(np.abs(ee).mean()),bias=float(ee.mean()),normalized_rmse=float(np.sqrt(np.mean(ee**2))/std),
                average_rmse=float(np.sqrt(np.mean((ee/h[sl])**2)))))
        residual_corr=float(np.corrcoef((v-v.mean(1,keepdims=True)).ravel(),(g-g.mean(1,keepdims=True)).ravel())[0,1])
        comparison['groups'][mode]['time_demeaned_value_correlation']=residual_corr
        for ax,key in zip(axes,['coverage','discovery_fraction','disconnection_ratio']):
            mean,lo,hi=scene_band(field(key).mean(0))
            ax.plot(np.arange(1,401),mean*100,label=mode,color=COLORS[mi])
            ax.fill_between(np.arange(1,401),lo*100,hi*100,color=COLORS[mi],alpha=.14)
            ax.set(title=key,xlabel='Episode step',ylabel='Percent');ax.legend()
        # Fixed representative scenes, not selected for performance.
        for scene in [10000,10001]:
            path=Path(paths[0]).parent/f'trace_{scene}.npz';sources[str(path)]=digest(path)
            with np.load(path) as d:
                for at in [0,150,200,400]:
                    positions=d['uav_xy'][at];u=len(positions);hops=d['B'][at].min(1);connected=hops<=u
                    counts=d['C'][at].sum(1);coverage=counts.sum()/population
                    relay_dependents=np.maximum(0,(coverage-d['fault_coverage'][at])*population-counts)
                    topologies.append(dict(group=mode,scene_seed=scene,time=at,coverage=float(coverage),
                        sensed=float(d['sensed'][at].any(0).mean()),connected=int(connected.sum()),direct=int((hops==1).sum()),
                        multihop=int(((hops>=2)&connected).sum()),max_hop=int(hops[connected].max()) if connected.any() else 0,
                        connected_hull_km2=hull_area(positions[connected])/1e6,
                        critical_relays=int((relay_dependents>.5).sum())))
    save_figure(fig,out/'figures','coverage_scene_intervals')
    write_csv(out/'stage_diagnosis.csv',stages);write_csv(out/'coverage_accounting.csv',accounting)
    write_csv(out/'critic_diagnosis.csv',critic);write_csv(out/'topology_diagnosis.csv',topologies)
    write_csv(out/'peak_timing.csv',peaks)
    probe=initial_probe(out,comparison);sources[str(probe)]=digest(probe)
    with np.load(probe) as data:
        fig,axes=plt.subplots(1,3,figsize=(13,4));probe_summary={}
        for mi,mode in enumerate(MODES):
            probe_summary[mode]={key:float(data[f'{mode}_{key}'].mean()) for key in
                ['mean_norm','std_mean','command_m','variance_m2','outer90','entropy']}
            bins=data['radius_bins'];hist=data[f'{mode}_radius_histogram'];density=hist/hist.sum()/np.diff(bins)
            axes[0].plot((bins[:-1]+bins[1:])/2,density,label=mode,color=COLORS[mi])
            axes[1].plot(data[f'{mode}_std_mean'].mean(0),'-o',label=mode,color=COLORS[mi],ms=3)
            axes[2].plot(data[f'{mode}_outer90'].mean(0)*100,'-o',label=mode,color=COLORS[mi],ms=3)
        for ax,title,x,y in zip(axes,['Disk action radius: same initial observations','Latent std on own local input','Probability of >27m command'],
                ['Normalized command radius','UAV / actor index','UAV / actor index'],['Density','Gaussian std','Percent']):
            ax.set(title=title,xlabel=x,ylabel=y);ax.legend()
        save_figure(fig,out/'figures','initial_input_randomness')
    atomic_json(out/'initial_probe_summary.json',probe_summary)
    stage=lambda mode,name:next(r for r in stages if r['group']==mode and r['stage']==name)
    row=lambda mode:next(r for r in summary if r['group']==mode and r['checkpoint']=='last')
    lines=['# 独立Actor对照：验证与分析','',
        '两组均完成种子0、100k联合转移、49轮PPO，best均为100k，与last参数相同。冻结模型验证复现原880001日志，并额外使用880002/880003；下表是32个场景、3套动作采样的结果。保留测试集未使用，没有新增训练。','',
        '|指标|共享Actor|独立Actor|','|---|---:|---:|']
    for label,key,percent in [('全回合覆盖','coverage',True),('51–150步覆盖','steps51_150_coverage',True),
        ('后100步覆盖','last100_coverage',True),('终态覆盖','final_coverage',True),('维持差','maintenance',True),
        ('感知率','all_discovery_fraction',True),('断联率','all_disconnection_ratio',True),
        ('实际速度m/s','all_actual_speed_mean',False),('条件位移方差m²','command_variance_m2',False),
        ('近悬停概率','near_hover_probability',True),('微分熵nats','entropy_nats_per_agent',False)]:
        values=[float(row(mode)[key]) for mode in MODES]
        form=lambda x:(f'{100*x:+.2f} pp' if key=='maintenance' else f'{x:.2%}') if percent else f'{x:.3f}'
        lines.append(f'|{label}|{form(values[0])}|{form(values[1])}|')
    lines+=['','## 覆盖提升真实存在，但独立组没有胜出','',
        '两组初始覆盖均为27.23%。880001原选模口径下，训练后共享46.31%、独立40.37%；两者均有学习，并非best仍停在未训练模型。两个best都在预算末端，也不能据此宣布已经收敛。','',
        '|独立−共享|差值|95%配对场景区间|','|---|---:|---|']
    for entry in paired:
        if entry['checkpoint']=='last' and entry['metric'] in ['coverage','last100_coverage','final_coverage','maintenance']:
            lines.append(f'|{entry["metric"]}|{100*float(entry["mean"]):+.2f} pp|[{100*float(entry["low"]):+.2f}, {100*float(entry["high"]):+.2f}] pp|')
    lines+=['','独立组维持差更好，但其前期覆盖也更低。这一项不能单独作为“更会保持”的证据；后100步覆盖仍更低。终态差值区间跨零，不能宣称终态确定更差。区间仅反映场景差异，不是训练种子区间。','',
        '![覆盖与联网曲线及场景区间](figures/coverage_scene_intervals.png)','',
        '## 独立组更分散，回传成为主要服务损失','',
        '|全程诊断|共享Actor|独立Actor|','|---|---:|---:|']
    for label,key in [('感知UE数','discovery_fraction'),('因下行未达标淘汰UE','downlink_rejected_users'),
                      ('因回传不可达淘汰UE','backhaul_rejected_users'),('因容量淘汰UE','capacity_rejected_users')]:
        scale=150 if key=='discovery_fraction' else 1
        lines.append(f'|{label}|{scale*stage("shared","all")[key]:.2f}|{scale*stage("independent","all")[key]:.2f}|')
    s,i=stage('shared','all'),stage('independent','all')
    lines += ['',f'独立组感知贡献增加{100*(i["discovery_fraction"]-s["discovery_fraction"]):+.2f} pp，'
        f'下行淘汰贡献变化{-100*(i["downlink_rejected_users"]-s["downlink_rejected_users"])/150:+.2f} pp，'
        f'回传淘汰贡献变化{-100*(i["backhaul_rejected_users"]-s["backhaul_rejected_users"])/150:+.2f} pp。容量影响接近零。',
        '这是现有顺序判据下的精确人数核算，不是分别改变链路参数的因果实验。它说明“看见更多用户”没有转化为更多联网服务。','',
        '## 共享组依旧存在后期收缩；独立组的平坦不是有效部署保持','']
    for mode in MODES:
        early,late=stage(mode,'early51_150'),stage(mode,'late301_400')
        lines += [f'{mode}：51–150步到后100步，感知率{early["discovery_fraction"]:.2%}→{late["discovery_fraction"]:.2%}，'
            f'断联率{early["disconnection_ratio"]:.2%}→{late["disconnection_ratio"]:.2%}，'
            f'距边界150m内比例{early["boundary_within_150_fraction"]:.2%}→{late["boundary_within_150_fraction"]:.2%}，'
            f'100m内近邻比例{early["uav_close_neighbor_fraction"]:.2%}→{late["uav_close_neighbor_fraction"]:.2%}。','']
    lines+=['共享组在网络逐渐接通的同时感知范围退化，支持“继续运动导致空间服务范围收缩”的解释。独立组聚集更少、间距更大，但断联更多；它没有先形成同样好的覆盖再成功保持下来。','',
        '固定场景10000：共享在t150已达到56.7%覆盖、14架联网；独立只有27.3%、9架联网。t400两者恰好同为56.7%，因此单看终态会掩盖过程差异。',
        '固定场景10001：共享终态15架联网但感知仅38.7%，右边界形成局部密集部署；独立终态感知88.0%却只有10架联网，覆盖28.0%，存在大面积可感知但断网的子网。两图说明两种不同失效形态，不能把代表场景当作全部32场景。','',
        '![共享场景10001](shared/figures/topology_last_10001.png)',
        '![独立场景10001](independent/figures/topology_last_10001.png)','',
        '本次共享组平均曲线的20步平滑峰值窗口中心约为'
        f'{peaks[0]["smooth_window_center"]:.1f}步，独立组约{peaks[1]["smooth_window_center"]:.1f}步。'
        '这不再是两组都在同一时段见顶；独立组后者只是较低平台上的宽峰。原先100–150步现象依赖具体策略，不能视为环境固定时刻的强制事件。',
        '训练进度上覆盖率上升，与同一个400步回合内后期退化可以同时存在：前者说明策略相比初始化有进步，后者说明部署保持仍不足。','',
        '## 熵下降不代表物理动作随机性下降','',
        '独立组微分熵较低，但条件位移方差更大、速度更快、近悬停更少。连续微分熵受分布形状与圆盘变换影响，不能直接替代位移方差。',
        '额外探针只在相同32个初始布局采样动作分布，每状态256次，不执行环境动作：','',
        '|初始局部输入上的已训练策略|共享Actor|独立Actor|','|---|---:|---:|']
    for label,key in [('高斯标准差均值','std_mean'),('平均指令距离m','command_m'),('指令超过27m的概率','outer90')]:
        vals=[probe_summary[m][key] for m in MODES]
        lines.append(f'|{label}|{vals[0]:.4f}|{vals[1]:.4f}|')
    lines+=['','![同初始输入的动作分布](figures/initial_input_randomness.png)','',
        '同输入下独立网络的均值与标准差确实分化，但这没有形成更高的团队覆盖；不能把分化当作成功的中继/服务分工。','',
        '## Critic与优化诊断','',
        '|模型|全程价值RMSE|后100步RMSE|末步RMSE|去时间均值相关性|','|---|---:|---:|---:|---:|']
    for mode in MODES:
        get=lambda stage:next(x for x in critic if x['group']==mode and x['stage']==stage)
        lines.append(f'|{mode}|{get("all")["rmse"]:.3f}|{get("last100")["rmse"]:.3f}|{get("last1")["rmse"]:.3f}|{comparison["groups"][mode]["time_demeaned_value_correlation"]:.3f}|')
    lines+=['','两组平均累计价值曲线较贴近MC，但逐状态误差依旧存在。独立组critic的归一化loss更低，不能据此说预测更准；回报和归一化尺度不同。当前没有对新策略做反事实价值差检验，因此不能宣布critic的动作判断问题已解决。',
        '最后5个日志点：共享KL约0.0094、PPO clipping约10.0%；独立KL约0.0205、clipping约20.3%。独立组单次策略变化更大。同学习率并不保证相同更新幅度，每个独立网络每轮只能利用自己的轨迹，参数量增加也不代表样本效率提高。','',
        '## 后续建议','',
        '本轮不支持把取消actor共享作为当前问题的解决办法。以共享纯覆盖组作为较强基准，保留独立结果用于解释协作与回传损失。',
        '下一步可先在冻结策略下区分连续高斯的均值移动与标准差造成的随机移动，再设计一个控制变量实验；若继续独立actor，应先匹配KL更新幅度或检验更大的有效采样量，而非同时改变多项奖励。',
        '本轮两组best都在100k，无法确认收敛；训练预算可能影响结果。但当前相同预算下共享更好，且双方仍大量移动，不能把部署保持问题简单归因于步数不足。未追加任何训练。','']
    (out/'interpretation.md').write_text('\n'.join(lines))
    sources[str(out/'comparison.json')]=digest(out/'comparison.json')
    sources[str(Path(__file__))]=digest(Path(__file__))
    atomic_json(out/'interpretation_sources.json',sources)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(DEFAULT_OUTPUT))
    args=parser.parse_args();analyze(args.output)
