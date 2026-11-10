"""Compare two saved actor-sharing budgets without collecting trajectories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_mappo_actor_sharing import hull_area, scene_band
from check_mappo_generalization import interval
from compare_online_time_factor import write_csv
from report_mappo_actor_sharing import MODES
from report_mappo_bridge import digest
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json


STAGES = {'all': slice(None), 'early51_150': slice(50, 150),
          'late301_400': slice(300, 400), 'last20': slice(-20, None),
          'terminal': slice(-1, None)}


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def analyze(out, baseline):
    comparison=json.loads((out/'comparison.json').read_text())
    previous=json.loads((baseline/'comparison.json').read_text())
    assert comparison['action_seeds']==previous['action_seeds']
    def budget_label(record):
        steps=record['groups']['shared']['manifest']['env_steps']
        assert steps==record['groups']['independent']['manifest']['env_steps']
        return f'{steps//1000000}M' if steps%1000000==0 else f'{steps/1000:g}k'
    old_label,new_label=budget_label(previous),budget_label(comparison)
    if old_label==new_label:raise ValueError('Expected different training budgets')
    figure_name=f'extension_{old_label}_vs_{new_label}'.lower()
    sources={str(p):digest(p) for p in [out/'comparison.json',baseline/'comparison.json',
        out/'per_scene.csv',baseline/'per_scene.csv',out/'summary.csv',baseline/'summary.csv',Path(__file__)]}
    # The earlier evaluations are frozen artifacts, even though their run
    # directories have subsequently been extended. Never reload those weights.
    summaries={label:table(root/'summary.csv') for label,root in [(old_label,baseline),(new_label,out)]}
    scenes={label:table(root/'per_scene.csv') for label,root in [(old_label,baseline),(new_label,out)]}
    changes=[];checkpoint_changes=[];stages=[];critics=[];topologies=[];peaks=[];datasets={}
    stability=[];stability_summary=[]
    for budget,record in [(old_label,previous),(new_label,comparison)]:
        for mode in MODES:
            for checkpoint in ['best','last']:
                rows=[]
                for path in record['groups'][mode]['checkpoints'][checkpoint]['files']:
                    sources[path]=digest(Path(path))
                    with np.load(path) as d:
                        cov=d['rewards'];seeds=d['scene_seeds'];action_seed=int(Path(path).parent.name.removeprefix('seed_'))
                        smooth=np.stack([np.convolve(c,np.ones(20)/20,'valid') for c in cov])
                        drawdown=(np.maximum.accumulate(smooth,axis=1)-smooth).max(1)
                        peak_gap=smooth.max(1)-cov[:,-100:].mean(1)
                        for i,seed in enumerate(seeds):
                            rows.append(dict(budget=budget,group=mode,checkpoint=checkpoint,
                                scene_seed=int(seed),action_seed=action_seed,
                                max_rolling20_drawdown=float(drawdown[i]),
                                own_peak_to_last100_gap=float(peak_gap[i])))
                stability.extend(rows)
                stability_summary.append(dict(budget=budget,group=mode,checkpoint=checkpoint,
                    rollout_count=len(rows),mean_max_drawdown=float(np.mean([r['max_rolling20_drawdown'] for r in rows])),
                    drawdown_over10pp_fraction=float(np.mean([r['max_rolling20_drawdown']>.1 for r in rows])),
                    drawdown_over20pp_fraction=float(np.mean([r['max_rolling20_drawdown']>.2 for r in rows])),
                    mean_own_peak_to_last100_gap=float(np.mean([r['own_peak_to_last100_gap'] for r in rows]))))
    for mode in MODES:
        old={int(r['scene_seed']):r for r in scenes[old_label] if r['group']==mode and r['checkpoint']=='last'}
        new={int(r['scene_seed']):r for r in scenes[new_label] if r['group']==mode and r['checkpoint']=='last'}
        assert old.keys()==new.keys()
        for key in ['coverage','steps51_150_coverage','last100_coverage','final_coverage','maintenance',
                    'all_discovery_fraction','all_disconnection_ratio','all_boundary_within_150_fraction',
                    'all_uav_close_neighbor_fraction','all_actual_speed_mean','command_variance_m2',
                    'near_hover_probability','t150_links_retention','t200_links_retention',
                    't150_connected_nodes_retention','t200_connected_nodes_retention']:
            delta=np.array([float(new[s][key])-float(old[s][key]) for s in sorted(old)])
            changes.append(dict(group=mode,comparison=f'{new_label}-{old_label}',metric=key,**interval(delta)))
        best={int(r['scene_seed']):r for r in scenes[new_label] if r['group']==mode and r['checkpoint']=='best'}
        assert best.keys()==new.keys()
        for key in ['coverage','steps51_150_coverage','last100_coverage','final_coverage','maintenance',
                    'all_disconnection_ratio','all_actual_speed_mean','command_variance_m2']:
            delta=np.array([float(new[s][key])-float(best[s][key]) for s in sorted(new)])
            checkpoint_changes.append(dict(group=mode,comparison='last-best',metric=key,**interval(delta)))
        for ck in ['best','last']:
            data=[]
            for path in comparison['groups'][mode]['checkpoints'][ck]['files']:
                sources[path]=digest(Path(path))
                with np.load(path) as d:data.append({k:d[k].copy() for k in d.files})
            names=list(data[0]['diagnostic_names']);a=np.stack([d['diagnostics'] for d in data])
            datasets[mode,ck]=data
            field=lambda k:a[...,names.index(k)]
            users=comparison['groups'][mode]['config']['env']['num_users']
            np.testing.assert_allclose(field('coverage'),field('discovery_fraction')-
                sum(field(k) for k in ['downlink_rejected_users','backhaul_rejected_users',
                                      'capacity_rejected_users'])/users,atol=1e-12)
            for stage,sl in STAGES.items():
                stages.append(dict(group=mode,checkpoint=ck,stage=stage,
                    **{k:float(field(k)[:,:,sl].mean()) for k in names}))
            v=np.stack([d['values'][...,0] for d in data]);g=np.stack([d['value_labels'] for d in data])
            h=data[0]['horizon_factors'];error=v-g
            corr=float(np.corrcoef((v-v.mean(1,keepdims=True)).ravel(),
                                  (g-g.mean(1,keepdims=True)).ravel())[0,1])
            for stage,sl in STAGES.items():
                e=error[:,:,sl];std=float(g[:,:,sl].std());rmse=float(np.sqrt((e**2).mean()))
                critics.append(dict(group=mode,checkpoint=ck,stage=stage,rmse=rmse,
                    mae=float(np.abs(e).mean()),bias=float(e.mean()),
                    normalized_rmse=rmse/std if std else np.nan,
                    average_rmse=float(np.sqrt(((e/h[sl])**2).mean())),
                    time_demeaned_correlation=corr))
            curve=field('coverage').mean((0,1));smooth=np.convolve(curve,np.ones(20)/20,mode='valid')
            peaks.append(dict(group=mode,checkpoint=ck,raw_peak_step=int(curve.argmax()+1),
                rolling20_window_end=int(smooth.argmax()+20),peak_coverage=float(smooth.max()),
                last100_coverage=float(curve[-100:].mean()),
                peak_to_last100_gap_pp=float(100*(smooth.max()-curve[-100:].mean()))))
            for scene in [10000,10001]:
                path=Path(comparison['groups'][mode]['checkpoints'][ck]['files'][0]).parent/f'trace_{scene}.npz'
                sources[str(path)]=digest(path)
                with np.load(path) as d:
                    for t in [0,150,200,400]:
                        xy=d['uav_xy'][t];u=len(xy);hops=d['B'][t].min(1);connected=hops<=u
                        dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
                        np.fill_diagonal(dist,np.inf)
                        edge=np.minimum(xy,np.array([3500,3500])-xy).min(1)
                        topologies.append(dict(group=mode,checkpoint=ck,scene_seed=scene,time=t,
                            coverage=float(d['C'][t].sum()/users),sensed=float(d['sensed'][t].any(0).mean()),
                            connected=int(connected.sum()),direct=int((hops==1).sum()),
                            multihop=int(((hops>=2)&connected).sum()),
                            max_hop=int(hops[connected].max()) if connected.any() else 0,
                            connected_hull_km2=hull_area(xy[connected])/1e6,
                            boundary150_fraction=float((edge<150).mean()),
                            close100_fraction=float((dist.min(1)<100).mean())))
    for name,rows in [('extension_paired_differences.csv',changes),('stage_diagnosis.csv',stages),
                      ('best_last_paired_differences.csv',checkpoint_changes),('critic_diagnosis.csv',critics),
                      ('topology_diagnosis.csv',topologies),('peak_timing.csv',peaks),
                      ('trajectory_stability.csv',stability),('trajectory_stability_summary.csv',stability_summary)]:
        write_csv(out/name,rows)
    style();fig,axes=plt.subplots(2,3,figsize=(14,8))
    fields=['coverage','discovery_fraction','disconnection_ratio',
            'boundary_within_150_fraction','uav_close_neighbor_fraction','actual_speed_mean']
    for index,mode in enumerate(MODES):
        for label,record in [(old_label,previous),(new_label,comparison)]:
            paths=record['groups'][mode]['checkpoints']['last']['files'];curves=[]
            for path in paths:
                sources[path]=digest(Path(path))
                with np.load(path) as d:
                    names=list(d['diagnostic_names']);curves.append(d['diagnostics'][...,[names.index(k) for k in fields]])
            curve=np.mean(curves,axis=0)
            for j,ax in enumerate(axes.flat):
                scale=1 if fields[j]=='actual_speed_mean' else 100
                ax.plot(np.arange(1,401),curve[:,:,j].mean(0)*scale,
                        label=f'{mode} {label}',color=COLORS[index],linestyle='--' if label==old_label else '-')
                ax.set(title=fields[j],xlabel='Episode step',ylabel='m/s' if scale==1 else '%');ax.legend(fontsize=7)
    save_figure(fig,out/'figures',figure_name)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for index,mode in enumerate(MODES):
        for checkpoint in ['best','last']:
            data=datasets[mode,checkpoint];names=list(data[0]['diagnostic_names'])
            curves=np.mean([d['diagnostics'] for d in data],axis=(0,1))
            step=comparison['groups'][mode]['checkpoints'][checkpoint]['step']
            for ax,key in zip(axes.flat,fields):
                scale=1 if key=='actual_speed_mean' else 100
                ax.plot(np.arange(1,401),curves[:,names.index(key)]*scale,
                    label=f'{mode} {checkpoint} ({step/1000:g}k)',color=COLORS[index],
                    linestyle='--' if checkpoint=='best' else '-')
                ax.set(title=key,xlabel='Episode step',ylabel='m/s' if scale==1 else '%')
                ax.legend(fontsize=7)
    save_figure(fig,out/'figures','best_vs_last_deployment')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i,mode in enumerate(MODES):
        data=datasets[mode,'last'];names=list(data[0]['diagnostic_names'])
        for ax,key in zip(axes,['coverage','discovery_fraction','disconnection_ratio']):
            curves=np.mean([d['diagnostics'][...,names.index(key)] for d in data],axis=0)
            mean,lo,hi=scene_band(curves)
            ax.plot(np.arange(1,401),mean*100,label=mode,color=COLORS[i])
            ax.fill_between(np.arange(1,401),lo*100,hi*100,color=COLORS[i],alpha=.15)
            ax.set(title=key,xlabel='Episode step',ylabel='%');ax.legend()
    save_figure(fig,out/'figures','coverage_scene_intervals')
    row=lambda label,mode:next(r for r in summaries[label] if r['group']==mode and r['checkpoint']=='last')
    stage=lambda mode,name:next(r for r in stages if r['group']==mode and r['checkpoint']=='last' and r['stage']==name)
    manifest=comparison['groups']['shared']['manifest']
    lines=[f'# {new_label}共享／独立Actor验证与续训分析','',
        f'两组均累计完成{manifest["env_steps"]:,}条联合转移、{manifest["updates"]}轮PPO，模型冻结后按32个相同验证场景和三套动作采样种子（880001–880003）评估。保留测试集未使用，没有新增训练。',
        '共享组在147,104步从完整检查点恢复到新目录；曲线只继承该恢复点之前的原日志，再拼接新分支，排除已经回退的后续记录。',
        f'{old_label}结果读取此前保存的审计数据，没有使用已被续训覆盖的旧模型文件。','',
        '|组别|预算|全回合覆盖|51–150步|后100步|终态|维持差|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for label in [old_label,new_label]:
            r=row(label,mode)
            lines.append(f'|{mode}|{label}|'+ '|'.join(f'{100*float(r[k]):+.2f} pp' if k=='maintenance' else f'{float(r[k]):.2%}' for k in
                ['coverage','steps51_150_coverage','last100_coverage','final_coverage','maintenance'])+'|')
    lines+=['',f'![{old_label}与{new_label}的回合内比较](figures/{figure_name}.png)','',
        '![当前预算best与last部署表现](figures/best_vs_last_deployment.png)','',
        f'|同组{new_label}−{old_label}|全回合覆盖变化及95%区间|后100步变化及95%区间|','|---|---:|---:|']
    for mode in MODES:
        vals=[]
        for key in ['coverage','last100_coverage']:
            r=next(r for r in changes if r['group']==mode and r['metric']==key)
            vals.append(f'{r["mean"]*100:+.2f} pp [{r["low"]*100:+.2f}, {r["high"]*100:+.2f}]')
        lines.append(f'|{mode}|'+ '|'.join(vals)+'|')
    lines+=['','区间为先平均动作采样重复、再进行2000次配对场景bootstrap（种子950001）的结果，不是训练种子置信区间。',
        '维持差必须结合早期和后期绝对覆盖解读；单独变平、移动减少或终态改善，不能证明同一有效部署保持不变。','',
        '|当前预算模型|20步平滑峰值|峰值窗口结束步|后100步|峰值减后100步|',
        '|---|---:|---:|---:|---:|']
    for r in peaks:
        lines.append(f'|{r["group"]} {r["checkpoint"]}|{r["peak_coverage"]:.2%}|{r["rolling20_window_end"]}|'
                     f'{r["last100_coverage"]:.2%}|{r["peak_to_last100_gap_pp"]:.2f} pp|')
    lines+=['','平滑峰值基于跨场景、跨动作重复的平均曲线；窗口宽20步。它补充早期−后期指标，避免早期仍在展开掩盖真正的峰值后下降。','',
        '|单条轨迹稳定性（last）|平均最大回落|回落超过10pp的比例|回落超过20pp的比例|',
        '|---|---:|---:|---:|']
    for r in stability_summary:
        if r['checkpoint']=='last':
            lines.append(f'|{r["group"]} {r["budget"]}|{r["mean_max_drawdown"]*100:.2f} pp|'
                         f'{r["drawdown_over10pp_fraction"]:.2%}|{r["drawdown_over20pp_fraction"]:.2%}|')
    lines+=['','最大回落先对每条400步轨迹作20步均值，再计算任一先前峰值到其后低点的最大差值；后续恢复也仍计入回落。',
        '10/20个百分点阈值仅作探索性描述，比例单位是32场景×3动作重复的轨迹，不是96个独立训练样本。平均曲线变平不等于每条轨迹都保持稳定。','',
        '|当前预算last−best|全回合覆盖变化及95%区间|后100步变化及95%区间|','|---|---:|---:|']
    for mode in MODES:
        vals=[]
        for key in ['coverage','last100_coverage']:
            r=next(r for r in checkpoint_changes if r['group']==mode and r['metric']==key)
            vals.append(f'{r["mean"]*100:+.2f} pp [{r["low"]*100:+.2f}, {r["high"]*100:+.2f}]')
        lines.append(f'|{mode}|'+ '|'.join(vals)+'|')
    lines+=['','best由选模验证集选择；这些区间是同一验证集的诊断，不能视为独立测试集上的无偏选模检验。','',
        f'|{new_label} last物理诊断|共享|独立|','|---|---:|---:|']
    for label,key,scale in [('感知率','discovery_fraction',100),('断联率','disconnection_ratio',100),
                            ('150m内靠边比例','boundary_within_150_fraction',100),
                            ('100m内近邻比例','uav_close_neighbor_fraction',100),
                            ('实际速度m/s','actual_speed_mean',1),('下行淘汰人数','downlink_rejected_users',1),
                            ('回传淘汰人数','backhaul_rejected_users',1),('容量淘汰人数','capacity_rejected_users',1)]:
        lines.append(f'|{label}|'+ '|'.join(f'{stage(m,"all")[key]*scale:.3f}'+ ('%' if scale==100 else '') for m in MODES)+'|')
    lines+=['','服务人数严格等于：感知人数−下行淘汰−回传淘汰−容量淘汰。这是当前判定次序下的核算，不是独立改变通信参数的因果实验。','',
        f'|动作随机性（{new_label} last）|共享|独立|','|---|---:|---:|']
    for label,key in [('条件位移总方差m²','command_variance_m2'),('近悬停概率','near_hover_probability'),
                      ('每机微分熵nats','entropy_nats_per_agent')]:
        lines.append(f'|{label}|'+ '|'.join(f'{float(row(new_label,m)[key]):.6f}' for m in MODES)+'|')
    lines+=['','条件方差是在同一局部输入上反复采样的二维指令位移方差之和，每20步64次隔离采样；不是不同UAV的位置方差。连续微分熵不能直接等同物理动作方差。','',
        f'|{new_label} last critic|全程RMSE|末20步RMSE|末步RMSE|全程归一化RMSE|去时间均值相关性|',
        '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        get=lambda name:next(r for r in critics if r['group']==mode and r['checkpoint']=='last' and r['stage']==name)
        a=get('all')
        lines.append(f'|{mode}|{a["rmse"]:.3f}|{get("last20")["rmse"]:.3f}|{get("terminal")["rmse"]:.3f}|{a["normalized_rmse"]:.3f}|{a["time_demeaned_correlation"]:.3f}|')
    lines+=['','critic标签使用完整纯覆盖MC，不含actor熵。两组各自在自己的on-policy状态分布上评估；平均预测曲线贴近不能证明动作价值差已经正确。本轮没有追加反事实动作价值实验。','',
        '## 本轮结论','',
        '下述结论按同预算last和保存的旧预算last比较，不用best替代last；best的完整结果见report.md。','']
    for mode in MODES:
        r=next(r for r in changes if r['group']==mode and r['metric']=='coverage')
        lines.append(f'{mode}从{old_label}到{new_label}平均覆盖变化{r["mean"]*100:+.2f}个百分点，'
            f'配对场景区间[{r["low"]*100:+.2f}, {r["high"]*100:+.2f}]个百分点。')
    delta=(float(row(new_label,'independent')['coverage'])-float(row(new_label,'shared')['coverage']))*100
    lines += [f'本预算独立−共享的平均覆盖差为{delta:+.2f}个百分点。', '']
    for mode in MODES:
        early,late=stage(mode,'early51_150'),stage(mode,'late301_400')
        r=row(new_label,mode)
        lines.append(f'{mode}：早期覆盖{early["coverage"]:.2%}→后100步{late["coverage"]:.2%}，'
            f'感知{early["discovery_fraction"]:.2%}→{late["discovery_fraction"]:.2%}，'
            f'断联{early["disconnection_ratio"]:.2%}→{late["disconnection_ratio"]:.2%}，'
            f'150m内靠边{early["boundary_within_150_fraction"]:.2%}→{late["boundary_within_150_fraction"]:.2%}，'
            f'100m内近邻{early["uav_close_neighbor_fraction"]:.2%}→{late["uav_close_neighbor_fraction"]:.2%}。')
        lines.append(f'其t150既有链路后续平均保留率为{float(r["t150_links_retention"]):.2%}，'
            f'既有联网节点后续联网保留率为{float(r["t150_connected_nodes_retention"]):.2%}。'
            '前者要求同一条边继续存在，后者允许路由改变；二者都不能代替UE服务覆盖。')
        lines.append('')
    lines+=['部署维持应同时检查早期和后期绝对覆盖、感知、断联及靠边变化。既有链路替换也可能保持服务，不能将拓扑完全静止作为唯一成功标准。','',
        '|最后5次训练日志均值|共享|独立|','|---|---:|---:|']
    for key in ['value_loss','actor_loss','joint_entropy','ppo_approx_kl','ppo_clip_fraction']:
        values=[]
        for mode in MODES:
            logs=comparison['groups'][mode]['logs']['metrics.csv'][-5:]
            xs=[float(r[key]) for r in logs if r.get(key)]
            values.append(float(np.mean(xs)) if xs else np.nan)
        lines.append(f'|{key}|{values[0]:.6f}|{values[1]:.6f}|')
    lines+=['','低且平稳的归一化critic loss不等于策略已收敛；其监督目标与归一化尺度会随策略变化。训练coverage日志是最后采集步快照，应以完整验证回合曲线判断任务表现。',
        'best步数分别为：'+ '，'.join(f'{mode} {comparison["groups"][mode]["checkpoints"]["best"]["step"]:,}' for mode in MODES)+
        f'；last均为{manifest["env_steps"]:,}。收敛与持续退化需要结合完整验证曲线，不能只看最后两个点。',
        '本轮没有改变采样温度、训练参数或追加训练；后续诊断应区分运动均值、采样标准差与价值差估计。','',
        '固定场景10000、10001的初始／150／200／400步图：','',
        '![共享10000](shared/figures/topology_last_10000.png)',
        '![独立10000](independent/figures/topology_last_10000.png)',
        '![共享10001](shared/figures/topology_last_10001.png)',
        '![独立10001](independent/figures/topology_last_10001.png)','',
        '全部best/last、逐场景差值、拓扑统计与来源哈希见report.md、paired_differences.csv、extension_paired_differences.csv、topology_diagnosis.csv和extension_sources.json。','']
    (out/'interpretation.md').write_text('\n'.join(lines))
    assert all(digest(Path(p))==h for p,h in sources.items())
    atomic_json(out/'extension_sources.json',sources)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('analysis/mappo_actor_sharing_coverage_400k_seed0'))
    p.add_argument('--baseline',type=Path,default=Path('analysis/mappo_actor_sharing_coverage_100k_seed0'))
    args=p.parse_args();analyze(args.output,args.baseline)
