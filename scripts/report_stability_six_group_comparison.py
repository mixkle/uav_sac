"""Saved-data comparison: three 1M reward ablations and three 10M references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_research.coverage_stability import REWARD_ABLATIONS, coverage_drop_metrics, run_path

REFS=('baseline_best','dppo_best','teacher_topk')
NEW=tuple(REWARD_ABLATIONS)
GROUPS=REFS+NEW
NAMES=dict(baseline_best='Baseline 10M best',dppo_best='DPPO 10M best',
    teacher_topk='Baseline 10M teacher',disconnection='1M + disconnection',
    robustness='1M + robustness',combined='1M + both')
SHORT=('Baseline','DPPO','Teacher','Disc. 1M','Rob. 1M','Both 1M')
COLORS=dict(baseline_best='#D62728',dppo_best='#0072B2',teacher_topk='#E69F00',
    disconnection='#222222',robustness='#56B4E9',combined='#777777')
LINES=dict(baseline_best='-',dppo_best='--',teacher_topk='-.',
    disconnection=(0,(6,2)),robustness=':',combined=(0,(4,1,1,1)))
MARKERS=dict(baseline_best='o',dppo_best='s',teacher_topk='^',disconnection='x',robustness='v',combined='D')
SCENES=list(range(10000,10032))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_evaluation(out,group,kind,sources):
    from uav_research.config import config_from_dict
    if group in REFS:
        p=out/'references'/group
        prov=json.loads((p/'provenance.json').read_text())
        assert prov['scene_seeds']==SCENES and prov['action_seed']==880001 and prov['device']=='cuda:0'
        with np.load(p/'step_metrics.npz') as z:
            assert z['scene_seeds'].tolist()==SCENES and z['metrics'].shape[:2]==(32,400)
            s=dict(zip(z['names'].tolist(),z['metrics'].transpose(2,0,1)))
        weights=dict(disconnection_weight=0.,robustness_weight=0.)
        training_budget=10_000_000
    else:
        p=out/kind/group
        prov=json.loads((p/'stability_source.json').read_text())
        identity=prov['evaluation']
        assert identity['scenes']==SCENES and identity['action_seed']==880001
        assert identity['device']=='cuda:0' and identity['batch_size']==16
        assert digest(p/'reward_curves.npz')==prov['reward_curves_sha256']
        cfg=config_from_dict(json.loads((p/'evaluation_config.json').read_text()))
        assert cfg.env.horizon==400 and cfg.env.capacity==10 and cfg.env.num_uavs==18
        assert cfg.run.total_steps==1_000_000
        weights=REWARD_ABLATIONS[group]
        assert all(getattr(cfg.env,k)==v for k,v in weights.items())
        with np.load(p/'reward_curves.npz') as z:
            assert z['scene_seeds'].tolist()==SCENES and z['raw'].shape==(32,400,4)
            raw=z['raw'].copy()
            np.testing.assert_allclose(z['reward_weights'],[1,.1,weights['robustness_weight'],-weights['disconnection_weight']])
            np.testing.assert_allclose(z['contributions'],raw*z['reward_weights'],atol=1e-12)
            np.testing.assert_allclose(z['reward'],z['contributions'].sum(-1),atol=1e-12)
            s=dict(coverage=raw[:,:,0],tau=raw[:,:,1],robustness=raw[:,:,2],disconnection_ratio=raw[:,:,3],reward=z['reward'].copy())
            s.update(dict(zip(z['diagnostic_names'].tolist(),z['diagnostics'].transpose(2,0,1))))
        training_budget=1_000_000
    source=sources[group] if group in REFS else sources[group][kind]
    assert digest(source['checkpoint'])==source['sha256']
    assert (prov['sha256'] if group in REFS else prov['evaluation']['sha256'])==source['sha256']
    assert all(np.isfinite(v).all() for v in s.values())
    rho,tau,j,d=[s[k] for k in ('coverage','tau','robustness','disconnection_ratio')]
    np.testing.assert_allclose(s['reward'],rho+.1*tau+weights['robustness_weight']*j-weights['disconnection_weight']*d,atol=2e-7)
    assert (j>=0).all() and (j<=(17/18)*rho+1e-12).all()
    s['common_reward']=rho+.1*tau
    s['throughput_bps']=tau*750e6
    rows=[]
    for i,scene in enumerate(SCENES):
        row=dict(group=group,scene_seed=scene,**coverage_drop_metrics(rho[i]))
        row.update(throughput_bps=float(s['throughput_bps'][i].mean()),
            last100_throughput_bps=float(s['throughput_bps'][i,-100:].mean()),
            common_return=float(s['common_reward'][i].sum()),training_return=float(s['reward'][i].sum()),
            robustness=float(j[i].mean()),retention_ratio=float(j[i].sum()/rho[i].sum()) if rho[i].sum() else 0.,
            disconnection_ratio=float(d[i].mean()),final_disconnection_ratio=float(d[i,-1]),
            discovery_fraction=float(s['discovery_fraction'][i].mean()),
            reflection_fraction=float(s['boundary_reflection_fraction'][i].mean()),
            boundary150_fraction=float(s['boundary_within_150_fraction'][i].mean()))
        drop=np.maximum(0,rho[i,:-1]-rho[i,1:]);events=drop>=.1-1e-12
        net_disconnect_increase=np.diff(d[i])>1e-12
        row.update(large_drop_with_net_disconnect_rise=int((events&net_disconnect_increase).sum()),
                   largest_drop_time=int(drop.argmax()+2),
                   largest_drop_net_disconnect_change=float(np.diff(d[i])[drop.argmax()]))
        rows.append(row)
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('group','scene_seed')}
    summary['max_drop_1_scene_p95']=float(np.quantile([r['max_drop_1'] for r in rows],.95))
    summary['curve_peak_time']=int(rho.mean(0).argmax()+1)
    summary['curve_peak_coverage']=float(rho.mean(0).max())
    summary['final_below30_scenes']=int((rho[:,-1]<.3).sum())
    summary['training_budget']=training_budget
    summary['checkpoint_step']=source['env_steps']
    summary['pooled_served_rate_bps']=summary['throughput_bps']/(150*summary['coverage']) if summary['coverage'] else None
    count=sum(r['drop_events_10pp'] for r in rows)
    summary['drop_with_net_disconnect_rise_fraction']=sum(r['large_drop_with_net_disconnect_rise'] for r in rows)/count if count else None
    if group in NEW:
        records=read(Path(run_path(group))/'validation.csv')
        record=next(x for x in records if int(x['env_steps'])==source['env_steps'])
        for key,target in [('coverage','coverage'),('return','training_return'),('throughput_bps','throughput_bps')]:
            np.testing.assert_allclose(summary[target],float(record[key]),atol=1e-6,rtol=0)
    return s,summary,rows,p


def main(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style,save_figure

    out=args.output
    meta=json.loads((out/'report_sources.json').read_text())
    assert meta['groups']==list(GROUPS)
    for path,sha in meta['input_files'].items():assert digest(out/path)==sha,f'Changed input: {path}'
    for path,sha in meta.get('run_log_sources',{}).items():assert digest(path)==sha,f'Changed training log: {path}'
    series,summaries,per_scene,paths={},{},{},{}
    for group in GROUPS:
        series[group],summaries[group],per_scene[group],paths[group]=load_evaluation(out,group,'best',meta['sources'])
    supplemental={g:load_evaluation(out,g,'last',meta['sources']) for g in NEW}
    draws=np.random.default_rng(950001).integers(32,size=(2000,32))
    paired=[]
    for group in NEW:
        for ref in REFS:
            for key in ('coverage','last100_coverage','throughput_bps','common_return','max_drop_1','max_drop_5',
                        'drop_events_10pp','min20_coverage','disconnection_ratio','robustness','maintain_coverage'):
                delta=np.array([a[key]-b[key] for a,b in zip(per_scene[group],per_scene[ref])])
                lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
                paired.append(dict(group=group,reference=ref,metric=key,difference=float(delta.mean()),low=float(lo),high=float(hi)))
    cases={}
    abrupt_cases={}
    for group in GROUPS:
        cases[group]={}
        abrupt_cases[group]={}
        for scene in (10000,10001):
            with np.load(paths[group]/f'trace_{scene}.npz') as z:
                c=z['C'];feasible=z['sensed']&(z['Y']>=-3)&(z['B'].min(2)[:,:,None]<=18)
                assert (c.sum(1)<=1).all() and (c.sum(2)<=10).all() and not (c.astype(bool)&~feasible).any()
                np.testing.assert_allclose(c.sum((1,2))/150,z['coverage'])
                cases[group][str(scene)]=[dict(time=t,coverage=float(z['coverage'][t]),connected=int((z['B'][t].min(1)<=18).sum()),
                    seen=int(z['sensed'][t].any(0).sum()),eligible=int(feasible[t].any(0).sum())) for t in (0,150,200,399,400)]
                # Diagnostic frame selection is the largest drop within each
                # predeclared scene; not a newly selected validation scene.
                t=int(np.argmax(z['coverage'][1:-1]-z['coverage'][2:])+2)
                linked_before=z['B'][t-1].min(1)<=18;linked_after=z['B'][t].min(1)<=18
                served_before=c[t-1].any(0);served_after=c[t].any(0)
                lost=served_before&~served_after;owner=c[t-1].argmax(0)
                a,b=z['A'][t-1].astype(bool),z['A'][t].astype(bool)
                abrupt_cases[group][str(scene)]=dict(time=t,
                    coverage_before=float(z['coverage'][t-1]),coverage_after=float(z['coverage'][t]),
                    connected_before=int(linked_before.sum()),connected_after=int(linked_after.sum()),
                    newly_disconnected_ids=np.flatnonzero(linked_before&~linked_after).tolist(),
                    lost_users=int(lost.sum()),gained_users=int((~served_before&served_after).sum()),
                    lost_users_whose_previous_server_disconnected=int((lost&~linked_after[owner]).sum()),
                    removed_uu_edges=np.argwhere(np.triu(a&~b,1)).tolist(),
                    added_uu_edges=np.argwhere(np.triu(~a&b,1)).tolist(),
                    removed_ug_edges=np.argwhere((z['B'][t-1]==1)&(z['B'][t]!=1)).tolist(),
                    pre_failure_mean=float(z['fault_coverage'][t-1].mean()),
                    pre_failure_worst=float(z['fault_coverage'][t-1].min()))
    for scene in (10000,10001):
        with np.load(paths['baseline_best']/f'trace_{scene}.npz') as z:
            initial=z['uav_xy'][0].copy();ue_path=z['user_xy'].copy()
        for group in GROUPS[1:]:
            with np.load(paths[group]/f'trace_{scene}.npz') as z:
                np.testing.assert_array_equal(initial,z['uav_xy'][0]);np.testing.assert_array_equal(ue_path,z['user_xy'])
    result=dict(groups=list(GROUPS),summary=summaries,paired=paired,
        supplemental_last={g:x[1] for g,x in supplemental.items()},fixed_scenes=cases,abrupt_fixed_cases=abrupt_cases,palette=COLORS,
        uncertainty='2000 paired validation-scene bootstrap draws; seed 950001; not training-seed confidence')
    (out/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    write_csv(out/'best_scene_metrics.csv',[r for g in GROUPS for r in per_scene[g]])
    write_csv(out/'paired_scene_differences.csv',paired)
    write_csv(out/'new_groups_last_scene_metrics.csv',[r for g in NEW for r in supplemental[g][2]])
    style();plt.rcParams['axes.prop_cycle']=plt.cycler(color=[COLORS[g] for g in GROUPS]);figdir=out/'figures'
    def line(ax,group,y):
        ax.plot(np.arange(1,len(y)+1),y,color=COLORS[group],ls=LINES[group],marker=MARKERS[group],
                markevery=45,markersize=3,lw=1.5,label=NAMES[group])
    for filename,panels in [
        ('task_curves',[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Total throughput (Mbps)'),('common_reward',1,'Common reward: coverage + 0.1 tau'),('discovery_fraction',100,'Instantaneous sensing (%)')]),
        ('deployment',[('disconnection_ratio',100,'Disconnected UAV (%)'),('robustness',100,'Single-failure retained coverage (%)'),('boundary_within_150_fraction',100,'Within 150 m of edge (%)'),('boundary_reflection_fraction',100,'Boundary reflection (%)')])]:
        fig,axes=plt.subplots(2,2,figsize=(13,7.5))
        for ax,(key,scale,label) in zip(axes.flat,panels):
            for group in GROUPS:line(ax,group,series[group][key].mean(0)*scale)
            ax.set(xlabel='Episode step (post-action)',ylabel=label)
        axes.flat[0].legend(fontsize=7,ncol=2);save_figure(fig,figdir,filename)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for ax,key,scale,label in zip(axes,('max_drop_1','max_drop_5','drop_events_10pp'),(100,100,1),
             ('Mean episode maximum 1-step drop (pp)','Mean episode maximum 5-step drop (pp)','Single-step drops >=10 pp / episode')):
        for i,g in enumerate(GROUPS):
            values=np.array([r[key]*scale for r in per_scene[g]]);mean=values.mean()
            lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
            ax.bar(i,mean,color=COLORS[g]);ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],color='black',capsize=3)
        ax.set_xticks(range(6),SHORT,rotation=35,ha='right');ax.set(ylabel=label)
    fig.suptitle('Best models; error bars: validation-scene bootstrap; budgets differ')
    save_figure(fig,figdir,'coverage_drops')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for g in GROUPS:
        for ax,key in zip(axes,('max_drop_1','min20_coverage')):
            v=np.sort([r[key]*100 for r in per_scene[g]])
            ax.plot(v,np.arange(1,33)/32,color=COLORS[g],ls=LINES[g],label=NAMES[g])
    axes[0].set(xlabel='Episode maximum single-step drop (pp)',ylabel='Scene CDF');axes[1].set(xlabel='Episode minimum 20-step coverage (%)',ylabel='Scene CDF')
    axes[0].legend(fontsize=7);save_figure(fig,figdir,'stability_distribution')
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for ax,key,scale,label in zip(axes,('coverage','last100_coverage','throughput_bps'),(100,100,1e-6),('Mean coverage (%)','Last-100 coverage (%)','Throughput (Mbps)')):
        for i,g in enumerate(GROUPS):
            values=np.array([r[key]*scale for r in per_scene[g]]);mean=values.mean()
            lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
            ax.bar(i,mean,color=COLORS[g]);ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],color='black',capsize=3)
        ax.set_xticks(range(6),SHORT,rotation=35,ha='right');ax.set(ylabel=label)
    save_figure(fig,figdir,'performance')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for g in NEW:
        records=read(Path(run_path(g))/'validation.csv');x=np.array([int(r['env_steps'])/1e6 for r in records])
        common=np.array([400*(float(r['coverage'])+.1*float(r['throughput_bps'])/750e6) for r in records])
        for ax,y,label in [(axes[0,0],common,'Validation common episode return'),(axes[0,1],np.array([float(r['coverage'])*100 for r in records]),'Validation coverage (%)'),
                           (axes[1,0],np.array([float(r['return']) for r in records]),'Own training-objective validation return'),(axes[1,1],np.array([float(r['disconnection_ratio'])*100 for r in records]),'Validation disconnected UAV (%)')]:
            ax.plot(x,y,color=COLORS[g],ls=LINES[g],label=NAMES[g]);ax.set(xlabel='Online joint transitions (millions)',ylabel=label)
    axes[0,0].legend(fontsize=7);fig.suptitle('New 1M runs; own returns have different definitions')
    save_figure(fig,figdir,'new_runs_validation')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for g in NEW:
        records=read(Path(run_path(g))/'metrics.csv');x=np.array([int(r['env_steps'])/1e6 for r in records])
        for ax,key in zip(axes.flat,('actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction')):
            y=np.array([float(r[key]) for r in records]);n=10
            assert np.isfinite(y).all()
            ax.plot(x[n-1:],np.convolve(y,np.ones(n)/n,'valid'),color=COLORS[g],ls=LINES[g],label=NAMES[g]);ax.set(xlabel='Joint transitions (millions)',ylabel=key)
    axes[0,0].legend(fontsize=7);save_figure(fig,figdir,'new_runs_loss')
    for scene in (10000,10001):
        i=SCENES.index(scene)
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        for g in GROUPS:
            line(axes[0],g,series[g]['coverage'][i]*100)
            line(axes[1],g,series[g]['disconnection_ratio'][i]*100)
        axes[0].set(ylabel='Coverage (%)');axes[1].set(ylabel='Disconnected UAV (%)',xlabel='Episode step')
        axes[0].legend(fontsize=7,ncol=2);fig.suptitle(f'Fixed scene {scene}: raw traces, no temporal smoothing')
        save_figure(fig,figdir,f'fixed_scene_events_{scene}')
    for scene in (10000,10001):
        fig,axes=plt.subplots(6,4,figsize=(16,20))
        for row,g in enumerate(GROUPS):
            with np.load(paths[g]/f'trace_{scene}.npz') as z:
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,users,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions']
                    seen,served=z['sensed'][t].any(0),c.any(0)
                    for mask,marker,col in [(~seen,'.','#aaaaaa'),(seen&~served,'x','#E69F00'),(served,'o',COLORS[g])]:
                        ax.scatter(users[mask,0],users[mask,1],c=col,marker=marker,s=9,linewidths=.6)
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.55,alpha=.6)
                        for k in range(3):
                            if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],'--',color='#555555',lw=.7)
                        for m in np.flatnonzero(c[i]):ax.plot([xy[i,0],users[m,0]],[xy[i,1],users[m,1]],color=COLORS[g],lw=.4,alpha=.18)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=30,edgecolors='black',linewidths=.4)
                    ax.scatter(bs[:,0],bs[:,1],marker='*',c='black',s=85,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[g]} | t={t} | {served.mean():.1%}',xlabel='x (m)',ylabel='y (m)')
        handles=[Line2D([],[],ls='none',marker=m,color=c,label=l) for m,c,l in [('o','#333333','Served UE (group color)'),('x','#E69F00','Sensed only'),('.','#aaaaaa','Unseen'),('^','#333333','Connected UAV'),('s','#333333','Disconnected UAV'),('*','black','BS')]]
        fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8,bbox_to_anchor=(.5,-.005));fig.suptitle(f'Fixed validation scene {scene}: six best policies')
        save_figure(fig,figdir,f'topology_best_{scene}')
    write_report(out,result)
    print(json.dumps(dict(output=str(out),summary={g:{k:s[k] for k in ('coverage','max_drop_1','drop_events_10pp','throughput_bps')} for g,s in summaries.items()})))


def write_report(out,result):
    s=result['summary'];lines=['# 稳定性奖励三组与10M三组：六组best对照','',
        '本报告主图只比较六组best。新增三组各训练1M；原baseline、DPPO各训练10M，teacher冻结baseline 10M best，没有独立训练。新增三组last仅作补充表，不增加主图实验组。',
        '统一32验证场景10000–10031、400步真正终止、采样种子880001、批次16、CUDA FP32。保留测试集未使用。新增best/last复验与各自训练日志一致。','',
        '训练目标：原三组为rho＋0.1tau；新增组分别加−0.1D、＋0.5J、＋0.5J−0.1D。主表共同回报始终按rho＋0.1tau计算，不能把含鲁棒奖励的较大训练回报直接视为更好的服务。',
        '新增best按各自训练目标选择，原best按原任务回报选择；单训练种子、验证集选模。预算、选模目标及训练机制不同，六组比较不能单独识别奖励项的因果贡献。','',
        '## 主要任务表现','', '| 组别 | best实际步数 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 共同回报 |', '|---|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        x=s[g];lines.append(f"| {NAMES[g]} | {x['checkpoint_step']:,} | {x['coverage']:.3%} | {x['last100_coverage']:.3%} | {x['final_coverage']:.3%} | {x['throughput_bps']/1e6:.2f} | {x['common_return']:.3f} |")
    lines+=['','## 主要发现','',
        f"鲁棒性组best平均覆盖{s['robustness']['coverage']:.3%}，与DPPO的{s['dppo_best']['coverage']:.3%}接近；但吞吐分别为{s['robustness']['throughput_bps']/1e6:.2f}与{s['dppo_best']['throughput_bps']/1e6:.2f}Mbps。不能只用覆盖或加入J后的训练回报判断优劣。",
        f"新增三组best相对baseline 10M的最大单步跌幅均值变化分别为{100*(s['disconnection']['max_drop_1']-s['baseline_best']['max_drop_1']):+.2f}、{100*(s['robustness']['max_drop_1']-s['baseline_best']['max_drop_1']):+.2f}、{100*(s['combined']['max_drop_1']-s['baseline_best']['max_drop_1']):+.2f}个百分点；当前六组比较中，没有看到新增项让该严重度指标低于baseline 10M。预算不同，不能直接解释为奖励项导致恶化。",
        f"断联惩罚组和鲁棒性组平均断联比例分别为{s['disconnection']['disconnection_ratio']:.2%}、{s['robustness']['disconnection_ratio']:.2%}，低于baseline的{s['baseline_best']['disconnection_ratio']:.2%}；与此同时严重覆盖损失仍存在。平均联网与服务尾部稳定性不是同一个目标。",
        f"鲁棒性组绝对J为{s['robustness']['robustness']:.3%}，baseline为{s['baseline_best']['robustness']:.3%}；相应保留比例为{s['robustness']['retention_ratio']:.3%}与{s['baseline_best']['retention_ratio']:.3%}。绝对J提高没有表现为更高的相对服务保留。"]
    lines+=['','## 覆盖骤降','',
        '从动作后t1–t400计算，不含reset→首动作。先逐场景计算最大跌幅，再跨场景平均；不是对平均覆盖曲线求跌幅。5步是端点差；阈值10个百分点另附5/20个百分点敏感性数据。','',
        '| 组别 | 最大单步跌幅均值 | 单步最大跌幅的场景95分位 | 最大5步跌幅均值 | 每回合≥10pp次数 | 最低20步覆盖均值 |', '|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        x=s[g];lines.append(f"| {NAMES[g]} | {100*x['max_drop_1']:.2f}pp | {100*x['max_drop_1_scene_p95']:.2f}pp | {100*x['max_drop_5']:.2f}pp | {x['drop_events_10pp']:.2f} | {x['min20_coverage']:.2%} |")
    lines+=['','## 对三个原参考的配对差值','',
        '差值为新增组减参考；骤降指标负值表示更少，但必须结合绝对覆盖和吞吐判断，防止“低覆盖伪稳定”。2000次场景配对bootstrap，种子950001；不是训练种子置信区间，未调整多重比较。','',
        '| 新增组 | 参考 | 指标 | 差值 | 95%场景区间 |','|---|---|---|---:|---:|']
    for p in result['paired']:
        if p['metric'] not in ('coverage','max_drop_1','drop_events_10pp','throughput_bps'):continue
        scale=1e-6 if p['metric']=='throughput_bps' else (1 if p['metric']=='drop_events_10pp' else 100)
        lines.append(f"| {NAMES[p['group']]} | {NAMES[p['reference']]} | {p['metric']} | {p['difference']*scale:+.3f} | [{p['low']*scale:+.3f}, {p['high']*scale:+.3f}] |")
    lines+=['','## 联网、鲁棒性与回合内保持','',
        '| 组别 | 断联UAV比例 | 故障保留覆盖J | 保留比例J/rho | 第51–150步覆盖 | 后期−前期覆盖 | 平均曲线峰值时刻 |', '|---|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        x=s[g];lines.append(f"| {NAMES[g]} | {x['disconnection_ratio']:.3%} | {x['robustness']:.3%} | {x['retention_ratio']:.3%} | {x['early51_150_coverage']:.3%} | {100*x['maintain_coverage']:+.3f}pp | {x['curve_peak_time']} |")
    lines+=['','保留比例先按每场景sum(J)/sum(rho)计算，再平均。最低20步覆盖可能发生在初期展开，需与后期绝对覆盖一同判断。新增三组中，单步覆盖下降≥10个百分点的事件均同时出现净断联UAV比例增加；这是一致性证据，不代表所有服务损失都只能由回传解释。']
    lines+=['','断联惩罚考察当前联网比例；J是平均假想单UAV故障后的固定关联服务保留。两者不直接约束连续动作造成的多链路失效。鲁棒项同时包含覆盖偏好：rho＋lambda J＝(1＋lambda)rho−lambda(rho−J)，因此提高J不等于单独学到冗余。','',
        '## 新增组last补充','', '| 组别 | 平均覆盖 | 后100步覆盖 | 吞吐Mbps | 最大单步跌幅均值 | 每回合≥10pp次数 |','|---|---:|---:|---:|---:|---:|']
    for g,x in result['supplemental_last'].items():lines.append(f"| {NAMES[g]} last | {x['coverage']:.3%} | {x['last100_coverage']:.3%} | {x['throughput_bps']/1e6:.2f} | {100*x['max_drop_1']:.2f}pp | {x['drop_events_10pp']:.2f} |")
    lines+=['',f"鲁棒性组last相对best平均覆盖变化{100*(result['supplemental_last']['robustness']['coverage']-s['robustness']['coverage']):+.2f}个百分点，≥10pp骤降次数由{s['robustness']['drop_events_10pp']:.2f}增至{result['supplemental_last']['robustness']['drop_events_10pp']:.2f}次/回合。继续训练到预算终点没有保持住best表现；本轮不自动延长训练。"]
    lines+=['','## 固定场景终态','', '| 组别 | 场景10000覆盖/联网 | 场景10001覆盖/联网 |','|---|---|---|']
    for g in GROUPS:
        a=result['fixed_scenes'][g]['10000'][-1];b=result['fixed_scenes'][g]['10001'][-1]
        lines.append(f"| {NAMES[g]} | {a['coverage']:.2%} / {a['connected']}/18 | {b['coverage']:.2%} / {b['connected']}/18 |")
    lines+=['','## 固定场景10000：覆盖骤降前后','',
        '以下时刻是该固定场景内的最大单步损失，属于明确挑选的事件诊断；完整场景分布仍以上表为准。','',
        '| 新增组 | 动作后时刻 | 覆盖前→后 | 联网UAV前→后 | 失去服务UE | 其中原服务UAV断联 | 丢失UU/UG链路数 |', '|---|---:|---|---|---:|---:|---|']
    for g in NEW:
        a=result['abrupt_fixed_cases'][g]['10000']
        lines.append(f"| {NAMES[g]} | {a['time']} | {a['coverage_before']:.2%}→{a['coverage_after']:.2%} | {a['connected_before']}→{a['connected_after']} | {a['lost_users']} | {a['lost_users_whose_previous_server_disconnected']} | {len(a['removed_uu_edges'])}/{len(a['removed_ug_edges'])} |")
    a=result['abrupt_fixed_cases']['robustness']['10000']
    lines+=[ '',f"特别是鲁棒性组：第{a['time']-1}步覆盖{a['coverage_before']:.2%}、J={a['pre_failure_mean']:.2%}，假想单UAV故障的最坏保留覆盖仍为{a['pre_failure_worst']:.2%}；但下一步同时丢失{len(a['removed_uu_edges'])}条UU与{len(a['removed_ug_edges'])}条UG链路，覆盖降至{a['coverage_after']:.2%}。这说明本例的静态单节点故障指标未覆盖实际运动导致的同时多链路失效，不能仅靠提高平均J判断动态部署可靠性。"]
    lines+=['','新增三组仍使用原集中式关联，没有移动前Top-K提议和新候选等待。它们同样发生大规模断联，因此本轮观察到的服务崩落不依赖Top-K时序；Top-K可以影响其他组的服务恢复，但不能作为新增三组崩落的解释。']
    lines+=['','## 来源与结论边界','',
        '原三组复用已校验的GPU评估，新三组在同设备与采样口径重新验证；模型、日志、原始结果与评估文件的来源SHA256保留。teacher仍是新baseline 10M best＋SINR Top-K闭环，不是DPPO预训练时使用的旧教师。',
        '该比较不含同预算原1M baseline；因此六组结果可比较当前最终表现，若要严格估计新增奖励的作用，仍应回到1M对1M对照。没有追加训练或修改物理与算法设置。','',
        '图表提供PDF/SVG/PNG；红/蓝/橙保留原三组，新三组使用黑/浅蓝/灰，并增加线型与标记，避免紫色及红绿对照。','',
        '- [覆盖、吞吐与共同奖励](figures/task_curves.png)','- [覆盖骤降](figures/coverage_drops.png)',
        '- [稳定性分布](figures/stability_distribution.png)','- [联网与鲁棒性](figures/deployment.png)',
        '- [性能汇总](figures/performance.png)','- [新三组验证曲线](figures/new_runs_validation.png)',
        '- [新三组loss](figures/new_runs_loss.png)','- [场景10000拓扑](figures/topology_best_10000.png)',
        '- [场景10001拓扑](figures/topology_best_10001.png)',
        '- [场景10000逐步覆盖与断联](figures/fixed_scene_events_10000.png)',
        '- [场景10001逐步覆盖与断联](figures/fixed_scene_events_10001.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_stability_six_group_1m_vs_10m_seed0'))
    main(parser.parse_args())
