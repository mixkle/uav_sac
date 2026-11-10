"""Frozen 1M baseline, mu=0/.1/.5/2/5; primary last, supplementary best."""
import argparse
import json
from pathlib import Path

import numpy as np
import report_mappo_neighbors as helpers
import report_mappo_stability_rewards as old
from uav_research.coverage_stability import DISCONNECTION_SWEEP,BASELINE_RUN,run_path

GROUPS=('mu0','mu01','mu05','mu2','mu5')
WEIGHTS=dict(zip(GROUPS,[0.,.1,.5,2.,5.]))
RUNS=dict(mu0=Path(BASELINE_RUN),mu01=Path(run_path('disconnection')),
          **{key:Path(run_path(group)) for key,group in zip(('mu05','mu2','mu5'),DISCONNECTION_SWEEP)})
NAMES={g:f'mu = {w:g}' for g,w in WEIGHTS.items()}
COLORS=dict(zip(GROUPS,['#555555','#E69F00','#009E73','#56B4E9','#0072B2']))
LINES=dict(zip(GROUPS,[':',(0,(5,2)),'-.','--','-']))
MARKERS=dict(zip(GROUPS,['o','^','s','D','x']))
DEFAULT=Path('analysis/mappo_disconnection_weight_sweep_1m_seed0')


def evaluate(args):
    group=next(k for k in DISCONNECTION_SWEEP if k==args.evaluate)
    old.REWARD_ABLATIONS={**old.REWARD_ABLATIONS,**DISCONNECTION_SWEEP}
    old.evaluate_all(args,{group:Path(run_path(group))})
    # Same 32 CUDA scenes should reproduce each saved checkpoint's validation row.
    for kind in args.checkpoints:
        p=args.output/kind/group
        prov=json.loads((p/'stability_source.json').read_text())
        row=next(r for r in helpers.read(Path(run_path(group))/'validation.csv') if int(r['env_steps'])==prov['checkpoint_step'])
        data=np.load(p/'reward_curves.npz');raw=data['raw'];reward=data['reward']
        measured=dict(coverage=float(raw[:,:,0].mean()),throughput_bps=float(raw[:,:,1].mean()*750e6),
                      return_=float(reward.sum(1).mean()))
        diffs={key:measured[src]-float(row[key]) for key,src in [('coverage','coverage'),('throughput_bps','throughput_bps'),('return','return_')]}
        assert max(abs(v) for v in diffs.values())<1e-6,diffs
        prov['validation_difference']=diffs
        (p/'stability_source.json').write_text(json.dumps(prov,indent=2)+'\n')
        print(json.dumps(dict(group=group,kind=kind,step=prov['checkpoint_step'],**measured,validation_difference=diffs)),flush=True)


def evaluation_path(out,group,kind):
    if group=='mu0':return Path('analysis/mappo_neighbors_1m_vs_10m_seed0')/kind/'neighbors3'
    if group=='mu01':return Path('analysis/mappo_stability_six_group_1m_vs_10m_seed0')/kind/'disconnection'
    mapping=dict(zip(('mu05','mu2','mu5'),DISCONNECTION_SWEEP))
    return out/kind/mapping[group]


def load_evaluation(out,g,kind):
    from uav_research.coverage_stability import coverage_drop_metrics,ablation_config
    from uav_research.config import load_config
    p=evaluation_path(out,g,kind)
    if (p/'step_metrics.npz').exists():
        prov=json.loads((p/'provenance.json').read_text());identity=prov
        with np.load(p/'step_metrics.npz') as z:
            assert z['scene_seeds'].tolist()==helpers.SCENES and z['metrics'].shape[:2]==(32,400)
            s=dict(zip(z['names'].tolist(),z['metrics'].transpose(2,0,1)))
        step=prov['env_steps']
    else:
        prov=json.loads((p/'stability_source.json').read_text());identity=prov['evaluation'];step=prov['checkpoint_step']
        assert helpers.digest(p/'reward_curves.npz')==prov['reward_curves_sha256']
        with np.load(p/'reward_curves.npz') as z:
            assert z['scene_seeds'].tolist()==helpers.SCENES and z['raw'].shape==(32,400,4)
            raw=z['raw'];weights=np.array([1,.1,0,-WEIGHTS[g]])
            np.testing.assert_array_equal(z['reward_weights'],weights)
            np.testing.assert_allclose(z['contributions'],raw*weights,atol=1e-12)
            np.testing.assert_allclose(z['reward'],z['contributions'].sum(-1),atol=1e-12)
            s=dict(coverage=raw[:,:,0],tau=raw[:,:,1],robustness=raw[:,:,2],disconnection_ratio=raw[:,:,3],reward=z['reward'].copy())
            s.update(dict(zip(z['diagnostic_names'].tolist(),z['diagnostics'].transpose(2,0,1))))
    scenes=identity.get('scene_seeds',identity.get('scenes'))
    assert scenes==helpers.SCENES and identity['action_seed']==880001 and identity['device']=='cuda:0'
    assert identity.get('batch_size',16)==16
    cp=RUNS[g]/f'{kind}.pt';assert helpers.digest(cp)==identity['sha256']
    cfg=load_config(RUNS[g]/'config.yaml');baseline=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml').as_dict()
    baseline['env']['disconnection_weight']=WEIGHTS[g];baseline['run']['device']=cfg.run.device
    assert cfg.as_dict()==baseline,'Unexpected scenario or optimizer change'
    assert all(np.isfinite(v).all() for v in s.values())
    s['common_reward']=s['coverage']+.1*s['tau'];s['penalty']=-WEIGHTS[g]*s['disconnection_ratio']
    s['throughput_bps']=s['tau']*750e6
    np.testing.assert_allclose(s['reward'],s['common_reward']+s['penalty'],atol=2e-7)
    assert (s['robustness']<=(17/18)*s['coverage']+1e-12).all()
    rows=[]
    for i,scene in enumerate(helpers.SCENES):
        rho=s['coverage'][i];d=s['disconnection_ratio'][i]
        r=dict(group=g,scene_seed=scene,**coverage_drop_metrics(rho))
        drop=np.maximum(0,rho[:-1]-rho[1:]);large=drop>=.1-1e-12
        r.update(throughput_mbps=float(s['throughput_bps'][i].mean()/1e6),
            last100_throughput_mbps=float(s['throughput_bps'][i,-100:].mean()/1e6),
            training_return=float(s['reward'][i].sum()),common_return=float(s['common_reward'][i].sum()),
            disconnection_ratio=float(d.mean()),last100_disconnection_ratio=float(d[-100:].mean()),
            final_disconnection_ratio=float(d[-1]),robustness=float(s['robustness'][i].mean()),
            sensing=float(s['discovery_fraction'][i].mean()),boundary150_fraction=float(s['boundary_within_150_fraction'][i].mean()),
            reflection_fraction=float(s['boundary_reflection_fraction'][i].mean()),
            close_neighbor_fraction=float(s['uav_close_neighbor_fraction'][i].mean()),
            nearest_neighbor_m=float(s['uav_nearest_neighbor_mean'][i].mean()),
            large_drop_with_disconnect_rise=int((large&(np.diff(d)>1e-12)).sum()))
        rows.append(r)
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('group','scene_seed')}
    summary.update(checkpoint_step=step,scenes_with_10pp_drop=int(sum(r['any_drop_10pp'] for r in rows)),
        curve_peak_time=int(s['coverage'].mean(0).argmax()+1),curve_peak_coverage=float(s['coverage'].mean(0).max()),
        final_below30_scenes=int((s['coverage'][:,-1]<.3).sum()),
        max_drop_scene_p95=float(np.quantile([r['max_drop_1'] for r in rows],.95)))
    row=next(r for r in helpers.read(RUNS[g]/'validation.csv') if int(r['env_steps'])==step)
    diff={key:summary[target]*scale-float(row[key]) for key,target,scale in
          [('coverage','coverage',1),('return','training_return',1),('throughput_bps','throughput_mbps',1e6)]}
    # The original baseline was trained/evaluated on CPU. Its cache is the
    # deliberate CUDA re-evaluation, all other runs should reproduce CUDA logs.
    if cfg.run.device.startswith('cuda'):assert max(abs(v) for v in diff.values())<1e-6,diff
    source=dict(checkpoint=str(cp),sha256=identity['sha256'],env_steps=step,device='cuda:0',
        training_device=cfg.run.device,mu=WEIGHTS[g],validation_difference=diff,evaluation=str(p))
    return s,rows,summary,source,p


def rebuild(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    old_manifest=out/'report_sources.json'
    if old_manifest.exists():
        for name,sha in json.loads(old_manifest.read_text())['input_files'].items():
            assert helpers.digest(name)==sha,f'Changed evaluation input {name}'
    results={};data={};all_rows={};paths={};inputs={};pairs=[]
    draws=np.random.default_rng(950001).integers(32,size=(2000,32))
    for kind in ('last','best'):
        data[kind]={};all_rows[kind]={};paths[kind]={};summaries={};sources={}
        for g in GROUPS:
            ss,rr,su,source,p=load_evaluation(out,g,kind)
            data[kind][g],all_rows[kind][g],summaries[g],sources[g],paths[kind][g]=ss,rr,su,source,p
            for f in p.iterdir():
                if f.is_file():inputs[str(f)]=helpers.digest(f)
        comparisons=[(g,'mu0') for g in GROUPS[1:]]+[(g,'mu01') for g in ('mu05','mu2','mu5')]+[('mu2','mu05'),('mu5','mu05')]
        for g,ref in comparisons:
            for key in ('coverage','last100_coverage','final_coverage','throughput_mbps','common_return','max_drop_1','max_drop_5',
                        'drop_events_10pp','min20_coverage','disconnection_ratio','boundary150_fraction','close_neighbor_fraction','maintain_coverage'):
                delta=np.array([a[key]-b[key] for a,b in zip(all_rows[kind][g],all_rows[kind][ref])])
                lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
                pairs.append(dict(checkpoint=kind,group=g,reference=ref,metric=key,difference=float(delta.mean()),low=float(lo),high=float(hi)))
        results[kind]=dict(summary=summaries,sources=sources)
        helpers.write_csv(out/f'{kind}_scene_metrics.csv',[r for g in GROUPS for r in all_rows[kind][g]])
    cases={}
    for kind in ('last','best'):
        for scene in (10000,10001):
            initial,users=None,None
            for g in GROUPS:
                with np.load(paths[kind][g]/f'trace_{scene}.npz') as z:
                    if initial is None:initial,users=z['uav_xy'][0].copy(),z['user_xy'].copy()
                    else:
                        np.testing.assert_array_equal(initial,z['uav_xy'][0]);np.testing.assert_array_equal(users,z['user_xy'])
                    assert (z['C'].sum(1)<=1).all() and (z['C'].sum(2)<=10).all()
                    np.testing.assert_allclose(z['C'].sum((1,2))/150,z['coverage'])
                    t=int(np.argmax(z['coverage'][1:-1]-z['coverage'][2:])+2)
                    b0,b1=z['B'][t-1].min(1)<=18,z['B'][t].min(1)<=18
                    lost=z['C'][t-1].any(0)&~z['C'][t].any(0);owner=z['C'][t-1].argmax(0)
                    cases[f'{kind}/{g}/{scene}']=dict(time=t,coverage_before=float(z['coverage'][t-1]),coverage_after=float(z['coverage'][t]),
                        connected_before=int(b0.sum()),connected_after=int(b1.sum()),newly_disconnected_uavs=(np.flatnonzero(b0&~b1)+1).tolist(),
                        lost_users=int(lost.sum()),lost_users_previous_server_disconnected=int((lost&~b1[owner]).sum()),
                        final_coverage=float(z['coverage'][-1]),final_connected=int((z['B'][-1].min(1)<=18).sum()))
    result=dict(results=results,paired=pairs,fixed_scene_events=cases,primary='last',groups=list(GROUPS),weights=WEIGHTS,
        uncertainty='2000 paired scenario bootstrap, seed 950001; not training-seed CI')
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    helpers.write_csv(out/'paired_scene_differences.csv',pairs)
    manifest=dict(input_files=inputs,report_script_sha256=helpers.digest(__file__),
        log_files={str(RUNS[g]/name):helpers.digest(RUNS[g]/name) for g in GROUPS for name in ('validation.csv','metrics.csv','manifest.json','config.yaml')})
    (out/'report_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    make_figures(out,data,all_rows,paths,draws)
    write_report(out,result)
    print(json.dumps(results,ensure_ascii=False),flush=True)


def make_figures(out,data,rows,paths,draws):
    import matplotlib.pyplot as plt
    from uav_research.plotting import style,save_figure
    style();figdir=out/'figures'
    def line(ax,g,y,x=None):
        if x is None:x=np.arange(1,len(y)+1)
        ax.plot(x,y,c=COLORS[g],ls=LINES[g],marker=MARKERS[g],markevery=max(1,len(y)//8),ms=3,label=NAMES[g])
    for kind in ('last','best'):
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,key,scale,label in zip(axes.flat,('coverage','throughput_bps','disconnection_ratio','discovery_fraction'),
            (100,1e-6,100,100),('Coverage (%)','Throughput (Mbps)','Disconnected UAV (%)','Sensing (%)')):
            for g in GROUPS:line(ax,g,data[kind][g][key].mean(0)*scale)
            ax.set(xlabel='Episode step',ylabel=label)
        axes.flat[0].legend(ncol=2,fontsize=8);fig.suptitle(f'{kind}: each run budget 1M; no runtime speed/safety filter')
        save_figure(fig,figdir,f'coverage_and_service_{kind}')
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,key,scale,label in zip(axes.flat,('coverage','last100_coverage','max_drop_1','drop_events_10pp','disconnection_ratio','throughput_mbps'),
            (100,100,100,1,100,1),('Mean coverage (%)','Last-100 coverage (%)','Mean episode max 1-step drop (pp)',
            'Drops >=10 pp / episode','Disconnected UAV (%)','Throughput (Mbps)')):
            for i,g in enumerate(GROUPS):
                v=np.array([r[key]*scale for r in rows[kind][g]]);m=v.mean();lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
                ax.bar(i,m,color=COLORS[g]);ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],color='black',capsize=3)
            ax.set_xticks(range(5),['0','0.1','0.5','2','5']);ax.set(xlabel='Disconnection weight mu',ylabel=label)
        fig.suptitle(f'{kind}: 32-scene bootstrap intervals');save_figure(fig,figdir,f'summary_{kind}')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,key,scale,label in zip(axes.flat,('coverage','tau','penalty','common_reward'),(1,.1,1,1),
            ('Coverage contribution','Throughput contribution','Negative disconnection contribution','Common reward (coverage + 0.1 tau)')):
            for g in GROUPS:line(ax,g,data[kind][g][key].mean(0)*scale)
            ax.set(xlabel='Episode step',ylabel=label)
        axes.flat[0].legend(ncol=2,fontsize=8);save_figure(fig,figdir,f'reward_components_{kind}')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for g in GROUPS:
        records=helpers.read(RUNS[g]/'validation.csv');x=np.array([float(r['env_steps']) for r in records])/1e6
        for ax,key,scale,label in zip(axes.flat,('coverage','throughput_bps','disconnection_ratio','return'),
            (100,1e-6,100,1),('Validation coverage (%)','Validation throughput (Mbps)','Disconnected UAV (%)',
            'Own training return (different objectives)')):
            line(ax,g,np.array([float(r[key]) for r in records])*scale,x)
            ax.set(xlabel='Joint transitions (millions)',ylabel=label)
    axes.flat[0].legend(fontsize=8,ncol=2);save_figure(fig,figdir,'validation_learning')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for g in GROUPS:
        rr=helpers.read(RUNS[g]/'metrics.csv');x=np.array([float(r['env_steps']) for r in rr])/1e6
        for ax,key in zip(axes.flat,('actor_loss','value_loss','joint_entropy','ppo_approx_kl')):
            if key not in rr[0]:continue
            y=np.array([float(r[key]) for r in rr]);ax.plot(x,y,c=COLORS[g],lw=.5,alpha=.15)
            sm=np.convolve(y,np.ones(15)/15,'valid');line(ax,g,sm,x[-len(sm):]);ax.set(xlabel='Joint transitions (millions)',ylabel=key)
    axes.flat[0].legend(fontsize=8,ncol=2);save_figure(fig,figdir,'loss_and_policy')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for g in GROUPS:
        for ax,kind in zip(axes,('last','best')):
            line(ax,g,data[kind][g]['uav_close_neighbor_fraction'].mean(0)*100)
            ax.set(xlabel='Episode step',ylabel='Nearest UAV within 100 m (%)',title=f'{kind}: aggregation diagnostic')
    axes[0].legend(fontsize=8,ncol=2);save_figure(fig,figdir,'aggregation')
    for kind in ('last','best'):
        for scene in (10000,10001):
            fig,axes=plt.subplots(5,4,figsize=(16,17))
            for row,g in enumerate(GROUPS):
                with np.load(paths[kind][g]/f'trace_{scene}.npz') as z:
                    for ax,t in zip(axes[row],(0,150,200,400)):
                        xy,ue,a,b,c=[z[k][t] for k in ('uav_xy','user_xy','A','B','C')];bs=z['bs_positions'];served=c.any(0)
                        ax.scatter(ue[~served,0],ue[~served,1],c='#aaaaaa',s=7)
                        ax.scatter(ue[served,0],ue[served,1],c=COLORS[g],s=10)
                        for i in range(18):
                            for j in range(i):
                                if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],c='#888888',lw=.6)
                            for k in range(3):
                                if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],ls='--',c='#888888',lw=.6)
                        linked=b.min(1)<=18
                        for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=32,edgecolors='black',linewidths=.5)
                        for i in range(18):ax.annotate(str(i+1),xy[i],fontsize=5,xytext=(2,2),textcoords='offset points')
                        ax.scatter(bs[:,0],bs[:,1],c='black',marker='*',s=70,clip_on=False)
                        ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[g]} | t={t} | {served.mean():.1%}')
            fig.suptitle(f'{kind}, scene {scene}: triangles connected; squares disconnected; stars BS',y=1.01)
            save_figure(fig,figdir,f'topology_{kind}_{scene}')


def write_report(out,r):
    lines=['# 断联惩罚权重0／0.1／0.5／2／5：1M训练验证','',
        '三组新训练均由用户完成，种子0、各1M条联合转移、489轮PPO。完整结果和恢复文件下载，冻结best／last，'
        '统一32场景10000–10031、采样种子880001、16环境、CUDA FP32／关闭TF32复验。原mu=0和0.1复用来源一致的同口径评估。'
        '未使用保留测试集，未重训或微调。','',
        '全部组为3邻居、容量10、共享连续Gaussian MAPPO、中心UE初始化、移动UE、时间因子critic。'
        '奖励为coverage＋0.1tau−mu D，鲁棒性与边界权重均为0。'
        '**本轮不应用近期的跳数减速或链路安全过滤器**，避免把执行干预混入奖励训练对照。','',
        '## 主要判断','']
    last=r['results']['last']['summary'];base,weak,medium,strong2,strong5=[last[g] for g in GROUPS]
    lines += [
        f"1. **当前单种子、1M预算下，mu=0.1的服务与稳定性取舍最好。** 相对mu=0，last覆盖{100*(weak['coverage']-base['coverage']):+.2f}pp、"
        f"吞吐{weak['throughput_mbps']-base['throughput_mbps']:+.2f}Mbps，同时断联率下降{100*(base['disconnection_ratio']-weak['disconnection_ratio']):.2f}pp，"
        f"最大单步跌幅均值下降{100*(base['max_drop_1']-weak['max_drop_1']):.2f}pp；相应场景区间未跨零。不是跨训练种子的最优权重证明。",
        f"2. **mu=0.5没有进一步提高服务能力。** 相对0.1，last覆盖{100*(medium['coverage']-weak['coverage']):+.2f}pp、"
        f"吞吐{medium['throughput_mbps']-weak['throughput_mbps']:+.2f}Mbps；平均断联率几乎相同（{weak['disconnection_ratio']:.2%}对{medium['disconnection_ratio']:.2%}）。"
        f"最大单步跌幅进一步降低{100*(weak['max_drop_1']-medium['max_drop_1']):.2f}pp，体现服务与风险的取舍。",
        f"3. **mu=2／5明显倾向保守联网。** last平均断联分别降至{strong2['disconnection_ratio']:.2%}／{strong5['disconnection_ratio']:.2%}，"
        f"但覆盖降至{strong2['coverage']:.2%}／{strong5['coverage']:.2%}，后100步仅{strong2['last100_coverage']:.2%}／{strong5['last100_coverage']:.2%}。"
        '当前结果支持“更强断联成本能减少断联，却可能牺牲展开与服务”，不支持“权重越大覆盖越好”。',
        f"4. **强惩罚组仍会退化，退化不一定由断联造成。** mu=2／5的后100步相对51–150步分别为"
        f"{100*strong2['maintain_coverage']:+.2f}pp／{100*strong5['maintain_coverage']:+.2f}pp。"
        f"感知率均值从mu=0的{base['sensing']:.2%}下降到{strong2['sensing']:.2%}／{strong5['sensing']:.2%}，"
        f"最近邻<100m的UAV比例升至{strong2['close_neighbor_fraction']:.2%}／{strong5['close_neighbor_fraction']:.2%}，"
        f"mu=0.1为{weak['close_neighbor_fraction']:.2%}。这些统计与固定拓扑的聚集现象一致。",'',
        '断联一架UAV时，mu=2／5的单步成本分别相当于16.67／41.67个UE的覆盖贡献（忽略吞吐），每架容量仅10。'
        '这使联网目标具有很强的取舍力度；D对所有断联UAV同等计数，不直接表达关键链路余量或展开价值。'
        '较小骤降还会受到整体覆盖较低的影响，需要与绝对覆盖和吞吐共同判断。','',
        '## 同预算last：主要比较','']
    def table(kind):
        v=['| mu | 实际检查点步数 | 平均覆盖 | 后100步 | 终态覆盖 | 吞吐Mbps | 断联率 | 最大单步跌幅均值 | ≥10pp次数/回合 | 共同服务回报 | 训练回报 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g,s in r['results'][kind]['summary'].items():
            v.append(f"| {WEIGHTS[g]:g} | {s['checkpoint_step']:,} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_mbps']:.2f} | {s['disconnection_ratio']:.2%} | {s['max_drop_1']*100:.2f}pp | {s['drop_events_10pp']:.2f} | {s['common_return']:.3f} | {s['training_return']:.3f} |")
        return v
    lines+=table('last')
    lines+=['','## 按自身奖励选出的best：辅助比较','',
        '各组best按自身包含断联惩罚的全回合验证回报选择，包含step0；没有重新按覆盖选择。'
        '共同服务回报始终为400步的coverage＋0.1tau之和。训练回报定义不同，不用于跨组直接排名。','']
    lines+=table('best')
    lines+=['','## 场景配对差值','',
        '以下为last相对原baseline（mu=0）的差值；2000次场景bootstrap、种子950001，不是训练种子置信区间。','']
    for g in GROUPS[1:]:
        for key in ('coverage','disconnection_ratio','max_drop_1','throughput_mbps'):
            p=next(p for p in r['paired'] if p['checkpoint']=='last' and p['group']==g and p['reference']=='mu0' and p['metric']==key)
            scale=1 if key=='throughput_mbps' else 100;unit='Mbps' if scale==1 else 'pp'
            lines.append(f"- mu={WEIGHTS[g]:g}, {key}: {scale*p['difference']:+.3f}{unit}，95%区间[{scale*p['low']:+.3f}, {scale*p['high']:+.3f}]{unit}。")
    lines+=['','## 固定场景的断链事件','',
        '场景10000／10001预先固定，各自挑最大单步跌幅作诊断，不把这些事件当成所有场景的统计代表。','',
        '| mu（last） | 场景 | 时刻 | 覆盖前→后 | 联网UAV前→后 | 失服务UE／原服务UAV断联 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        for scene in (10000,10001):
            e=r['fixed_scene_events'][f'last/{g}/{scene}']
            lines.append(f"| {WEIGHTS[g]:g} | {scene} | {e['time']} | {e['coverage_before']:.1%}→{e['coverage_after']:.1%} | {e['connected_before']}→{e['connected_after']} | {e['lost_users']}／{e['lost_users_previous_server_disconnected']} |")
    lines += ['',
        '场景10000的last终态：mu=2／5都保持18架联网，但覆盖仅36%／37.33%；原mu=0也有18架联网，覆盖70.67%。'
        '结合拓扑可见强惩罚组主要聚集在下方或左下接入区域，许多上方UE不再被覆盖。'
        '这两个终态的低覆盖不能归因于当时UAV断联。可达性只是服务条件之一，空间分布、感知、下行干扰和容量仍影响最终覆盖。','']
    lines+=['','## 来源与解释限制','',
        '新增三组best和last覆盖、训练回报、吞吐均复现各自训练期CUDA记录（最大绝对误差<1e-6）。'
        '原mu=0在CPU训练，本轮使用其CUDA复验覆盖56.43844%；没有混用历史CPU覆盖58.63125%。'
        '其余四组CUDA训练。训练设备及随机流差异仍需考虑，单训练种子不能支持唯一根因判断。','',
        '强惩罚可能改变联网与展开的取舍。评价时同时看共同覆盖、吞吐、尾段覆盖、极端跌幅与断联，'
        '并用聚集和固定拓扑检查是否通过牺牲展开换取联网。D是断联UAV比例，奖励没有直接约束链路余量或单步覆盖跌幅。','',
        '## 图表','',
        '- [Last覆盖／吞吐／断联曲线](figures/coverage_and_service_last.png)',
        '- [Last指标汇总与场景区间](figures/summary_last.png)',
        '- [Best覆盖／吞吐／断联曲线](figures/coverage_and_service_best.png)',
        '- [训练期验证曲线](figures/validation_learning.png)',
        '- [Loss与策略诊断](figures/loss_and_policy.png)',
        '- [Last奖励分解](figures/reward_components_last.png)',
        '- [Last场景10000拓扑](figures/topology_last_10000.png)',
        '- [Last场景10001拓扑](figures/topology_last_10001.png)',
        '- [Best场景10000拓扑](figures/topology_best_10000.png)',
        '- [Best场景10001拓扑](figures/topology_best_10001.png)','',
        '另存PDF／SVG；使用颜色、线型、标记区分，不使用紫色。保存逐场景表、原始轨迹、SHA256和配对区间。'
        '原两组数据就地读取，不复制或修改。重建：','',
        '```bash','.venv-dppo-compat/bin/python scripts/report_disconnection_weight_sweep.py --rebuild','```','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--evaluate',choices=list(DISCONNECTION_SWEEP))
    p.add_argument('--checkpoints',nargs='+',choices=['best','last'],default=['best','last'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rebuild',action='store_true')
    a=p.parse_args()
    if bool(a.evaluate)==bool(a.rebuild):p.error('Choose --evaluate or --rebuild')
    if a.evaluate:evaluate(a)
    else:rebuild(a)
