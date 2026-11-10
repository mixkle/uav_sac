"""Best-only comparison: six 1M neighbor baselines and the 10M DPPO model."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import report_mappo_neighbors as legacy

RUNS = {**legacy.RUNS, **{f'neighbors{k}': Path(f'runs/mappo_capacity10_neighbors{k}_1m_seed0') for k in (6, 7, 8)}}
GROUPS = (*RUNS, 'dppo10m')
NAMES = {g: f'Baseline 1M / N{g[-1]}' for g in RUNS} | {'dppo10m': 'DPPO 10M best'}
COLORS = dict(zip(GROUPS, ['#555555','#009E73','#E69F00','#111111','#56B4E9','#9A6324','#0072B2']))
LINES = dict(zip(GROUPS, [':','-','--','-.',(0,(1,2)),(0,(6,2)),'-']))
MARKERS = dict(zip(GROUPS, ['o','s','^','D','v','X','*']))
DEFAULT = Path('analysis/mappo_neighbors3to8_1m_vs_dppo10m_seed0')


def evaluate(args):
    legacy.RUNS.update(RUNS)
    args.checkpoint = 'best'
    legacy.evaluate_one(args)


def rebuild(args):
    from uav_research.diffusion_dppo.report import error_metrics
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cache = Path('analysis/mappo_neighbors_1m_vs_10m_seed0')
    for g in ('neighbors3', 'neighbors4', 'neighbors5', 'dppo10m'):
        section = 'references' if g == 'dppo10m' else 'best'
        target = out/section/g
        if not target.exists(): shutil.copytree(cache/section/g, target)
    old = out/'report_sources.json'
    if old.exists():
        for name, sha in json.loads(old.read_text())['evaluation_files'].items():
            assert legacy.digest(out/name) == sha, f'Changed evaluation: {name}'
    series, rows, summaries, paths, sources, inputs, critic, usage, events = ({ } for _ in range(9))
    for g in GROUPS:
        path = out/('best' if g in RUNS else 'references')/g
        prov = json.loads((path/'provenance.json').read_text())
        assert prov['scene_seeds'] == legacy.SCENES and prov['action_seed'] == 880001
        assert prov['device'] == 'cuda:0'
        cp = Path(str(prov['checkpoint']).removeprefix('/home/dym/uav_sac/'))
        assert legacy.digest(cp) == prov['sha256'], f'Checkpoint changed: {cp}'
        sources[g] = {**prov, 'checkpoint': str(cp)}
        series[g], rows[g], summaries[g] = legacy.series_and_rows(path, g)
        summaries[g]['checkpoint_step'] = int(prov['env_steps'])
        paths[g] = path
        for f in path.iterdir():
            if f.is_file(): inputs[str(f.relative_to(out))] = legacy.digest(f)
        with np.load(path/'critic_predictions.npz') as z:
            critic[g] = error_metrics(z['predicted'], z['mc_return'])
        if g in RUNS:
            with np.load(path/'neighbor_availability.npz') as z:
                degree, observed = z['data'][...,0], z['data'][...,1]
                k = int(g[-1])
                np.testing.assert_array_equal(observed, np.minimum(degree, k))
                usage[g] = dict(mean_degree=float(degree.mean()), mean_observed=float(observed.mean()),
                    slots_filled_fraction=float((degree>=k).mean()),
                    degree_exceeds_slots=float((degree>k).mean()), missing_neighbor_mean=float((degree-observed).mean()),
                    **{f'degree_at_least{n}': float((degree>=n).mean()) for n in range(3,9)})
            manifest = json.loads((RUNS[g]/'manifest.json').read_text())
            assert manifest['env_steps'] == 1_000_000 and manifest['status'] == 'completed'
            summaries[g]['actor_parameters'] = manifest['actor_parameters']
            summaries[g]['training_seconds'] = manifest['elapsed_seconds']
            initial = legacy.read(RUNS[g]/'validation.csv')[0]
            assert int(initial['env_steps']) == 0
            summaries[g]['step0_coverage'] = float(initial['coverage'])
            summaries[g]['step0_return'] = float(initial['return'])
        # Audit a fixed diagnostic event: largest single-step drop in each fixed scene.
        events[g] = {}
        for scene in (10000,10001):
            with np.load(path/f'trace_{scene}.npz') as z:
                c,b = z['C'],z['B']
                assert (c.sum(1)<=1).all() and (c.sum(2)<=10).all()
                np.testing.assert_allclose(c.sum((1,2))/150,z['coverage'])
                t = int(np.argmax(z['coverage'][1:-1]-z['coverage'][2:])+2)
                linked0,linked1 = b[t-1].min(1)<=18,b[t].min(1)<=18
                lost = c[t-1].any(0)&~c[t].any(0)
                owner = c[t-1].argmax(0)
                lost_edges = np.argwhere(np.triu(z['A'][t-1]&~z['A'][t],1))
                events[g][str(scene)] = dict(time=t,coverage_before=float(z['coverage'][t-1]),
                    coverage_after=float(z['coverage'][t]),connected_before=int(linked0.sum()),
                    connected_after=int(linked1.sum()),lost_users=int(lost.sum()),
                    lost_users_previous_server_disconnected=int((lost&~linked1[owner]).sum()),
                    newly_disconnected_uavs=(np.flatnonzero(linked0&~linked1)+1).tolist(),
                    lost_uu_edges=(lost_edges+1).tolist(),
                    snapshots=[dict(time=j,coverage=float(z['coverage'][j]),connected=int((b[j].min(1)<=18).sum()),
                        components=legacy.component_count(z['A'][j])) for j in (0,150,200,400)])
    # Identical initial UAVs and entire UE motion, independent of policies.
    for scene in (10000,10001):
        with np.load(paths['neighbors3']/f'trace_{scene}.npz') as z:
            initial,ue = z['uav_xy'][0].copy(),z['user_xy'].copy()
        for g in GROUPS:
            with np.load(paths[g]/f'trace_{scene}.npz') as z:
                np.testing.assert_array_equal(initial,z['uav_xy'][0])
                np.testing.assert_array_equal(ue,z['user_xy'])
    draws = np.random.default_rng(950001).integers(32,size=(2000,32))
    metrics = ('coverage','last100_coverage','final_coverage','throughput_mbps','return','max_drop_1',
               'max_drop_5','drop_events_10pp','maintain_coverage','disconnection_ratio')
    pairs = []
    comparisons = [(g,'neighbors3') for g in GROUPS if g!='neighbors3']
    comparisons += [('dppo10m',g) for g in RUNS if g!='neighbors3']
    for g,ref in comparisons:
        for key in metrics:
            d = np.array([a[key]-b[key] for a,b in zip(rows[g],rows[ref])])
            lo,hi = np.quantile(d[draws].mean(1),[.025,.975])
            pairs.append(dict(group=g,reference=ref,metric=key,difference=float(d.mean()),low=float(lo),high=float(hi)))
    result = dict(groups=list(GROUPS),best=summaries,sources=sources,paired=pairs,
        neighbor_usage=usage,critic=critic,fixed_scene_events=events,colors=COLORS,
        uncertainty='2000 paired scene bootstrap, seed 950001; not training-seed confidence intervals')
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    legacy.write_csv(out/'best_scene_metrics.csv',[r for g in GROUPS for r in rows[g]])
    legacy.write_csv(out/'paired_scene_differences.csv',pairs)
    (out/'report_sources.json').write_text(json.dumps(dict(evaluation_files=inputs,sources=sources,
        report_script_sha256=legacy.digest(__file__),helper_sha256=legacy.digest(legacy.__file__),
        training_log_files={str(RUNS[g]/n):legacy.digest(RUNS[g]/n) for g in RUNS
            for n in ('metrics.csv','validation.csv','manifest.json','config.yaml')}),indent=2)+'\n')
    make_figures(out,series,rows,paths,draws)
    write_report(out,result)
    print(json.dumps(dict(best=summaries,output=str(out)),ensure_ascii=False),flush=True)


def make_figures(out,series,rows,paths,draws):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style,save_figure
    style()
    figures = out/'figures'
    def line(ax,g,y,x=None):
        if x is None: x=np.arange(1,len(y)+1)
        ax.plot(x,y,color=COLORS[g],ls=LINES[g],marker=MARKERS[g],markevery=max(1,len(y)//8),
            ms=3.5,lw=1.5,label=NAMES[g])
    for name,panels in (
        ('task_curves_best',[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Total throughput (Mbps)'),
            ('reward',1,'Reward: coverage + 0.1 tau'),('discovery_fraction',100,'Sensing (%)')]),
        ('deployment_best',[('disconnection_ratio',100,'Disconnected UAV (%)'),
            ('boundary_within_150_fraction',100,'UAV within 150 m of boundary (%)'),
            ('uav_close_neighbor_fraction',100,'Nearest UAV within 100 m (%)'),('actual_speed_mean',1,'Actual speed (m/s)')])):
        fig,axes=plt.subplots(2,2,figsize=(14,8))
        for ax,(key,scale,label) in zip(axes.flat,panels):
            for g in GROUPS: line(ax,g,series[g][key].mean(0)*scale)
            ax.set(xlabel='Episode step (post-action)',ylabel=label)
        axes.flat[0].legend(fontsize=7,ncol=2)
        save_figure(fig,figures,name)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,key,scale,label in zip(axes.flat,
        ('coverage','last100_coverage','final_coverage','throughput_mbps','max_drop_1','disconnection_ratio'),
        (100,100,100,1,100,100),
        ('Mean coverage (%)','Last-100 coverage (%)','Final coverage (%)','Throughput (Mbps)',
         'Mean episode max single-step drop (pp)','Disconnected UAV (%)')):
        for i,g in enumerate(GROUPS):
            v=np.array([r[key]*scale for r in rows[g]]);lo,hi=np.quantile(v[draws].mean(1),[.025,.975]);m=v.mean()
            ax.bar(i,m,color=COLORS[g]);ax.errorbar(i,m,yerr=[[max(0,m-lo)],[max(0,hi-m)]],color='black',capsize=3)
        ax.set_xticks(range(7),['N3','N4','N5','N6','N7','N8','DPPO\n10M'])
        ax.set(ylabel=label)
    fig.suptitle('Best checkpoints: six 1M baselines vs DPPO 10M; 32-scene bootstrap intervals')
    save_figure(fig,figures,'best_summary')
    for filename,keys,scales,labels in (
        ('validation_learning_1m',('coverage','throughput_bps','return','disconnection_ratio'),
            (100,1e-6,1,100),('Validation coverage (%)','Throughput (Mbps)','Task return','Disconnected UAV (%)')),
        ('loss_and_policy_1m',('actor_loss','value_loss','joint_entropy','ppo_approx_kl'),
            (1,1,1,1),('Actor loss','Value loss','Mean actor entropy','Approximate KL'))):
        fig,axes=plt.subplots(2,2,figsize=(14,8))
        for g in RUNS:
            records=legacy.read(RUNS[g]/('validation.csv' if filename.startswith('validation') else 'metrics.csv'))
            x=np.array([float(r['env_steps']) for r in records])/1e6
            for ax,key,scale,label in zip(axes.flat,keys,scales,labels):
                if key not in records[0]: continue
                y=np.array([float(r[key])*scale for r in records])
                if filename.startswith('loss'):
                    ax.plot(x,y,color=COLORS[g],alpha=.12,lw=.5)
                    y=np.convolve(y,np.ones(15)/15,'valid')
                line(ax,g,y,x[-len(y):]);ax.set(xlabel='Joint transitions (millions)',ylabel=label)
        axes.flat[0].legend(fontsize=7,ncol=2)
        save_figure(fig,figures,filename)
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    for g in GROUPS:
        with np.load(paths[g]/'critic_predictions.npz') as z:
            line(axes[0],g,(z['predicted']-z['mc_return']).mean(0))
            line(axes[1],g,np.sqrt(((z['predicted']-z['mc_return'])**2).mean(0)))
    axes[0].set(xlabel='Pre-action time + 1',ylabel='Value prediction bias')
    axes[1].set(xlabel='Pre-action time + 1',ylabel='Value RMSE')
    axes[0].legend(fontsize=7,ncol=2);save_figure(fig,figures,'critic_best')
    for scene in (10000,10001):
        fig,ax=plt.subplots(figsize=(12,4.8))
        for g in GROUPS: line(ax,g,series[g]['coverage'][legacy.SCENES.index(scene)]*100)
        ax.set(xlabel='Episode step',ylabel='Coverage (%)',title=f'Fixed scene {scene}: unsmoothed coverage')
        ax.legend(fontsize=8,ncol=2);save_figure(fig,figures,f'fixed_scene_{scene}')
        fig,axes=plt.subplots(7,4,figsize=(16,23))
        for row,g in enumerate(GROUPS):
            with np.load(paths[g]/f'trace_{scene}.npz') as z:
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,ue,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions']
                    seen,served=z['sensed'][t].any(0),c.any(0)
                    for mask,marker,col in [(~seen,'.','#aaaaaa'),(seen&~served,'x','#777777'),(served,'o',COLORS[g])]:
                        ax.scatter(ue[mask,0],ue[mask,1],c=col,marker=marker,s=9,linewidths=.5)
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.55,alpha=.7)
                        for k in range(3):
                            if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],'--',color='#555555',lw=.65)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:
                        ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=28,edgecolors='black',linewidths=.5)
                    for i in range(18):ax.annotate(str(i+1),xy[i],fontsize=5,xytext=(2,2),textcoords='offset points')
                    ax.scatter(bs[:,0],bs[:,1],marker='*',c='black',s=75,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[g]} | t={t} | {served.mean():.1%}')
        fig.legend(handles=[Line2D([],[],ls='none',marker=m,color=c,label=l) for m,c,l in
            [('o','#333333','Served UE'),('x','#777777','Sensed only'),('.','#aaaaaa','Unseen'),
             ('^','#333333','Connected UAV'),('s','#333333','Disconnected UAV'),('*','black','BS')]],
            loc='lower center',ncol=6,fontsize=8,bbox_to_anchor=(.5,-.005))
        fig.suptitle(f'Fixed validation scene {scene}: best models; 1M vs 10M budgets',y=1.005)
        save_figure(fig,figures,f'topology_best_{scene}')


def write_report(out,result):
    s=result['best'];pairs=result['paired']
    lines=['# 1M baseline邻居数3–8与DPPO 10M：best模型比较','',
        '仅比较上述七组best，不加入baseline 10M、teacher、旧DPPO阶段或其他奖励消融。'
        '新6／7／8邻居完整运行已下载，冻结best复验；3／4／5和DPPO复用SHA256一致的同口径验证。','',
        '统一32场景10000–10031、400步、动作采样种子880001、批次16、CUDA FP32、关闭TF32。'
        '所有组使用容量10、中心UE初始化、移动UE，奖励为覆盖率＋0.1归一化吞吐，无断联／鲁棒性／边界奖励。'
        '保留测试集未使用。本轮未更新任何模型。','',
        '## 七组best结果','',
        '| 模型 | best实际步数 | 全程覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 | 断联率 | 最大单步跌幅均值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        v=s[g]
        lines.append(f"| {NAMES[g]} | {v['checkpoint_step']:,} | {v['coverage']:.3%} | {v['last100_coverage']:.3%} | {v['final_coverage']:.3%} | {v['throughput_mbps']:.2f} | {v['return']:.3f} | {v['disconnection_ratio']:.2%} | {100*v['max_drop_1']:.2f}pp |")
    n3,n8,dppo = (s[g] for g in ('neighbors3','neighbors8','dppo10m'))
    throughput_pair=next(p for p in pairs if p['group']=='dppo10m' and p['reference']=='neighbors8' and p['metric']=='throughput_mbps')
    lines+=['','## 主要结论','',
        f"1. **增加邻居信息对当前baseline有帮助，但不是单调关系。** 8邻居平均覆盖比3邻居高{100*(n8['coverage']-n3['coverage']):.2f}pp，"
        f"吞吐高{n8['throughput_mbps']-n3['throughput_mbps']:.2f}Mbps；7邻居却低于6邻居。不能据此认为每多一个邻居都必然提高收益。",
        f"2. **8邻居baseline与DPPO的覆盖差距已很小。** DPPO全程覆盖高{100*(dppo['coverage']-n8['coverage']):.2f}pp；"
        f"后100步为{dppo['last100_coverage']:.2%}对{n8['last100_coverage']:.2%}，终态为{dppo['final_coverage']:.2%}对{n8['final_coverage']:.2%}。"
        '覆盖差的配对场景区间跨零，当前验证没有显示明确覆盖优势，但也不能证明两者等价。',
        f"3. **DPPO更突出的差异是数据率。** 相对8邻居吞吐高{throughput_pair['difference']:.2f}Mbps"
        f"（{100*throughput_pair['difference']/n8['throughput_mbps']:.2f}%），场景区间"
        f"[{throughput_pair['low']:.2f}, {throughput_pair['high']:.2f}]Mbps。"
        '这一现象与DPPO可以选择关联和布局相容，但本对照没有隔离关联、策略结构和训练预算的贡献。',
        f"4. **平均部署维持改善，极端断链仍未解决。** 8邻居的后100步覆盖比51–150步高{100*n8['maintain_coverage']:.2f}pp，"
        f"但32/32场景仍发生至少一次≥10pp单步骤降；每回合次数从3邻居{n3['drop_events_10pp']:.2f}降到{n8['drop_events_10pp']:.2f}，"
        f"单回合最大跌幅均值仍为{100*n8['max_drop_1']:.2f}pp。DPPO也有{dppo['scenes_with_10pp_drop']}/32场景出现该类事件。",
        '5. **论文需要更强的邻居观测对照。** 不能只把DPPO与3邻居baseline的差值解释为新算法收益；'
        '后续正式算法对比应统一邻居信息与训练预算，保留当前3–8邻居结果作为观察范围敏感性分析。','']
    winner=max(RUNS,key=lambda g:s[g]['coverage'])
    lines+=['','## 邻居数的影响','',f"当前单种子运行中，平均覆盖最高的baseline为{winner[-1]}邻居（{s[winner]['coverage']:.3%}）。以下均为相对3邻居的配对场景差值，95%区间来自2000次场景bootstrap（种子950001）。",'']
    for g in list(RUNS)[1:]:
        p=next(p for p in pairs if p['group']==g and p['reference']=='neighbors3' and p['metric']=='coverage')
        lines.append(f"- {g[-1]}邻居：平均覆盖{100*p['difference']:+.3f}pp，区间[{100*p['low']:+.3f}, {100*p['high']:+.3f}]pp；吞吐{s[g]['throughput_mbps']-s['neighbors3']['throughput_mbps']:+.2f}Mbps。")
    lines+=['','不能把邻居更多直接等同于性能更好。邻居输入增加了可见信息，也扩大第一层参数量；'
        '每组只有一个训练种子，best选模还依赖固定验证集。区间只描述这些验证场景上的差值，不代表跨训练种子的显著性。','',
        '## 与DPPO 10M比较','']
    for g in RUNS:
        p=next(p for p in pairs if p['group']=='dppo10m' and p['reference']==g and p['metric']=='coverage')
        lines.append(f"- DPPO−N{g[-1]}：覆盖{100*p['difference']:+.3f}pp，场景区间[{100*p['low']:+.3f}, {100*p['high']:+.3f}]pp；吞吐{s['dppo10m']['throughput_mbps']-s[g]['throughput_mbps']:+.2f}Mbps。")
    lines+=['','DPPO在线预算10M，baseline六组各1M，且DPPO有示范／预训练、集合扩散actor和评分Top-K关联。'
        '本表是现有best的表现比较，不能凭它证明同预算算法优势，也不能把差异只归因于扩散模型。','',
        '## 部署维持与骤降','',
        '| 模型 | 后100步−51至150步 | 平均曲线峰值时刻 | 每回合≥10pp单步跌幅次数 | 有≥10pp跌幅的场景 | 靠边150m内 | 近邻<100m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        v=s[g]
        lines.append(f"| {NAMES[g]} | {100*v['maintain_coverage']:+.2f}pp | {v['mean_curve_peak_step']} | {v['drop_events_10pp']:.2f} | {v['scenes_with_10pp_drop']}/32 | {v['boundary150_fraction']:.2%} | {v['close_neighbor_fraction']:.2%} |")
    lines+=['','最大跌幅为每个回合内部相邻执行后状态覆盖差的最大值，再对32场景平均；不是平均覆盖曲线的最大跌幅。'
        '高平均覆盖或低平均断联不保证关键链路稳定。下面是预先固定场景中各策略的最大跌幅事件，属于诊断示例，不代表全部场景。','',
        '| 模型 | 场景 | 时刻 | 覆盖前→后 | 联网UAV前→后 | 失服务UE／其中原服务UAV断联 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        for scene,e in result['fixed_scene_events'][g].items():
            lines.append(f"| {NAMES[g]} | {scene} | {e['time']} | {e['coverage_before']:.1%}→{e['coverage_after']:.1%} | {e['connected_before']}→{e['connected_after']} | {e['lost_users']}／{e['lost_users_previous_server_disconnected']} |")
    lines+=['','### 固定拓扑的含义','',
        '| 模型 | 场景10000终态覆盖／联网数 | 场景10001终态覆盖／联网数 |','|---|---:|---:|']
    for g in GROUPS:
        a,b=[result['fixed_scene_events'][g][str(scene)]['snapshots'][-1] for scene in (10000,10001)]
        lines.append(f"| {NAMES[g]} | {a['coverage']:.2%}／{a['connected']} | {b['coverage']:.2%}／{b['connected']} |")
    lines+=['',
        '8邻居在这两个固定场景终态均有18架联网、78.67%覆盖，终态拓扑表现较好；'
        '但场景10001的t351→352仍有77.33%→4%的短暂崩落，110个失服务UE的原服务UAV全部同时断联。'
        '因此只看终态图会漏掉执行过程中的严重链路风险。',
        '7邻居在场景10001终态聚集且全部失去BS可达性，覆盖为零。'
        '这说明观察槽位变多没有自动转化为维持网关接入的策略；也不代表7邻居设定必然失败，仍需多训练种子验证。',
        '多个互不相连的UU分量若分别接入BS仍属合法服务拓扑，不能仅以UU图不连通判断部署失败。','']
    lines+=['','## 邻居槽位与训练来源','',
        '| 邻居上限 | 平均真实直连邻居 | 平均实际观察 | 所有槽位均可填满 | 真实邻居超过上限 | Actor参数 |',
        '|---|---:|---:|---:|---:|---:|']
    for g,u in result['neighbor_usage'].items():
        lines.append(f"| {g[-1]} | {u['mean_degree']:.2f} | {u['mean_observed']:.2f} | {u['slots_filled_fraction']:.2%} | {u['degree_exceeds_slots']:.2%} | {s[g]['actor_parameters']:,} |")
    lines+=['','这些可见率在各自策略产生的布局中统计，既受观测上限影响，也受策略的聚集／展开行为影响，不是严格同状态信息增益。'
        '新增邻居槽位仍限制在650m物理直连范围，补零不代表可观察远处UAV。','',
        '原3邻居在CPU训练，4–8邻居在CUDA训练；验证统一CUDA，但训练随机流和数值路径仍有差异。'
        '原3邻居历史CPU验证58.63125%，统一CUDA复验56.43844%，主表只使用后者；同一整数种子不保证CPU与CUDA高斯噪声相同。'
        '6／7／8邻居best复验覆盖、回报、吞吐与训练期验证一致（最大绝对差<1e-6）。','',
        '所有best均按完整回合任务回报（覆盖＋0.1吞吐）选择，不单独按覆盖最大化。本次没有另选模型。'
        '模型哈希、原始逐步数据、逐场景表、固定轨迹、MC预测及可重建来源保存在本目录。','',
        '## Critic检查','']
    for g,c in result['critic'].items():
        lines.append(f"- {NAMES[g]}：MC RMSE {c['all']['rmse']:.3f}，MAE {c['all']['mae']:.3f}，偏差{c['all']['bias']:+.3f}。")
    lines+=['','各组使用自己的on-policy轨迹；MC带采样噪声，不能将误差解释为同一状态上的网络比较。','',
        '## 图表','',
        '- [覆盖、吞吐、奖励与感知](figures/task_curves_best.png)',
        '- [七组汇总与骤降](figures/best_summary.png)',
        '- [断联、贴边、聚集与速度](figures/deployment_best.png)',
        '- [六组1M验证学习曲线](figures/validation_learning_1m.png)',
        '- [六组1M loss与策略诊断](figures/loss_and_policy_1m.png)',
        '- [固定场景10000拓扑](figures/topology_best_10000.png)',
        '- [固定场景10001拓扑](figures/topology_best_10001.png)',
        '- [Critic误差](figures/critic_best.png)','',
        '所有图另存PDF／SVG；使用颜色、线型和标记共同区分，不使用紫色。重建命令：','',
        '```bash','.venv-dppo-compat/bin/python scripts/report_mappo_neighbors_3to8.py --rebuild','```','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--evaluate', choices=list(RUNS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rebuild', action='store_true')
    a = p.parse_args()
    if bool(a.evaluate) == bool(a.rebuild): p.error('Choose --evaluate or --rebuild')
    if a.evaluate: evaluate(a)
    else: rebuild(a)
