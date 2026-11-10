"""Saved hop-speed trajectory: identify a cut and audit single-UAV interventions.

No training, no policy inference and no new stochastic trajectory sampling.
Counterfactuals keep post-action UE/other-UAV positions fixed, recompute physical
links and run the existing centralized association once.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.runtime import configure_runtime
from uav_research.coverage_stability import BASELINE_CONFIG

ROOT=Path('analysis/baseline3_best_hop_speed_025_k04')
DEFAULT=ROOT/'critical_link_scene10000_t151'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def connected(adjacency,direct):
    reach=direct.any(1).copy()
    for _ in range(len(reach)):
        after=reach|adjacency[:,reach].any(1)
        if np.array_equal(after,reach):break
        reach=after
    return reach


def audit(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'hop_scaled/trace_10000.npz';sha=digest(source)
    with np.load(source) as z:data={k:z[k].copy() for k in z.files}
    # This case is selected as the largest drop within the two saved scaled traces.
    candidates=[]
    for scene in (10000,10001):
        with np.load(ROOT/f'hop_scaled/trace_{scene}.npz') as z:
            t=int(np.argmax(z['coverage'][1:-1]-z['coverage'][2:])+2)
            candidates.append(dict(scene=scene,time=t,drop=float(z['coverage'][t-1]-z['coverage'][t])))
    chosen=max(candidates,key=lambda c:c['drop'])
    assert chosen['scene']==10000 and chosen['time']==151,chosen
    t=chosen['time'];cfg=load_config(BASELINE_CONFIG);cfg.run.device='cpu';configure_runtime(cfg)
    env=UAVEnvironment(cfg.env);u=cfg.env.num_uavs
    xy0,xy1=data['uav_xy'][t-1:t+1];a0,a1=data['A'][t-1:t+1];b0,b1=data['B'][t-1:t+1]
    ue=data['user_xy'][t];bs=data['bs_positions'];d0,d1=b0==1,b1==1
    before,after=b0.min(1)<=u,b1.min(1)<=u
    lost_nodes=np.flatnonzero(before&~after)
    # Validate saved physical topology and association before any counterfactual.
    for j in (t-1,t,t+1):
        env.set_positions(data['uav_xy'][j],data['user_xy'][j],j)
        for name in ('A','B','C'):np.testing.assert_array_equal(getattr(env,name),data[name][j])
    edge_tests=[]
    lost_edges=np.argwhere(np.triu(a0&~a1,1))
    for i,j in lost_edges:
        restore=a1.copy();restore[i,j]=restore[j,i]=True
        keep=connected(restore,d1)
        endpoints=[]
        for v,other in ((i,j),(j,i)):
            order=sorted([k for k in range(u) if k!=v and a0[v,k]],
                key=lambda k:(float(np.linalg.norm(xy0[k]-xy0[v])),k))
            endpoints.append(dict(uav=int(v+1),pre_position=xy0[v].tolist(),post_position=xy1[v].tolist(),
                displacement=(xy1[v]-xy0[v]).tolist(),distance_m=float(np.linalg.norm(xy1[v]-xy0[v])),
                raw_command=data['action'][t,v].tolist(),scale=float(data['speed_factor'][t,v]),
                pre_hops=int(b0[v].min()),post_hops=int(b1[v].min()),neighbor_rank=order.index(other)+1,
                distance_if_this_uav_held=float(np.linalg.norm(xy0[v]-xy1[other]))))
        remove=a0.copy();remove[i,j]=remove[j,i]=False
        edge_tests.append(dict(uav_pair=[int(i+1),int(j+1)],distance_before=float(np.linalg.norm(xy0[i]-xy0[j])),
            distance_after=float(np.linalg.norm(xy1[i]-xy1[j])),
            pre_margin_m=float(cfg.env.uu_range-np.linalg.norm(xy0[i]-xy0[j])),
            restored_edge_connected_count=int(keep.sum()),
            restored_edge_recovered_uavs=(np.flatnonzero(keep&~after)+1).tolist(),
            removing_this_edge_from_before_connected_count=int(connected(remove,d0).sum()),endpoints=endpoints))
    counterfactual=[];hold_frames={}
    for i in range(u):
        xy=xy1.copy();xy[i]=xy0[i]
        env.set_positions(xy,ue,t)
        reach=env.B.min(1)<=u
        counterfactual.append(dict(held_uav=i+1,coverage=float(env.info['coverage']),
            throughput_mbps=float(env.info['throughput_bps']/1e6),connected_count=int(reach.sum()),
            recovered_uavs=(np.flatnonzero(reach&~after)+1).tolist()))
        if i in (8,11):hold_frames[str(i+1)]={k:np.asarray(getattr(env,k)).copy() for k in ('uav_xy','A','B','C')}
    lost_users=data['C'][t-1].any(0)&~data['C'][t].any(0)
    owner=data['C'][t-1].argmax(0)
    result=dict(scene=10000,time_before=t-1,time_after=t,recovery_time=t+1,
        selected_among_saved_traces=candidates,source=str(source),source_sha256=sha,
        script_sha256=digest(__file__),source_checkpoint_sha256=json.loads((ROOT/'hop_scaled/provenance.json').read_text())['sha256'],
        coverage_before=float(data['coverage'][t-1]),coverage_after=float(data['coverage'][t]),
        coverage_recovery=float(data['coverage'][t+1]),connected_before=int(before.sum()),connected_after=int(after.sum()),
        connected_recovery=int((data['B'][t+1].min(1)<=u).sum()),
        newly_disconnected_uavs=(lost_nodes+1).tolist(),lost_users=int(lost_users.sum()),
        lost_users_previous_server_disconnected=int((lost_users&~after[owner]).sum()),
        lost_uu_edges=edge_tests,lost_ug_edges=(np.argwhere(d0&~d1)+1).tolist(),
        single_uav_hold=counterfactual,
        status='saved trajectory audit only; all counterfactual UE/other UAV post-action positions fixed')
    # Concrete causal checks, rather than choosing an actor from visual inspection.
    assert len(lost_edges)==2 and not result['lost_ug_edges']
    assert counterfactual[11]['connected_count']==18 and counterfactual[8]['connected_count']==9
    assert [r['held_uav'] for r in counterfactual if r['connected_count']>9]==[12]
    assert next(e for e in edge_tests if e['uav_pair']==[9,12])['restored_edge_connected_count']==18
    assert next(e for e in edge_tests if e['uav_pair']==[2,11])['restored_edge_connected_count']==9
    (out/'event.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(out/'counterfactual_hold9.npz',**hold_frames['9'],user_xy=ue,bs_positions=bs)
    np.savez_compressed(out/'counterfactual_hold12.npz',**hold_frames['12'],user_xy=ue,bs_positions=bs)
    make_figures(out,data,hold_frames,result,cfg.env.uu_range)
    write_report(out,result)
    assert digest(source)==sha
    print(json.dumps(result,ensure_ascii=False),flush=True)


def make_figures(out,data,holds,r,limit):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style,save_figure
    style();figdir=out/'figures';t=r['time_after']
    green,blue,orange='#009E73','#0072B2','#E69F00'
    def topology(ax,xy,ue,a,b,c,title):
        bs=data['bs_positions'];reach=b.min(1)<=18;served=c.any(0)
        previous=data['C'][t-1].any(0);lost=previous&~served
        ax.scatter(ue[~served&~lost,0],ue[~served&~lost,1],s=9,c='#bbbbbb',alpha=.8)
        ax.scatter(ue[served,0],ue[served,1],s=12,c=green,alpha=.8)
        ax.scatter(ue[lost,0],ue[lost,1],s=18,c=blue,marker='x',linewidths=.8)
        for i in range(18):
            for j in range(i):
                if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],c='#777777',lw=.8,alpha=.8,zorder=1)
            for k in range(3):
                if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],ls='--',c='#888888',lw=.8,zorder=1)
        ax.plot(xy[[8,11],0],xy[[8,11],1],c=orange,lw=2.5,ls='-' if a[8,11] else '--',zorder=2)
        for mask,marker,col in [(reach,'^',green),(~reach,'s',blue)]:
            ax.scatter(xy[mask,0],xy[mask,1],s=55,marker=marker,c=col,edgecolors='black',linewidths=.6,zorder=3)
        ax.scatter(xy[8,0],xy[8,1],s=115,marker='D',facecolors='white',edgecolors='black',linewidths=1.8,zorder=4)
        ax.scatter(xy[11,0],xy[11,1],s=140,marker='P',c=orange,edgecolors='black',linewidths=1.2,zorder=4)
        for i in range(18):ax.annotate(str(i+1),xy[i],fontsize=8,fontweight='bold',xytext=(5,4),textcoords='offset points',zorder=5)
        ax.scatter(bs[:,0],bs[:,1],c='black',marker='*',s=140,clip_on=False)
        ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
    fig,axes=plt.subplots(2,2,figsize=(13,12))
    for ax,j in zip(axes.flat[:3],(t-1,t,t+1)):
        xy,ue,a,b,c=[data[key][j] for key in ('uav_xy','user_xy','A','B','C')]
        topology(ax,xy,ue,a,b,c,f'Observed t={j}: coverage={c.any(0).mean():.2%}, connected={(b.min(1)<=18).sum()}/18')
    hold=holds['12'];c=hold['C']
    topology(axes.flat[3],hold['uav_xy'],data['user_xy'][t],hold['A'],hold['B'],c,
        f'Counterfactual t={t}: hold U12 only\ncoverage={c.any(0).mean():.2%}, connected=18/18')
    handles=[Line2D([],[],ls='none',marker=m,color=col,label=label) for m,col,label in
        [('^',green,'Connected UAV'),('s',blue,'Disconnected UAV'),('D','black','U9: gateway'),
         ('P',orange,'U12: relay'),('o',green,'Served UE'),('x',blue,'Lost service vs t150')]]
    handles.append(Line2D([],[],color=orange,lw=2.5,ls='--',label='U9-U12 (dashed = absent)'))
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.025),fontsize=9)
    fig.suptitle('Hop-scaled policy, fixed scene 10000: the t150 -> t151 cut and one-step recovery')
    save_figure(fig,figdir,'topology_before_after_recovery_hold12')
    # Detailed movement views show the physical wall at exactly 650 m.
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,v in zip(axes.flat[:2],(8,11)):
        p0,p1=data['uav_xy'][t-1:t+1,v];move=p1-p0
        ax.scatter(*p0,c=green,s=65,label='t150');ax.scatter(*p1,c=blue,s=65,label='t151')
        ax.annotate('',xy=p1,xytext=p0,arrowprops=dict(arrowstyle='->',color=orange,lw=2.5))
        pad=2.5;ax.set(xlim=(min(p0[0],p1[0])-pad,max(p0[0],p1[0])+pad),ylim=(min(p0[1],p1[1])-pad,max(p0[1],p1[1])+pad),aspect='equal',
            xlabel='x (m)',ylabel='y (m)',title=f'U{v+1}: actual movement {np.linalg.norm(move):.3f} m\ndx={move[0]:+.3f}, dy={move[1]:+.3f}')
        ax.legend()
    times=np.arange(140,161);dist=np.linalg.norm(data['uav_xy'][times,8]-data['uav_xy'][times,11],axis=1)
    ax=axes[1,0];ax.plot(times,dist,'o-',c=blue,label='Observed U9-U12 distance')
    ax.axhline(limit,c=orange,ls='--',label='650 m UU threshold');ax.axvline(t,c='black',ls=':')
    ax.set(xlabel='Episode state time',ylabel='Distance (m)',title='Repeated crossing of a hard communication threshold');ax.legend(fontsize=8)
    ax=axes[1,1];cases=['Before\nt150','Observed\nt151','Hold U9\nt151','Hold U12\nt151','Recovered\nt152']
    xy0,xy1=data['uav_xy'][t-1:t+1]
    vals=[np.linalg.norm(xy0[8]-xy0[11]),np.linalg.norm(xy1[8]-xy1[11]),
          np.linalg.norm(xy0[8]-xy1[11]),np.linalg.norm(xy1[8]-xy0[11]),np.linalg.norm(data['uav_xy'][t+1,8]-data['uav_xy'][t+1,11])]
    colors=[green if v<=limit else blue for v in vals]
    ax.bar(range(5),vals,color=colors);ax.axhline(limit,c=orange,ls='--');ax.set_xticks(range(5),cases)
    ax.set(ylabel='U9-U12 distance (m)',ylim=(625,657),title='Single-UAV hold interventions (truncated y-axis)')
    for i,v in enumerate(vals):ax.text(i,v+.25,f'{v:.2f}',ha='center',fontsize=9)
    save_figure(fig,figdir,'critical_link_movement_and_threshold')
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True);times=np.arange(130,171)
    axes[0].plot(times,100*data['coverage'][times],c=green,marker='o',ms=3,label='Observed coverage')
    axes[0].scatter([t],[100*holds['12']['C'].any(0).mean()],c=orange,marker='P',s=100,label='Hold U12 only at t151')
    axes[0].set(ylabel='Coverage (%)');axes[0].legend()
    count=(data['B'][times].min(2)<=18).sum(1)
    axes[1].step(times,count,where='mid',c=blue,label='Connected UAV')
    axes[1].scatter([t],[18],c=orange,marker='P',s=100,label='Hold U12 only at t151')
    axes[1].set(xlabel='Episode state time',ylabel='Connected UAV count',ylim=(0,19));axes[1].legend()
    for ax in axes:ax.axvline(t,c='black',ls=':')
    save_figure(fig,figdir,'event_timeline')


def write_report(out,r):
    edges=r['lost_uu_edges'];critical=next(e for e in edges if e['uav_pair']==[9,12])
    h12=r['single_uav_hold'][11];h9=r['single_uav_hold'][8]
    lines=['# 跳数缩放组：场景10000，第151次动作的关键链路失效','',
        '**具体触发动作来自UAV 12。** 它在t150→151移动约4.50m，将U9–U12距离推过650m硬阈值。'
        '固定其他17架UAV的动作后位置及同一步UE位置，仅让U12保持t150位置，全部18架仍可联网；逐架冻结18次中，只有冻结U12恢复了全部9架断联UAV。','',
        '这是从本轮跳数缩放组已保存的两个代表场景10000／10001中，挑选最大单步跌幅的案例；不声称是全部32场景中的最坏事件。'
        '完全读取已有轨迹，无重新训练、无actor调用、无随机轨迹重采样。','',
        '## 发生时间与后果','',
        '| 状态 | 覆盖率 | 联网UAV |','|---|---:|---:|',
        f"| t150：第151次动作执行前 | {r['coverage_before']:.2%}（104/150） | {r['connected_before']}/18 |",
        f"| t151：第151次动作执行后 | {r['coverage_after']:.2%}（47/150） | {r['connected_after']}/18 |",
        f"| t152：下一次动作执行后 | {r['coverage_recovery']:.2%}（101/150） | {r['connected_recovery']}/18 |",'',
        f"单步下降38个百分点，{r['lost_users']}个UE失去服务；其中{r['lost_users_previous_server_disconnected']}个UE原服务UAV在同一步断联。"
        f"新断联UAV编号为{r['newly_disconnected_uavs']}。本步没有直接UAV–BS接入边失效。",'',
        '## 失效链路与运动','',
        '| 失效UU边 | 运动前距离 | 运动后距离 | 仅恢复这条边后的联网数 |',
        '|---|---:|---:|---:|']
    for e in edges:lines.append(f"| U{e['uav_pair'][0]}–U{e['uav_pair'][1]} | {e['distance_before']:.3f}m | {e['distance_after']:.3f}m | {e['restored_edge_connected_count']}/18 |")
    lines+=['','U2–U11也在同一步失效，但两端仍有BS可达路径。单独恢复U2–U11不能恢复这个失联子网；恢复U9–U12则能恢复全部9架。'
        '此处“仅恢复一条边”是拓扑干预，用于检查路径依赖，不将超距边解释为合法可通信链路。','',
        f"关键边U9–U12原有安全余量仅{critical['pre_margin_m']:.3f}m。两端在执行前互为第2近邻，都在3邻居观察范围内，U9最短1跳、U12最短2跳。",'',
        '| UAV | 原始动作(dx,dy) | α | 实际位移(dx,dy)，m | 实际路程 |',
        '|---|---|---:|---|---:|']
    for p in critical['endpoints']:
        cmd=p['raw_command'];d=p['displacement']
        lines.append(f"| U{p['uav']} | ({cmd[0]:+.6f}, {cmd[1]:+.6f}) | {p['scale']:.6f} | ({d[0]:+.6f}, {d[1]:+.6f}) | {p['distance_m']:.3f}m |")
    lines+=['',
        'U12向远离U9的方向运动是本步主要触发因素；U9的运动实际上部分抵消了距离增长。'
        '两者实际都没有碰到场地边界，因此这次失效来自机间距离阈值，与边界反射无关。', '',
        '## 单步反事实：谁的运动触发了断链','',
        '| 同一步布局 | U9–U12距离 | 联网数 | 重新计算后的覆盖 |',
        '|---|---:|---:|---:|',
        f"| 实际执行 | {critical['distance_after']:.3f}m | 9/18 | {r['coverage_after']:.2%} |",
        f"| 仅冻结U9，其他照常 | {critical['endpoints'][0]['distance_if_this_uav_held']:.3f}m | {h9['connected_count']}/18 | {h9['coverage']:.2%} |",
        f"| 仅冻结U12，其他照常 | {critical['endpoints'][1]['distance_if_this_uav_held']:.3f}m | {h12['connected_count']}/18 | {h12['coverage']:.2%} |",'',
        '反事实固定t151的UE位置及其他UAV位置，只将指定UAV恢复到t150位置；重新计算原无线、路由和集中关联。'
        '它是一步部署干预，没有从t151继续生成新策略轨迹。UE继续运动及其他UAV位置变化仍会影响SINR与关联，'
        '所以恢复18架联网后覆盖为68%，没有机械地复制t150的69.33%。逐架冻结的完整18条记录在event.json中。','',
        '## 为什么减速后仍然会发生','',
        '跳数缩放降低位移幅度，却没有直接约束链路距离或使用安全余量。U9–U12执行前仅剩约1.40m余量，'
        'U12即使已经从原策略指令速度缩到49.74%，约4.50m的移动仍足以越过阈值。'
        '网络存在依赖这一桥接链路的9机子网，一条边断开会同时使多架UAV和大量UE失联。',
        't151断联后，U12下一步按规则恢复α=1，输出动作约(−0.6026,−0.1466)，向回移动；'
        't152时U9–U12距离降到631.72m，链路和18架UAV可达性恢复。'
        '这是短暂断链与恢复事件，附近还存在反复跨阈值，不能只看终态拓扑判断稳定性。',
        '这说明“关键邻居已经可见”和“按跳数减速”仍不足以保证链路保持；'
        '并不证明所有失败都由同一UAV、同一链路或同一种机制引起。','',
        '## 图表','',
        '- [实际t150／151／152拓扑与仅冻结U12的反事实](figures/topology_before_after_recovery_hold12.png)',
        '- [U9与U12真实位移、650m阈值及冻结对照](figures/critical_link_movement_and_threshold.png)',
        '- [事件附近覆盖与联网数量时间线](figures/event_timeline.png)','',
        '全部另存PDF／SVG。重建命令不重新采样：','',
        '```bash','.venv-dppo-compat/bin/python scripts/analyze_hop_speed_break.py','```','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    audit(p.parse_args())
