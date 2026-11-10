"""Frozen baseline: hop speed plus a local stationary-upstream safety filter."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import report_mappo_neighbors as legacy
from uav_research.link_safety_diagnostic import LinkSafetyVector

ROOT=Path('analysis/baseline3_best_hop_speed_025_k04')
DEFAULT=Path('analysis/baseline3_best_hop_speed_link_safety_delta5')
CHECKPOINT=Path('runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt')
GROUPS=('original','hop_scaled','link_safe')
LABELS=dict(original='Original speed',hop_scaled='Hop-scaled',link_safe='Hop + link safety (5 m)')
COLORS=dict(original='#555555',hop_scaled='#009E73',link_safe='#0072B2')
STYLES=dict(original=':',hop_scaled='--',link_safe='-')


def collect(args):
    from uav_research.config import config_from_dict
    from uav_research.runtime import load_checkpoint,atomic_json
    from uav_research.algorithms import make_learner
    from uav_research.diffusion_dppo.storage import configure
    from uav_research.diffusion_dppo.config import DiffusionSettings
    from uav_research.diffusion_dppo.evaluation import evaluate
    out=args.output;target=out/'link_safe';out.mkdir(parents=True,exist_ok=True)
    if (target/'step_metrics.npz').exists():raise FileExistsError('Use --rebuild; refusing to overwrite a saved rollout')
    sha=legacy.digest(CHECKPOINT);saved=load_checkpoint(CHECKPOINT);cfg=config_from_dict(saved['config'])
    assert cfg.env.neighbor_slots==3 and saved['env_steps']==671744
    cfg.run.device='cpu';cfg.run.cpu_threads=8;cfg.run.numba_threads=4;cfg.run.num_envs=16
    cfg.run.evaluation_action_seed=880001
    device=configure(cfg);learner=make_learner(cfg,device);learner.load_state_dict(saved['learner'])
    learner.actor.eval().requires_grad_(False);learner.value.eval().requires_grad_(False)
    weights={k:v.clone() for k,v in learner.actor.state_dict().items()}
    batches=[];noise=hashlib.sha256();first=[]
    class AuditedVector(LinkSafetyVector):
        def reset(self,seeds):
            result=super().reset(seeds)
            self.ue_hash=hashlib.sha256(np.stack([e.user_xy for e in self.envs]).tobytes())
            self.initial_hash=hashlib.sha256(np.stack([e.uav_xy for e in self.envs]).tobytes()+np.stack([e.user_xy for e in self.envs]).tobytes()).hexdigest()
            return result
        def step(self,actions):
            result=super().step(actions);self.ue_hash.update(np.stack([e.user_xy for e in self.envs]).tobytes())
            return result
    def factory(env,count,backend):
        v=AuditedVector(env,count,backend,margin=5);batches.append(v);return v
    class FrozenActor(torch.nn.Module):
        def __init__(self):super().__init__();self.actor=learner.actor
        @torch.no_grad()
        def sample(self,obs):
            noise.update(torch.get_rng_state().numpy().tobytes())
            action=self.actor.sample(obs.to_torch(device),False)[0].cpu().numpy()
            if len(first)<len(batches):first.append(hashlib.sha256(action.tobytes()).hexdigest())
            return action,None
    ecfg=SimpleNamespace(env=cfg.env,algo=cfg.algo,run=cfg.run,diffusion=DiffusionSettings())
    print('Frozen link-safety evaluation: delta=5 m; 32 scenes x 400 steps; CPU',flush=True)
    with patch('uav_research.vector.VectorEnvironment',factory):
        evaluate(ecfg,FrozenActor(),legacy.SCENES,target,greedy=True)
    audit=dict(initial_hashes=[v.initial_hash for v in batches],ue_path_hashes=[v.ue_hash.hexdigest() for v in batches],
        action_rng_sequence_sha256=noise.hexdigest(),first_action_hashes=first)
    reference=json.loads((ROOT/'hop_scaled/provenance.json').read_text())
    assert audit==reference['audit'],'UE paths/initial positions/action RNG mismatch'
    assert legacy.digest(CHECKPOINT)==sha
    assert all(torch.equal(weights[k],v) for k,v in learner.actor.state_dict().items())
    atomic_json(target/'provenance.json',dict(checkpoint=str(CHECKPOINT),sha256=sha,checkpoint_step=saved['env_steps'],
        device='cpu',scene_seeds=legacy.SCENES,action_seed=880001,batch_size=16,alpha_min=.25,k=.4,margin_m=5,
        parent='visible strictly lower-hop neighbor; tie: distance then ID',constraint='pre-action upstream position',
        infeasible='s=0; flag infeasible',no_visible_parent='apply hop scaling only',
        history_action='unscaled raw actor command',actor_updates=0,critic_queries=0,audit=audit,
        source_sha256={p:legacy.digest(p) for p in (__file__,'src/uav_research/link_safety_diagnostic.py','src/uav_research/hop_speed_diagnostic.py')}))
    rebuild(args)


def rebuild(args):
    from uav_research.plotting import style,save_figure
    import matplotlib.pyplot as plt
    out=args.output;paths={g:(out/g if g=='link_safe' else ROOT/g) for g in GROUPS}
    series={};rows={};summary={};sources={}
    for g,p in paths.items():
        sources[g]=json.loads((p/'provenance.json').read_text())
        assert sources[g]['device']=='cpu' and sources[g]['scene_seeds']==legacy.SCENES and sources[g]['action_seed']==880001
        assert legacy.digest(sources[g]['checkpoint'])==sources[g]['sha256']
        series[g],rows[g],summary[g]=legacy.series_and_rows(p,g)
        summary[g]['first100_coverage']=float(series[g]['coverage'][:,:100].mean())
        summary[g]['post150_coverage']=float(series[g]['coverage'][:,150:].mean())
    assert sources['link_safe']['audit']==sources['hop_scaled']['audit']==sources['original']['audit']
    s=series['link_safe'];active=float(s['link_constraint_count'].sum());infeasible=float(s['link_constraint_infeasible_count'].sum())
    diagnostics={k:float(s[k].sum()) for k in s if k.startswith('link_constraint_')}
    diagnostics.update(eligible_parent_events=active,infeasible_fraction=infeasible/active if active else 0,
        constrained_fraction=active/(32*400*18),
        eligible_no_upstream_events=float(s['link_constraint_no_upstream_count'].sum()),
        mean_scale=float(s['speed_factor_mean'].mean()))
    assert diagnostics['link_constraint_static_violation_count']==0
    assert diagnostics['link_constraint_reflected_static_violation_count']==0
    draws=np.random.default_rng(950001).integers(32,size=(2000,32));paired=[]
    for ref in ('original','hop_scaled'):
        for key in ('coverage','last100_coverage','final_coverage','throughput_mbps','return','disconnection_ratio','max_drop_1',
                    'max_drop_5','drop_events_10pp','maintain_coverage','speed_mps','boundary150_fraction'):
            d=np.array([a[key]-b[key] for a,b in zip(rows['link_safe'],rows[ref])]);lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
            paired.append(dict(reference=ref,metric=key,difference=float(d.mean()),low=float(lo),high=float(hi)))
    inputs={str(p/f):legacy.digest(p/f) for p in paths.values() for f in ('provenance.json','step_metrics.npz','trace_10000.npz','trace_10001.npz')}
    result=dict(summary=summary,diagnostics=diagnostics,paired=paired,sources=sources,input_sha256=inputs,old_event=audit_old_event(out))
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    legacy.write_csv(out/'scene_metrics.csv',[r for g in GROUPS for r in rows[g]])
    legacy.write_csv(out/'paired_differences.csv',paired)
    style();figdir=out/'figures'
    def line(ax,g,y):ax.plot(np.arange(1,len(y)+1),y,color=COLORS[g],ls=STYLES[g],label=LABELS[g])
    for name,panels in (
        ('task_and_deployment',[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Throughput (Mbps)'),
            ('disconnection_ratio',100,'Disconnected UAV (%)'),('actual_speed_mean',1,'Actual speed (m/s)'),
            ('boundary_within_150_fraction',100,'Within 150 m of boundary (%)'),('speed_factor_mean',1,'Mean applied scale')]),
        ('stability_and_sensing',[('discovery_fraction',100,'Sensing (%)'),('uav_close_neighbor_fraction',100,'Nearest UAV within 100 m (%)'),
            ('newly_disconnected_fraction',18,'Newly disconnected UAV / step'),('boundary_reflection_fraction',100,'Boundary reflection (%)')])):
        fig,axes=plt.subplots(2,3 if len(panels)==6 else 2,figsize=(15 if len(panels)==6 else 12,8))
        for ax,(key,scale,label) in zip(axes.flat,panels):
            for g in GROUPS:line(ax,g,series[g][key].mean(0)*scale)
            ax.set(xlabel='Episode step',ylabel=label)
        axes.flat[0].legend(fontsize=8);save_figure(fig,figdir,name)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,scale,label in zip(axes,('coverage','max_drop_1','disconnection_ratio'),(100,100,100),
        ('Mean coverage (%)','Mean episode max single-step drop (pp)','Disconnected UAV (%)')):
        for i,g in enumerate(GROUPS):
            v=np.array([r[key] for r in rows[g]])*scale;m=v.mean();lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
            ax.bar(i,m,color=COLORS[g]);ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],color='black',capsize=3)
        ax.set_xticks(range(3),['Original','Hop only','Hop + safety']);ax.set(ylabel=label)
    save_figure(fig,figdir,'summary')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,key,label in zip(axes.flat,
        ('link_constraint_count','link_constraint_infeasible_count','link_constraint_moving_margin_violation_count','link_constraint_actual_broken_count'),
        ('Selected upstream constraints / step','Infeasible scalar constraints / step',
         'Selected links outside 645 m after both move','Selected links broken (>650 m) after both move')):
        ax.plot(np.arange(1,401),s[key].mean(0),c=COLORS['link_safe']);ax.set(xlabel='Episode step',ylabel=label)
    save_figure(fig,figdir,'constraint_diagnostics')
    for scene in (10000,10001):
        fig,axes=plt.subplots(3,4,figsize=(16,12))
        for row,g in enumerate(GROUPS):
            with np.load(paths[g]/f'trace_{scene}.npz') as z:
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,ue,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions'];served=c.any(0)
                    ax.scatter(ue[~served,0],ue[~served,1],s=7,c='#aaaaaa')
                    ax.scatter(ue[served,0],ue[served,1],s=10,c=COLORS[g])
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],c='#888888',lw=.6)
                        for k in range(3):
                            if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],'--',c='#888888',lw=.6)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=35,edgecolors='black',linewidths=.5)
                    for i in range(18):ax.annotate(str(i+1),xy[i],fontsize=5,xytext=(2,2),textcoords='offset points')
                    ax.scatter(bs[:,0],bs[:,1],c='black',marker='*',s=70,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{LABELS[g]} | t={t} | {served.mean():.1%}')
        fig.suptitle(f'Fixed scene {scene}: triangles connected; squares disconnected; stars BS')
        save_figure(fig,figdir,f'topology_{scene}')
        fig,ax=plt.subplots(figsize=(11,4))
        for g in GROUPS:line(ax,g,series[g]['coverage'][legacy.SCENES.index(scene)]*100)
        ax.set(xlabel='Episode step',ylabel='Coverage (%)',title=f'Fixed scene {scene}');ax.legend()
        save_figure(fig,figdir,f'coverage_{scene}')
    write_report(out,result)
    print(json.dumps(dict(summary=summary,diagnostics=diagnostics),ensure_ascii=False),flush=True)


def audit_old_event(out):
    from uav_research.config import load_config
    from uav_research.environment import UAVEnvironment,reflect
    from uav_research.hop_speed_diagnostic import hop_speed_factor
    from uav_research.link_safety_diagnostic import visible_parent,maximum_link_scale
    cfg=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    env=UAVEnvironment(cfg.env)
    source=ROOT/'hop_scaled/trace_10000.npz'
    with np.load(source) as z:
        t=151;xy=z['uav_xy'][t-1];env.set_positions(xy,z['user_xy'][t-1],t-1)
        upper=hop_speed_factor(env.B.min(1),18);scales=upper.copy()
        parents=np.full(18,-1,np.int64);feasible=np.zeros(18,bool)
        nominal=z['action'][t]*cfg.env.max_speed*cfg.env.dt
        for i in range(18):
            j,_,_=visible_parent(env.obs.fixed,cfg.env,i,require_lower_hop=True)
            if j>=0:
                parents[i]=j
                scales[i],feasible[i]=maximum_link_scale(xy[i]-xy[j],nominal[i],upper[i],645)
        new=reflect(xy+nominal*scales[:,None],[3500,3500])
        env.set_positions(new,z['user_xy'][t],t)
        result=dict(source=str(source),source_sha256=legacy.digest(source),t=t,
            u12_parent=int(parents[11]+1),u12_feasible=bool(feasible[11]),u12_scale=float(scales[11]),
            distance_9_12=float(np.linalg.norm(new[8]-new[11])),coverage=float(env.info['coverage']),
            connected=int((env.B.min(1)<=18).sum()),
            scope='All-UAV filter on old t150 positions/saved t151 raw commands; old t151 UE fixed; one-step diagnostic only')
        assert result['u12_parent']==9 and not result['u12_feasible'] and result['u12_scale']==0
        assert result['connected']==18
        (out/'old_event_one_step.json').write_text(json.dumps(result,indent=2)+'\n')
        np.savez_compressed(out/'old_event_one_step.npz',uav_xy=new,user_xy=env.user_xy,A=env.A,B=env.B,C=env.C,
            scale=scales,parents=parents,feasible=feasible,bs_positions=cfg.env.bs_positions)
        return result


def write_report(out,r):
    lines=['# 跳数缩放＋局部链路安全约束：δ=5m','',
        '冻结容量10、3邻居baseline 1M best（实际671,744步），不训练、不查询critic。'
        '比较原速度、仅跳数缩放、跳数缩放＋链路约束；旧两组复用上一轮相同CPU口径原始数据，不重新采样。'
        '新组仍为32场景10000–10031、动作种子880001、批次16、400步、CPU8线程／Numba4线程。'
        '新旧初始位置、完整UE路径、动作随机状态序列及首步原始动作逐项哈希一致。','',
        '## 实施规则','',
        '- 沿用alpha_min=0.25、k=0.4的跳数缩放，断联保持alpha=1。',
        '- 多跳联网UAV只在当前可见3邻居中选严格更低跳数、且跳数最小者。平局按距离、编号。未读取不可见邻居或上游动作。',
        '- 直连BS、断联以及无可见更低跳数邻居的UAV仅按原跳数缩放，不加链路约束；最后一种情况单独记录。',
        '- 安全距离为650−5=645m。保持actor方向，求满足 ||p_i+s*delta_p_actor−p_j_old||≤645、0≤s≤alpha 的最大s。',
        '- 无可行解时s=0并明确记录不可行；停住不等于已满足645m约束。若原位在安全圈外、动作向内且能够进入，求解器会保留相应可行区间。',
        '- 所有UAV同时、独立处理；上游使用动作前位置。它不是联合动作约束，不能保证动作后相对移动上游的链路安全。',
        '- 保留原动作历史；缩放后镜面反射；UE、无线、关联、奖励均不变。actor、critic和生产训练配置均未修改。','',
        '## 任务与稳定性','',
        '| 模式 | 平均覆盖 | 前100步 | 后100步 | 终态覆盖 | 吞吐Mbps | 断联率 | 最大单步跌幅均值 | 实际速度m/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g,v in r['summary'].items():
        lines.append(f"| {LABELS[g]} | {v['coverage']:.3%} | {v['first100_coverage']:.3%} | {v['last100_coverage']:.3%} | {v['final_coverage']:.3%} | {v['throughput_mbps']:.2f} | {v['disconnection_ratio']:.2%} | {100*v['max_drop_1']:.2f}pp | {v['speed_mps']:.2f} |")
    h,l=r['summary']['hop_scaled'],r['summary']['link_safe']
    lines+=['',
        f"**相对仅按跳数减速，当前冻结策略的平均服务与稳定性均改善。** 平均覆盖{100*(l['coverage']-h['coverage']):+.2f}pp、"
        f"吞吐{l['throughput_mbps']-h['throughput_mbps']:+.2f}Mbps，断联率{100*(l['disconnection_ratio']-h['disconnection_ratio']):+.2f}pp，"
        f"最大单步跌幅均值{100*(l['max_drop_1']-h['max_drop_1']):+.2f}pp；这些配对场景差值区间均未跨零。",
        f"后100步覆盖{100*(l['last100_coverage']-h['last100_coverage']):+.2f}pp，但终态覆盖{100*(l['final_coverage']-h['final_coverage']):+.2f}pp，"
        '两项区间均跨零，不能认定所有阶段都有收益。相比原速度，前期展开仍较慢。',
        f"≥10pp单步骤降次数由每回合{h['drop_events_10pp']:.2f}降到{l['drop_events_10pp']:.2f}，"
        f"新组仍有{l['scenes_with_10pp_drop']}/32场景发生该类骤降，链路风险没有消失。",'']
    d=r['diagnostics'];active=d['eligible_parent_events'];feasible=active-d['link_constraint_infeasible_count']
    lines+=['','## 约束是否实际成立','',
        f"- 有选定上游的约束机会：{int(active):,}个UAV时隙，占全部时隙{d['constrained_fraction']:.2%}。",
        f"- 无可行缩放解：{int(d['link_constraint_infeasible_count']):,}次，占有上游机会{d['infeasible_fraction']:.2%}，均取s=0。",
        f"- 缺少可见更低跳数邻居而跳过：{int(d['eligible_no_upstream_events']):,}次。",
        f"- s小于跳数缩放上限而受到限制：{int(d['link_constraint_limited_count']):,}次。",
        f"- 对判为可行的{int(feasible):,}次，相对上游旧位置的公式违反次数为{int(d['link_constraint_static_violation_count'])}；边界反射后为{int(d['link_constraint_reflected_static_violation_count'])}。",
        f"- 双方移动后，选定链路距离超过645m：{int(d['link_constraint_moving_margin_violation_count']):,}次；超过650m实际断边：{int(d['link_constraint_actual_broken_count']):,}次。",
        f"- 其中公式本来可行、但上游移动后实际断边：{int(d['link_constraint_feasible_but_actual_broken_count']):,}次。",'',
        '这些是所选父边在UAV时隙中的计数，可重复出现于同一对节点，不是独立场景或唯一断链事件数量。'
        '公式可行、无可行解退回0、无上游跳过与上游移动后的实际断链分别统计，避免把未满足的约束算成安全。','',
        '直接接入BS的上游仍可移动最多7.5m/步，已经超过5m裕量；多跳上游的允许位移还可能更大。'
        '因此即使子节点严格落在上游旧位置的645m圆内，双方移动后的距离仍可能超过650m。',
        '', '## 上一轮t150→151断链的同起点检查','',
        f"保持上一轮场景10000在t150的位置、保存的下一动作和t151的UE位置，用本轮过滤器只做一次几何与关联重算。"
        f"U12仍选U9作上游，但原距离648.60m已大于645m，沿原动作方向无可行s，按约定取0。"
        f"其他UAV也执行本轮过滤后，U9–U12距离为{r['old_event']['distance_9_12']:.3f}m，"
        f"联网{r['old_event']['connected']}/18、覆盖{r['old_event']['coverage']:.2%}（原事件31.33%）。",
        '此例阻止了原先的大规模断联，但依靠的是“不可行时停住”回退；没有满足645m安全圈，不能记作可行约束成功。'
        '这只是固定原起点的一步诊断，与新策略从t0完整执行到t151的状态不同。','',
        '## 与仅跳数减速的配对差值','',
        '32场景配对bootstrap 2000次、种子950001；不是训练种子置信区间。','']
    for p in r['paired']:
        if p['reference']!='hop_scaled':continue
        key=p['metric'];scale=100 if key in ('coverage','last100_coverage','final_coverage','disconnection_ratio','max_drop_1','max_drop_5','maintain_coverage','boundary150_fraction') else 1
        unit='pp' if scale==100 else ('Mbps' if key=='throughput_mbps' else '')
        lines.append(f"- {key}: {scale*p['difference']:+.3f}{unit}，95%区间[{scale*p['low']:+.3f}, {scale*p['high']:+.3f}]{unit}。")
    lines+=['','## 判断边界','',
        '先联合检查覆盖、吞吐、展开速度、断联和骤降，不能把冻结／减速本身当作收益。'
        '本轮严格实施相对旧上游位置的用户约束，未增加上游速度预测、动作交换、BS安全距离或下游依赖约束。'
        '上游直接联网UAV可能离开BS接入区，父节点选择也可能逐步切换，因此局部旧位置约束并非全网连通性保证。'
        '本轮是冻结策略的执行干预，不等同于使用该动力学重新训练。'
        '三个控制组均为CPU采样；原速度58.63125%复现历史CPU口径，未混用CUDA的56.43844%。','',
        '## 图表','',
        '- [覆盖、吞吐、断联和速度](figures/task_and_deployment.png)',
        '- [稳定性、感知与反射](figures/stability_and_sensing.png)',
        '- [三组指标汇总](figures/summary.png)',
        '- [可行性与移动上游诊断](figures/constraint_diagnostics.png)',
        '- [场景10000拓扑](figures/topology_10000.png)',
        '- [场景10001拓扑](figures/topology_10001.png)','',
        '全部图另存PDF／SVG。保存全部逐步标量及两个代表场景的上游选择、缩放系数和可行标记；原两组数据不复制、不修改。',
        '重建命令：','', '```bash','.venv-dppo-compat/bin/python scripts/diagnose_link_safety.py --rebuild','```','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--rebuild',action='store_true');a=p.parse_args()
    if a.rebuild:rebuild(a)
    else:collect(a)
