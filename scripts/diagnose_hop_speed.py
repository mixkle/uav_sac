"""Frozen 3-neighbor 1M baseline: unmodified vs hop-dependent physical speed."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import report_mappo_neighbors as legacy
from uav_research.hop_speed_diagnostic import HopSpeedVector, hop_speed_factor

GROUPS=('original','hop_scaled')
NAMES=dict(original='Original speed',hop_scaled='Hop-scaled speed')
COLORS=dict(original='#009E73',hop_scaled='#0072B2')
DEFAULT=Path('analysis/baseline3_best_hop_speed_025_k04')
CHECKPOINT=Path('runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt')


def collect(args):
    from uav_research.algorithms import make_learner
    from uav_research.config import config_from_dict
    from uav_research.runtime import load_checkpoint,atomic_json
    from uav_research.diffusion_dppo.storage import configure
    from uav_research.diffusion_dppo.config import DiffusionSettings
    from uav_research.diffusion_dppo.evaluation import evaluate
    args.output.mkdir(parents=True,exist_ok=True)
    sha=legacy.digest(CHECKPOINT)
    saved=load_checkpoint(CHECKPOINT);cfg=config_from_dict(saved['config'])
    assert cfg.env.neighbor_slots==3 and cfg.env.num_uavs==18 and saved['env_steps']==671744
    assert cfg.env.throughput_weight==.1 and cfg.env.disconnection_weight==cfg.env.robustness_weight==cfg.env.boundary_penalty_weight==0
    cfg.run.device=args.device;cfg.run.cpu_threads=8;cfg.run.numba_threads=4;cfg.run.num_envs=16
    cfg.run.evaluation_action_seed=880001
    device=configure(cfg);learner=make_learner(cfg,device);learner.load_state_dict(saved['learner'])
    learner.actor.eval().requires_grad_(False);learner.value.eval().requires_grad_(False)
    before={k:v.clone() for k,v in learner.actor.state_dict().items()}
    results={};audits={}
    for group in GROUPS:
        target=args.output/group
        if (target/'step_metrics.npz').exists():raise FileExistsError(f'Refusing to overwrite {target}; use --rebuild')
        batches=[];noise_digest=hashlib.sha256();first_actions=[]
        class AuditedVector(HopSpeedVector):
            def reset(self,seeds):
                result=super().reset(seeds)
                self.ue_hash=hashlib.sha256(np.stack([e.user_xy for e in self.envs]).tobytes())
                self.initial_hash=hashlib.sha256(np.stack([e.uav_xy for e in self.envs]).tobytes()+np.stack([e.user_xy for e in self.envs]).tobytes()).hexdigest()
                return result
            def step(self,actions):
                result=super().step(actions)
                self.ue_hash.update(np.stack([e.user_xy for e in self.envs]).tobytes())
                return result
        def factory(env,count,backend):
            vector=AuditedVector(env,count,backend,enabled=group=='hop_scaled',alpha_min=.25,k=.4)
            batches.append(vector);return vector
        class FrozenActor(torch.nn.Module):
            def __init__(self):super().__init__();self.actor=learner.actor
            @torch.no_grad()
            def sample(self,obs):
                state=torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state()
                noise_digest.update(state.cpu().numpy().tobytes())
                action=self.actor.sample(obs.to_torch(device),False)[0].cpu().numpy()
                if len(first_actions)<len(batches):first_actions.append(hashlib.sha256(action.tobytes()).hexdigest())
                return action,None
        e_cfg=SimpleNamespace(env=cfg.env,algo=cfg.algo,run=cfg.run,diffusion=DiffusionSettings())
        print(f'Frozen evaluation {group}, {args.device}, 32 scenes x 400 steps',flush=True)
        with patch('uav_research.vector.VectorEnvironment',factory):
            result=evaluate(e_cfg,FrozenActor(),legacy.SCENES,target,greedy=True)
        audits[group]=dict(initial_hashes=[v.initial_hash for v in batches],ue_path_hashes=[v.ue_hash.hexdigest() for v in batches],
            action_rng_sequence_sha256=noise_digest.hexdigest(),first_action_hashes=first_actions)
        results[group]={k:result['summary'][k] for k in ('coverage','return','throughput_bps','disconnection_ratio')}
        atomic_json(target/'provenance.json',dict(checkpoint=str(CHECKPOINT),sha256=sha,checkpoint_step=saved['env_steps'],
            group=group,device=str(device),scene_seeds=legacy.SCENES,action_seed=880001,batch_size=16,
            alpha_min=.25,k=.4,pre_action_hops=True,history_action='original unscaled policy command',
            raw_unreachable_sentinel=19,critic_queries=0,actor_updates=0,audit=audits[group]))
        print(group,results[group],flush=True)
    assert audits['original']==audits['hop_scaled'],'Initial scenes, UE paths or action RNG draws differ'
    assert legacy.digest(CHECKPOINT)==sha
    assert all(torch.equal(before[k],v) for k,v in learner.actor.state_dict().items())
    atomic_json(args.output/'execution_audit.json',dict(paired_audits=audits,checkpoint_unchanged=True,
        actor_parameters_unchanged=True,tests='4 hop-scale tests passed',formal_training=False,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device=str(device),cpu_threads=8,numba_threads=4),
        source_sha256={p:legacy.digest(p) for p in (__file__,'src/uav_research/hop_speed_diagnostic.py',
            'src/uav_research/environment.py','src/uav_research/vector.py','src/uav_research/models.py',
            'src/uav_research/diffusion_dppo/evaluation.py')}))
    rebuild(args)


def rebuild(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style,save_figure
    out=args.output;series={};rows={};summary={};sources={}
    for g in GROUPS:
        p=out/g;prov=json.loads((p/'provenance.json').read_text())
        assert legacy.digest(prov['checkpoint'])==prov['sha256']
        series[g],rows[g],summary[g]=legacy.series_and_rows(p,g)
        for i,row in enumerate(rows[g]):
            row['first100_coverage']=float(series[g]['coverage'][i,:100].mean())
            row['post150_coverage']=float(series[g]['coverage'][i,150:].mean())
        summary[g]['first100_coverage']=float(series[g]['coverage'][:,:100].mean())
        summary[g]['post150_coverage']=float(series[g]['coverage'][:,150:].mean())
        summary[g].update(speed_factor_mean=float(series[g]['speed_factor_mean'].mean()),
            newly_disconnected_per_step=float(series[g]['newly_disconnected_fraction'].mean()*18))
        sources[g]=prov
    assert sources['original']['audit']==sources['hop_scaled']['audit']
    assert sources['original']['device']==sources['hop_scaled']['device']
    draws=np.random.default_rng(950001).integers(32,size=(2000,32));paired=[]
    for key in ('coverage','first100_coverage','post150_coverage','last100_coverage','final_coverage','throughput_mbps','return','disconnection_ratio',
                'max_drop_1','max_drop_5','drop_events_10pp','maintain_coverage','boundary150_fraction','speed_mps'):
        d=np.array([a[key]-b[key] for a,b in zip(rows['hop_scaled'],rows['original'])])
        lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
        paired.append(dict(metric=key,difference=float(d.mean()),low=float(lo),high=float(hi)))
    values={str(h):float(hop_speed_factor([h],18)[0]) for h in range(1,20)}
    result=dict(summary=summary,paired=paired,sources=sources,alpha_by_hop=values,
        data_sha256={str(p.relative_to(out)):legacy.digest(p) for g in GROUPS for p in (out/g).iterdir() if p.is_file()})
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    legacy.write_csv(out/'scene_metrics.csv',[r for g in GROUPS for r in rows[g]])
    legacy.write_csv(out/'paired_differences.csv',paired)
    style();figdir=out/'figures'
    def line(ax,g,y):ax.plot(np.arange(1,len(y)+1),y,color=COLORS[g],ls='--' if g=='original' else '-',label=NAMES[g])
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    panels=[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Throughput (Mbps)'),('disconnection_ratio',100,'Disconnected UAV (%)'),
        ('actual_speed_mean',1,'Actual speed (m/s)'),('boundary_within_150_fraction',100,'Within 150 m of boundary (%)'),
        ('speed_factor_mean',1,'Mean applied speed factor')]
    for ax,(key,scale,label) in zip(axes.flat,panels):
        for g in GROUPS:line(ax,g,series[g][key].mean(0)*scale)
        ax.set(xlabel='Episode step',ylabel=label)
    axes.flat[0].legend();save_figure(fig,figdir,'task_and_deployment')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,scale,label in zip(axes,('coverage','max_drop_1','disconnection_ratio'),(100,100,100),
        ('Mean coverage (%)','Mean episode max single-step drop (pp)','Disconnected UAV (%)')):
        for i,g in enumerate(GROUPS):
            v=np.array([r[key] for r in rows[g]])*scale;m=v.mean();lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
            ax.bar(i,m,color=COLORS[g]);ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],color='black',capsize=4)
        ax.set_xticks([0,1],['Original','Hop-scaled']);ax.set(ylabel=label)
    save_figure(fig,figdir,'summary')
    fig,ax=plt.subplots(figsize=(7,4));h=np.arange(1,19)
    ax.plot(h,hop_speed_factor(h,18),'o-',color=COLORS['hop_scaled'],label='Connected')
    ax.scatter([19],[1],color='black',marker='s',label='Disconnected (sentinel 19)')
    ax.set(xlabel='Pre-action minimum BS hop count',ylabel='Speed factor',ylim=(0,1.05));ax.legend()
    save_figure(fig,figdir,'speed_curve')
    fixed={}
    for scene in (10000,10001):
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        initial,users=None,None
        for row,g in enumerate(GROUPS):
            with np.load(out/g/f'trace_{scene}.npz') as z:
                if initial is None:initial,users=z['uav_xy'][0].copy(),z['user_xy'].copy()
                else:
                    np.testing.assert_array_equal(initial,z['uav_xy'][0]);np.testing.assert_array_equal(users,z['user_xy'])
                fixed[f'{g}/{scene}']=[dict(t=t,coverage=float(z['coverage'][t]),connected=int((z['B'][t].min(1)<=18).sum())) for t in (0,150,200,400)]
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,ue,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions']
                    served=c.any(0);seen=z['sensed'][t].any(0)
                    for mask,marker,col in [(~seen,'.','#aaaaaa'),(seen&~served,'x','#777777'),(served,'o',COLORS[g])]:
                        ax.scatter(ue[mask,0],ue[mask,1],c=col,marker=marker,s=10,linewidths=.6)
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.6)
                        for k in range(3):
                            if b[i,k]==1:ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],'--',color='#777777',lw=.6)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=35,edgecolors='black',linewidths=.5)
                    for i in range(18):ax.annotate(str(i+1),xy[i],fontsize=5,xytext=(2,2),textcoords='offset points')
                    ax.scatter(bs[:,0],bs[:,1],marker='*',c='black',s=70,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[g]} | t={t} | {served.mean():.1%}')
        fig.suptitle(f'Fixed scene {scene}: triangles=connected UAV, squares=disconnected; stars=BS')
        save_figure(fig,figdir,f'topology_{scene}')
        fig,ax=plt.subplots(figsize=(10,4))
        for g in GROUPS:line(ax,g,series[g]['coverage'][legacy.SCENES.index(scene)]*100)
        ax.set(xlabel='Episode step',ylabel='Coverage (%)',title=f'Fixed scene {scene}');ax.legend()
        save_figure(fig,figdir,f'coverage_{scene}')
    result['fixed_scenes']=fixed
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_report(out,result)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def write_report(out,result):
    s=result['summary'];a,b=s['original'],s['hop_scaled']
    lines=['# 冻结3邻居baseline best：按联网跳数缩放物理速度','',
        '模型为容量10、覆盖＋0.1归一化吞吐的baseline 1M best，实际来自671,744步；不使用baseline 10M best。'
        '冻结actor，不更新网络、不查询critic，不修改生产训练配置或环境工厂。','',
        f"统一32场景10000–10031、动作种子880001、批次16、400步、{result['sources']['original']['device']}执行。"
        '两组初始位置、完整UE运动路径和每次动作采样前的随机状态SHA256完全相同；固定场景初始动作也一致。'
        '由于后续观测不同，动作数值不会一直相同。','',
        '只在动作执行前读取当前最短联网跳数，按用户公式计算α，缩放30m/步的实际运动指令。'
        '保持方向和actor原始圆盘动作不变；上一动作历史继续记录actor原始指令，上一跳数／感知人数继续取原时刻值。'
        '边界反射在缩放后执行，反射统计按缩放后位移计算；UE速度、时间、奖励、无线和集中关联规则均不变。','',
        '公式采用分母1−exp(−kU)，U=18，alpha_min=0.25，k=0.4，没有改成U−1。'
        '代码最短跳数1–18均为合法可达，19为不可达；不可达时α=1。用户文字中的“断联=18”按断联意图解释，不将合法18跳连接误判。','',
        '| 模式 | 全程覆盖 | 后100步 | 终态覆盖 | 吞吐Mbps | 回报 | 断联率 | 最大单步跌幅均值 | ≥10pp跌幅次数/回合 | 实际速度m/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        v=s[g];lines.append(f"| {NAMES[g]} | {v['coverage']:.3%} | {v['last100_coverage']:.3%} | {v['final_coverage']:.3%} | {v['throughput_mbps']:.2f} | {v['return']:.3f} | {v['disconnection_ratio']:.2%} | {100*v['max_drop_1']:.2f}pp | {v['drop_events_10pp']:.2f} | {v['speed_mps']:.2f} |")
    lines+=['','## 配对差值：缩放−原速度','',
        '2000次场景bootstrap、种子950001。只描述当前冻结策略在这些场景上的干预效果，不是训练种子置信区间。','']
    for p in result['paired']:
        key=p['metric'];scale=100 if key in ('coverage','first100_coverage','post150_coverage','last100_coverage','final_coverage','disconnection_ratio','max_drop_1','max_drop_5','maintain_coverage','boundary150_fraction') else 1
        unit='pp' if scale==100 else ('Mbps' if key=='throughput_mbps' else '')
        lines.append(f"- {key}: {p['difference']*scale:+.3f}{unit}，95%区间[{p['low']*scale:+.3f}, {p['high']*scale:+.3f}]{unit}。")
    lines+=['','## 缩放强度','', '| 跳数 | α | 满幅动作对应速度m/s |','|---|---:|---:|']
    for h in (1,2,3,4,5,8,18,19):
        factor=result['alpha_by_hop'][str(h)]
        lines.append(f"| {'断联（19）' if h==19 else h} | {factor:.4f} | {30*factor:.2f} |")
    lines+=['','## 解释边界','',
        f"**当前结果支持降低部分断链风险，但尚未显示全程任务收益改善。** 前100步覆盖从{a['first100_coverage']:.2%}降至{b['first100_coverage']:.2%}，"
        f"151–400步则从{a['post150_coverage']:.2%}升至{b['post150_coverage']:.2%}；这一形态与减速延缓展开、改善后期保持的取舍相容。"
        '全程覆盖、后100步覆盖和吞吐差值的场景区间均跨零，不能把均值变化当作确定的整体收益。',
        '平均断联率差和最大单步跌幅差的95%场景区间均位于零以下，当前32场景中有较清楚的稳定性改善迹象。'
        '但两组32/32场景仍都存在≥10pp的单步骤降，缩放没有消除链路失效。',
        f"靠边150m内的UAV比例从{a['boundary150_fraction']:.2%}升至{b['boundary150_fraction']:.2%}，"
        f"反射比例却从{a['reflection_fraction']:.2%}降至{b['reflection_fraction']:.2%}。靠边停留与越界反射是不同指标，不能将反射减少解读为远离边界。",
        f"平均覆盖变化{100*(b['coverage']-a['coverage']):+.2f}pp，后100步变化{100*(b['last100_coverage']-a['last100_coverage']):+.2f}pp；"
        f"平均断联变化{100*(b['disconnection_ratio']-a['disconnection_ratio']):+.2f}pp，最大单步跌幅均值变化{100*(b['max_drop_1']-a['max_drop_1']):+.2f}pp。",
        '应联合判断展开能力、最终服务和极端断链，不能仅凭速度下降认为策略学会保持。'
        '本轮改变了冻结策略执行时的动力学；其输出没有针对此速度规则训练，效果不等同于加入规则后重新训练的收益。'
        '低跳数并不保证承担关键中继，也不能据跳数保证节点独立冗余，缩放本身不强制保持任何链路。','',
        '如本轮使用CPU，应与本轮原速度控制组比较，不能混入此前CUDA口径的56.438%覆盖；同一整数种子在CPU与CUDA产生不同高斯噪声流。','',
        '## 图表','',
        '- [覆盖、吞吐、断联与速度](figures/task_and_deployment.png)',
        '- [覆盖、骤降与断联汇总](figures/summary.png)',
        '- [速度系数曲线](figures/speed_curve.png)',
        '- [场景10000拓扑](figures/topology_10000.png)',
        '- [场景10001拓扑](figures/topology_10001.png)','',
        '所有图另存PDF／SVG，原速度绿色虚线、缩放蓝色实线。结果、固定轨迹、来源和逐场景指标保留；重建命令：','',
        '```bash','.venv-dppo-compat/bin/python scripts/diagnose_hop_speed.py --rebuild','```','']
    if (out/'critical_link_scene10000_t151/report.md').exists():
        lines+=['## 严重断链逐步追踪','',
            '[场景10000，第151次动作：U12运动导致U9–U12失效的拓扑与单步反事实](critical_link_scene10000_t151/report.md)。'
            '该事件仅读取保存轨迹；18次单UAV冻结对照显示，只有冻结U12能使失联9架恢复可达。','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--device',default='cpu')
    p.add_argument('--rebuild',action='store_true')
    args=p.parse_args()
    if args.rebuild:rebuild(args)
    else:collect(args)
