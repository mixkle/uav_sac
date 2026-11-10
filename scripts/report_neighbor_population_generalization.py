"""Frozen 1M neighbor-3/4/5 actors: UE/UAV population transfer, no critic or training."""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from uav_research.population_generalization import SCENARIOS,SUBSET_SEED

RUNS={
    'neighbors3':Path('runs/mappo_coverage_throughput_capacity10_1m_seed0'),
    'neighbors4':Path('runs/mappo_capacity10_neighbors4_1m_seed0'),
    'neighbors5':Path('runs/mappo_capacity10_neighbors5_1m_seed0'),
}
SCENES=list(range(660000,660064))
NAMES=dict(original='18 UAV / 150 UE',ue60='18 UAV / 90 UE',ue80='18 UAV / 120 UE',
           uav14='14 UAV / 150 UE',uav10='10 UAV / 150 UE')
COLORS=dict(neighbors3='#444444',neighbors4='#009E73',neighbors5='#E69F00')
LINES=dict(neighbors3=':',neighbors4='-',neighbors5='--')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def check_saved_prefix(actor,cfg,device,group):
    """Eight steps must reproduce previous original-size CUDA execution exactly."""
    from uav_research.population_generalization import PopulationVector,paired_action
    from uav_research.runtime import isolated_rng,seed_all
    cache=Path('analysis/mappo_neighbors_1m_vs_10m_seed0/best')/group
    source=json.loads((cache/'provenance.json').read_text())
    assert digest(RUNS[group]/'best.pt')==source['sha256']
    with isolated_rng(),np.load(cache/'trace_10000.npz') as a,np.load(cache/'trace_10001.npz') as b:
        seed_all(880001)
        with PopulationVector(cfg.env,cfg.env,16) as vector:
            obs,_,_=vector.reset(list(range(10000,10016)))
            for t in range(8):
                action=paired_action(actor,obs,cfg.env,cfg.env,vector.active_ids)
                obs,_,_,_,_,_=vector.step(action)
                for env,z in zip(vector.envs[:2],(a,b)):
                    np.testing.assert_array_equal(env.last_action,z['action'][t+1])
                    np.testing.assert_array_equal(env.uav_xy,z['uav_xy'][t+1])
                    np.testing.assert_array_equal(env.user_xy,z['user_xy'][t+1])
    return dict(scenes=list(range(10000,10016)),steps=8,exact_action_and_trace=True)


def evaluate_scenario(actor,cfg,scenario,output):
    from uav_research.population_generalization import PopulationVector,paired_action
    from uav_research.runtime import isolated_rng,seed_all,atomic_json
    from uav_research.topk_diagnostic import atomic_npz
    from uav_research.coverage_stability import coverage_drop_metrics
    from uav_research.diffusion_dppo.storage import Progress
    e=copy.deepcopy(cfg.env);e.num_uavs,e.num_users=SCENARIOS[scenario]
    output.mkdir(parents=True,exist_ok=True)
    cfg_check=copy.deepcopy(cfg);cfg_check.env=e;cfg_check.validate()
    progress=Progress(30);started=time.perf_counter()
    all_metrics,all_agents,rows,mappings=[],[],[],[]
    keys=None
    with isolated_rng():
        seed_all(880001)
        for start in range(0,len(SCENES),16):
            seeds=SCENES[start:start+16]
            with PopulationVector(e,cfg.env,len(seeds)) as vector:
                obs,_,_=vector.reset(seeds)
                for env in vector.envs:
                    mappings.append(dict(scene_seed=env.initial_seed,
                        active_uav_ids=(env.active_ids+1).tolist(),active_ue_ids=(env.user_ids+1).tolist()))
                frames=[[env.trace_frame()] if start+i<2 else [] for i,env in enumerate(vector.envs)]
                metrics,agents=[],[]
                for t in range(e.horizon):
                    action=paired_action(actor,obs,e,cfg.env,vector.active_ids)
                    obs,_,reward,done,truncated,infos=vector.step(action)
                    if keys is None:keys=sorted(k for k,v in infos[0].items() if np.isscalar(v))
                    metrics.append([[info[k] for k in keys] for info in infos])
                    agent_rows=[]
                    for i,env in enumerate(vector.envs):
                        assert (env.C.sum(0)<=1).all() and (env.C.sum(1)<=e.capacity).all()
                        assert not (env.C.astype(bool)&~(env.sensed&(env.Y>=e.sinr_threshold_db)&
                            (env.B.min(1)[:,None]<=e.num_uavs))).any()
                        agent_rows.append(np.stack([env.C.sum(1),env.B.min(1),env.A.sum(1)-1,env.sensed.sum(1)],-1))
                        if frames[i]:frames[i].append(env.trace_frame())
                    agents.append(agent_rows)
                    expected=np.array([info['coverage']+.1*info['tau'] for info in infos])
                    np.testing.assert_allclose(reward,expected,rtol=2e-7,atol=2e-7)
                    if truncated.any() or (t==e.horizon-1 and not done.all()):
                        raise AssertionError('Incorrect finite-task termination')
                    progress(f'{scenario} scenes={seeds[0]}..{seeds[-1]} t={t+1}/{e.horizon}')
                data=np.asarray(metrics).transpose(1,0,2);all_metrics.append(data)
                all_agents.append(np.asarray(agents).transpose(1,0,2,3))
                for i,seed in enumerate(seeds):
                    s=dict(zip(keys,data[i].T));rho=s['coverage'];throughput=s['throughput_bps']/1e6
                    ceiling=min(1,e.num_uavs*e.capacity/e.num_users)
                    row=dict(scene_seed=seed,**coverage_drop_metrics(rho))
                    row.update(throughput_mbps=float(throughput.mean()),last100_throughput_mbps=float(throughput[-100:].mean()),
                        served_users=float(rho.mean()*e.num_users),
                        throughput_per_ue_mbps=float(throughput.mean()/e.num_users),
                        capacity_ceiling=ceiling,coverage_over_capacity_ceiling=float(rho.mean()/ceiling),
                        disconnection_ratio=float(s['disconnection_ratio'].mean()),
                        sensing=float(s['discovery_fraction'].mean()),
                        nearest_neighbor_m=float(s['uav_nearest_neighbor_mean'].mean()),
                        close_neighbor_fraction=float(s['uav_close_neighbor_fraction'].mean()),
                        boundary150_fraction=float(s['boundary_within_150_fraction'].mean()),
                        capacity_rejected_users=float(s['capacity_rejected_users'].mean()),
                        backhaul_rejected_users=float(s['backhaul_rejected_users'].mean()),
                        downlink_rejected_users=float(s['downlink_rejected_users'].mean()),
                        sensing_rejected_users=float(s['sensing_rejected_users'].mean()),
                        actual_speed_mps=float(s['actual_speed_mean'].mean()),return_=float(s['reward'].sum()))
                    row['return']=row.pop('return_');rows.append(row)
                    if frames[i]:
                        atomic_npz(output/f'trace_{seed}.npz',**{
                            k:np.asarray([frame[k] for frame in frames[i]]) for k in frames[i][0]},
                            bs_positions=np.asarray(e.bs_positions),width=e.width,height=e.height,scene_seed=seed)
    data=np.concatenate(all_metrics)
    assert data.shape[:2]==(64,400) and np.isfinite(data).all()
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='scene_seed'}
    summary.update(scenes=64,num_uavs=e.num_uavs,num_users=e.num_users,
        elapsed_seconds=time.perf_counter()-started,
        served_rate_mbps=summary['throughput_mbps']/summary['served_users'] if summary['served_users'] else None,
        physical_observation_dim=e.fixed_obs_dim,actor_observation_dim=cfg.env.fixed_obs_dim)
    write_csv(output/'episodes.csv',rows)
    atomic_npz(output/'step_metrics.npz',metrics=data,names=np.asarray(keys),scene_seeds=SCENES)
    atomic_npz(output/'agent_metrics.npz',metrics=np.concatenate(all_agents),
        names=['served_users','hop','direct_degree','sensed_users'],scene_seeds=SCENES)
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'scene_mapping.json',mappings)
    atomic_json(output/'physical_config.json',cfg_check.as_dict())
    return summary


def evaluate_all(args):
    import torch
    from uav_research.config import config_from_dict
    from uav_research.models import GaussianActor
    from uav_research.runtime import load_checkpoint,atomic_json,configure_runtime
    group=args.evaluate;cp=RUNS[group]/'best.pt';before=digest(cp)
    saved=load_checkpoint(cp);cfg=config_from_dict(saved['config'])
    assert cfg.run.total_steps==1_000_000 and cfg.env.num_uavs==18 and cfg.env.num_users==150
    assert cfg.env.neighbor_slots==int(group[-1]) and cfg.env.capacity==10
    assert cfg.env.throughput_weight==.1 and cfg.env.robustness_weight==cfg.env.disconnection_weight==0
    assert cfg.algo.actor_sharing=='shared' and cfg.algo.mappo_state_dependent_std
    original_device=cfg.run.device;cfg.run.device=args.device
    device=configure_runtime(cfg);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a=cfg.algo
    actor=GaussianActor(cfg.env.fixed_obs_dim,a.hidden,a.log_std_min,a.log_std_max,
        state_dependent_std=True,ue_hidden=a.ue_hidden,count_scale=cfg.env.capacity).to(device)
    actor.load_state_dict(saved['learner']['actor']);actor.eval().requires_grad_(False)
    checkpoint_step=int(saved['env_steps'])
    del saved
    prefix=check_saved_prefix(actor,cfg,device,group)
    if args.check:
        print(json.dumps(dict(group=group,check='passed',prefix=prefix,formal_training=False)),flush=True)
        return
    actor_before={k:v.clone() for k,v in actor.state_dict().items()}
    for scenario in args.scenarios:
        out=args.output/group/scenario
        print(f'Evaluate {group}, {scenario}; frozen best step={checkpoint_step}',flush=True)
        summary=evaluate_scenario(actor,cfg,scenario,out)
        assert digest(cp)==before
        assert all(torch.equal(v,actor_before[k]) for k,v in actor.state_dict().items())
        assert all(p.grad is None for p in actor.parameters())
        atomic_json(out/'provenance.json',dict(checkpoint=str(cp),sha256=before,
            env_steps=checkpoint_step,training_budget=1_000_000,checkpoint_kind='best',
            training_device=original_device,device=args.device,scene_seeds=SCENES,action_seed=880001,
            batch_size=16,subset_seed=SUBSET_SEED,neighbor_slots=cfg.env.neighbor_slots,scenario=scenario,
            full_noise_bank_uavs=18,original_users_motion_bank=150,critic_used=False,weights_frozen=True,
            prefix_reproduction=prefix,script_sha256=digest(__file__),
            adapter_sha256=digest('src/uav_research/population_generalization.py')))
        print(json.dumps(dict(group=group,scenario=scenario,**{k:summary[k] for k in
            ('coverage','last100_coverage','throughput_mbps','disconnection_ratio','capacity_ceiling')})),flush=True)


def rebuild(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style,save_figure
    out=args.output;series,summary,rows,sources={},{},{},{}
    previous=out/'report_sources.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        for name,sha in old['evaluation_files'].items():
            assert digest(out/name)==sha,f'Changed evaluation file: {name}'
    for g in RUNS:
        series[g],summary[g],rows[g],sources[g]={},{},{},{}
        for scenario in SCENARIOS:
            p=out/g/scenario;prov=json.loads((p/'provenance.json').read_text())
            assert prov['sha256']==digest(RUNS[g]/'best.pt') and prov['weights_frozen'] and not prov['critic_used']
            assert prov['scene_seeds']==SCENES and prov['action_seed']==880001 and prov['batch_size']==16
            with np.load(p/'step_metrics.npz') as z:
                assert z['scene_seeds'].tolist()==SCENES
                s=dict(zip(z['names'].tolist(),z['metrics'].transpose(2,0,1)))
            np.testing.assert_allclose(s['reward'],s['coverage']+.1*s['tau'],atol=2e-7)
            u,m=SCENARIOS[scenario];ceiling=min(1,u*10/m)
            assert (s['coverage']<=ceiling+1e-12).all()
            rejected=sum(s[k] for k in ('sensing_rejected_users','downlink_rejected_users',
                                       'backhaul_rejected_users','capacity_rejected_users'))
            np.testing.assert_allclose(rejected+s['coverage']*m,m,atol=1e-10)
            summary[g][scenario]=json.loads((p/'summary.json').read_text())
            with (p/'episodes.csv').open() as f:rr=list(csv.DictReader(f))
            rows[g][scenario]=[{k:float(v) for k,v in r.items()} for r in rr]
            assert [int(r['scene_seed']) for r in rows[g][scenario]]==SCENES
            series[g][scenario]=s;sources[g][scenario]=prov
    # Paired cohorts: identical retained UE paths, UAV initial positions and IDs.
    for seed in SCENES[:2]:
        with np.load(out/'neighbors3'/'original'/f'trace_{seed}.npz') as z:
            reference_uav=z['uav_xy'][0].copy();reference_ue=z['user_xy'].copy()
        for g in RUNS:
            with np.load(out/g/'original'/f'trace_{seed}.npz') as z:
                full_uav=z['uav_xy'][0].copy();full_ue=z['user_xy'].copy()
                np.testing.assert_array_equal(full_uav,reference_uav)
                np.testing.assert_array_equal(full_ue,reference_ue)
            for scenario,(u,m) in SCENARIOS.items():
                with np.load(out/g/scenario/f'trace_{seed}.npz') as z:
                    ids=z['active_uav_ids'][0];ue_ids=z['active_ue_ids'][0]
                    np.testing.assert_array_equal(z['uav_xy'][0],full_uav[ids])
                    np.testing.assert_array_equal(z['user_xy'],full_ue[:,ue_ids])
                    assert z['C'].shape==(401,u,m) and z['A'].shape==(401,u,u)
                    assert (z['C'].sum(1)<=1).all() and (z['C'].sum(2)<=10).all()
                    np.testing.assert_allclose(z['C'].sum((1,2))/m,z['coverage'])
    draws=np.random.default_rng(950001).integers(64,size=(2000,64));paired=[]
    for g in RUNS:
        for scenario in tuple(SCENARIOS)[1:]:
            for key in ('coverage','last100_coverage','final_coverage','served_users','throughput_mbps',
                        'throughput_per_ue_mbps','disconnection_ratio','max_drop_1','maintain_coverage'):
                d=np.array([a[key]-b[key] for a,b in zip(rows[g][scenario],rows[g]['original'])])
                lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                paired.append(dict(group=g,scenario=scenario,reference='same actor original size',metric=key,
                    difference=float(d.mean()),low=float(lo),high=float(hi)))
    for scenario in SCENARIOS:
        for g in ('neighbors4','neighbors5'):
            for key in ('coverage','last100_coverage','throughput_mbps','disconnection_ratio','max_drop_1'):
                d=np.array([a[key]-b[key] for a,b in zip(rows[g][scenario],rows['neighbors3'][scenario])])
                lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                paired.append(dict(group=g,scenario=scenario,reference='neighbors3',metric=key,
                    difference=float(d.mean()),low=float(lo),high=float(hi)))
    result=dict(summary=summary,paired=paired,sources=sources,scenarios=SCENARIOS,scene_seeds=SCENES,
        action_seed=880001,bootstrap=dict(unit='diagnostic scene',replicates=2000,seed=950001,training_seed_ci=False))
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_csv(out/'per_scene_metrics.csv',[dict(group=g,scenario=scenario,**r) for g in RUNS for scenario in SCENARIOS for r in rows[g][scenario]])
    write_csv(out/'paired_scene_differences.csv',paired)
    style();figdir=out/'figures';xs=np.arange(5)
    for stem,panels in [
        ('population_summary',[('coverage',100,'Coverage (%)'),('last100_coverage',100,'Last-100 coverage (%)'),
            ('throughput_mbps',1,'Total throughput (Mbps)'),('disconnection_ratio',100,'Disconnected UAV (%)')]),
        ('capacity_and_service',[('served_users',1,'Mean served UE'),('coverage_over_capacity_ceiling',100,'Coverage / capacity-only upper bound (%)'),
            ('throughput_per_ue_mbps',1,'Total throughput / all UE (Mbps)'),('max_drop_1',100,'Mean episode maximum single-step drop (pp)')])]:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,(key,scale,label) in zip(axes.flat,panels):
            for gi,g in enumerate(RUNS):
                means=[];errs=[]
                for scenario in SCENARIOS:
                    v=np.array([r[key]*scale for r in rows[g][scenario]])
                    means.append(v.mean());lo,hi=np.quantile(v[draws].mean(1),[.025,.975]);errs.append([max(0,v.mean()-lo),max(0,hi-v.mean())])
                ax.errorbar(xs+(gi-1)*.12,means,yerr=np.array(errs).T,color=COLORS[g],ls=LINES[g],
                            marker=['o','s','^'][gi],capsize=3,label=f'{g[-1]} neighbors')
            ax.set_xticks(xs,[NAMES[s].replace(' / ','\n') for s in SCENARIOS]);ax.set(ylabel=label)
        axes.flat[0].legend();fig.suptitle('Frozen 1M best actors; 64 paired diagnostic scenes; no fine-tuning')
        save_figure(fig,figdir,stem)
    for metric,scale,label in [('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Total throughput (Mbps)'),('disconnection_ratio',100,'Disconnected UAV (%)')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,scenario in zip(axes.flat,SCENARIOS):
            for gi,g in enumerate(RUNS):ax.plot(np.arange(1,401),series[g][scenario][metric].mean(0)*scale,
                color=COLORS[g],ls=LINES[g],marker=['o','s','^'][gi],markevery=65,markersize=3,label=f'{g[-1]} neighbors')
            ax.set(title=NAMES[scenario],xlabel='Episode step',ylabel=label)
            if scenario=='uav10' and metric=='coverage':ax.axhline(100*2/3,color='#777777',ls='-.',label='Capacity ceiling 66.67%')
        axes.flat[-1].axis('off');axes.flat[0].legend();save_figure(fig,figdir,f'{metric}_curves')
    # Stage attribution is descriptive, not an exact recoverable-gain decomposition.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    stage_keys=('sensing_rejected_users','downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users')
    for ax,g in zip(axes,RUNS):
        base=np.zeros(5)
        for key,col,label in zip(stage_keys,('#aaaaaa','#0072B2','#E69F00','#222222'),('Sensing','Downlink','Backhaul','Capacity')):
            v=np.array([summary[g][s][key]/SCENARIOS[s][1]*100 for s in SCENARIOS])
            ax.bar(xs,v,bottom=base,color=col,label=label);base+=v
        ax.set_xticks(xs,['Original','UE90','UE120','UAV14','UAV10'],rotation=25);ax.set(title=f'{g[-1]} neighbors',ylabel='Rejected UE (%)')
    axes[0].legend(fontsize=7);save_figure(fig,figdir,'service_stage_rejections')
    for scenario in ('ue60','uav14','uav10'):
        for seed in SCENES[:2]:
            fig,axes=plt.subplots(3,3,figsize=(12,12))
            for row,g in enumerate(RUNS):
                with np.load(out/g/scenario/f'trace_{seed}.npz') as z:
                    u,m=SCENARIOS[scenario]
                    for ax,t in zip(axes[row],(0,150,400)):
                        xy,ue,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions'];served=c.any(0);seen=z['sensed'][t].any(0)
                        for mask,marker,col in [(~seen,'.','#aaaaaa'),(seen&~served,'x','#777777'),(served,'o',COLORS[g])]:
                            ax.scatter(ue[mask,0],ue[mask,1],s=9,c=col,marker=marker,linewidths=.6)
                        for i in range(u):
                            for j in range(i):
                                if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.6)
                            for h in range(3):
                                if b[i,h]==1:ax.plot([xy[i,0],bs[h,0]],[xy[i,1],bs[h,1]],'--',color='#555555',lw=.7)
                            ax.annotate(str(int(z['active_uav_ids'][t,i])+1),xy[i],fontsize=6,xytext=(2,2),textcoords='offset points')
                        linked=b.min(1)<=u
                        for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=30,edgecolors='black',linewidths=.4)
                        ax.scatter(bs[:,0],bs[:,1],c='black',marker='*',s=80,clip_on=False)
                        ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{g[-1]} neighbors | t{t} | {served.mean():.1%}')
            fig.suptitle(f'{NAMES[scenario]}, fixed diagnostic scene {seed}; labels retain original UAV IDs',y=1.01)
            fig.legend(handles=[Line2D([],[],marker=m,color=c,ls='none',label=l) for m,c,l in
                [('o','#333333','Served UE'),('x','#777777','Sensed only'),('^','#333333','Connected UAV'),('s','#333333','Disconnected UAV'),('*','black','BS')]],
                loc='lower center',ncol=5,bbox_to_anchor=(.5,-.015),fontsize=8)
            save_figure(fig,figdir,f'topology_{scenario}_{seed}')
    write_report(out,result)
    files={str(p.relative_to(out)):digest(p) for g in RUNS for p in (out/g).rglob('*') if p.is_file()}
    (out/'report_sources.json').write_text(json.dumps(dict(evaluation_files=files,
        script_sha256=digest(__file__),adapter_sha256=digest('src/uav_research/population_generalization.py')),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),summary=summary),ensure_ascii=False),flush=True)


def write_report(out,result):
    summary=result['summary']
    lines=['# 冻结3／4／5邻居actor：UE与UAV数量的泛化诊断','',
        '只使用三组1M运行原best：3邻居671,744步、4邻居620,544步、5邻居980,992步。冻结全部actor参数，不重新选模、不微调、不使用critic。'
        '本轮没有加入10M模型。','',
        '采用64个新诊断场景660000–660063，未用于这三组训练选模；它们从此属于开发诊断集，不再声称是未接触的最终测试集。'
        '保留测试集未使用。原场景与四种变化统一CUDA FP32、关闭TF32、采样种子880001、批次16、400步。'
        '每场景一条采样轨迹；置信区间为2000次场景级配对bootstrap、种子950001，不是训练种子区间。','',
        '## 控制与适配','',
        '- 场地3500×3500 m、3 BS、容量10、UU650 m、UG1500 m和其他无线参数不变，奖励仍为覆盖＋0.1归一化吞吐。',
        '- 从同一18 UAV、150 UE初始布局取子集。UE保留原编号前90／120个；UAV按场景种子和固定种子972001随机抽取嵌套的14／10架，按原编号排序，不筛选覆盖或联网情况。',
        '- UE运动始终先生成原150人的速度／方向，再选保留UE；动作噪声始终先抽取18个UAV，再按原编号选取。保留节点的外部随机输入配对，策略状态和动作均值仍可随场景变化。',
        '- UAV减少时，只有actor输入保留18个编号槽位：缺失邻接位补零，邻居ID仍除以18，可达跳数仍除以19、断网仍编码1；历史跳数同步转换。UE集合不补到150，不依赖全局UE数。',
        '- 物理环境真正只有14／10架：移除节点不发射、不干扰、不路由、不占容量。原critic输入维度不同，完全不调用。',
        '- 原规模入口在16个旧验证场景的前8步复现原CUDA动作和两条代表轨迹，逐元素一致；新群体规模使用同一个冻结actor。','',
        '## 结果','',
        '| 场景 | 邻居数 | 平均覆盖 | 后100步 | 终态覆盖 | 实际服务UE数 | 总吞吐Mbps | 每UE吞吐Mbps | 断联率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        for g in RUNS:
            s=summary[g][scenario]
            lines.append(f"| {NAMES[scenario]} | {g[-1]} | {s['coverage']:.2%} | {s['last100_coverage']:.2%} | {s['final_coverage']:.2%} | {s['served_users']:.2f} | {s['throughput_mbps']:.2f} | {s['throughput_per_ue_mbps']:.3f} | {s['disconnection_ratio']:.2%} |")
    lines+=['','原规模数字来自这64个新场景，不能与此前32个验证场景的表逐行当作同一测量。'
        'UE减少时覆盖率分母和奖励中吞吐归一化分母都随真实人数变化，不能仅凭覆盖率上升断言网络服务了更多人。','',
        '## 容量与物理难度','',
        '18／14／10架的总容量分别180／140／100。150 UE下14架最多覆盖93.33%、10架最多66.67%；'
        '容量上界只考虑人数，实际还受发现、下行及回传限制。“覆盖／容量上界”不是相对最优部署性能。'
        '减少UAV既降低中继密度和总容量，也降低满负载同频干扰；结果不要求单调下降。','',
        '| 场景 | 邻居数 | 覆盖／容量上界 | 最大单步跌幅均值 | 后100步−51至150步 |',
        '|---|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        for g in RUNS:
            s=summary[g][scenario]
            lines.append(f"| {NAMES[scenario]} | {g[-1]} | {s['coverage_over_capacity_ceiling']:.2%} | {s['max_drop_1']*100:.2f}pp | {s['maintain_coverage']*100:+.2f}pp |")
    lines+=['','## 相对同一actor原规模的覆盖变化','',
        '| 变化 | 邻居数 | 覆盖差 | 场景95%区间 |','|---|---:|---:|---:|']
    for p in result['paired']:
        if p['reference']=='same actor original size' and p['metric']=='coverage':
            lines.append(f"| {NAMES[p['scenario']]} | {p['group'][-1]} | {100*p['difference']:+.2f}pp | [{100*p['low']:+.2f}, {100*p['high']:+.2f}]pp |")
    lines+=['','## 结果解读','',
        '**UE数量减少：当前三份策略没有出现明显失效。** '
        '3邻居在90／120 UE下平均覆盖分别提升2.59／3.31个百分点，5邻居提升2.15／2.22个百分点；'
        '4邻居分别提升0.31／0.76个百分点，其场景区间跨零，宜表述为基本保持。'
        '这支持本轮UE减量范围内的变长集合执行与服务能力迁移，但没有测试UE数量增加。','',
        'UE减少同时减轻容量压力。例如5邻居的容量淘汰从原规模平均6.11人降至90 UE的0.30人。'
        '其实际服务人数从90.38降至56.17，总吞吐从494.71降至304.70 Mbps；覆盖比例升高并不意味着总服务量增加。','',
        '**UAV数量减少：三份策略都明显损失覆盖，5邻居的这份模型保留得更好。** '
        '14架时平均覆盖为49.38／48.40／51.73%，10架时为33.02／33.54／38.59%。'
        '5邻居相对3邻居在14架时高2.35个百分点（场景区间[0.58,4.10]），10架时高5.57个百分点（[3.95,7.22]）；'
        '这是当前单种子检查点的场景证据，不是多训练种子结论。','',
        '网络变稀疏后，断联和感知／回传限制增加。10架场景中，3邻居断联率35.21%、5邻居23.82%；'
        '仅因回传而未服务的UE平均分别38.50／24.47人。另一方面5邻居未感知UE为49.65人，高于3邻居44.84人，'
        '说明5邻居这份策略的收益更多体现在回传服务保留，不能简单解释为感知范围展开更大。','',
        '**总容量上限不能单独解释下降，但目前也不能把全部下降判为策略泛化失败。** '
        '10架最多服务100人，而实际仅服务约49.53／50.30／57.88人；剩余缺口同时包含感知、下行、回传和局部容量限制。'
        '减少节点使任务客观变难；若要量化“泛化不足”相对可达到水平的差距，需要另外在14／10架目标场景从头训练同预算对照，本轮没有执行。','',
        '**增加邻居数的优势并非在所有新布局中都稳定。** 原规模64场景中，4／5邻居相对3邻居的平均覆盖优势约1.00／1.18个百分点，'
        '对应场景区间均跨零；UE减少时5邻居与3邻居也很接近。不能将上轮32验证场景的优势直接推广为所有场景的确定优势。'
        'UAV减少时5邻居模型的相对优势更明确，值得作为后续重复种子的候选。','']
    lines+=['','## 结论边界与文件','',
        '本实验评估冻结策略在节点规模变化和明确输入适配下的迁移表现。未见布局上的表现与原规模相近，才支持相应范围内的泛化；'
        '任务变容易、容量上限改变和干扰变化都需要一起解释。单种子结果不能证明结构对任意规模都可泛化，也不能把失败全部归因于邻居槽位。','',
        '保存每场景的保留节点映射、原始逐步指标、代表轨迹及来源SHA256。物理服务人数与容量、配对UE轨迹、奖励和观测转换均有断言检查。','',
        '- [总体表现](figures/population_summary.png)',
        '- [容量与实际服务能力](figures/capacity_and_service.png)',
        '- [覆盖过程](figures/coverage_curves.png)',
        '- [吞吐过程](figures/throughput_bps_curves.png)',
        '- [断联过程](figures/disconnection_ratio_curves.png)',
        '- [服务条件淘汰](figures/service_stage_rejections.png)',
        '- [UE90固定场景拓扑](figures/topology_ue60_660000.png)',
        '- [14 UAV固定场景拓扑](figures/topology_uav14_660000.png)',
        '- [10 UAV固定场景拓扑](figures/topology_uav10_660000.png)','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate',choices=list(RUNS))
    p.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--output',type=Path,default=Path('analysis/mappo_neighbors_population_generalization_1m_seed0'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--check',action='store_true',help='Only verify old original-size execution; no new full evaluation')
    a=p.parse_args()
    if bool(a.evaluate)==bool(a.rebuild):p.error('Choose one of --evaluate / --rebuild')
    if a.evaluate:evaluate_all(a)
    else:rebuild(a)
