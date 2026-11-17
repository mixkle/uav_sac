"""Frozen A3 timing audit, bounded raw blocks, scene-paired saved-data report."""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import psutil
from .association_timing import TimingEnvironment, PROTOCOLS
from .environment import physical_links
from .runtime import resolve_output, atomic_json, atomic_save, isolated_rng, seed_all
from .topk_diagnostic import atomic_npz, file_hash
from .diffusion_dppo.config import from_dict
from .diffusion_dppo.storage import configure, load, Progress, ensure_budget
from .diffusion_dppo.observations import stack
from .diffusion_dppo.policy import make_service_actor
from .diffusion_dppo.evaluation import frame
from .system_ablation.report import episode_metrics, write_csv
from .diffusion_followup.runtime import actor_digest

ACTION_SEEDS = [880001, 880002, 880003]
SCENES = list(range(10000, 10032))
DEFAULT_OUTPUT = 'analysis/t1_t2_t3_post_motion_association'


def raw_record(env, action, chain, index, groups, previous):
    return dict(uav_before=env.pre_positions[0], user_before=env.pre_positions[1],
        uav_after=env.uav_xy.copy(), user_after=env.user_xy.copy(),
        before=np.packbits(env.timing_before), after=np.packbits(env.timing_after),
        y_before=env.pre_y, y_after=env.Y.copy(), rates=env.rates.copy(),
        A=env.A.copy(), B=env.B.copy(), history=env.history.copy(),
        previous_owner=previous, previous_service_age=env.pre_ages.copy(),
        previous_last_owner=env.pre_last_owner.copy(), movement=action.movement.copy(),
        learned_score_matrix=env.last_scores.copy(),
        associations=np.stack([np.packbits(env.shadow_associations[name]) for name in PROTOCOLS]))


def flush_raw(path, rows, chains, first_step, chosen):
    if not rows:
        return
    arrays = {k: np.stack([[e[k] for e in step] for step in rows]) for k in rows[0][0]}
    # Full sampled augmented actions allow exact reconstruction without a second
    # actor call; candidate offsets route each score and each denoising sample.
    arrays.update(chain_movement=np.stack([c.movement for c in chains]),
        chain_scores=np.concatenate([c.scores for c in chains]),
        chain_score_offsets=np.r_[0, np.cumsum([len(c.scores) for c in chains])],
        chain_candidate_offsets=np.stack([c.candidate_offsets for c in chains]),
        chain_logp=np.stack([c.logp for c in chains]),
        candidate_ids=np.concatenate([x['ids'] for x in _RAW_ACTIONS]),
        candidate_id_offsets=np.r_[0,np.cumsum([len(x['ids']) for x in _RAW_ACTIONS])],
        scene_seeds=np.asarray(chosen), first_step=first_step,
        protocols=np.asarray(PROTOCOLS), shape=np.asarray(rows[0][0]['y_after'].shape))
    atomic_npz(path, **arrays)


_RAW_ACTIONS = []  # case-local block buffer, never used by the environment


@torch.no_grad()
def run_case(checkpoint, protocol, action_seed, output, device='cpu', scenes=None,
             steps=None, raw=True):
    source = resolve_output(f'runs/system_ablation_1m/A3_seed0/{checkpoint}.pt')
    digest = file_hash(source)
    data = load(source, kinds={'dppo_model'})
    cfg = from_dict(data['config']); cfg.run.device = device
    cfg.run.cpu_threads = 8
    cfg.run.numba_threads = 4
    cfg.diffusion.actor_update_devices = []
    if cfg.diffusion.ablation['association'] != 'learned_topk':
        raise ValueError('The frozen source must use the original learned Top-K')
    configure(cfg)
    chosen_scenes = SCENES if scenes is None else list(scenes)
    horizon = cfg.env.horizon if steps is None else steps
    if not 1 <= horizon <= cfg.env.horizon:
        raise ValueError('Invalid diagnostic horizon')
    target = Path(output)/'rollout'/checkpoint/protocol/str(action_seed)
    stamp = dict(checkpoint=checkpoint, checkpoint_sha256=digest, protocol=protocol,
        actor_action_seed=action_seed, scene_seeds=chosen_scenes, horizon=horizon,
        source_online_steps=data['env_steps'], raw_blocks=bool(raw and protocol=='T1'), version=1)
    if (target/'complete.json').exists():
        if json.loads((target/'complete.json').read_text()) != stamp:
            raise ValueError('Existing timing result has a different source/contract')
        print('reuse', target, flush=True)
        return
    ensure_budget(cfg,target,estimated_bytes=2**30)
    target.mkdir(parents=True,exist_ok=True)
    actor = make_service_actor(cfg).to(device)
    actor.load_state_dict(data['learner']['actor']); actor.eval(); actor.requires_grad_(False)
    actor_hash = actor_digest(actor)
    progress = Progress(30)
    metrics, one_step, mechanics, traces = [], [], [], {}
    user_hash = hashlib.sha256()
    started = time.perf_counter();peak_rss=psutil.Process().memory_info().rss
    keys = None
    with isolated_rng():
        seed_all(action_seed)
        for begin in range(0,len(chosen_scenes),cfg.run.num_envs):
            chosen = chosen_scenes[begin:begin+cfg.run.num_envs]
            envs = [TimingEnvironment(cfg.env,protocol,shadow=protocol=='T1') for _ in chosen]
            resets = [e.reset(s) for e,s in zip(envs,chosen)]
            obs = stack([r[0] for r in resets])
            local_frames = {i:[frame(e)] for i,e in enumerate(envs) if chosen[i] in (10000,10001)}
            if raw and protocol=='T1':
                atomic_save(target/f'initial_snapshots_{begin}.pt',
                    dict(scenes=chosen,snapshots=[e.snapshot() for e in envs],config=cfg.as_dict()))
            batch_metrics=[];raw_rows=[];raw_chains=[];closed_rows=[];_RAW_ACTIONS.clear()
            # All protocols draw their UE motion from the same per-scene RNG;
            # a digest of the whole UE path is checked across closed-loop arms.
            user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
            for t in range(horizon):
                action,chain = actor.sample(obs)
                if not np.isfinite(chain.logp).all():
                    raise FloatingPointError('Nonfinite frozen-policy transition probability')
                items=action.split(cfg.env.num_uavs)
                previous=[e.owner().copy() for e in envs]
                if raw and protocol=='T1' and t in (0,150,200,399):
                    atomic_save(target/f'pre_snapshots_b{begin}_t{t}.pt',
                        dict(scenes=chosen,snapshots=[e.snapshot() for e in envs],action=action,chain=chain))
                prepared=[e.prepare(a) for e,a in zip(envs,items)]
                links=physical_links(cfg.env,np.stack([e.uav_xy for e in envs]),np.stack([e.user_xy for e in envs]))
                results=[e.finish(p,tuple(x[i] for x in links)) for i,(e,p) in enumerate(zip(envs,prepared))]
                if any(r[4] or r[3] != (t+1==cfg.env.horizon) for r in results):
                    raise AssertionError('Terminal/truncation semantics changed')
                user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
                infos=[dict(r[-1],near_hover_fraction=float((np.linalg.norm(e.actual_displacement,axis=1)<1).mean()))
                       for e,r in zip(envs,results)]
                if keys is None:
                    keys=sorted(k for k,v in infos[0].items() if np.isscalar(v))
                batch_metrics.append([[r[k] for k in keys] for r in infos])
                for i,e in enumerate(envs):
                    if protocol=='T1':
                        for name,row in e.shadow_rows.items():
                            one_step.append(dict(checkpoint=checkpoint,reference_pool='A3_T1',
                                scene_seed=chosen[i],actor_action_seed=action_seed,step=t+1,protocol=name,**row))
                        p=e.shadow_rows
                        mechanics.append(dict(checkpoint=checkpoint,reference_pool='A3_T1',scene_seed=chosen[i],
                            actor_action_seed=action_seed,step=t+1,**e.timing,
                            invalidated_proposals=p['T2']['invalidated_proposals'],
                            unscored_waiting_edges=e.timing['new_candidate_edges'],
                            rerank_added_served_ues=(p['P2']['coverage']-p['P1']['coverage'])*cfg.env.num_users,
                            new_candidate_added_served_ues=(p['T3']['coverage']-p['P2']['coverage'])*cfg.env.num_users))
                    if i in local_frames:
                        local_frames[i].append(frame(e))
                closed_rows.append([dict(uav_xy=e.uav_xy.copy(),user_xy=e.user_xy.copy(),
                    association=np.packbits(e.C),proposals=np.packbits(e.proposals),
                    feasible_before=np.packbits(e.timing_before),feasible_after=np.packbits(e.timing_after),
                    movement=a.movement.copy(),scores=e.last_scores.copy(),
                    actual_displacement=e.actual_displacement.copy(),B=e.B.copy())
                    for e,a in zip(envs,items)])
                if len(closed_rows)==50 or t==horizon-1:
                    atomic_npz(target/'closed_loop'/f'b{begin}_t{t+1-len(closed_rows):03d}.npz',
                        **{k:np.stack([[e[k] for e in row] for row in closed_rows]) for k in closed_rows[0][0]},
                        scene_seeds=np.asarray(chosen),first_step=t+2-len(closed_rows),
                        association_shape=np.asarray(envs[0].C.shape))
                    closed_rows.clear()
                if raw and protocol=='T1':
                    raw_rows.append([raw_record(e,a,chain,i,cfg.env.num_uavs,previous[i]) for i,(e,a) in enumerate(zip(envs,items))])
                    raw_chains.append(chain);_RAW_ACTIONS.append(dict(ids=action.ue_ids.copy()))
                    if len(raw_rows)==50 or t==horizon-1:
                        flush_raw(target/'raw'/f'b{begin}_t{t+1-len(raw_rows):03d}.npz',raw_rows,raw_chains,
                                  t+2-len(raw_rows),chosen)
                        raw_rows.clear();raw_chains.clear();_RAW_ACTIONS.clear()
                obs=stack([r[0] for r in results])
                peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
                if peak_rss>12*2**30:
                    raise MemoryError('Timing diagnostic exceeds 12 GiB RSS budget')
                progress(f'{checkpoint}/{protocol}/{action_seed} scenes={chosen[0]}..{chosen[-1]} t={t+1}/{horizon}')
            metrics.append(np.asarray(batch_metrics,dtype=np.float64).transpose(1,0,2))
            for i,frames in local_frames.items():
                atomic_npz(target/f'trace_{chosen[i]}.npz',
                    **{k:np.asarray([f[k] for f in frames]) for k in frames[0]},
                    bs_positions=np.asarray(cfg.env.bs_positions),width=cfg.env.width,height=cfg.env.height)
    array=np.concatenate(metrics)
    if not np.isfinite(array).all():
        raise FloatingPointError('Nonfinite timing result')
    per_episode=episode_metrics(array,keys,chosen_scenes)
    names=['direct_switches','service_interruptions','service_reconnections','completed_service_spells',
           'completed_service_duration_sum','service_age_mean','proposal_count','invalidated_proposals']
    for row,episode in zip(per_episode,array):
        for n in names:
            x=episode[:,keys.index(n)]
            row['last100_'+n+'_per_step']=float(x[-100:].mean())
            row['last100_'+n+'_window_sum']=float(x[-100:].sum())
        for window,size in [('all',len(episode)),('last100',100)]:
            spells=episode[-size:,keys.index('completed_service_spells')].sum()
            duration=episode[-size:,keys.index('completed_service_duration_sum')].sum()
            row[window+'_completed_service_duration_mean']=float(duration/spells) if spells else None
    write_csv(target/'per_episode.csv',per_episode)
    atomic_npz(target/'step_metrics.npz',metrics=array,names=np.asarray(keys),scene_seeds=chosen_scenes)
    # CSV includes every numeric production diagnostic; source identity is in
    # provenance.json, rather than repeating a 64-character SHA on every row.
    write_csv(target/'per_step.csv',[
        dict(scene_seed=s,step=t+1,**dict(zip(keys,x)))
        for s,episode in zip(chosen_scenes,array) for t,x in enumerate(episode)])
    if protocol=='T1':
        write_csv(target/'paired_step_metrics.csv',one_step)
        write_csv(target/'mechanism_breakdown.csv',mechanics)
    assert actor_hash==actor_digest(actor) and all(p.grad is None for p in actor.parameters())
    assert file_hash(source)==digest
    atomic_json(target/'provenance.json',dict(stamp,config=cfg.as_dict(),
        frozen_actor_sha256=actor_hash,weights_unchanged=True,optimizer_created=False,
        elapsed_seconds=time.perf_counter()-started,ue_path_sha256=user_hash.hexdigest(),
        peak_rss_gib=peak_rss/2**30,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
        next_action='Protocol-dependent observation; subsequent motion can differ',
        raw_format='T1 blocks: exact float64 physics, packed masks/associations; initial complete RNG and saved full chains'))
    atomic_json(target/'complete.json',stamp)
    print('complete',target,flush=True)


def _read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def bootstrap(values):
    values=np.asarray(values,float)
    draws=np.random.default_rng(950001).integers(len(values),size=(2000,len(values)))
    return (float(values.mean()),*np.quantile(values[draws].mean(1),[.025,.975]).tolist())


def rebuild(output,require_complete=True):
    root=Path(output);all_episodes=[];all_steps=[];all_mechanics=[];sources=[];series={}
    required={(c,p,s) for c in ('best','last') for p in ('T1','T2','T3') for s in ACTION_SEEDS}
    found=set()
    for p in sorted((root/'rollout').glob('*/*/*/complete.json')):
        stamp=json.loads(p.read_text());case=p.parent
        found.add((stamp['checkpoint'],stamp['protocol'],stamp['actor_action_seed']))
        meta=json.loads((case/'provenance.json').read_text());sources.append(meta)
        for row in _read(case/'per_episode.csv'):
            all_episodes.append(dict(checkpoint=stamp['checkpoint'],protocol=stamp['protocol'],
                actor_action_seed=stamp['actor_action_seed'],**row))
        if (case/'paired_step_metrics.csv').exists():
            all_steps+=_read(case/'paired_step_metrics.csv')
            all_mechanics+=_read(case/'mechanism_breakdown.csv')
        with np.load(case/'step_metrics.npz') as z:
            series[(stamp['checkpoint'],stamp['protocol'],stamp['actor_action_seed'])]=(z['metrics'].copy(),z['names'].tolist())
    if require_complete and found!=required:
        raise ValueError(f'Missing formal cases: {sorted(required-found)}')
    # Matching all UE paths is stronger than merely matching the initial seed.
    for c in ('best','last'):
        for s in ACTION_SEEDS:
            hashes=[x['ue_path_sha256'] for x in sources if x['checkpoint']==c and x['actor_action_seed']==s]
            if len(set(hashes))>1:
                raise AssertionError('Exogenous UE paths diverged across protocols')
    perf=[];diffs=[];values={}
    for c,p in sorted({(r['checkpoint'],r['protocol']) for r in all_episodes}):
        subset=[r for r in all_episodes if r['checkpoint']==c and r['protocol']==p]
        scenes=sorted({int(r['scene_seed']) for r in subset})
        keys=[k for k in subset[0] if k not in ('checkpoint','protocol','scene_seed','actor_action_seed')]
        mean={}
        for key in keys:
            try:
                mean[key]=np.array([np.mean([float(r[key]) for r in subset if int(r['scene_seed'])==s]) for s in scenes])
            except (TypeError,ValueError):
                continue
        values[c,p]=mean
        row=dict(checkpoint=c,protocol=p,scenes=len(scenes),action_repetitions=len(subset)//len(scenes))
        for key,a in mean.items():
            d,lo,hi=bootstrap(a);row[key]=d;row[key+'_low']=lo;row[key+'_high']=hi
        perf.append(row)
    main_metrics=['last100_coverage','last50_coverage','final_coverage','last100_throughput_mbps',
        'final_throughput_mbps','last100_coverage_std_pp','last100_coverage_p05','last100_drop_events_10pp',
        'last100_connected_fraction','episode_return','last100_direct_switches_window_sum',
        'last100_service_interruptions_window_sum','last100_service_reconnections_window_sum',
        'last100_completed_service_duration_sum_window_sum','last100_completed_service_spells_window_sum']
    for c in ('best','last'):
        for left,right in [('T2','T1'),('T3','T2'),('T3','T1')]:
            if (c,left) not in values or (c,right) not in values:continue
            for key in main_metrics:
                d,lo,hi=bootstrap(values[c,left][key]-values[c,right][key])
                diffs.append(dict(checkpoint=c,comparison=left+' - '+right,metric=key,difference=d,low=lo,high=hi))
    write_csv(root/'rollout/per_episode.csv',all_episodes)
    write_csv(root/'one_step/paired_step_metrics.csv',all_steps)
    write_csv(root/'one_step/mechanism_breakdown.csv',all_mechanics)
    write_csv(root/'summary/performance.csv',perf)
    write_csv(root/'summary/paired_differences.csv',diffs)
    write_csv(root/'summary/paired_bootstrap_ci.csv',diffs)
    # Immediate effect: average within scene/action/window before bootstrapping.
    immediate=[];scene_immediate=[]
    for c in ('best','last'):
        for window,lo_t,hi_t in [('first100',1,100),('middle200',101,300),('last100',301,400),('all',1,400)]:
            subset=[r for r in all_steps if r['checkpoint']==c and lo_t<=int(r['step'])<=hi_t]
            if not subset:continue
            scenes=sorted({int(r['scene_seed']) for r in subset})
            for left,right in [('T2','T1'),('P1','T2'),('P2','P1'),('T3','P2'),('T3','T2')]:
                for key in ['coverage','throughput_bps','reward','direct_switches','service_interruptions']:
                    grouped={}
                    for r in subset:
                        if r['protocol'] not in (left,right):continue
                        ident=(int(r['scene_seed']),int(r['actor_action_seed']),int(r['step']))
                        grouped.setdefault(ident,{})[r['protocol']]=float(r[key])
                    v=[np.mean([d[left]-d[right] for ident,d in grouped.items() if ident[0]==s]) for s in scenes]
                    scene_immediate.extend(dict(checkpoint=c,reference_pool='A3_T1',window=window,
                        comparison=left+' - '+right,metric=key,scene_seed=s,difference=float(x))
                        for s,x in zip(scenes,v))
                    d,low,high=bootstrap(v)
                    immediate.append(dict(checkpoint=c,reference_pool='A3_T1',window=window,
                        comparison=left+' - '+right,metric=key,difference=d,low=low,high=high))
    write_csv(root/'summary/immediate_paired_effects.csv',immediate)
    write_csv(root/'one_step/scene_paired_effects.csv',scene_immediate)
    mechanism_summary=[]
    for c in ('best','last'):
        for window,lo_t,hi_t in [('first100',1,100),('middle200',101,300),('last100',301,400),('all',1,400)]:
            rows=[r for r in all_mechanics if r['checkpoint']==c and lo_t<=int(r['step'])<=hi_t]
            if not rows:continue
            scenes=sorted({int(r['scene_seed']) for r in rows})
            for key in ('candidate_before_edges','candidate_after_edges','new_candidate_edges',
                        'lost_candidate_edges','new_candidate_unique_ues','invalidated_proposals',
                        'rerank_added_served_ues','new_candidate_added_served_ues'):
                means=[np.mean([float(r[key]) for r in rows if int(r['scene_seed'])==s]) for s in scenes]
                mean,low,high=bootstrap(means)
                mechanism_summary.append(dict(checkpoint=c,window=window,metric=key,mean=mean,low=low,high=high))
    write_csv(root/'summary/mechanism_statistics.csv',mechanism_summary)
    # Stream the combined per-step table instead of materializing all numeric
    # fields for 230,400 states as Python dictionaries at once.
    with (root/'rollout/per_step.csv').open('w',newline='') as combined:
        writer=None
        for meta in sources:
            case=root/'rollout'/meta['checkpoint']/meta['protocol']/str(meta['actor_action_seed'])
            with (case/'per_step.csv').open() as f:
                for row in csv.DictReader(f):
                    record=dict(checkpoint=meta['checkpoint'],protocol=meta['protocol'],
                        actor_action_seed=meta['actor_action_seed'],**row)
                    if writer is None:
                        writer=csv.DictWriter(combined,fieldnames=list(record));writer.writeheader()
                    writer.writerow(record)
    _figures(root,perf,series,immediate,diffs)
    atomic_json(root/'provenance.json',dict(evaluations=sources,bootstrap_repeats=2000,bootstrap_seed=950001,
        training=False,heldout_test_used=False,reference_pool='A3_T1; optional F2b state pool not run',
        external_ue_paths_identical=True,closed_loop_motion_may_differ=True))
    lines=['# T1–T3冻结关联时序实验','',
        'A3 best/last冻结；32开发验证场景×3动作种子；未训练、未重选模型、未使用保留测试集。',
        'T1学习评分；T2运动前SINR旧提议；T3运动后本地SINR与完整可服务集合重新一轮Top-K。',
        '后100步为301–400；时间σ逐轨迹计算；事件数为窗口累计。区间为场景bootstrap，不是训练种子区间。','',
        '|模型|协议|后100覆盖%|后100吞吐Mbps|时间σ pp|终态覆盖%|终态吞吐Mbps|回报|后100中断次数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in perf:
        lines.append(f"|{r['checkpoint']}|{r['protocol']}|{r['last100_coverage']*100:.3f}|{r['last100_throughput_mbps']:.2f}|{r['last100_coverage_std_pp']:.3f}|{r['final_coverage']*100:.3f}|{r['final_throughput_mbps']:.2f}|{r['episode_return']:.3f}|{r['last100_service_interruptions_window_sum']:.2f}|")
    lines+=['','一步效应见summary/immediate_paired_effects.csv；三个影子差值按指定顺序相加，不解释为独立无交互贡献。',
        'T3假设运动后即时测量和一次服务交换，不建模其信令时延；仍是一轮提议，不是最大匹配或补位迭代。',
        '读取保存的CSV、NPZ可以重建报告；权重哈希和UE路径一致性检查记录在provenance.json。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return perf,diffs,immediate


def _figures(root,perf,series,immediate,diffs):
    import matplotlib.pyplot as plt
    from .plotting import style,save_figure
    style();colors=['#009E73','#0072B2','#E69F00'];styles=['-','--','-.']
    for c in ('best','last'):
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for i,p in enumerate(('T1','T2','T3')):
            cases=[v for (kind,mode,_),v in series.items() if kind==c and mode==p]
            if not cases:continue
            for ax,key,scale,label in zip(axs,['coverage','throughput_bps'],[100,1e-6],['Coverage (%)','Throughput (Mbps)']):
                y=np.mean([m[:,:,names.index(key)].mean(0) for m,names in cases],0)
                ax.plot(np.arange(1,len(y)+1),y*scale,c=colors[i],ls=styles[i],label=p)
                ax.set(xlabel='Episode step',ylabel=label)
        axs[0].legend();save_figure(fig,root/'figures',c+'_coverage_throughput_traces')
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,key,scale,label in zip(axs,['coverage','throughput_bps'],[100,1e-6],['Coverage change (pp)','Throughput change (Mbps)']):
            pairs=[('P1 - T2','Ownership score'),('P2 - P1','Old-set reselect'),('T3 - P2','New candidates')]
            vals=[next((r for r in immediate if r['checkpoint']==c and r['window']=='last100' and r['comparison']==pair and r['metric']==key),None) for pair,_ in pairs]
            if all(r is not None for r in vals):
                y=np.array([r['difference'] for r in vals])*scale
                low=np.array([r['low'] for r in vals])*scale;high=np.array([r['high'] for r in vals])*scale
                ax.bar(range(3),y,color=colors,yerr=[y-low,high-y],capsize=3)
                ax.set(xticks=range(3),xticklabels=[s for _,s in pairs],ylabel=label)
                ax.axhline(0,c='k',lw=.6)
        save_figure(fig,root/'figures',c+'_timing_decomposition')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,metric,ikey,scale,label in zip(axs,['last100_coverage','last100_throughput_mbps'],['coverage','throughput_bps'],[100,1],['Coverage change (pp)','Throughput change (Mbps)']):
            for j,pair in enumerate(['T2 - T1','T3 - T2']):
                closed=next(r for r in diffs if r['checkpoint']==c and r['comparison']==pair and r['metric']==metric)
                now=next(r for r in immediate if r['checkpoint']==c and r['comparison']==pair and r['metric']==ikey and r['window']=='last100')
                vals=[now['difference']*(100 if ikey=='coverage' else 1e-6),closed['difference']*scale]
                ax.bar(np.arange(2)+j*.3,vals,width=.3,label=pair,color=colors[j])
            ax.set(xticks=[.15,1.15],xticklabels=['Same-state immediate','Closed loop'],ylabel=label);ax.axhline(0,c='k',lw=.6)
        axs[0].legend();save_figure(fig,root/'figures',c+'_immediate_vs_closed_loop')
        fig,axs=plt.subplots(1,3,figsize=(12,3.7))
        for ax,key,label in zip(axs,['last100_coverage','last100_throughput_mbps','last100_service_interruptions_window_sum'],['Last-100 coverage (%)','Last-100 throughput (Mbps)','Interruptions / 100-step episode window']):
            chosen=[r for r in perf if r['checkpoint']==c]
            factor=100 if key=='last100_coverage' else 1
            ax.bar(range(len(chosen)),[r[key]*factor for r in chosen],color=colors)
            ax.set(xticks=range(len(chosen)),xticklabels=[r['protocol'] for r in chosen],ylabel=label)
        save_figure(fig,root/'figures',c+'_last100_metrics')
        fig,axs=plt.subplots(1,4,figsize=(15,3.7))
        for ax,key,label in zip(axs,
                ['last100_direct_switches_per_step','last100_service_interruptions_per_step',
                 'last100_service_reconnections_per_step','last100_completed_service_duration_mean'],
                ['Direct switches / step','Interruptions / step','Reconnections / step',
                 'Ended service duration (steps)']):
            chosen=[r for r in perf if r['checkpoint']==c]
            ax.bar(range(len(chosen)),[r[key] for r in chosen],color=colors)
            ax.set(xticks=range(len(chosen)),xticklabels=[r['protocol'] for r in chosen],ylabel=label)
        save_figure(fig,root/'figures',c+'_service_continuity')
    # Raw topology renderer uses only saved production-compatible frames.
    for path in root.glob('rollout/*/*/880001/trace_*.npz'):
        with np.load(path) as z:
            if len(z['uav_xy'])<401:continue
            fig,axs=plt.subplots(1,4,figsize=(16,4))
            for ax,t in zip(axs,[0,150,200,400]):
                u,m=z['uav_xy'][t],z['user_xy'][t];served=z['C'][t].any(0);online=z['B'][t].min(1)<=len(u)
                ax.scatter(m[~served,0],m[~served,1],c='#aaaaaa',s=6)
                ax.scatter(m[served,0],m[served,1],c=colors[0],s=8)
                ax.scatter(u[online,0],u[online,1],c=colors[1],marker='^',s=25)
                ax.scatter(u[~online,0],u[~online,1],c='k',marker='x',s=25)
                bs=z['bs_positions'];ax.scatter(bs[:,0],bs[:,1],c='k',marker='s',s=40)
                for i,j in zip(*np.where(np.triu(z['A'][t],1))):ax.plot(u[[i,j],0],u[[i,j],1],c='#888888',lw=.6)
                for i,j in zip(*np.where(z['C'][t])):
                    ax.plot([u[i,0],m[j,0]],[u[i,1],m[j,1]],c=colors[0],lw=.3,alpha=.35)
                for i,g in zip(*np.where(z['B'][t]==1)):ax.plot([u[i,0],bs[g,0]],[u[i,1],bs[g,1]],c=colors[1],ls='--',lw=.6)
                ax.set(xlim=(0,float(z['width'])),ylim=(0,float(z['height'])),aspect='equal')
                ax.set_title(f't={t}, coverage={served.mean():.1%}, online={online.sum()}/18',fontsize=9)
            save_figure(fig,root/'figures/topology',f'{path.parts[-4]}_{path.parts[-3]}_{path.stem}')
