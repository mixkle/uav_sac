"""Frozen 10M best / 10M best / new-teacher comparison; no training."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


GROUPS = ('baseline_best', 'dppo_best', 'teacher_topk')
NAMES = dict(baseline_best='Baseline 10M best', dppo_best='DPPO 10M best',
             teacher_topk='Baseline 10M teacher')
COLORS = dict(baseline_best='#D62728', dppo_best='#0072B2', teacher_topk='#E69F00')
LINES = dict(baseline_best='-', dppo_best='--', teacher_topk='-.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def evaluate_group(args):
    import torch
    from uav_research.algorithms import make_learner
    from uav_research.config import config_from_dict, load_config
    from uav_research.runtime import load_checkpoint, atomic_json, source_digest
    from uav_research.models import disk_transform
    from uav_research.diffusion_dppo.storage import configure
    from uav_research.diffusion_dppo.data import teacher_action
    from uav_research.diffusion_dppo.evaluation import evaluate

    path = args.baseline_run / 'best.pt'
    before = digest(path)
    saved = load_checkpoint(path)
    original = config_from_dict(saved['config'])
    assert original.algo.name == 'mappo' and original.run.total_steps == 10_000_000
    assert original.algo.actor_sharing == 'shared'
    cfg = load_config(args.dppo_run / 'config.yaml')
    assert cfg.env == original.env
    cfg.run = copy.deepcopy(original.run)
    cfg.run.device = args.device
    cfg.diffusion.actor_update_devices = []
    cfg.diffusion.teacher_checkpoint = str(path)
    cfg.diffusion.teacher_step = int(saved['env_steps'])
    device = configure(cfg)
    learner = make_learner(original, device)
    learner.load_state_dict(saved['learner'])
    learner.actor.eval().requires_grad_(False)

    class FrozenExecution(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = learner.actor

        @torch.no_grad()
        def sample(self, obs, generator=None):
            # Independent moment RNG cannot consume execution randomness.
            if generator is not None:
                raw = obs.legacy() if hasattr(obs, 'legacy') else obs
                distribution = self.actor.distribution(raw.to_torch(device))
                noise = torch.randn(distribution.loc.shape, device=device,
                                    dtype=distribution.loc.dtype, generator=generator)
                movement = disk_transform(distribution.loc + distribution.scale * noise)[0]
                return SimpleNamespace(movement=movement.cpu().numpy()), None
            if args.evaluate == 'teacher_topk':
                action, _, _ = teacher_action(self.actor, obs, cfg)
                return action, None
            movement = self.actor.sample(obs.to_torch(device), False)[0]
            return movement.cpu().numpy(), None

    out = args.output / args.evaluate
    result = evaluate(cfg, FrozenExecution(), cfg.run.validation_seeds, out,
                      critic=learner if args.evaluate == 'baseline_best' else None,
                      moment_repeats=args.moment_repeats,
                      greedy=args.evaluate == 'baseline_best')
    assert digest(path) == before
    if args.evaluate == 'baseline_best':
        row = next(r for r in read(args.baseline_run / 'validation.csv')
                   if int(r['env_steps']) == saved['env_steps'])
        differences = {k: abs(result['summary'][k] - float(row[k]))
                       for k in ('coverage', 'return', 'throughput_bps')}
        assert max(differences.values()) < 1e-6, differences
    else:
        differences = None
    provenance = dict(checkpoint=str(path), sha256=before, env_steps=saved['env_steps'],
        training_budget=original.run.total_steps, action_seed=cfg.run.evaluation_action_seed,
        scene_seeds=cfg.run.validation_seeds, batch_size=cfg.run.num_envs,
        device=args.device, moment_repeats=args.moment_repeats,
        evaluation_source_digest=source_digest(), evaluation_script_sha256=digest(__file__),
        mode=args.evaluate, validation_reproduction=differences,
        score_rule='sigmoid((pre_move_downlink_SINR_db - threshold_db) / 10)',
        association='baseline_centralized' if args.evaluate == 'baseline_best' else 'pre_move_topk_post_move_gate',
        teacher_is_posthoc_comparison=True, not_original_dppo_pretraining_teacher=True)
    atomic_json(out / 'provenance.json', provenance)
    print(json.dumps(dict(group=args.evaluate, step=saved['env_steps'],
        **{k: result['summary'][k] for k in ('coverage', 'throughput_bps', 'return')})), flush=True)


def rebuild(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style, save_figure
    from uav_research.diffusion_dppo.report import error_metrics
    from uav_research.diffusion_dppo.evaluation import mc_returns

    out = args.output
    sources = json.loads((out / 'report_sources.json').read_text())
    assert sources['labels'] == list(GROUPS)
    for name,sha in sources.get('evaluation_files',{}).items():
        assert digest(out/name)==sha, f'Evaluation artifact changed: {name}'
    for path,sha in sources.get('run_log_sources',{}).items():
        assert digest(path)==sha, f'Training log changed: {path}'
    models, series, episodes, critic, moments, cases, stability = {}, {}, {}, {}, {}, {}, {}
    for name in GROUPS:
        p = out / name
        prov = json.loads((p / 'provenance.json').read_text())
        source = sources['sources'][name]
        assert prov['sha256'] == digest(source['path']) == source['sha256']
        assert prov['scene_seeds'] == list(range(10000, 10032))
        assert prov['action_seed'] == 880001 and prov['device'] == 'cuda:0'
        models[name] = json.loads((p / 'summary.json').read_text())
        episodes[name] = sorted(read(p / 'episodes.csv'), key=lambda x: int(x['scene_seed']))
        assert [int(x['scene_seed']) for x in episodes[name]] == list(range(10000, 10032))
        with np.load(p / 'step_metrics.npz') as z:
            assert z['metrics'].shape[:2] == (32, 400)
            series[name] = s = dict(zip(z['names'].tolist(), z['metrics'].transpose(2, 0, 1)))
        assert all(np.isfinite(x).all() for x in s.values())
        np.testing.assert_allclose(s['reward'], s['coverage'] + .1*s['tau'], atol=2e-7)
        np.testing.assert_allclose(s['tau'], s['throughput_bps']/750e6, atol=1e-12)
        np.testing.assert_allclose(s['coverage'] + (s['sensing_rejected_users'] +
            s['downlink_rejected_users'] + s['backhaul_rejected_users'] +
            s['capacity_rejected_users'])/150, 1, atol=1e-12)
        for key in ('coverage', 'return', 'throughput_bps'):
            expected = s['reward'].sum(1).mean() if key == 'return' else s[key].mean()
            np.testing.assert_allclose(expected, models[name][key], atol=1e-8)
        models[name]['early51_150_coverage'] = float(s['coverage'][:, 50:150].mean())
        models[name]['maintain_coverage'] = float(s['coverage'][:, -100:].mean()-s['coverage'][:, 50:150].mean())
        curve = s['coverage'].mean(0)
        models[name]['peak_mean_curve_step'] = int(curve.argmax()+1)
        models[name]['peak_mean_curve_coverage'] = float(curve.max())
        models[name]['pooled_served_user_rate_bps'] = models[name]['throughput_bps']/(150*models[name]['coverage'])
        losses = np.maximum(0, -np.diff(s['coverage'], axis=1))
        # Changes between post-action states; the initial t0->t1 transition is excluded.
        worst = losses.max(1)
        stability[name] = dict(max_step_drop_mean=float(worst.mean()),
            max_step_drop_p95=float(np.quantile(worst, .95)),
            large_drop_events_mean=float((losses >= .1-1e-12).sum(1).mean()),
            final_below30=int((s['coverage'][:, -1] < .3).sum()),
            final_at_least70=int((s['coverage'][:, -1] >= .7).sum()),
            worst4_episode_coverage=float(np.sort(s['coverage'].mean(1))[:4].mean()),
            lower_tail_count=4, lower_tail_fraction=.125)
        cp = p / 'critic_predictions.npz'
        if cp.exists():
            assert name != 'teacher_topk', 'Centralized teacher critic is not a Top-K value estimate'
            with np.load(cp) as z:
                np.testing.assert_allclose(z['mc_return'], mc_returns(s['reward'], .995), atol=1e-9)
                np.testing.assert_allclose(z['mc_return'], z['mc_coverage']+z['mc_throughput'], atol=1e-5)
                critic[name] = dict(cumulative=error_metrics(z['predicted'], z['mc_return']),
                                    average=error_metrics(z['predicted_average'], z['mc_average']))
        with np.load(p / 'action_moments.npz') as z:
            moments[name] = dict(conditional_mean_command_norm_m=float(np.linalg.norm(z['moments'][:,:,:,0],axis=-1).mean()),
                conditional_variance_trace_m2=float(z['moments'][:,:,:,1].sum(-1).mean()))
        cases[name] = {}
        for scene in (10000, 10001):
            with np.load(p / f'trace_{scene}.npz') as z:
                c = z['C']
                feasible = z['sensed'] & (z['Y'] >= -3) & (z['B'].min(2)[:,:,None] <= 18)
                assert not (c.astype(bool)&~feasible).any()
                assert (c.sum(1)<=1).all() and (c.sum(2)<=10).all()
                np.testing.assert_allclose(c.sum((1,2))/150,z['coverage'])
                cases[name][str(scene)] = [dict(time=t, coverage=float(z['coverage'][t]),
                    connected=int((z['B'][t].min(1)<=18).sum()),
                    sensed=int(z['sensed'][t].any(0).sum()),
                    eligible=int(feasible[t].any(0).sum()),
                    proposals=int(z['proposals'][t].sum()) if name!='baseline_best' else None)
                    for t in (0,150,200,399,400)]
    # Paired initial physical states and exogenous UE paths must agree.
    for scene in (10000,10001):
        with np.load(out/'baseline_best'/f'trace_{scene}.npz') as z:
            initial_uav, user_path = z['uav_xy'][0].copy(), z['user_xy'].copy()
        for name in GROUPS[1:]:
            with np.load(out/name/f'trace_{scene}.npz') as z:
                np.testing.assert_array_equal(initial_uav,z['uav_xy'][0])
                np.testing.assert_array_equal(user_path,z['user_xy'])
    paired=[]
    draws=np.random.default_rng(950001).integers(32,size=(2000,32))
    for left,right in [('dppo_best','baseline_best'),('teacher_topk','baseline_best'),('dppo_best','teacher_topk')]:
        for key in ('coverage','throughput_bps','return','last100_coverage','final_coverage',
                    'maintain_coverage','disconnection_ratio','boundary_reflection_fraction'):
            delta=np.asarray([float(a[key])-float(b[key]) for a,b in zip(episodes[left],episodes[right])])
            low,high=np.quantile(delta[draws].mean(1),[.025,.975])
            paired.append(dict(comparison=f'{left} - {right}',metric=key,difference=float(delta.mean()),low=float(low),high=float(high)))
    with (out/'paired_scene_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    training={}
    for name,run in [('baseline_best',args.baseline_run),('dppo_best',args.dppo_run)]:
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['status'] in ('complete','completed') and manifest['env_steps']==10_000_000
        rows=read(run/'validation.csv')
        selected=sources['sources'][name]['env_steps']
        record=next(r for r in rows if int(r['env_steps'])==selected)
        for key in ('coverage','return','throughput_bps'):
            np.testing.assert_allclose(models[name][key],float(record[key]),atol=1e-6,rtol=0)
        metric_rows=read(run/'metrics.csv')
        counts=[int(r['env_steps']) for r in metric_rows]
        assert counts[-1]==10_000_000 and all(y>=x for x,y in zip(counts,counts[1:]))
        # Legacy train() writes a final status row after its normal last log.
        # This repeats a completed update; it is not another optimizer step.
        for i,(x,y) in enumerate(zip(counts,counts[1:])):
            if x==y:
                assert x==10_000_000
                for k in ('updates','actor_loss','value_loss'):
                    assert metric_rows[i][k]==metric_rows[i+1][k]
        assert len(set(counts))==manifest['updates']
        windows=[]
        for start in (0,2_000_000,4_000_000,6_000_000,8_000_000,9_000_000):
            end=min(start+2_000_000,10_000_000)
            subset=[r for r in rows if start<int(r['env_steps'])<=end]
            windows.append(dict(start=start,end=end,**{k:float(np.mean([float(r[k]) for r in subset])) for k in ('coverage','return')}))
        training[name]=dict(manifest=manifest,validation_windows=windows,
                            metric_csv_rows=len(counts),unique_updates=len(set(counts)))
    result=dict(models=models,paired=paired,critic=critic,moments=moments,stability=stability,
        fixed_scenes=cases,training=training,palette=COLORS,groups=list(GROUPS),
        uncertainty='2000 paired validation-scene bootstrap draws; seed 950001; not training-seed CI')
    (out/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    style()
    plt.rcParams['axes.prop_cycle']=plt.cycler(color=[COLORS[x] for x in GROUPS])
    figdir=out/'figures'
    def line(ax,name,y,**kw):
        ax.plot(np.arange(1,len(y)+1),y,color=COLORS[name],ls=LINES[name],label=NAMES[name],lw=1.7,**kw)
    plots={
      'task_curves':[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Total throughput (Mbps)'),('reward',1,'Task reward'),('discovery_fraction',100,'Instantaneous sensing (%)')],
      'deployment':[('disconnection_ratio',100,'Disconnected UAV (%)'),('boundary_within_150_fraction',100,'Within 150 m of edge (%)'),('boundary_reflection_fraction',100,'Boundary reflection (%)'),('uav_close_neighbor_fraction',100,'Close UAV (%)')],
      'movement':[('actual_speed_mean',1,'Actual speed (m/s)'),('action_norm_mean',30,'Mean command amplitude (m)'),('near_hover_fraction',100,'Actual move <1 m (%)'),('uav_nearest_neighbor_mean',1,'Nearest UAV distance (m)')],
      'service_bottlenecks':[('sensing_rejected_users',1,'Unseen UE'),('downlink_rejected_users',1,'Downlink rejected UE'),('backhaul_rejected_users',1,'Backhaul rejected UE'),('capacity_rejected_users',1,'Capacity / protocol unserved UE')],
      'protocol':[('proposal_acceptance',100,'Proposal acceptance (%)'),('invalidated_proposals',1,'Invalidated proposals'),('duplicate_proposals',1,'Duplicate valid proposals'),('over_capacity_fraction',100,'Candidates > capacity (%)')]}
    for filename,panels in plots.items():
        fig,axes=plt.subplots(2,2,figsize=(11,6.7))
        for ax,(key,scale,title) in zip(axes.flat,panels):
            for name in GROUPS:
                if key in series[name]:line(ax,name,series[name][key].mean(0)*scale)
            ax.set(xlabel='Episode step (post-action)',ylabel=title)
        axes.flat[0].legend(fontsize=7)
        if filename=='protocol':fig.suptitle('Top-K protocol diagnostics; centralized baseline has no proposals')
        save_figure(fig,figdir,filename)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,key,scale,title in zip(axes,('coverage','tau','reward'),(1,.1,1),('Coverage contribution','Throughput contribution','Total task reward')):
        for name in GROUPS:line(ax,name,series[name][key].mean(0)*scale)
        ax.set(xlabel='Episode step',ylabel=title)
    axes[0].legend(fontsize=7);save_figure(fig,figdir,'reward_components')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,scale,title in zip(axes,('coverage','throughput_bps','return'),(100,1e-6,1),('Mean coverage (%)','Throughput (Mbps)','Episode return')):
        for i,name in enumerate(GROUPS):
            a=np.array([float(r[key])*scale for r in episodes[name]])
            mean=a.mean();low,high=np.quantile(a[draws].mean(1),[.025,.975])
            ax.bar(i,mean,color=COLORS[name],alpha=.85)
            ax.errorbar(i,mean,yerr=[[mean-low],[high-mean]],color='#222222',capsize=3)
        ax.set_xticks(range(3),['Baseline','DPPO','Teacher']);ax.set(ylabel=title)
    fig.suptitle('10M best checkpoints; error bars: validation-scene bootstrap')
    save_figure(fig,figdir,'performance')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for name in GROUPS:
        with np.load(out/name/'action_moments.npz') as z:
            for ax,y in [(axes[0],np.linalg.norm(z['moments'][:,:,:,0],axis=-1).mean((0,2))),
                         (axes[1],z['moments'][:,:,:,1].sum(-1).mean((0,2)))]:
                ax.plot(z['times'],y,color=COLORS[name],ls=LINES[name],label=NAMES[name])
    axes[0].set(ylabel='Conditional mean command norm (m)',xlabel='State time');axes[1].set(ylabel='Conditional variance trace (m²)',xlabel='State time')
    axes[0].legend(fontsize=7);save_figure(fig,figdir,'action_randomness')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,name in enumerate(GROUPS):
        st=stability[name]
        for ax,key,scale in [(axes[0],'max_step_drop_mean',100),(axes[1],'large_drop_events_mean',1)]:
            ax.bar(i,st[key]*scale,color=COLORS[name],alpha=.85)
        drops=np.maximum(0,-np.diff(series[name]['coverage'],axis=1)).max(1)
        axes[2].plot(np.sort(drops)*100,np.arange(1,33)/32,color=COLORS[name],ls=LINES[name],label=NAMES[name])
    for ax,title in zip(axes[:2],('Mean episode maximum single-step drop (pp)','Drops >=10 pp per episode')):
        ax.set_xticks(range(3),['Baseline','DPPO','Teacher']);ax.set(ylabel=title)
    axes[2].set(xlabel='Episode maximum coverage drop (pp)',ylabel='Scene CDF');axes[2].legend(fontsize=6)
    save_figure(fig,figdir,'stability')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,run in [('baseline_best',args.baseline_run),('dppo_best',args.dppo_run)]:
        rows=read(run/'validation.csv');x=np.array([int(r['env_steps'])/1e6 for r in rows])
        for ax,key,scale in [(axes[0],'return',1),(axes[1],'coverage',100)]:
            y=np.array([float(r[key])*scale for r in rows]);ax.plot(x,y,color=COLORS[name],alpha=.25,lw=.5)
            ax.plot(x[9:],np.convolve(y,np.ones(10)/10,'valid'),color=COLORS[name],ls=LINES[name],label=NAMES[name])
            ax.scatter([sources['sources'][name]['env_steps']/1e6],[models[name][key]*scale],color=COLORS[name],s=30)
            ax.set(xlabel='Online joint transitions (millions)',ylabel='Validation return' if key=='return' else 'Validation coverage (%)')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle('Training histories; line: 10-evaluation trailing mean; dot: selected best')
    save_figure(fig,figdir,'validation_learning')
    fig,axes=plt.subplots(2,2,figsize=(11,6.5))
    for name,run in [('baseline_best',args.baseline_run),('dppo_best',args.dppo_run)]:
        rows=read(run/'metrics.csv');x=np.array([int(r['env_steps'])/1e6 for r in rows])
        for ax,key in zip(axes.flat,('actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction')):
            y=np.array([float(r[key]) for r in rows]);n=50
            ax.plot(x[n-1:],np.convolve(y,np.ones(n)/n,'valid'),color=COLORS[name],ls=LINES[name],label=NAMES[name])
            ax.set(xlabel='Online joint transitions (millions)',ylabel=key)
    axes.flat[0].legend(fontsize=7);fig.suptitle('Training updates; actor loss definitions differ between PPO and DPPO')
    save_figure(fig,figdir,'loss_and_policy_diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ('baseline_best','dppo_best'):
        with np.load(out/name/'critic_predictions.npz') as z:
            for ax,pred,truth in [(axes[0],'predicted','mc_return'),(axes[1],'predicted_average','mc_average')]:
                ax.plot(np.arange(400),z[pred].mean(0),color=COLORS[name],label=NAMES[name]+' V')
                ax.plot(np.arange(400),z[truth].mean(0),color=COLORS[name],ls=':',label=NAMES[name]+' MC')
    axes[0].set(ylabel='Cumulative task value',xlabel='State time before action');axes[1].set(ylabel='Discounted average task reward',xlabel='State time before action')
    axes[0].legend(fontsize=6);save_figure(fig,figdir,'critic_mc')
    for scene in (10000,10001):
        fig,axes=plt.subplots(3,4,figsize=(15,11.2))
        for row,name in enumerate(GROUPS):
            with np.load(out/name/f'trace_{scene}.npz') as z:
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,users,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')]
                    seen,served=z['sensed'][t].any(0),c.any(0);bs=z['bs_positions']
                    for mask,marker,color in [(~seen,'.','#999999'),(seen&~served,'x','#E69F00'),(served,'o',COLORS[name])]:
                        ax.scatter(users[mask,0],users[mask,1],marker=marker,color=color,s=10,linewidths=.6)
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]:ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.6,alpha=.7)
                        for g in range(3):
                            if b[i,g]==1:ax.plot([xy[i,0],bs[g,0]],[xy[i,1],bs[g,1]],'--',color='#777777',lw=.8)
                        for m in np.flatnonzero(c[i]):ax.plot([xy[i,0],users[m,0]],[xy[i,1],users[m,1]],color=COLORS[name],lw=.4,alpha=.18)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:ax.scatter(xy[mask,0],xy[mask,1],marker=marker,color=COLORS[name],s=32,edgecolors='black',linewidths=.5)
                    ax.scatter(bs[:,0],bs[:,1],marker='*',color='black',s=90,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[name]} | t={t} | {served.mean():.1%}',xlabel='x (m)',ylabel='y (m)')
        handles=[Line2D([],[],ls='none',marker=m,color=c,label=l) for m,c,l in [('o','#333333','Served UE (group color)'),('x','#E69F00','Sensed only'),('.','#999999','Unseen'),('^','#333333','Connected UAV'),('s','#333333','Disconnected UAV'),('*','black','BS')]]
        fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8,bbox_to_anchor=(.5,-.01));fig.suptitle(f'Fixed validation scene {scene}')
        save_figure(fig,figdir,f'topology_best_{scene}')
    write_report(out,result,sources)
    print(json.dumps(dict(output=str(out),models={k:{m:v[m] for m in ('coverage','return','throughput_bps')} for k,v in models.items()})))


def write_report(out,result,sources):
    m=result['models']
    lines=['# Baseline 10M best、DPPO 10M best与新teacher验证','',
        '仅比较用户指定三组。红色：baseline 10M best；蓝色：DPPO 10M best；橙色：baseline 10M teacher。','',
        '32个验证场景10000–10031，400步真正终止，采样动作种子880001，批次16，CUDA FP32。正式保留测试集未使用。',
        '场景、容量10、中心UE初始化、移动UE、观察历史和物理参数一致。奖励为覆盖率＋0.1×归一化吞吐，归一化分母750 Mbps；Mbps为全系统总吞吐。','',
        '## 模型与teacher定义','',
        f"baseline完成10M从零训练，best实际位于{sources['sources']['baseline_best']['env_steps']:,}步；DPPO完成10M在线训练，best实际位于{sources['sources']['dppo_best']['env_steps']:,}步。两组均按32场景全回合任务回报选模。",
        'teacher组直接冻结上述baseline best的同一个运动actor，局部评分为sigmoid((移动前下行SINR−门槛)/10)，使用移动前Top-K、移动后门控；不补位、不二次推理。初始关联沿用baseline。它没有独立训练。',
        '**本次新teacher用于后验协议比较；原DPPO示范仍来自旧1M baseline的671,744步模型，不是本次10M teacher。没有重新采集示范或训练DPPO。**','',
        '| 模型 | 全回合覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 |', '|---|---:|---:|---:|---:|---:|']
    for name in GROUPS:
        s=m[name];lines.append(f"| {NAMES[name]} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_bps']/1e6:.3f} | {s['return']:.4f} |")
    lines+=['','## 核心解读','']
    d,b,t=[m[x] for x in ('dppo_best','baseline_best','teacher_topk')]
    lines.append(f"同为10M训练预算，DPPO best相对baseline best：平均覆盖变化{100*(d['coverage']-b['coverage']):+.3f}个百分点，吞吐变化{(d['throughput_bps']-b['throughput_bps'])/1e6:+.3f}Mbps（{100*(d['throughput_bps']/b['throughput_bps']-1):+.2f}%），任务回报变化{d['return']-b['return']:+.4f}。相比此前1M baseline，预算差异已缩小；本次仅报告新10M对照。")
    lines.append(f"同一个baseline actor换成SINR评分Top-K闭环执行后：平均覆盖变化{100*(t['coverage']-b['coverage']):+.3f}个百分点，吞吐变化{(t['throughput_bps']-b['throughput_bps'])/1e6:+.3f}Mbps，任务回报变化{t['return']-b['return']:+.4f}。该差值反映协议与随之发生的闭环状态变化。")
    lines.append(f"DPPO相对新teacher：平均覆盖变化{100*(d['coverage']-t['coverage']):+.3f}个百分点，吞吐变化{(d['throughput_bps']-t['throughput_bps'])/1e6:+.3f}Mbps，任务回报变化{d['return']-t['return']:+.4f}。它不能单独分离扩散表示、学习评分和运动策略的贡献。")
    cov=next(p for p in result['paired'] if p['comparison']=='dppo_best - baseline_best' and p['metric']=='coverage')
    lines.append(f"**DPPO对baseline的全回合覆盖差值95%场景区间为[{100*cov['low']:+.3f}, {100*cov['high']:+.3f}]个百分点，{'包含零；目前不能据此声称覆盖提升具有稳健的统计证据' if cov['low']<=0<=cov['high'] else '未包含零'}。** 吞吐与回报的配对区间见下表；best在同一验证集上选择，区间还有选模偏差。")
    lines.append(f"**后期覆盖没有超过baseline。** 后100步分别为baseline {b['last100_coverage']:.3%}、DPPO {d['last100_coverage']:.3%}，终态分别为{b['final_coverage']:.3%}、{d['final_coverage']:.3%}。DPPO较高的全程覆盖伴随更快的早期展开，不能据此认定部署维持已优于baseline。")
    lines.append(f"按全部服务用户时隙汇总，每个已服务UE的平均数据率为baseline {b['pooled_served_user_rate_bps']/1e6:.3f}Mbps、DPPO {d['pooled_served_user_rate_bps']/1e6:.3f}Mbps、teacher {t['pooled_served_user_rate_bps']/1e6:.3f}Mbps。该比值为总吞吐之和除以服务UE时隙之和；说明数据率差异不只来自服务人数，但不能直接归因于学习评分。")
    lines+=['','## 配对场景差值','',
        '差值为左减右；2000次配对场景bootstrap、种子950001。区间仅描述这32个验证场景，不是训练种子置信区间；同一验证集选best存在选择偏差。','',
        '| 比较 | 指标 | 差值 | 95%场景区间 |','|---|---|---:|---:|']
    for p in result['paired']:
        if p['metric'] not in ('coverage','throughput_bps','return','last100_coverage','maintain_coverage'):continue
        scale,unit=(1e-6,' Mbps') if p['metric']=='throughput_bps' else ((1,'') if p['metric']=='return' else (100,' 个百分点'))
        label=p['comparison'].replace('dppo_best','DPPO').replace('baseline_best','Baseline').replace('teacher_topk','Teacher')
        lines.append(f"| {label} | {p['metric']} | {p['difference']*scale:+.3f}{unit} | [{p['low']*scale:+.3f}, {p['high']*scale:+.3f}] |")
    lines+=['','## 部署维持与服务瓶颈','',
        '| 模型 | 第51–150步覆盖 | 后期−前期覆盖 | 覆盖曲线峰值时刻 | 断联比例 | 边界反射 |','|---|---:|---:|---:|---:|---:|']
    for name in GROUPS:
        s=m[name];lines.append(f"| {NAMES[name]} | {s['early51_150_coverage']:.3%} | {100*s['maintain_coverage']:+.3f}个百分点 | {s['peak_mean_curve_step']} | {s['disconnection_ratio']:.3%} | {s['boundary_reflection_fraction']:.3%} |")
    lines+=['','| 模型 | 未感知UE | 下行不足UE | 回传不足UE | 容量/协议未服务UE |','|---|---:|---:|---:|---:|']
    for name in GROUPS:
        s=m[name];vals=[s[k] for k in ('sensing_rejected_users','downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users')]
        lines.append(f"| {NAMES[name]} | "+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['','最后一列不能全部解释为容量浪费；Top-K组还包含重复提议、移动后失效和新候选等待造成的缺口。teacher换协议会改变后续本机服务标记与动作，因此是闭环协议对照，不是固定运动轨迹下的纯关联求解比较。','',
        '## 覆盖骤降与固定场景','',
        '相邻覆盖变化从动作后状态t1–t400计算，不含初始t0→t1；大幅下降阈值为10个百分点。','',
        '| 模型 | 每回合最大单步跌幅均值 | 最大跌幅95分位 | 每回合大幅下降次数 | 终态低于30%场景数 |','|---|---:|---:|---:|---:|']
    for name in GROUPS:
        s=result['stability'][name];lines.append(f"| {NAMES[name]} | {100*s['max_step_drop_mean']:.2f}个百分点 | {100*s['max_step_drop_p95']:.2f}个百分点 | {s['large_drop_events_mean']:.2f} | {s['final_below30']}/32 |")
    lines+=['','DPPO的平均断联比例较低，但每回合最大单步覆盖损失仍很大；平均联网率不能替代严重服务崩落的评估。上述统计未将事件全部归因为回传失效，还可能包含下行、容量及提议时序影响。']
    for scene in ('10000','10001'):
        lines+=['',f'场景{scene}的终态：']
        for name in GROUPS:
            s=result['fixed_scenes'][name][scene][-1]
            lines.append(f"- {NAMES[name]}：覆盖{s['coverage']:.2%}，联网{s['connected']}/18，感知{s['sensed']}人，可服务候选{s['eligible']}人，提议{s['proposals'] if s['proposals'] is not None else '不适用'}条。")
    lines+=['','## Critic与训练成本','',
        '仅对原关联baseline和DPPO自己的on-policy轨迹检查MC；teacher切换协议后原critic不是该teacher的价值估计，因此不将其误差作为teacher预测准确性。','',
        '| 模型 | 累计价值RMSE | 标准差归一化RMSE |','|---|---:|---:|']
    for name in ('baseline_best','dppo_best'):
        c=result['critic'][name]['cumulative'];lines.append(f"| {NAMES[name]} | {c['all']['rmse']:.4f} | {c['normalized_rmse']:.4f} |")
    for name in ('baseline_best','dppo_best'):
        man=result['training'][name]['manifest'];seconds=man['elapsed_seconds']
        eval_s=man.get('evaluation_seconds',man.get('phase_seconds',{}).get('evaluation',0))
        lines.append(f"\n{NAMES[name]}在线训练总墙钟{seconds/3600:.2f}小时，其中验证{eval_s/3600:.2f}小时。")
    lines+=['','## 归因边界','',
        '在线预算已统一为10M；DPPO仍多出示范和预训练，actor容量、观察无线特征、学习率、PPO阈值、熵机制及关联协议不同。该比较支持系统级性能判断，不能把全部差值单独归因于扩散模型。',
        'teacher与baseline同一actor但关联协议不同；DPPO与teacher的差异同时包含运动策略、学习评分与训练流程。只有一个训练种子；本报告不增加训练、调参或测试集评估。','',
        '图表均提供PDF/SVG/PNG；重建只读取已保存数据，无需重新采样。','',
        '- [任务曲线](figures/task_curves.png)','- [性能汇总](figures/performance.png)',
        '- [奖励分量](figures/reward_components.png)','- [部署与断联](figures/deployment.png)','- [骤降与尾部稳定性](figures/stability.png)',
        '- [服务瓶颈](figures/service_bottlenecks.png)','- [协议诊断](figures/protocol.png)',
        '- [训练验证曲线](figures/validation_learning.png)','- [Loss及策略诊断](figures/loss_and_policy_diagnostics.png)',
        '- [Critic与MC](figures/critic_mc.png)','- [场景10000拓扑](figures/topology_best_10000.png)',
        '- [场景10001拓扑](figures/topology_best_10001.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,default=Path('runs/mappo_coverage_throughput_capacity10_10m_seed0'))
    parser.add_argument('--dppo-run',type=Path,default=Path('runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0'))
    parser.add_argument('--output',type=Path,default=Path('analysis/baseline_dppo_10m_best_teacher_seed0'))
    parser.add_argument('--evaluate',choices=('baseline_best','teacher_topk'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--moment-repeats',type=int,default=64)
    args=parser.parse_args()
    evaluate_group(args) if args.evaluate else rebuild(args)
