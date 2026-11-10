"""Frozen 3/4/5-neighbor 1M evaluation and comparison with two 10M best models."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCENES = list(range(10000, 10032))
RUNS = {
    'neighbors3': Path('runs/mappo_coverage_throughput_capacity10_1m_seed0'),
    'neighbors4': Path('runs/mappo_capacity10_neighbors4_1m_seed0'),
    'neighbors5': Path('runs/mappo_capacity10_neighbors5_1m_seed0'),
}
GROUPS = (*RUNS, 'baseline10m', 'dppo10m')
NAMES = dict(neighbors3='Baseline 1M / 3 neighbors', neighbors4='Baseline 1M / 4 neighbors',
             neighbors5='Baseline 1M / 5 neighbors', baseline10m='Baseline 10M best', dppo10m='DPPO 10M best')
COLORS = dict(neighbors3='#444444', neighbors4='#009E73', neighbors5='#E69F00',
              baseline10m='#D62728', dppo10m='#0072B2')
LINES = dict(neighbors3=':', neighbors4='-', neighbors5='--', baseline10m='-.', dppo10m=(0, (6, 2)))
MARKERS = dict(neighbors3='o', neighbors4='s', neighbors5='^', baseline10m='D', dppo10m='x')
REFS = dict(baseline10m='baseline_best', dppo10m='dppo_best')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate_one(args):
    import torch
    from uav_research.algorithms import make_learner
    from uav_research.config import config_from_dict, load_config
    from uav_research.runtime import load_checkpoint, atomic_json
    from uav_research.diffusion_dppo.storage import configure
    from uav_research.diffusion_dppo.evaluation import evaluate
    from uav_research.diffusion_dppo.config import DiffusionSettings

    group, kind = args.evaluate, args.checkpoint
    run = RUNS[group]
    checkpoint = run/f'{kind}.pt'
    sha = digest(checkpoint)
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['env_steps'] == 1_000_000
    saved = load_checkpoint(checkpoint)
    cfg = config_from_dict(saved['config'])
    baseline = load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    expected = baseline.as_dict()
    expected['env']['neighbor_slots'] = int(group[-1])
    expected['run']['device'] = cfg.run.device
    assert cfg.as_dict() == expected, 'Changes beyond neighbor slots/device'
    original_device = cfg.run.device
    cfg.run.device = args.device
    cfg.run.validation_seeds = SCENES
    cfg.run.evaluation_action_seed = 880001
    cfg.run.evaluation_policy = 'sample'
    cfg.run.num_envs = 16
    device = configure(cfg)
    model = make_learner(cfg, device)
    model.load_state_dict(saved['learner'])
    model.actor.eval().requires_grad_(False)
    model.value.eval().requires_grad_(False)
    neighbor_data = []

    class FrozenExecution(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = model.actor

        @torch.no_grad()
        def sample(self, obs):
            e = cfg.env
            start = e.self_dim + 2*e.num_bs
            ids = obs.fixed[..., start:e.current_obs_dim:e.neighbor_dim]
            observed = (ids > 0).sum(-1)
            adjacency = obs.fixed[..., 5:5+e.num_uavs] > 0
            # The physical adjacency matrix includes its diagonal; self is not a neighbor.
            degree = adjacency.sum(-1) - np.diagonal(adjacency, axis1=-2, axis2=-1)
            np.testing.assert_array_equal(observed, np.minimum(degree, e.neighbor_slots))
            neighbor_data.append(np.stack([degree, observed], -1).astype(np.uint8))
            action = self.actor.sample(obs.to_torch(device), False)[0]
            return action.cpu().numpy(), None

    out = args.output/kind/group
    print(f'Evaluating {kind}/{group}: step={saved["env_steps"]}', flush=True)
    # Reuse the rich read-only evaluator with the legacy centralized environment.
    # Its runtime guards need diffusion settings; the MAPPO model/config remain unchanged.
    evaluation_cfg = SimpleNamespace(env=cfg.env, algo=cfg.algo, run=cfg.run, diffusion=DiffusionSettings())
    result = evaluate(evaluation_cfg, FrozenExecution(), SCENES, out, critic=model, greedy=True)
    # Evaluation loops over batches first, then time; retain this ordering explicitly.
    neighbors = np.asarray(neighbor_data).reshape(2, 400, 16, 18, 2).transpose(0, 2, 1, 3, 4).reshape(32, 400, 18, 2)
    np.savez_compressed(out/'neighbor_availability.npz', data=neighbors,
                       names=['physical_direct_degree', 'observed_neighbor_count'], scene_seeds=SCENES)
    assert digest(checkpoint) == sha
    training_row = next(r for r in read(run/'validation.csv') if int(r['env_steps']) == saved['env_steps'])
    differences = {k: result['summary'][k] - float(training_row[k]) for k in ('coverage', 'return', 'throughput_bps')}
    if original_device.startswith('cuda'):
        assert max(abs(v) for v in differences.values()) < 1e-6, differences
    atomic_json(out/'provenance.json', dict(checkpoint=str(checkpoint), sha256=sha, env_steps=saved['env_steps'],
        training_budget=1_000_000, neighbor_slots=cfg.env.neighbor_slots, scene_seeds=SCENES,
        action_seed=880001, batch_size=16, device=args.device, training_device=original_device,
        checkpoint_kind=kind, validation_difference=differences, script_sha256=digest(__file__)))
    print(json.dumps(dict(group=group, kind=kind, step=saved['env_steps'],
        **{k: result['summary'][k] for k in ('coverage', 'return', 'throughput_bps')},
        validation_difference=differences)), flush=True)


def series_and_rows(path, group):
    from uav_research.coverage_stability import coverage_drop_metrics
    with np.load(path/'step_metrics.npz') as z:
        assert z['scene_seeds'].tolist() == SCENES and z['metrics'].shape[:2] == (32, 400)
        s = dict(zip(z['names'].tolist(), z['metrics'].transpose(2, 0, 1)))
    assert all(np.isfinite(v).all() for v in s.values())
    np.testing.assert_allclose(s['reward'], s['coverage']+.1*s['tau'], atol=2e-7)
    rows = []
    for i, scene in enumerate(SCENES):
        row = dict(group=group, scene_seed=scene, **coverage_drop_metrics(s['coverage'][i]))
        row.update(throughput_mbps=float(s['throughput_bps'][i].mean()/1e6),
            last100_throughput_mbps=float(s['throughput_bps'][i,-100:].mean()/1e6),
            return_=float(s['reward'][i].sum()),
            disconnection_ratio=float(s['disconnection_ratio'][i].mean()),
            robustness=float(s['robustness'][i].mean()),
            sensing=float(s['discovery_fraction'][i].mean()),
            boundary150_fraction=float(s['boundary_within_150_fraction'][i].mean()),
            reflection_fraction=float(s['boundary_reflection_fraction'][i].mean()),
            close_neighbor_fraction=float(s['uav_close_neighbor_fraction'][i].mean()),
            nearest_neighbor_m=float(s['uav_nearest_neighbor_mean'][i].mean()),
            speed_mps=float(s['actual_speed_mean'][i].mean()))
        row['return'] = row.pop('return_')
        rows.append(row)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('group', 'scene_seed')}
    summary['mean_curve_peak_step'] = int(s['coverage'].mean(0).argmax()+1)
    summary['mean_curve_peak_coverage'] = float(s['coverage'].mean(0).max())
    summary['final_below30_scenes'] = int((s['coverage'][:,-1] < .3).sum())
    summary['scenes_with_10pp_drop'] = int(sum(r['any_drop_10pp'] for r in rows))
    return s, rows, summary


def rebuild(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from uav_research.plotting import style, save_figure
    from uav_research.diffusion_dppo.report import error_metrics

    out = args.output
    for group, name in REFS.items():
        target = out/'references'/group
        if not target.exists():
            shutil.copytree(args.references/name, target)
    series, rows, summary, sources, paths, critic = {}, {}, {}, {}, {}, {}
    inputs = {}
    for group in GROUPS:
        p = out/('best' if group in RUNS else 'references')/group
        prov = json.loads((p/'provenance.json').read_text())
        assert prov['scene_seeds'] == SCENES and prov['action_seed'] == 880001 and prov['device'] == 'cuda:0'
        cp = Path(prov['checkpoint'])
        if cp.is_absolute() and str(cp).startswith('/home/dym/uav_sac/'):
            cp = Path(str(cp).removeprefix('/home/dym/uav_sac/'))
        assert digest(cp) == prov['sha256']
        sources[group] = dict(checkpoint=str(cp), **prov)
        paths[group] = p
        series[group], rows[group], summary[group] = series_and_rows(p, group)
        for f in p.iterdir():
            if f.is_file(): inputs[str(f.relative_to(out))] = digest(f)
        with np.load(p/'critic_predictions.npz') as z:
            critic[group] = error_metrics(z['predicted'], z['mc_return'])
        summary[group]['checkpoint_step'] = int(prov['env_steps'])
    supplemental, neighbor_usage = {}, {}
    for group in RUNS:
        p = out/'last'/group
        prov = json.loads((p/'provenance.json').read_text())
        assert digest(RUNS[group]/'last.pt') == prov['sha256'] and prov['env_steps'] == 1_000_000
        _, r, su = series_and_rows(p, group)
        supplemental[group] = dict(summary=su, rows=r)
        neighbor_usage[group] = {}
        for kind in ('best', 'last'):
            q = out/kind/group
            with np.load(q/'neighbor_availability.npz') as z:
                d = z['data'][...,0]
                observed = z['data'][...,1]
                neighbor_usage[group][kind] = dict(
                    mean_degree=float(d.mean()), mean_observed=float(observed.mean()),
                    degree_at_least4=float((d>=4).mean()), degree_at_least5=float((d>=5).mean()),
                    degree_exceeds_slots=float((d>int(group[-1])).mean()),
                    missing_neighbor_mean=float((d-observed).mean()))
            for f in q.iterdir():
                if f.is_file(): inputs[str(f.relative_to(out))] = digest(f)
    # Fixed scenario geometry and UE randomness must be identical across all five policies.
    cases = {}
    for scene in (10000, 10001):
        initial, ue = None, None
        for group in GROUPS:
            with np.load(paths[group]/f'trace_{scene}.npz') as z:
                if initial is None: initial, ue = z['uav_xy'][0].copy(), z['user_xy'].copy()
                else:
                    np.testing.assert_array_equal(initial, z['uav_xy'][0])
                    np.testing.assert_array_equal(ue, z['user_xy'])
                c, b = z['C'], z['B']
                assert (c.sum(1)<=1).all() and (c.sum(2)<=10).all()
                np.testing.assert_allclose(c.sum((1,2))/150, z['coverage'])
                cases[f'{group}/{scene}'] = [dict(time=t, coverage=float(z['coverage'][t]),
                    connected=int((b[t].min(1)<=18).sum()),
                    uu_links=int(np.triu(z['A'][t],1).sum()),
                    components=int(component_count(z['A'][t]))) for t in (0,150,200,400)]
    draws = np.random.default_rng(950001).integers(32, size=(2000,32))
    paired = []
    for kind in ('best', 'last'):
        source = rows if kind=='best' else {g:x['rows'] for g,x in supplemental.items()}
        for group in ('neighbors4', 'neighbors5'):
            for ref in (('neighbors3','baseline10m','dppo10m') if kind=='best' else ('neighbors3',)):
                for key in ('coverage','last100_coverage','final_coverage','throughput_mbps','return',
                            'max_drop_1','max_drop_5','drop_events_10pp','maintain_coverage','disconnection_ratio'):
                    d = np.array([a[key]-b[key] for a,b in zip(source[group],source[ref])])
                    low,high = np.quantile(d[draws].mean(1), [.025,.975])
                    paired.append(dict(checkpoint=kind,group=group,reference=ref,metric=key,
                                       difference=float(d.mean()),low=float(low),high=float(high)))
    result = dict(groups=list(GROUPS),best=summary,last={g:v['summary'] for g,v in supplemental.items()},
        paired=paired,neighbor_usage=neighbor_usage,critic_best=critic,fixed_scenes=cases,
        sources=sources,colors=COLORS,uncertainty='2000 paired scene bootstrap; seed 950001; not training-seed CI')
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_csv(out/'best_scene_metrics.csv',[r for g in GROUPS for r in rows[g]])
    write_csv(out/'last_scene_metrics.csv',[r for g in RUNS for r in supplemental[g]['rows']])
    write_csv(out/'paired_scene_differences.csv',paired)
    manifest = dict(evaluation_files=inputs,sources=sources,
        training_log_files={str(RUNS[g]/n):digest(RUNS[g]/n) for g in RUNS for n in ('metrics.csv','validation.csv','manifest.json','config.yaml')})
    (out/'report_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    style()
    figdir = out/'figures'
    def line(ax,g,y):
        ax.plot(np.arange(1,len(y)+1),y,color=COLORS[g],ls=LINES[g],marker=MARKERS[g],
                markevery=45,markersize=3,lw=1.5,label=NAMES[g])
    for filename,panels in [
        ('task_curves_best',[('coverage',100,'Coverage (%)'),('throughput_bps',1e-6,'Total throughput (Mbps)'),
            ('reward',1,'Reward: coverage + 0.1 tau'),('discovery_fraction',100,'Sensing (%)')]),
        ('deployment_best',[('disconnection_ratio',100,'Disconnected UAV (%)'),('boundary_within_150_fraction',100,'UAV within 150 m of boundary (%)'),
            ('uav_close_neighbor_fraction',100,'Nearest UAV within 100 m (%)'),('actual_speed_mean',1,'Actual speed (m/s)')])]:
        fig,axes = plt.subplots(2,2,figsize=(13,7.5))
        for ax,(key,scale,label) in zip(axes.flat,panels):
            for g in GROUPS: line(ax,g,series[g][key].mean(0)*scale)
            ax.set(xlabel='Episode step (post-action)',ylabel=label)
        axes.flat[0].legend(fontsize=7,ncol=2)
        save_figure(fig,figdir,filename)
    fig,axes = plt.subplots(1,3,figsize=(14,4.4))
    for ax,key,scale,label in zip(axes,('coverage','last100_coverage','max_drop_1'),(100,100,100),
            ('Mean coverage (%)','Last-100 coverage (%)','Mean episode max single-step drop (pp)')):
        for i,g in enumerate(GROUPS):
            v=np.array([r[key]*scale for r in rows[g]]);m=v.mean();lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
            ax.bar(i,m,color=COLORS[g]);ax.errorbar(i,m,yerr=[[max(0,m-lo)],[max(0,hi-m)]],color='black',capsize=3)
        ax.set_xticks(range(5),['1M / N3','1M / N4','1M / N5','Base 10M','DPPO 10M'],rotation=25,ha='right');ax.set(ylabel=label)
    fig.suptitle('Best models; scene bootstrap intervals; training budgets differ')
    save_figure(fig,figdir,'best_summary')
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for g in RUNS:
        p=out/'last'/g
        ss,_,_=series_and_rows(p,g)
        line(axes[0],g,ss['coverage'].mean(0)*100);line(axes[1],g,ss['throughput_bps'].mean(0)/1e6)
    axes[0].set(ylabel='Coverage (%)',xlabel='Episode step');axes[1].set(ylabel='Throughput (Mbps)',xlabel='Episode step')
    axes[0].legend(fontsize=8);fig.suptitle('Same-budget last models: three neighbor counts, each 1M')
    save_figure(fig,figdir,'same_budget_last')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for g in RUNS:
        records=read(RUNS[g]/'validation.csv');x=np.array([float(r['env_steps']) for r in records])/1e6
        for ax,key,scale in zip(axes.flat,('coverage','throughput_bps','return','disconnection_ratio'),(100,1e-6,1,100)):
            ax.plot(x,[float(r[key])*scale for r in records],color=COLORS[g],ls=LINES[g],label=NAMES[g])
            ax.set(xlabel='Joint transitions (millions)',ylabel=key)
    axes.flat[0].legend(fontsize=8);save_figure(fig,figdir,'validation_learning_1m')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for g in RUNS:
        records=read(RUNS[g]/'metrics.csv');x=np.array([float(r['env_steps']) for r in records])/1e6
        for ax,key in zip(axes.flat,('actor_loss','value_loss','joint_entropy','ppo_approx_kl')):
            if key in records[0]:
                y=np.array([float(r[key]) for r in records]);ax.plot(x,y,color=COLORS[g],alpha=.2,lw=.6)
                smooth=np.convolve(y,np.ones(15)/15,'valid') if len(y)>=15 else y
                ax.plot(x[len(x)-len(smooth):],smooth,color=COLORS[g],ls=LINES[g],label=NAMES[g])
            ax.set(xlabel='Joint transitions (millions)',ylabel=key)
    axes.flat[0].legend(fontsize=8);save_figure(fig,figdir,'loss_and_policy_1m')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for g in GROUPS:
        with np.load(paths[g]/'critic_predictions.npz') as z:
            axes[0].plot(np.arange(400),z['mc_return'].mean(0),color=COLORS[g],ls=LINES[g],label=NAMES[g]+' MC')
            axes[0].plot(np.arange(400),z['predicted'].mean(0),color=COLORS[g],ls=':',lw=1)
            axes[1].plot(np.arange(400),(z['predicted']-z['mc_return']).mean(0),color=COLORS[g],ls=LINES[g],label=NAMES[g])
    axes[0].set(xlabel='Pre-action time',ylabel='Task value (dotted: prediction)');axes[1].set(xlabel='Pre-action time',ylabel='Prediction bias')
    axes[1].legend(fontsize=7);save_figure(fig,figdir,'critic_best')
    for scene in (10000,10001):
        fig,axes=plt.subplots(5,4,figsize=(16,17))
        for row,g in enumerate(GROUPS):
            with np.load(paths[g]/f'trace_{scene}.npz') as z:
                for ax,t in zip(axes[row],(0,150,200,400)):
                    xy,ue,c,a,b=[z[k][t] for k in ('uav_xy','user_xy','C','A','B')];bs=z['bs_positions']
                    seen,served=z['sensed'][t].any(0),c.any(0)
                    for mask,marker,col in [(~seen,'.','#aaaaaa'),(seen&~served,'x','#777777'),(served,'o',COLORS[g])]:
                        ax.scatter(ue[mask,0],ue[mask,1],c=col,marker=marker,s=10,linewidths=.6)
                    for i in range(18):
                        for j in range(i):
                            if a[i,j]: ax.plot(xy[[i,j],0],xy[[i,j],1],color='#777777',lw=.6,alpha=.7)
                        for k in range(3):
                            if b[i,k]==1: ax.plot([xy[i,0],bs[k,0]],[xy[i,1],bs[k,1]],'--',color='#555555',lw=.7)
                    linked=b.min(1)<=18
                    for mask,marker in [(linked,'^'),(~linked,'s')]:
                        ax.scatter(xy[mask,0],xy[mask,1],c=COLORS[g],marker=marker,s=30,edgecolors='black',linewidths=.5)
                    ax.scatter(bs[:,0],bs[:,1],marker='*',c='black',s=80,clip_on=False)
                    ax.set(xlim=(0,3500),ylim=(0,3500),aspect='equal',title=f'{NAMES[g]} | t={t} | {served.mean():.1%}')
        fig.legend(handles=[Line2D([],[],ls='none',marker=m,color=c,label=l) for m,c,l in
            [('o','#333333','Served UE (group color)'),('x','#777777','Sensed only'),('.','#aaaaaa','Unseen'),
             ('^','#333333','Connected UAV'),('s','#333333','Disconnected UAV'),('*','black','BS')]],
            loc='lower center',ncol=6,fontsize=8,bbox_to_anchor=(.5,-.002))
        fig.suptitle(f'Fixed validation scene {scene}: best checkpoints, differing budgets')
        save_figure(fig,figdir,f'topology_best_{scene}')
        fig,ax=plt.subplots(figsize=(11,4))
        for g in GROUPS: line(ax,g,series[g]['coverage'][SCENES.index(scene)]*100)
        ax.set(xlabel='Episode step',ylabel='Coverage (%)',title=f'Fixed scene {scene}: unsmoothed coverage')
        ax.legend(fontsize=8,ncol=2);save_figure(fig,figdir,f'fixed_scene_{scene}')
    write_report(out,result)
    print(json.dumps(dict(output=str(out),best=summary,last=result['last']),ensure_ascii=False),flush=True)


def component_count(adjacency):
    remaining=set(range(len(adjacency)));count=0
    while remaining:
        count+=1;front=[remaining.pop()]
        while front:
            v=front.pop();others=remaining.intersection(np.flatnonzero(adjacency[v]).tolist())
            remaining.difference_update(others);front.extend(others)
    return count


def write_report(out, result):
    lines=['# 邻居观测3／4／5：1M baseline与10M参考比较','',
        '本轮仅冻结模型验证，不训练。32个验证场景10000–10031、动作采样种子880001、批次16、400步、CUDA FP32，关闭TF32。'
        '1M三组重新验证best和last；10M两组仅复用来源哈希一致、相同口径的best评估。不加入teacher或其他奖励消融组。','',
        '所有组奖励均为覆盖率＋0.1归一化吞吐，无鲁棒性与断联惩罚。主表按此前习惯展示各运行best；'
        '邻居数的同预算判断同时查看1M last。10M组预算不同，DPPO另有示范及预训练、服务评分关联和不同UE特征，不能将跨组差异单独归因于邻居数或算法。','',
        '原3邻居1M在CPU训练，4／5邻居在CUDA训练；本轮统一CUDA验证，但训练设备与采样随机流仍是额外差异。'
        '单训练种子只能提供当前运行的证据。下列区间按32场景配对bootstrap 2000次，种子950001；不是训练种子置信区间。','',
        '## Best：五组比较','',
        '| 组别 | best实际步数 | 平均覆盖 | 后100步 | 终态覆盖 | 吞吐Mbps | 回报 | 断联率 | 最大单步跌幅均值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g,s in result['best'].items():
        lines.append(f"| {NAMES[g]} | {s['checkpoint_step']:,} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_mbps']:.2f} | {s['return']:.3f} | {s['disconnection_ratio']:.2%} | {s['max_drop_1']*100:.2f}pp |")
    lines+=['','## 同预算last：各1M','',
        '| 邻居数 | 平均覆盖 | 后100步 | 终态覆盖 | 吞吐Mbps | 回报 | 最大单步跌幅均值 | 维持差（后100步−51至150步） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g,s in result['last'].items():
        lines.append(f"| {g[-1]} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_mbps']:.2f} | {s['return']:.3f} | {s['max_drop_1']*100:.2f}pp | {s['maintain_coverage']*100:+.2f}pp |")
    lines+=['','## 邻居数效应与稳定性','']
    for kind in ('best','last'):
        for g in ('neighbors4','neighbors5'):
            p=next(p for p in result['paired'] if p['checkpoint']==kind and p['group']==g and p['reference']=='neighbors3' and p['metric']=='coverage')
            lines.append(f"- {kind}，{g[-1]}邻居−3邻居：平均覆盖{p['difference']*100:+.3f}pp，场景95%区间[{p['low']*100:+.3f}, {p['high']*100:+.3f}]pp。")
    for g,s in result['best'].items():
        usage=result['neighbor_usage'].get(g,{}).get('best')
        text=f"- {NAMES[g]}：平均覆盖曲线峰值t{s['mean_curve_peak_step']}、{s['mean_curve_peak_coverage']:.2%}；后100步−51至150步为{s['maintain_coverage']*100:+.2f}pp；32场景中{s['scenes_with_10pp_drop']}个出现≥10pp单步跌幅。靠边150m内{s['boundary150_fraction']:.2%}，近邻<100m{s['close_neighbor_fraction']:.2%}。"
        if usage: text+=f"真实直连度≥4／5的UAV时隙占比{usage['degree_at_least4']:.2%}／{usage['degree_at_least5']:.2%}；超出观测槽位比例{usage['degree_exceeds_slots']:.2%}。"
        lines.append(text)
    lines+=['','## 固定场景终态','', '| 组别 | 场景10000覆盖／联网UAV | 场景10001覆盖／联网UAV |','|---|---:|---:|']
    for g in GROUPS:
        a,b=[result['fixed_scenes'][f'{g}/{scene}'][-1] for scene in (10000,10001)]
        lines.append(f"| {NAMES[g]} | {a['coverage']:.2%} / {a['connected']} | {b['coverage']:.2%} / {b['connected']} |")
    lines+=['','拓扑固定展示t0／150／200／400，按场景预先选定；UE运动和初始UAV位置跨组逐元素一致。'
        '三角形为联网UAV、方形为断联UAV、星号为BS、实线为UU、虚线为直接BS接入。所有直接链路均绘制，并非路由树。'
        '靠边率降低、邻居可见更多或断联率降低本身均不能代替覆盖与维持改善。','',
        '## 价值与来源','']
    for g,m in result['critic_best'].items():
        a=m['all']
        lines.append(f"- {NAMES[g]}：本策略完整回合MC对比，critic RMSE {a['rmse']:.3f}、MAE {a['mae']:.3f}、偏差{a['bias']:+.3f}。")
    lines+=['','各critic使用各自on-policy状态，误差不能解释为同状态上的网络对比。'
        '完整配置、检查点SHA256、逐场景指标、配对差值与原始预测均保留；重建不重新采样。','',
        '## 图表','',
        '- [五组覆盖／吞吐／奖励](figures/task_curves_best.png)',
        '- [同预算last](figures/same_budget_last.png)',
        '- [Best汇总与骤降](figures/best_summary.png)',
        '- [部署与贴边](figures/deployment_best.png)',
        '- [1M验证学习曲线](figures/validation_learning_1m.png)',
        '- [Loss与PPO诊断](figures/loss_and_policy_1m.png)',
        '- [场景10000拓扑](figures/topology_best_10000.png)',
        '- [场景10001拓扑](figures/topology_best_10001.png)',
        '- [critic与MC](figures/critic_best.png)','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('analysis/mappo_neighbors_1m_vs_10m_seed0'))
    p.add_argument('--evaluate',choices=list(RUNS))
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--references',type=Path,default=Path('analysis/baseline_dppo_10m_best_teacher_seed0'))
    p.add_argument('--rebuild',action='store_true')
    a=p.parse_args()
    if bool(a.evaluate)==bool(a.rebuild):p.error('Choose one of --evaluate / --rebuild')
    if a.evaluate:evaluate_one(a)
    else:rebuild(a)
