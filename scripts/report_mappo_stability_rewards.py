"""Uniform frozen evaluation of baseline 1M and three reward ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_research.coverage_stability import (
    BASELINE_RUN, REWARD_ABLATIONS, run_path, coverage_drop_metrics,
)

GROUPS = ('baseline', *REWARD_ABLATIONS)
NAMES = dict(baseline='Baseline 1M', disconnection='Disconnection penalty',
             robustness='Robustness reward', combined='Both additions')
COLORS = dict(baseline='#444444', disconnection='#D62728',
              robustness='#0072B2', combined='#E69F00')
SCENES = list(range(10000, 10032))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate_all(args, runs):
    import torch
    from uav_research.config import config_from_dict
    from uav_research.algorithms import make_learner
    from uav_research.runtime import configure_runtime, load_checkpoint, atomic_json
    from uav_research.evaluate import evaluate

    for kind in args.checkpoints:
        for group,run in runs.items():
            checkpoint=run/f'{kind}.pt'
            manifest=json.loads((run/'manifest.json').read_text())
            if manifest['env_steps']!=1_000_000 or manifest['status'] not in ('complete','completed'):
                raise ValueError(f'Expected a completed 1M run: {run}')
            sha=digest(checkpoint)
            out=args.output/kind/group
            cache=out/'stability_source.json'
            desired=dict(sha256=sha,scenes=SCENES,action_seed=880001,batch_size=16,device=args.device,
                         checkpoint=str(checkpoint.resolve()),kind=kind)
            cached=json.loads(cache.read_text()) if cache.exists() else {}
            curve=out/'reward_curves.npz'
            if (cached.get('evaluation')==desired and curve.exists() and
                    digest(curve)==cached.get('reward_curves_sha256')):
                print(f'Using verified-source evaluation cache: {kind}/{group}',flush=True)
                continue
            saved=load_checkpoint(checkpoint)
            cfg=config_from_dict(saved['config'])
            expected=REWARD_ABLATIONS.get(group,dict(disconnection_weight=0.,robustness_weight=0.))
            if (cfg.env.throughput_weight!=.1 or cfg.env.reward_primary!='coverage' or
                    cfg.env.boundary_penalty_weight!=0 or cfg.run.total_steps!=1_000_000 or
                    any(getattr(cfg.env,k)!=v for k,v in expected.items())):
                raise ValueError(f'Incorrect reward/budget for {group}')
            cfg.run.device=args.device
            cfg.run.validation_seeds=SCENES
            cfg.run.evaluation_action_seed=880001
            cfg.run.evaluation_policy='sample'
            cfg.run.num_envs=16
            device=configure_runtime(cfg)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            model=make_learner(cfg,device)
            model.load_state_dict(saved['learner'])
            model.actor.eval()
            print(f'Evaluating {kind}/{group}; checkpoint step={saved["env_steps"]}',flush=True)
            evaluate(cfg,model,SCENES,out,trace_count=2,reward_curves=True)
            if digest(checkpoint)!=sha:raise RuntimeError('Frozen checkpoint changed during evaluation')
            atomic_json(cache,dict(evaluation=desired,checkpoint_step=saved['env_steps'],
                weights=expected,selection='own training reward' if kind=='best' else 'fixed 1M budget',
                reward_curves_sha256=digest(out/'reward_curves.npz')))
            del model
            if device.type=='cuda':torch.cuda.empty_cache()


def rebuild(args, runs):
    import matplotlib.pyplot as plt
    from uav_research.plotting import style,save_figure
    from uav_research.config import config_from_dict

    style()
    plt.rcParams['axes.prop_cycle']=plt.cycler(color=[COLORS[g] for g in GROUPS])
    draws=np.random.default_rng(950001).integers(32,size=(2000,32))
    all_results={}
    for kind in args.checkpoints:
        series,rows,sources={},[],{}
        for group,run in runs.items():
            out=args.output/kind/group
            source=json.loads((out/'stability_source.json').read_text())
            if digest(run/f'{kind}.pt')!=source['evaluation']['sha256']:
                raise ValueError('Cached evaluation belongs to another checkpoint')
            assert source['evaluation']['scenes']==SCENES
            assert source['evaluation']['action_seed']==880001
            assert digest(out/'reward_curves.npz')==source['reward_curves_sha256']
            sources[group]=source
            cfg=config_from_dict(json.loads((out/'evaluation_config.json').read_text()))
            with np.load(out/'reward_curves.npz') as z:
                assert z['scene_seeds'].tolist()==SCENES and z['raw'].shape==(32,400,4)
                raw=z['raw'].copy();reward=z['reward'].copy();contributions=z['contributions'].copy()
                np.testing.assert_allclose(contributions.sum(-1),reward,atol=1e-12)
                np.testing.assert_allclose(raw*z['reward_weights'],contributions,atol=1e-12)
                np.testing.assert_allclose(z['reward_weights'],[1,.1,cfg.env.robustness_weight,-cfg.env.disconnection_weight])
                s=dict(coverage=raw[:,:,0],tau=raw[:,:,1],robustness=raw[:,:,2],disconnection_ratio=raw[:,:,3],
                       training_reward=reward,common_reward=raw[:,:,0]+.1*raw[:,:,1])
                s.update(dict(zip(z['diagnostic_names'].tolist(),z['diagnostics'].transpose(2,0,1))))
                series[group]=s
            for i,scene in enumerate(SCENES):
                row=dict(group=group,scene_seed=scene,**coverage_drop_metrics(s['coverage'][i]))
                row.update(training_return=float(reward[i].sum()),common_return=float(s['common_reward'][i].sum()),
                    throughput_bps=float(s['tau'][i].mean()*750e6),
                    robustness=float(s['robustness'][i].mean()),disconnection_ratio=float(s['disconnection_ratio'][i].mean()))
                rows.append(row)
        summary={g:{k:float(np.mean([r[k] for r in rows if r['group']==g]))
                    for k in rows[0] if k not in ('group','scene_seed')} for g in GROUPS}
        paired=[]
        for group in GROUPS[1:]:
            left=[r for r in rows if r['group']==group];right=[r for r in rows if r['group']=='baseline']
            for key in ('coverage','last100_coverage','throughput_bps','common_return',
                        'max_drop_1','max_drop_5','max_drawdown','drop_events_10pp','min20_coverage','disconnection_ratio'):
                delta=np.array([a[key]-b[key] for a,b in zip(left,right)])
                low,high=np.quantile(delta[draws].mean(1),[.025,.975])
                paired.append(dict(group=group,metric=key,difference=float(delta.mean()),low=float(low),high=float(high)))
        write_csv(args.output/f'{kind}_scene_metrics.csv',rows)
        write_csv(args.output/f'{kind}_paired_scene_differences.csv',paired)
        all_results[kind]=dict(summary=summary,paired=paired,sources=sources)
        figdir=args.output/'figures'
        for stem,panels in [('coverage_and_service',[('coverage',100,'Coverage (%)'),('tau',750,'Throughput (Mbps)'),
             ('disconnection_ratio',100,'Disconnected UAV (%)'),('robustness',100,'Retained fault coverage (%)')]),
             ('reward_components',[('coverage',1,'Coverage contribution'),('tau',.1,'Throughput contribution'),
             ('robustness',None,'Robustness contribution'),('disconnection_ratio',None,'Disconnection contribution')])]:
            fig,axes=plt.subplots(2,2,figsize=(11,6.5))
            for ax,(key,scale,label) in zip(axes.flat,panels):
                for group in GROUPS:
                    factor=scale
                    if factor is None:factor=REWARD_ABLATIONS.get(group,{}).get('robustness_weight',0) if key=='robustness' else -REWARD_ABLATIONS.get(group,{}).get('disconnection_weight',0)
                    ax.plot(np.arange(1,401),series[group][key].mean(0)*factor,color=COLORS[group],label=NAMES[group])
                ax.set(xlabel='Episode step',ylabel=label)
            axes.flat[0].legend(fontsize=7);save_figure(fig,figdir,f'{stem}_{kind}')
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,key,scale,label in zip(axes,('max_drop_1','max_drop_5','drop_events_10pp'),(100,100,1),
                    ('Mean episode maximum 1-step drop (pp)','Mean episode maximum 5-step drop (pp)','Drops >=10 pp per episode')):
            for i,group in enumerate(GROUPS):
                a=np.array([r[key]*scale for r in rows if r['group']==group]);mean=a.mean()
                lo,hi=np.quantile(a[draws].mean(1),[.025,.975])
                ax.bar(i,mean,color=COLORS[group]);ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],color='black',capsize=3)
            ax.set_xticks(range(4),['Base','Disc.','Rob.','Both']);ax.set(ylabel=label)
        save_figure(fig,figdir,f'coverage_drops_{kind}')
    result=dict(results=all_results,primary='last',thresholds_pp=[5,10,20],
        uncertainty='2000 paired validation-scene bootstrap draws; seed 950001; not training-seed CI')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# Baseline 1M：断联与鲁棒性奖励的覆盖骤降对照','',
        '主要比较相同1M预算last；best按各组自身训练奖励选择，作为补充。跨组统一任务指标为覆盖＋0.1×归一化吞吐，不比较不同定义训练回报的数值高低。',
        '32场景10000–10031、采样种子880001、批次16。单训练种子；区间是场景区间，非训练种子置信区间。保留测试集未使用。','',
        '骤降从动作后t1–t400计算，不含reset到第一步；1步、5步均指端点覆盖差，单位百分点。指标越低越好需同时检查平均覆盖、后100步覆盖与吞吐，防止以低覆盖静止来减少骤降。','']
    for kind,data in all_results.items():
        lines += [f'## {kind}','', '| 组别 | 覆盖 | 后100步覆盖 | 吞吐Mbps | 最大单步跌幅均值 | 最大5步跌幅均值 | 每回合≥10pp次数 |','|---|---:|---:|---:|---:|---:|---:|']
        for group,s in data['summary'].items():
            lines.append(f"| {NAMES[group]} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['throughput_bps']/1e6:.2f} | {100*s['max_drop_1']:.2f}pp | {100*s['max_drop_5']:.2f}pp | {s['drop_events_10pp']:.2f} |")
        lines += ['',f'- [覆盖与服务曲线](figures/coverage_and_service_{kind}.png)',f'- [覆盖骤降](figures/coverage_drops_{kind}.png)',f'- [奖励分量](figures/reward_components_{kind}.png)','']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),checkpoints=args.checkpoints)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=4,type=Path,default=[Path(BASELINE_RUN),*[Path(run_path(g)) for g in REWARD_ABLATIONS]])
    p.add_argument('--output',type=Path,default=Path('analysis/mappo_capacity10_stability_rewards_1m_seed0'))
    p.add_argument('--checkpoints',nargs='+',choices=['last','best'],default=['last','best'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rebuild',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs=dict(zip(GROUPS,a.runs))
    if not a.rebuild:evaluate_all(a,runs)
    rebuild(a,runs)
