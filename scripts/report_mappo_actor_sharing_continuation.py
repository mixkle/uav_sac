"""Validate independent 1.2M, reusing frozen shared/independent 1M audits."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_actor_sharing import publish, scene_rows, training_logs, validate_configs
from report_mappo_bridge import digest
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import config_from_dict, load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, load_checkpoint, source_digest


DEFAULT_RUN='runs/continuous_mappo_coverage_independent_central_history_100k_seed0'
DEFAULT_BASELINE='analysis/mappo_actor_sharing_coverage_1m_seed0'
DEFAULT_OUTPUT='analysis/mappo_actor_sharing_independent_1p2m_vs_shared_1m_seed0'


def same_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b


def build(run,out,baseline):
    run,out,baseline=map(Path,[run,out,baseline])
    previous=json.loads((baseline/'comparison.json').read_text());cfg=load_config(run/'config.yaml')
    reference=config_from_dict(previous['groups']['shared']['config'])
    checked=copy.deepcopy(cfg);checked.run.total_steps=reference.run.total_steps
    validate_configs([reference,checked])
    assert reference.run.total_steps==1000000 and cfg.run.total_steps==1200000
    action_seeds=previous['action_seeds'];assert action_seeds==[880001,880002,880003]
    configure_runtime(cfg);manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['env_steps']!=1200000:
        raise ValueError('Requires completed independent 1.2M run')
    if json.loads((run/'metadata.json').read_text())['source_hash']!=source_digest():
        raise ValueError('Training implementation differs from current implementation')
    logs,history_sources,lineage=training_logs(run)
    sources=[baseline/'comparison.json',Path(__file__),Path('scripts/report_mappo_actor_sharing.py')]
    sources+=list(Path('src/uav_research').glob('*.py'))+history_sources
    sources+=[run/name for name in ['config.yaml','manifest.json','metadata.json','best_selection.json',
                                   'best.pt','last.pt']]
    for mode in ['shared','independent']:
        for checkpoint in previous['groups'][mode]['checkpoints'].values():
            for file in checkpoint['files']:
                sources.append(Path(file));sources+=list(Path(file).parent.glob('trace_*.npz'))
    hashes={str(p):digest(p) for p in dict.fromkeys(sources)}
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    selection=json.loads((run/'best_selection.json').read_text())
    best,last=[load_checkpoint(run/f'{name}.pt','cpu',mmap=True) for name in ['best','last']]
    identical=(best['env_steps']==last['env_steps'] and same_tree(best['learner'],last['learner']))
    verification=dict(last_step=last['env_steps'],best_step=best['env_steps'],
        ppo_updates=last['learner']['updates'],best_last_learner_identical=identical,
        training_code_matches=True,source_checkpoint_count=2)
    step_counts=lambda opt:sorted({int(v['step']) for v in opt['state'].values() if 'step' in v})
    verification['actor_optimizer_steps']=[step_counts(o) for o in last['learner']['actor_optimizers']]
    verification['critic_optimizer_steps']=step_counts(last['learner']['value_opt'])
    assert verification['ppo_updates']==587
    assert all(x==[23480] for x in verification['actor_optimizer_steps'])
    assert verification['critic_optimizer_steps']==[23480]
    del best,last
    independent=dict(run=str(run),config=cfg.as_dict(),manifest=manifest,selection=selection,
        logs=logs,resume_lineage=lineage,checkpoints=dict(initial=copy.deepcopy(
            previous['groups']['independent']['checkpoints']['initial'])))
    for ck in ['last','best']:
        if ck=='best' and identical:
            independent['checkpoints'][ck]=copy.deepcopy(independent['checkpoints']['last'])
            independent['checkpoints'][ck]['reused_identical_checkpoint']='last'
            continue
        _,policy=policy_from_checkpoint(run/f'{ck}.pt','cpu');step=policy.provenance['checkpoint_env_steps']
        policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
        before=actor_digest(policy.actor);files=[]
        for seed in action_seeds:
            audit_cfg=copy.deepcopy(cfg);audit_cfg.run.evaluation_action_seed=seed
            target=out/'independent'/ck/f'seed_{seed}'
            data,_=audit_policy(audit_cfg,policy,target,previous['samples'],per_agent=True,
                deployment_times=(150,200),trace_seeds=(10000,10001) if seed==action_seeds[0] else ())
            np.testing.assert_allclose(data['rewards'],data['diagnostics'][...,0],atol=1e-12)
            np.testing.assert_allclose(data['agent_diagnostics'][...,0].sum(-1),data['rewards'],atol=1e-14)
            if seed==action_seeds[0]:
                logged=[r for r in logs['validation.csv'] if int(r['env_steps'])==step][-1]
                np.testing.assert_allclose(data['rewards'].mean(),float(logged['coverage']),atol=1e-8,rtol=1e-8)
                if step==selection['env_steps']:
                    np.testing.assert_allclose(data['rewards'].sum(1).mean(),selection['selection_score'],atol=1e-8)
            files.append(str(target/'audit.npz'))
            print(f'independent/{ck}/{seed}: {data["rewards"].mean():.4%}, step {step}',flush=True)
        assert actor_digest(policy.actor)==before and all(p.grad is None for p in policy.actor.parameters())
        independent['checkpoints'][ck]=dict(step=step,files=files,actor_sha256=before)
    assert all(digest(Path(p))==h for p,h in hashes.items())
    comparison=dict(groups=dict(shared=copy.deepcopy(previous['groups']['shared']),independent=independent),
        baseline_comparison=str(baseline/'comparison.json'),comparison_scope='shared_1m_independent_1p2m',
        action_seeds=action_seeds,samples=previous['samples'],source_sha256=hashes,
        training_seed_count=1,elapsed_seconds=time.perf_counter()-started)
    atomic_json(out/'comparison.json',comparison);atomic_json(out/'verification.json',verification)
    publish(out,comparison)
    return comparison


def explain(out):
    out=Path(out);comparison=json.loads((out/'comparison.json').read_text())
    previous=json.loads(Path(comparison['baseline_comparison']).read_text())
    variants=[('shared_1m',comparison['groups']['shared']),
              ('independent_1m',previous['groups']['independent']),
              ('independent_1p2m',comparison['groups']['independent'])]
    summary=[];scenes={};datasets={};stage_rows=[];critic_rows=[];stability=[];sources={}
    for label,group in variants:
        files=group['checkpoints']['last']['files'];scenes[label]=scene_rows(files);data=[]
        for file in files:
            sources[file]=digest(Path(file))
            with np.load(file) as d:data.append({k:d[k].copy() for k in d.files})
        datasets[label]=data
        means={k:float(np.mean([r[k] for r in scenes[label]])) for k in scenes[label][0] if k!='scene_seed'}
        summary.append(dict(group=label,training_budget=group['manifest']['env_steps'],**means))
        a=np.stack([d['diagnostics'] for d in data]);names=list(data[0]['diagnostic_names'])
        np.testing.assert_allclose(a[...,0],a[...,names.index('discovery_fraction')]-sum(
            a[...,names.index(k)] for k in ['downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users'])/150,atol=1e-12)
        for stage,sl in [('all',slice(None)),('early51_150',slice(50,150)),('last100',slice(-100,None))]:
            stage_rows.append(dict(group=label,stage=stage,**{n:float(a[:,:,sl,i].mean()) for i,n in enumerate(names)}))
        v=np.stack([d['values'][...,0] for d in data]);g=np.stack([d['value_labels'] for d in data]);h=data[0]['horizon_factors']
        corr=float(np.corrcoef((v-v.mean(1,keepdims=True)).ravel(),(g-g.mean(1,keepdims=True)).ravel())[0,1])
        for stage,sl in [('all',slice(None)),('last100',slice(-100,None)),('last20',slice(-20,None)),('last1',slice(-1,None))]:
            e=(v-g)[:,:,sl];std=g[:,:,sl].std();rmse=float(np.sqrt((e**2).mean()))
            critic_rows.append(dict(group=label,stage=stage,rmse=rmse,mae=float(np.abs(e).mean()),
                bias=float(e.mean()),normalized_rmse=rmse/std if std else np.nan,
                average_rmse=float(np.sqrt(((e/h[sl])**2).mean())),time_demeaned_correlation=corr))
        cov=np.stack([d['rewards'] for d in data]);drawdowns=[];gaps=[]
        for trace in cov.reshape(-1,400):
            smooth=np.convolve(trace,np.ones(20)/20,'valid')
            drawdowns.append((np.maximum.accumulate(smooth)-smooth).max());gaps.append(smooth.max()-trace[-100:].mean())
        curve=cov.mean((0,1));smooth=np.convolve(curve,np.ones(20)/20,'valid')
        stability.append(dict(group=label,mean_max_drawdown=float(np.mean(drawdowns)),
            drawdown_over20pp_fraction=float(np.mean(np.array(drawdowns)>.2)),
            individual_peak_to_last100_gap=float(np.mean(gaps)),mean_curve_peak=float(smooth.max()),
            mean_curve_peak_end=int(smooth.argmax()+20),mean_curve_peak_to_last100_gap=float(smooth.max()-curve[-100:].mean())))
    changes=[]
    metrics=['coverage','steps51_150_coverage','last100_coverage','final_coverage','maintenance',
        'all_discovery_fraction','all_disconnection_ratio','all_boundary_within_150_fraction',
        'all_actual_speed_mean','command_variance_m2']
    for base in ['independent_1m','shared_1m']:
        old,new=scenes[base],scenes['independent_1p2m']
        assert [r['scene_seed'] for r in old]==[r['scene_seed'] for r in new]
        for key in metrics:
            delta=np.array([r[key] for r in new])-np.array([r[key] for r in old])
            changes.append(dict(comparison=f'independent_1p2m-{base}',metric=key,**interval(delta)))
    for name,rows in [('budget_overview.csv',summary),('continuation_paired_differences.csv',changes),
                      ('stage_diagnosis.csv',stage_rows),('critic_diagnosis.csv',critic_rows),
                      ('trajectory_stability_summary.csv',stability)]:write_csv(out/name,rows)
    style();fig,axes=plt.subplots(2,3,figsize=(14,8))
    fields=['coverage','discovery_fraction','disconnection_ratio','actual_speed_mean',
            'boundary_within_150_fraction','uav_close_neighbor_fraction']
    for color,(label,_) in zip(COLORS,variants):
        data=datasets[label];names=list(data[0]['diagnostic_names']);a=np.mean([d['diagnostics'] for d in data],axis=(0,1))
        for ax,key in zip(axes.flat,fields):
            scale=1 if key=='actual_speed_mean' else 100
            ax.plot(np.arange(1,401),a[:,names.index(key)]*scale,label=label,color=color)
            ax.set(title=key,xlabel='Episode step',ylabel='m/s' if scale==1 else '%');ax.legend(fontsize=7)
    save_figure(fig,out/'figures','continuation_deployment')
    lines=['# 独立1.2M验证：与独立1M、共享1M比较','',
        '物理环境、奖励、观测、优化参数和验证协议不变。本次只重新验证独立1.2M，复用已保存的共享/独立1M审计。',
        '跨组比较的预算为独立1.2M、共享1M，独立多使用20%环境样本，不能作为同预算优劣证明。',
        '仍为32个场景、三套动作种子，先平均每场景动作重复，再作2000次配对场景bootstrap（950001）；不是训练种子区间。保留测试集未使用。','',
        '|last模型|全回合覆盖|51–150步|后100步|终态|维持差|','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["group"]}|'+ '|'.join(f'{r[k]*100:+.2f} pp' if k=='maintenance' else f'{r[k]:.2%}'
            for k in ['coverage','steps51_150_coverage','last100_coverage','final_coverage','maintenance'])+'|')
    lines+=['','|差值|指标|均值及95%场景区间（百分点）|','|---|---|---:|']
    for r in changes:
        if r['metric'] in ['coverage','last100_coverage','final_coverage','maintenance']:
            lines.append(f'|{r["comparison"]}|{r["metric"]}|{r["mean"]*100:+.2f} [{r["low"]*100:+.2f}, {r["high"]*100:+.2f}]|')
    lines+=['','![续训前后及共享基准](figures/continuation_deployment.png)','',
        '独立1.2M的best与last是否完全相同见verification.json；相同时只采集一套三次重复，两个检查点引用同一份数据。共享best属于共享1M运行内选出的历史最优模型，并不要求best本身恰好位于1M。','',
        '|last模型|速度m/s|条件位移方差m²|近悬停概率|150m内靠边|','|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["group"]}|{r["all_actual_speed_mean"]:.3f}|{r["command_variance_m2"]:.3f}|{r["near_hover_probability"]:.3%}|{r["all_boundary_within_150_fraction"]:.2%}|')
    lines+=['','|last模型|20步平滑平均曲线峰值|峰值减后100步|单轨迹平均最大回落|','|---|---:|---:|---:|']
    for r in stability:
        lines.append(f'|{r["group"]}|{r["mean_curve_peak"]:.2%}|{r["mean_curve_peak_to_last100_gap"]*100:.2f} pp|{r["mean_max_drawdown"]*100:.2f} pp|')
    lines+=['','最大回落是在单条轨迹的20步平滑曲线上，计算任意先前峰值到后续低点的最大差值，即使后来恢复也计入。平均曲线变平不等于逐场景部署稳定。','',
        '|last critic|全程MC RMSE|后100步RMSE|末步RMSE|去时间均值相关性|','|---|---:|---:|---:|---:|']
    for label,_ in variants:
        get=lambda stage:next(r for r in critic_rows if r['group']==label and r['stage']==stage)
        lines.append(f'|{label}|{get("all")["rmse"]:.3f}|{get("last100")["rmse"]:.3f}|{get("last1")["rmse"]:.3f}|{get("all")["time_demeaned_correlation"]:.3f}|')
    lines+=['','critic标签为完整纯覆盖MC，不含actor熵；各模型使用自己的on-policy轨迹，不能替代同状态动作价值差诊断。',
        '训练/验证奖励、loss、每UAV诊断、best/last拓扑及原始数据索引见report.md与comparison.json。',
        '图表可通过本脚本--rebuild重建，不重新采样。','']
    (out/'interpretation.md').write_text('\n'.join(lines))
    sources[comparison['baseline_comparison']]=digest(Path(comparison['baseline_comparison']))
    for p in [out/'comparison.json',Path(__file__)]:sources[str(p)]=digest(p)
    assert all(digest(Path(p))==h for p,h in sources.items())
    atomic_json(out/'analysis_sources.json',sources)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=DEFAULT_RUN)
    p.add_argument('--baseline',default=DEFAULT_BASELINE);p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--rebuild',action='store_true');args=p.parse_args()
    if args.rebuild:
        out=Path(args.output);publish(out,json.loads((out/'comparison.json').read_text()))
    else:build(args.run,args.output,args.baseline)
    explain(args.output)
