"""Frozen shared/independent actor comparison; no training or reserved tests."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest, log_plots
from report_mappo_history import per_scene
from report_mappo_training import topology
from uav_research.algorithms import make_learner
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import COLORS, plt, save_figure, style, read_csv
from uav_research.runtime import configure_runtime, seed_all, isolated_rng, atomic_json, load_checkpoint


MODES=['shared','independent']
BASE='continuous_mappo_coverage_{}_central_history_100k'
DEFAULT_RUNS=[f'runs/{BASE.format(mode)}_seed0' for mode in MODES]
DEFAULT_OUTPUT='analysis/mappo_actor_sharing_coverage_100k_seed0'


def training_logs(run, ancestors=()):
    """Recover only the inherited prefix when a run resumes in a new directory."""
    run=Path(run)
    if run.resolve() in ancestors:
        raise ValueError('Cyclic resume lineage')
    logs={name:[] for name in ['metrics.csv','validation.csv']}
    sources=[];lineage=[]
    sessions=run/'sessions.json'
    if sessions.exists():
        sources.append(sessions)
        for session in json.loads(sessions.read_text()):
            resume=session.get('resume_from')
            if not resume:continue
            checkpoint=Path(resume);parent=checkpoint.parent
            if parent.resolve()==run.resolve():continue
            restored=load_checkpoint(checkpoint,'cpu',mmap=True)
            cutoff=restored['env_steps'];del restored
            inherited,files,history=training_logs(parent,(*ancestors,run.resolve()))
            for name,rows in inherited.items():
                logs[name]=[r for r in rows if int(r['env_steps'])<=cutoff]
            sources.extend(files)
            sources.extend([checkpoint/'metadata.pt',parent/'config.yaml'])
            lineage=history+[dict(run=str(parent),inherited_through_step=cutoff)]
    for name in logs:
        path=run/name;sources.append(path)
        combined={int(r['env_steps']):r for r in logs[name]}
        combined.update({int(r['env_steps']):r for r in read_csv(path)})
        logs[name]=[combined[step] for step in sorted(combined)]
    return logs,list(dict.fromkeys(sources)),lineage


def validate_configs(configs):
    if len(configs)!=2 or [c.algo.actor_sharing for c in configs]!=MODES:
        raise ValueError('Expected shared baseline followed by independent experiment')
    a,b=[copy.deepcopy(c) for c in configs];b.algo.actor_sharing='shared'
    if a.as_dict()!=b.as_dict():raise ValueError('Only actor_sharing may differ in this comparison')
    if not (a.algo.name=='mappo' and a.env.action_space=='continuous'
            and a.algo.mappo_state_dependent_std and a.algo.critic_parameterization=='time_factor'
            and a.env.observation_history=='one_step' and a.env.observation_coordinates=='relative'
            and a.env.user_initialization=='uniform_interior' and a.env.reward_primary=='coverage'
            and all(getattr(a.env,k)==0 for k in ['throughput_weight','robustness_weight',
                                                'disconnection_weight','boundary_penalty_weight'])
            and a.run.evaluation_policy=='sample' and a.run.best_metric=='task_return'):
        raise ValueError('Requires the paired central-UE, history, pure-coverage continuous MAPPO setup')


def scene_rows(files):
    repetitions=[];seeds=None
    for path in files:
        with np.load(path) as d:
            if seeds is None:seeds=d['scene_seeds'].copy()
            np.testing.assert_array_equal(seeds,d['scene_seeds'])
            stats=per_scene(d)
            cov=d['diagnostics'][...,list(d['diagnostic_names']).index('coverage')]
            w=min(20,cov.shape[1])
            smooth=np.stack([np.convolve(row,np.ones(w)/w,mode='valid') for row in cov])
            stats['peak_step_rolling20']=smooth.argmax(1)+w
            stats['peak_coverage_rolling20']=smooth.max(1)
            for i in range(d['agent_diagnostics'].shape[2]):
                stats[f'agent_{i}_coverage']=d['agent_diagnostics'][:,:,i,0].mean(1)
                stats[f'agent_{i}_command_m']=d['agent_diagnostics'][:,:,i,1].mean(1)
                stats[f'agent_{i}_actual_m']=d['agent_diagnostics'][:,:,i,2].mean(1)
                for j,name in enumerate(d['moment_names']):stats[f'agent_{i}_{name}']=d['moments'][:,:,i,j].mean(1)
            for name,key in [('common_mean_spread','common_means'),('common_std_spread','common_stds')]:
                x=d[key];stats[name]=np.square(x-x.mean(2,keepdims=True)).sum(-1).mean((1,2))
            repetitions.append(stats)
    stats={k:np.mean([r[k] for r in repetitions],axis=0) for k in repetitions[0]}
    return [dict(scene_seed=int(seed),**{k:float(v[i]) for k,v in stats.items()}) for i,seed in enumerate(seeds)]


def build(runs,output,samples=64,action_seeds=(880001,880002,880003)):
    paths=list(map(Path,runs));configs=[load_config(p/'config.yaml') for p in paths];validate_configs(configs)
    action_seeds=list(action_seeds)
    if not action_seeds or action_seeds[0]!=configs[0].run.evaluation_action_seed:
        raise ValueError('First action seed must reproduce the logged validation protocol')
    if set(configs[0].run.validation_seeds)&set(configs[0].run.test_seeds):raise ValueError('Overlapping scenes')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=[p/name for p in paths for name in ['config.yaml','last.pt','best.pt','manifest.json',
                                               'metrics.csv','validation.csv','best_selection.json']]
    source+=list(Path('src/uav_research').glob('*.py'))
    source += [Path(__file__),Path('scripts/report_mappo_history.py'),Path('scripts/report_mappo_bridge.py')]
    histories=[training_logs(p) for p in paths]
    for _,files,_ in histories:source.extend(files)
    source=list(dict.fromkeys(source))
    hashes={str(p):digest(p) for p in source};groups={}
    for mode,run,cfg,(logs,_,lineage) in zip(MODES,paths,configs,histories):
        configure_runtime(cfg)
        manifest=json.loads((run/'manifest.json').read_text());selection=json.loads((run/'best_selection.json').read_text())
        if manifest['status']!='completed' or manifest['env_steps']!=cfg.run.total_steps:
            raise ValueError('Report requires a completed declared training budget')
        group=dict(run=str(run),config=cfg.as_dict(),manifest=manifest,selection=selection,
                   logs=logs,resume_lineage=lineage,checkpoints={})
        for ck in ['initial','best','last']:
            if ck=='initial':
                with isolated_rng():seed_all(cfg.run.seed);policy=make_learner(cfg,'cpu')
                step=0
            else:
                _,policy=policy_from_checkpoint(run/f'{ck}.pt','cpu')
                step=policy.provenance['checkpoint_env_steps']
            policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
            before=actor_digest(policy.actor);files=[]
            for seed in action_seeds[:1] if ck=='initial' else action_seeds:
                audit_cfg=copy.deepcopy(cfg);audit_cfg.run.evaluation_action_seed=seed
                target=out/mode/ck/f'seed_{seed}'
                data,critic=audit_policy(audit_cfg,policy,target,samples,per_agent=True,
                    deployment_times=tuple(t for t in [150,200] if t<cfg.env.horizon),
                    trace_seeds=tuple(cfg.run.validation_seeds[:2]) if seed==action_seeds[0] else ())
                np.testing.assert_allclose(data['rewards'],data['diagnostics'][...,0],atol=1e-12)
                np.testing.assert_allclose(data['agent_diagnostics'][...,0].sum(-1),data['rewards'],atol=1e-14)
                files.append(str(target/'audit.npz'))
                if seed==action_seeds[0]:
                    logged=[r for r in group['logs']['validation.csv'] if int(r['env_steps'])==step]
                    if not logged:raise ValueError('Checkpoint has no corresponding validation log')
                    np.testing.assert_allclose(data['rewards'].mean(),float(logged[-1]['coverage']),atol=1e-8,rtol=1e-8)
                    np.testing.assert_allclose(data['rewards'].sum(1).mean(),float(logged[-1]['return']),atol=1e-8,rtol=1e-8)
                    if ck=='best':
                        np.testing.assert_allclose(data['rewards'].sum(1).mean(),selection['selection_score'],atol=1e-8,rtol=1e-8)
                        assert step==selection['env_steps']
                print(f'{mode}/{ck}/{seed}: {data["rewards"].mean():.4%} coverage, step {step}',flush=True)
            assert actor_digest(policy.actor)==before and all(p.grad is None for p in policy.actor.parameters())
            group['checkpoints'][ck]=dict(step=step,files=files,actor_sha256=before)
        groups[mode]=group
    assert hashes=={str(p):digest(p) for p in source}
    comparison=dict(groups=groups,action_seeds=action_seeds,source_sha256=hashes,training_seed_count=1,
                    samples=samples,elapsed_seconds=time.perf_counter()-started)
    atomic_json(out/'comparison.json',comparison);publish(out,comparison)
    return comparison


def publish(out,comparison):
    out=Path(out);groups=comparison['groups'];style();figures=out/'figures'
    log_plots(groups,figures)
    rows=[];differences=[];summary=[]
    for ck in ['initial','best','last']:
        paired={}
        for mode,g in groups.items():
            values=scene_rows(g['checkpoints'][ck]['files']);paired[mode]=values
            rows.extend([dict(group=mode,checkpoint=ck,checkpoint_step=g['checkpoints'][ck]['step'],**v) for v in values])
            means={k:float(np.mean([v[k] for v in values])) for k in values[0] if k!='scene_seed'}
            means.update(group=mode,checkpoint=ck,checkpoint_step=g['checkpoints'][ck]['step'])
            for prefix in ['all','first100','middle100','last100','last20','last1']:
                key=f'{prefix}_value_mse'
                if key in means:means[f'{prefix}_value_rmse']=float(np.sqrt(means[key]))
            summary.append(means)
        assert [r['scene_seed'] for r in paired['shared']]==[r['scene_seed'] for r in paired['independent']]
        for key in ['coverage','last100_coverage','final_coverage','maintenance','steps51_150_coverage',
                    'train_return','peak_step_rolling20','peak_coverage_rolling20',
                    'common_mean_spread','common_std_spread','all_disconnection_ratio',
                    'all_actual_speed_mean','all_boundary_reflection_fraction','all_uav_close_neighbor_fraction']:
            delta=np.array([r[key] for r in paired['independent']])-np.array([r[key] for r in paired['shared']])
            differences.append(dict(checkpoint=ck,comparison='independent-shared',metric=key,**interval(delta)))
        fig,axes=plt.subplots(2,3,figsize=(13,7));vfig,vaxes=plt.subplots(1,2,figsize=(10,3.5))
        for mode,g in groups.items():
            datasets=[]
            for path in g['checkpoints'][ck]['files']:
                with np.load(path) as d:datasets.append({key:d[key].copy() for key in d.files})
            diag=np.mean([d['diagnostics'] for d in datasets],axis=(0,1));names=list(datasets[0]['diagnostic_names'])
            for ax,key in zip(axes.flat,['coverage','discovery_fraction','disconnection_ratio','actual_speed_mean',
                                         'boundary_within_150_fraction','uav_close_neighbor_fraction']):
                ax.plot(np.arange(1,len(diag)+1),diag[:,names.index(key)],label=mode)
                ax.set(title=key,xlabel='Episode step');ax.legend(fontsize=7)
            pred=np.mean([d['values'][...,0] for d in datasets],axis=(0,1))
            target=np.mean([d['value_labels'] for d in datasets],axis=(0,1));h=datasets[0]['horizon_factors']
            for ax,scale in zip(vaxes,[1,h]):
                ax.plot(pred/scale,label=f'{mode}: V');ax.plot(target/scale,'--',label=f'{mode}: MC')
                ax.set(xlabel='State time before action',ylabel='Cumulative / average coverage');ax.legend(fontsize=7)
            if ck!='initial':
                cfg=config_from_dict(g['config'])
                for scene in cfg.run.validation_seeds[:2]:
                    trace=Path(g['checkpoints'][ck]['files'][0]).parent/f'trace_{scene}.npz'
                    frame_steps=sorted({t for t in [0,150,200,cfg.env.horizon] if t<=cfg.env.horizon})
                    topology(trace,out/mode/'figures',ck,g['checkpoints'][ck]['step'],cfg,frame_steps=frame_steps)
                agent_fig,agent_axes=plt.subplots(2,3,figsize=(13,7))
                agent=np.mean([d['agent_diagnostics'] for d in datasets],axis=(0,1,2))
                moments=np.mean([d['moments'] for d in datasets],axis=(0,1,2))
                for ax,y,title in zip(agent_axes.flat,[agent[:,0],agent[:,1],agent[:,2],moments[:,0],moments[:,1],moments[:,3]],
                    ['Coverage contribution','Command distance (m)','Actual distance (m)','Conditional displacement variance (m²)',
                     'Near-hover probability','Differential entropy (nats)']):
                    ax.bar(np.arange(len(y)),y);ax.set(title=title,xlabel='Fixed UAV / actor index')
                save_figure(agent_fig,figures,f'{mode}_{ck}_per_agent')
        save_figure(fig,figures,f'deployment_{ck}');save_figure(vfig,figures,f'critic_{ck}')
    write_csv(out/'per_scene.csv',rows);write_csv(out/'summary.csv',summary);write_csv(out/'paired_differences.csv',differences)
    # Per-agent optimizer diagnostics are logged independently of team critic.
    for mode,g in groups.items():
        logs=list({r['env_steps']:r for r in g['logs']['metrics.csv']}.values())
        fig,axes=plt.subplots(2,4,figsize=(16,7))
        for ax,key in zip(axes.flat,['coverage','actor_loss','entropy','ppo_approx_kl','ppo_clip_fraction',
                                     'actor_grad_norm','encoder_grad_norm','history_input_grad_norm']):
            for i in range(g['config']['env']['num_uavs']):
                values=[float(r.get(f'agent_{i}_{key}') or np.nan) for r in logs]
                if np.isfinite(values).any():ax.plot([int(r['env_steps']) for r in logs],values,lw=.8,alpha=.7)
            ax.set(title=key,xlabel='Joint transitions')
        save_figure(fig,figures,f'{mode}_per_agent_training')
    scope=comparison.get('comparison_scope','equal_budget')
    selection_note=('本报告比较共享1M与独立1.2M的last，独立组多用20%环境样本，不是同预算对照；best作为补充。'
                    if scope=='shared_1m_independent_1p2m' else '主要比较同预算last；')
    report=['# 共享／独立actor纯覆盖PPO对照','',
        '两组使用团队覆盖奖励、团队GAE和单输出集中式时间因子critic。独立组不是标准IPPO；每个UAV拥有独立编码器/策略/Adam，编号固定。',
        selection_note+'best按全回合覆盖回报选择，包含step0。initial只使用原选模动作种子，best/last先平均动作重复，再按场景配对bootstrap。','',
        '|组别|检查点|步数|平均覆盖|前期51–150|后100步|终态|维持差|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        report.append(f'|{row["group"]}|{row["checkpoint"]}|{row["checkpoint_step"]}|{row["coverage"]:.2%}|{row["steps51_150_coverage"]:.2%}|{row["last100_coverage"]:.2%}|{row["final_coverage"]:.2%}|{row["maintenance"]:+.2%}|')
    report+=['','![部署维持](figures/deployment_last.png)','![价值预测与MC](figures/critic_last.png)','',
             '|组别|Actor参数量|Critic参数量|训练总耗时秒|峰值RSS MiB|','|---|---:|---:|---:|---:|']
    for mode,g in groups.items():
        m=g['manifest'];report.append(f'|{mode}|{m["actor_parameters"]}|{m["critic_parameters"]}|{m["elapsed_seconds"]:.2f}|{m["peak_rss_mib"]:.1f}|')
    report+=['','参数容量和每个网络接收的数据量也发生变化，因此即便独立组获胜，也不能单独证明共享梯度冲突是唯一根因。共享组的每UAV梯度栏为空，整体共享梯度在常规日志中；独立组各自记录梯度。',
        '学习曲线的训练覆盖是最后采集步的环境均值，不是完整训练回合均值。critic两组使用各自on-policy轨迹，MC不包含actor熵。历史actor与物理状态critic的既有限制保留。',
        'common_mean_spread/common_std_spread在完全相同局部输入上比较各网络的latent高斯均值/标准差，只有表示分化，不等于有效分工。条件位移方差每20步使用隔离RNG的多次采样（实际数量见comparison.json）。',
        '场景区间使用2000次bootstrap（950001），不是训练种子区间；保留测试集未使用。原始逐步审计、每UAV贡献/动作/条件方差、同输入分布参数、拓扑与逐场景表均保存，可通过--rebuild重建。','']
    (out/'report.md').write_text('\n'.join(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs=2,default=DEFAULT_RUNS)
    parser.add_argument('--output',default=DEFAULT_OUTPUT);parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--rebuild',action='store_true');args=parser.parse_args()
    if args.rebuild:
        out=Path(args.output);publish(out,json.loads((out/'comparison.json').read_text()))
    else:build(args.runs,args.output,args.samples)
