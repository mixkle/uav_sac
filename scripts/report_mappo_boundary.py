"""Four-way M1 initialization / directional boundary-cost diagnostic (read-only)."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from uav_research.config import config_from_dict, load_config
from uav_research.plotting import plt, read_csv, save_figure, style, COLORS
from uav_research.runtime import atomic_json
from compare_online_time_factor import bootstrap_difference, write_csv
from report_mappo_bridge import audit_run, digest, SOURCE_FILES
from report_mappo_training import topology

LABELS = ['M1', 'M1-C', 'M1-B', 'M1-CB']
PREFIX = 'continuous_mappo_full_reward_time_factor_relative'
DEFAULT_RUNS = [f'runs/{PREFIX}{suffix}_100k_seed0' for suffix in ['', '_central', '_boundary', '_central_boundary']]
METRICS = ['task_return','return','coverage','last100_coverage','final_coverage',
           'discovery_fraction','disconnection_ratio','robustness','throughput_bps',
           'boundary_penalty_return','boundary_reflection_fraction','boundary_within_150_fraction']


def validate_configs(configs):
    base=configs['M1']
    if not (base.algo.name=='mappo' and base.algo.critic_parameterization=='time_factor'
            and base.algo.mappo_state_dependent_std and base.env.action_space=='continuous'
            and base.env.observation_coordinates=='relative'
            and base.env.user_initialization=='uniform' and base.env.boundary_penalty_weight==0
            and base.run.evaluation_policy=='sample'):
        raise ValueError('Baseline must be original relative-coordinate continuous M1')
    for label,cfg in configs.items():
        expected=copy.deepcopy(base)
        if label!='M1': expected.run.best_metric='task_return'
        if label in ['M1-C','M1-CB']:
            expected.env.user_initialization='uniform_interior';expected.env.user_initial_margin_fraction=1/7
        expected.env.boundary_penalty_weight=.05 if label in ['M1-B','M1-CB'] else 0.
        if expected.as_dict()!=cfg.as_dict():
            raise ValueError(f'{label} changed settings outside initialization/boundary/selection')


def paired_results(groups):
    result=[]
    comparisons=[('M1','M1-C'),('M1','M1-B'),('M1-C','M1-CB')]
    for ck in ['best','last']:
        episodes={k:g['checkpoints'][ck]['episodes'] for k,g in groups.items()}
        seeds=[r['scene_seed'] for r in episodes['M1']]
        assert all([r['scene_seed'] for r in rows]==seeds for rows in episodes.values())
        for key in METRICS:
            values={k:np.array([r[key] for r in rows]) for k,rows in episodes.items()}
            for left,right in comparisons:
                result.append(dict(checkpoint=ck,comparison=f'{right}-{left}',metric=key,
                                   **bootstrap_difference(values[left],values[right],key)))
            left=values['M1-B']-values['M1'];right=values['M1-CB']-values['M1-C']
            result.append(dict(checkpoint=ck,comparison='(M1-CB - M1-C) - (M1-B - M1)',metric=key,
                               **bootstrap_difference(left,right,key)))
        # Scenario-paired differences in MC prediction error; each policy is on-policy.
        errors={}
        for label in LABELS:
            item=groups[label]['checkpoints'][ck]
            errors[label]=np.array(item['scene_squared_value_errors'])
        for left,right in comparisons:
            # Stored scene MSE, re-sqrt for the existing scene bootstrap RMSE routine.
            result.append(dict(checkpoint=ck,comparison=f'{right}-{left}',metric='critic_rmse',
                               **bootstrap_difference(np.sqrt(errors[left])[:,None],np.sqrt(errors[right])[:,None],'rmse')))
    return result


def series_plot(groups,figures,fields,name,xkey='env_steps',log='validation.csv'):
    fig,axes=plt.subplots((len(fields)+2)//3,3,figsize=(13,3.5*((len(fields)+2)//3)),squeeze=False)
    for label,g in groups.items():
        rows=g['logs'][log];x=np.array([float(r[xkey]) for r in rows])/(1000 if xkey=='env_steps' else 60)
        for ax,key in zip(axes.flat,fields):
            fallback='return' if key=='task_return' else key
            y=[float(r.get(key,r.get(fallback,'')) or np.nan) for r in rows]
            ax.plot(x,y,label=label)
    for ax,key in zip(axes.flat,fields):
        ax.set(title=key,xlabel='Joint transitions (k)' if xkey=='env_steps' else 'Wall time (min)');ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(fields):]:ax.set_visible(False)
    save_figure(fig,figures,name)


def publish(out,comparison):
    out=Path(out);figures=out/'figures';style();groups=comparison['groups']
    series_plot(groups,figures,['coverage','discovery_fraction','disconnection_ratio','task_return','return','robustness'],'validation_progress')
    series_plot(groups,figures,['coverage','task_return','boundary_penalty_return'],'wall_time_progress',xkey='elapsed_seconds')
    series_plot(groups,figures,['actor_loss','value_loss','joint_entropy','ppo_approx_kl','ppo_clip_fraction','actor_grad_norm'],'loss_and_policy_diagnostics',log='metrics.csv')
    series_plot(groups,figures,['reward','task_reward','boundary_cost','boundary_penalty','boundary_within_150_fraction','boundary_reflection_fraction'],'training_reward_and_boundary',log='metrics.csv')
    progress=copy.deepcopy(groups)
    for g in progress.values():
        e=g['config']['env']
        for filename,rows in g['logs'].items():
            if filename not in ['metrics.csv','validation.csv']:continue
            for row in rows:
                coverage=float(row['coverage'])
                tau=(float(row['tau']) if row.get('tau') else float(row.get('throughput_bps',0))/(e['num_users']*e['rate_reference_bps']))
                row.update(coverage_contribution=coverage,throughput_contribution=e['throughput_weight']*tau,
                           robustness_contribution=e['robustness_weight']*float(row['robustness']),
                           disconnection_contribution=-e['disconnection_weight']*float(row['disconnection_ratio']))
                if filename=='metrics.csv':row['boundary_contribution']=-float(row.get('boundary_penalty',0) or 0)
                else:
                    row['boundary_contribution']=-float(row.get('boundary_penalty_return',0) or 0)/e['horizon']
    for log,name in [('metrics.csv','training'),('validation.csv','validation')]:
        series_plot(progress,figures,['coverage_contribution','throughput_contribution','robustness_contribution',
                                     'disconnection_contribution','boundary_contribution'],f'{name}_reward_components_progress',log=log)
    policy_rows=[];critic_rows=[];moment_rows=[]
    for ck in ['best','last']:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        bfig,baxes=plt.subplots(2,3,figsize=(13,7))
        cfig,caxes=plt.subplots(2,3,figsize=(13,7))
        rfig,raxes=plt.subplots(2,3,figsize=(13,7))
        mfig,maxes=plt.subplots(1,3,figsize=(13,3.6))
        fields=['coverage','discovery_fraction','disconnection_ratio','uav_nearest_neighbor_mean','uav_close_neighbor_fraction','actual_speed_mean']
        bfields=['boundary_within_50_fraction','boundary_within_100_fraction','boundary_within_150_fraction',
                 'boundary_outward_fraction','boundary_reflection_fraction','boundary_penalty']
        for color,(label,g) in zip(COLORS,groups.items()):
            item=g['checkpoints'][ck]
            with np.load(out/label/ck/'audit.npz') as d:
                names=list(d['diagnostic_names']);diag=d['diagnostics'];h=d['horizon_factors']
                pred=d['values'][...,0];mc=d['value_labels'];t=np.arange(len(h))
                for ax,key in zip(axes.flat,fields):ax.plot(t,diag[...,names.index(key)].mean(0),label=label,color=color)
                for ax,key in zip(baxes.flat,bfields):ax.plot(t,diag[...,names.index(key)].mean(0),label=label,color=color)
                for ax,y in zip(caxes.flat,[pred.mean(0),mc.mean(0),(pred/h).mean(0),(mc/h).mean(0),pred[:,-20:].mean(0),mc[:,-20:].mean(0)]):
                    ax.plot(np.arange(len(y)),y,label=label,color=color)
                cnames=list(d['contribution_names'])
                for j,name in enumerate(['coverage','throughput','robustness','disconnection','boundary']):
                    y=d['contributions'][...,cnames.index(name)].mean(0) if name in cnames else np.zeros(len(h))
                    raxes.flat[j].plot(t,y,label=label,color=color)
                raxes.flat[5].plot(t,d['rewards'].mean(0),label=label,color=color)
                maxes[0].hist(d['boundary_distances'].ravel(),bins=np.linspace(0,g['config']['env']['width']/2,40),histtype='step',density=True,label=label)
                for ax,key in zip(maxes[1:],['edge','interior']):
                    n=diag[...,names.index(key+'_users')].sum(0);s=diag[...,names.index(key+'_served_users')].sum(0)
                    ax.plot(t,np.divide(s,n,out=np.full_like(s,np.nan),where=n>0),label=label)
                stages={'all':slice(None),'last100':slice(-100,None),'last1':slice(-1,None)}
                for stage,sl in stages.items():
                    row=dict(group=label,checkpoint=ck,stage=stage,checkpoint_step=item['step'])
                    row.update({name:float(diag[:,sl,j].mean()) for j,name in enumerate(names)})
                    row.update(reward=float(d['rewards'][:,sl].mean()),task_reward=float(d['task_rewards'][:,sl].mean()))
                    row.update({str(name):float(d['movement'][:,sl,j].mean()) for j,name in enumerate(d['movement_names'])})
                    for region in ['edge','interior']:
                        n=diag[:,sl,names.index(region+'_users')].sum();s=diag[:,sl,names.index(region+'_served_users')].sum()
                        row[region+'_coverage']=float(s/n) if n else None
                    near=diag[:,sl,names.index('boundary_buffer_fraction')].sum();outward=diag[:,sl,names.index('boundary_outward_fraction')].sum()
                    row['outward_given_buffer']=float(outward/near) if near else None
                    policy_rows.append(row)
                for j,name in enumerate(d['moment_names']):
                    moment_rows.append(dict(group=label,checkpoint=ck,metric=str(name),mean=float(d['moments'][...,j].mean())))
            for scale in ['value','average']:
                for stage,m in item['critic'][scale]['stages'].items():critic_rows.append(dict(group=label,checkpoint=ck,scale=scale,stage=stage,**m))
            cfg=config_from_dict(g['config'])
            for scene in cfg.run.validation_seeds[:2]:
                topology(out/label/f'validation_{ck}'/f'trace_{scene}.npz',out/label/'figures',ck,item['step'],cfg,boundary_overlay=True)
        for axis_list,titles in [(axes.flat,fields),(baxes.flat,bfields),(caxes.flat,['Predicted V','MC G (training reward)','Predicted V/H','MC G/H','Predicted V: last 20','MC G: last 20']),
                                 (raxes.flat,['Coverage','Throughput','Robustness','Disconnection','Boundary cost contribution','Total training reward']),
                                 (maxes,['Boundary distance density','Edge UE coverage (<150 m)','Interior UE coverage'])]:
            for ax,title in zip(axis_list,titles):ax.set(title=title);ax.legend(fontsize=7)
        for f,name in [(fig,'deployment'),(bfig,'boundary'),(cfig,'critic'),(rfig,'reward_components'),(mfig,'distance_and_edge_service')]:save_figure(f,figures,f'{name}_{ck}')
    write_csv(out/'policy_metrics.csv',policy_rows);write_csv(out/'critic_metrics.csv',critic_rows)
    write_csv(out/'action_moments.csv',moment_rows);write_csv(out/'paired_differences.csv',comparison['paired'])
    af,aa=plt.subplots(1,3,figsize=(13,3.6))
    for label in LABELS:
        with np.load(out/label/'last/audit.npz') as d:
            names=list(d['moment_names'])
            for ax,key in zip(aa,['command_variance_m2','near_hover_probability','mean_command_displacement_m']):
                ax.plot(d['moment_times'],d['moments'][...,names.index(key)].mean((0,2)),label=label)
    for ax,key in zip(aa,['Conditional command variance (m²)','P(command <1 m)','Mean command displacement (m)']):
        ax.set(title=key,xlabel='Episode step');ax.legend(fontsize=7)
    save_figure(af,figures,'action_distribution_last')
    base=groups['M1']['config'];env=base['env'];center=groups['M1-C']['config']['env']
    density=env['num_users']/(env['width']*env['height']/1e6)
    ratio=1/(1-2*center['user_initial_margin_fraction'])**2
    lines=['# M1 边界惩罚与中心初始化对照','',
           f"配置：{env['width']:g}×{env['height']:g} m，{env['num_uavs']} UAV、{env['num_users']} UE，{env['horizon']}步回合，每组{base['run']['total_steps']}条联合转移。",
           ('本报告为小场景实现验收，不用于判断正式实验效果。' if base['run']['total_steps']!=100000 else '正式100k单种子对照。'),
           f'中心初始化只改变初始位置，UE仍运动。中心初始密度{density*ratio:.2f}人/km²，原场景约{density:.2f}人/km²，约{ratio:.2f}倍；与BS距离也变化。',
           '相同种子用于配对，不代表相同物理场景。best按原四项任务回报选择，last是主要同预算比较。',
           '边界惩罚使用动作前位置和原始指令；反射比例与靠边停留比例不同。向外比例按全部UAV步计，outward_given_buffer另按缓冲区内UAV步计。','',
           '|组别|step 0覆盖|last覆盖|覆盖增益(pp)|last后100覆盖|last终态覆盖|last任务回报|best步数|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in groups.items():
        initial=float(g['initial']['coverage']);last=g['checkpoints']['last']['summary']['mean']
        lines.append(f"|{label}|{initial:.2%}|{last['coverage']:.2%}|{100*(last['coverage']-initial):+.2f}|{last['last100_coverage']:.2%}|{last['final_coverage']:.2%}|{last['task_return']:.3f}|{g['checkpoints']['best']['step']}|")
    lines+=['','![训练验证](figures/validation_progress.png)','![Loss和策略](figures/loss_and_policy_diagnostics.png)',
            '![部署](figures/deployment_last.png)','![边界诊断](figures/boundary_last.png)',
            '![奖励分量](figures/reward_components_last.png)','![价值预测](figures/critic_last.png)',
            '![边缘服务](figures/distance_and_edge_service_last.png)','',
            '## Last 配对差值及交互作用','', '|比较|指标|差值|场景95%区间|','|---|---|---:|---|']
    for row in comparison['paired']:
        if row['checkpoint']=='last' and row['metric'] in ['task_return','coverage','last100_coverage','boundary_reflection_fraction','critic_rmse']:
            lines.append(f"|{row['comparison']}|{row['metric']}|{row['delta']:+.5f}|[{row['low']:+.5f}, {row['high']:+.5f}]|")
    lines+=['','区间来自配对场景bootstrap 2000次、seed950001，不是训练种子置信区间。best含验证选模影响。',
            'critic以各自on-policy完整训练奖励MC为标签，含负边界成本，不含actor熵正则；原任务MC单独保存在audit.npz。',
            '仅减少反射不足以说明任务改善；同时检查覆盖、后期维持、悬停、边缘服务及联网比例。',
            'step 0为各组自己的初始化分布；比较各组训练增益，不能将中心组的密度变化归因于奖励。','', '## 固定场景拓扑','']
    for label,g in groups.items():
        for ck in ['best','last']:
            for seed in g['config']['run']['validation_seeds'][:2]:lines.append(f'![{label} {ck} {seed}]({label}/figures/topology_{ck}_{seed}.png)')
    lines+=['','橙色浅带为边界缓冲区；绿色虚线为中心组UE初始化范围，后续不限制UE运动。',
            '来源配置、训练/验证日志快照、耗时与峰值内存、源码和检查点哈希见comparison.json。图表可从本目录保存数据重建。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def build(runs,output,samples=64):
    out=Path(output)
    if (out/'comparison.json').exists():raise ValueError('Report exists; use --rebuild-only or a new output')
    if len(runs)!=4:raise ValueError('Expected M1 M1-C M1-B M1-CB in that order')
    runs=dict(zip(LABELS,map(Path,runs)))
    configs={k:load_config(p/'config.yaml') for k,p in runs.items()};validate_configs(configs)
    files=[p/name for p in runs.values() for name in SOURCE_FILES]
    files += [p for p in Path('src/uav_research').glob('*.py')]+[Path(__file__),Path('scripts/report_mappo_bridge.py'),Path('scripts/report_mappo_training.py')]
    hashes={str(p):digest(p) for p in files}
    groups={k:audit_run(k,p,out,samples) for k,p in runs.items()}
    for label,g in groups.items():
        for ck,item in g['checkpoints'].items():
            with np.load(out/label/ck/'audit.npz') as d:
                item['scene_squared_value_errors']=((d['values'][...,0]-d['value_labels'])**2).mean(1).tolist()
    comparison=dict(groups=groups,paired=paired_results(groups),source_sha256=hashes,
                    training_seed_count=1,bootstrap_repeats=2000,bootstrap_seed=950001,moment_samples=samples)
    if not all(digest(p)==v for p,v in hashes.items()):raise ValueError('Source changed during report')
    atomic_json(out/'comparison.json',comparison);publish(out,comparison);return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=4,default=DEFAULT_RUNS,metavar=('M1','M1-C','M1-B','M1-CB'))
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_boundary_central_100k_seed0'))
    parser.add_argument('--moment-samples',type=int,default=64)
    parser.add_argument('--rebuild-only',action='store_true')
    args=parser.parse_args()
    if args.rebuild_only:publish(args.output,json.loads((args.output/'comparison.json').read_text()))
    else:build(args.runs,args.output,args.moment_samples)


if __name__=='__main__':main()
