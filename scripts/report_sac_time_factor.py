"""Read-only full-reward SAC comparison, soft-Q MC audit and policy moments.

Each Q is checked on its own frozen policy's sampled actions with fixed alpha.
The current action's entropy is never included in a Q(s,a) MC label.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from uav_research.config import load_config
from uav_research.critic_calibration import discounted_episode_returns
from uav_research.critic_time_factor import horizon_factors, error_metrics
from uav_research.evaluate import policy_from_checkpoint, evaluate
from uav_research.observations import as_tensor_observation
from uav_research.plotting import plt, style, save_figure, read_csv
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json
from uav_research.sac_diagnostics import soft_q_returns, action_moments
from uav_research.vector import VectorEnvironment
from report_mappo_training import topology, component_plots

STAGES={'all':slice(None),'first100':slice(0,100),'middle100':slice(150,250),
        'last100':slice(-100,None),'last20':slice(-20,None),'last1':slice(-1,None)}
DIAG=['coverage','tau','throughput_bps','robustness','disconnection_ratio','discovery_fraction',
      'actual_speed_mean','uav_nearest_neighbor_mean','uav_close_neighbor_fraction',
      'boundary_reflection_fraction','sensing_rejected_users','downlink_rejected_users',
      'backhaul_rejected_users','capacity_rejected_users','near_zero_actual_displacement_fraction']
MOMENTS=['command_variance_m2','near_hover_probability','latent_std','mean_command_speed']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def scores(pred,label,h):
    result={}
    for scale,x,y in [('value',pred,label),('average_soft_q',pred/h,label/h)]:
        result[scale]=error_metrics(x,y)
        for key,item in result[scale]['stages'].items():
            std=float(y[:,STAGES[key]].std())
            item['label_std']=std
            item['normalized_rmse']=item['rmse']/std if std>0 else None
    return result


def audit(run,checkpoint,output,moment_samples=64):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(run/'config.yaml');configure_runtime(cfg)
    path=run/f'{checkpoint}.pt';before=digest(path)
    cfg,policy=policy_from_checkpoint(path,'cpu')
    if cfg.algo.name!='sac' or cfg.env.action_space!='continuous':raise ValueError('Requires continuous SAC')
    if set(cfg.run.validation_seeds)&set(cfg.run.test_seeds):raise ValueError('Validation/test overlap')
    alpha=float(policy.log_alpha.exp().detach());policy.actor.eval();policy.critic.eval()
    all_q,all_rewards,all_terms,all_logp,all_diag,all_actions,all_moments=[],[],[],[],[],[],[]
    weights=np.array([1.,cfg.env.throughput_weight,cfg.env.robustness_weight,-cfg.env.disconnection_weight])
    moment_times=list(range(0,cfg.env.horizon,20))
    with isolated_rng(),torch.inference_mode():
        torch.manual_seed(cfg.run.evaluation_action_seed)
        for start in range(0,len(cfg.run.validation_seeds),cfg.run.num_envs):
            seeds=cfg.run.validation_seeds[start:start+cfg.run.num_envs]
            with VectorEnvironment(cfg.env,len(seeds),'batch') as vector:
                obs,state,_=vector.reset(seeds)
                qs,rs,terms,logps,diags,actions,moments=[],[],[],[],[],[],[]
                for t in range(cfg.env.horizon):
                    tensor_obs=as_tensor_observation(obs,'cpu')
                    action,lp,_=policy.actor.sample(tensor_obs)
                    if t in moment_times:
                        # Diagnostics must not advance the rollout's action RNG.
                        with isolated_rng():
                            torch.manual_seed(970001+start*cfg.env.horizon+t)
                            mm=action_moments(policy.actor,tensor_obs,cfg.env.max_speed*cfg.env.dt,moment_samples)
                            moments.append(np.stack([mm[k].numpy() for k in MOMENTS],-1))
                    q=policy.critic(torch.as_tensor(state),action)
                    qs.append(torch.stack(q,-1).numpy());logps.append(lp.sum(-1).numpy())
                    actions.append(action.numpy())
                    obs,state,reward,done,truncated,infos=vector.step(action.numpy())
                    raw=np.array([[i[k] for k in ['reward_primary','tau','robustness','disconnection_ratio']] for i in infos])
                    contribution=raw*weights
                    np.testing.assert_allclose(contribution.sum(-1),reward,rtol=1e-6,atol=1e-7)
                    # Environment reward is float32 in vector API; retain exact info for MC.
                    actual=np.array([i['reward'] for i in infos])
                    np.testing.assert_allclose(contribution.sum(-1),actual,atol=1e-12,rtol=1e-12)
                    assert np.all(done==(t==cfg.env.horizon-1)) and not truncated.any()
                    rs.append(actual);terms.append(contribution)
                    diagnostic_infos=[dict(i,near_zero_actual_displacement_fraction=float(
                        (np.linalg.norm(env.actual_displacement,axis=-1)<1).mean()))
                        for i,env in zip(infos,vector.envs)]
                    diags.append(np.array([[i[k] for k in DIAG] for i in diagnostic_infos]))
                all_q.append(np.stack(qs).transpose(1,0,2));all_rewards.append(np.stack(rs).T)
                all_terms.append(np.stack(terms).transpose(1,0,2));all_logp.append(np.stack(logps).T)
                all_diag.append(np.stack(diags).transpose(1,0,2))
                all_actions.append(np.stack(actions).transpose(1,0,2,3))
                all_moments.append(np.stack(moments).transpose(1,0,2,3))
    q=np.concatenate(all_q);r=np.concatenate(all_rewards);lp=np.concatenate(all_logp)
    contributions=np.concatenate(all_terms)
    env_g=discounted_episode_returns(r,cfg.algo.discount)
    soft_g=soft_q_returns(r,lp,cfg.algo.discount,alpha)
    components=np.stack([discounted_episode_returns(contributions[...,i],cfg.algo.discount) for i in range(4)],-1)
    np.testing.assert_allclose(components.sum(-1),env_g,atol=1e-10,rtol=1e-12)
    np.testing.assert_array_equal(soft_g[:,-1],r[:,-1])
    h=horizon_factors(np.arange(cfg.env.horizon,0,-1),cfg.algo.discount)
    data=dict(q_values=q,rewards=r,contributions=contributions,component_returns=components,
              environment_returns=env_g,soft_returns=soft_g,entropy_returns=soft_g-env_g,
              joint_logp=lp,alpha=np.array(alpha),horizon_factors=h,
              actions=np.concatenate(all_actions),diagnostics=np.concatenate(all_diag),diagnostic_names=np.array(DIAG),
              moments=np.concatenate(all_moments),moment_names=np.array(MOMENTS),moment_times=np.array(moment_times),
              scene_seeds=np.array(cfg.run.validation_seeds),action_seed=np.array(cfg.run.evaluation_action_seed))
    if not all(np.isfinite(v).all() for v in data.values() if np.asarray(v).dtype.kind in 'fc'):
        raise ValueError('Nonfinite frozen-policy diagnostics')
    result={name:scores(p,soft_g,h) for name,p in [('q1',q[...,0]),('q2',q[...,1]),('min_q',q.min(-1))]}
    policy_metrics={}
    diag=data['diagnostics']
    for stage in ['all','last100','last1']:
        sl=STAGES[stage]
        row={k:float(diag[:,sl,i].mean()) for i,k in enumerate(DIAG)}
        rho=diag[:,sl,0];rob=diag[:,sl,3]
        row.update(reward=float(r[:,sl].mean()),retention_ratio=float(rob.sum()/rho.sum()) if rho.sum() else None,
                   zero_coverage_episodes=int((rho.sum(1)==0).sum()),
                   near_zero_command_fraction=float((np.linalg.norm(data['actions'][:,sl],axis=-1)*cfg.env.max_speed*cfg.env.dt<1).mean()))
        policy_metrics[stage]=row
    result.update(alpha=alpha,policy=policy_metrics,
                  environment_return_mean=float(env_g.mean()),entropy_return_mean=float((soft_g-env_g).mean()),
                  soft_return_mean=float(soft_g.mean()),checkpoint_sha256=before,
                  checkpoint_env_steps=policy.provenance['checkpoint_env_steps'],
                  moments={k:float(data['moments'][...,i].mean()) for i,k in enumerate(MOMENTS)},
                  protocol='Frozen current actor and alpha; own on-policy sampled actions; future entropy only; no training')
    assert before==digest(path)
    np.savez_compressed(output/'audit.npz',**data);atomic_json(output/'metrics.json',result)
    return data,result


def plot_audit(data,output,name):
    style();output=Path(output)
    h=data['horizon_factors'];g=data['soft_returns'];q=data['q_values']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax in axes[0,:2]:
        ax.plot(g.mean(0),label='Sampled soft-Q MC')
        for i in range(2):ax.plot(q[...,i].mean(0),label=f'Q{i+1}')
    axes[0,1].set_xlim(max(0,len(h)-20),len(h)-1)
    axes[0,2].plot((g/h).mean(0),label='MC / H')
    for i in range(2):axes[0,2].plot((q[...,i]/h).mean(0),label=f'Q{i+1} / H')
    for i in range(2):axes[1,0].plot(np.sqrt(((q[...,i]-g)**2).mean(0)),label=f'Q{i+1} RMSE')
    axes[1,1].plot(data['environment_returns'].mean(0),label='Environment MC')
    axes[1,1].plot(data['entropy_returns'].mean(0),label='Future entropy MC')
    axes[1,1].plot(g.mean(0),label='Soft-Q MC')
    axes[1,2].plot(np.abs(q[...,0]-q[...,1]).mean(0),label='Mean |Q1 - Q2|')
    for ax,title in zip(axes.flat,['Cumulative soft-Q','Final 20 pre-action states','Average soft-Q',
                                  'Q prediction RMSE','Environment and entropy returns','Twin-Q disagreement']):
        ax.set(title=title,xlabel='Pre-action time');ax.legend(fontsize=7)
    fig.suptitle(name);save_figure(fig,output,'soft_q_diagnostics')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(indices,title) in zip(axes.flat,[([0],'Command variance (m²)'),([1],'P(command < 1 m)'),([2],'Latent std'),([3],'Mean command displacement (m)')]):
        for i in indices:ax.plot(data['moment_times'],data['moments'][...,i].mean((0,2)),label=MOMENTS[i])
        ax.set(title=title,xlabel='Pre-action time');ax.legend(fontsize=7)
    save_figure(fig,output,'policy_randomness')


def training_plots(runs,output):
    style()
    for table,fields,name in [
        ('metrics.csv',['q_loss','actor_loss','alpha','joint_entropy','td_abs','q_grad_norm'],'sac_loss_diagnostics'),
        ('validation.csv',['return','coverage','throughput_bps','robustness','disconnection_ratio','discovery_fraction'],'validation_comparison')]:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for label,run in runs.items():
            rows=list({r['env_steps']:r for r in read_csv(run/table)}.values())
            x=np.array([int(r['env_steps']) for r in rows])/1000
            for ax,k in zip(axes.flat,fields):
                y=np.array([float(r[k]) if r[k] else np.nan for r in rows])
                ax.plot(x,y,label=label)
        for ax,k in zip(axes.flat,fields):ax.set(title=k,xlabel='Joint transitions (k)');ax.legend()
        save_figure(fig,output,name)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for label,run in runs.items():
        rows=read_csv(run/'validation.csv');x=np.array([float(r['elapsed_seconds']) for r in rows])/60
        for ax,k in zip(axes,['return','coverage','disconnection_ratio']):
            ax.plot(x,[float(r[k]) for r in rows],label=label);ax.set(title=k,xlabel='Wall time (min)');ax.legend()
    save_figure(fig,output,'wall_time_comparison')


def publish(output):
    """Rebuild all figures and paired statistics from saved audits, without inference."""
    from compare_online_time_factor import bootstrap_difference
    output=Path(output);summary=json.loads((output/'comparison.json').read_text())
    runs={k:Path(v['run']) for k,v in summary['groups'].items()}
    training_plots(runs,output/'figures')
    for mode,run in runs.items():
        cfg=load_config(run/'config.yaml')
        component_plots(run,output/mode/'figures',cfg,read_csv(run/'metrics.csv'),read_csv(run/'validation.csv'))
        for ck,item in summary['groups'][mode]['checkpoints'].items():
            for trace in item['traces']:
                topology(Path(trace),output/mode/'figures',ck,item['audit']['checkpoint_env_steps'],cfg)
    paired=[]
    for ck in ['best','last']:
        datasets={mode:np.load(output/mode/ck/'audit.npz') for mode in runs}
        for mode,d in datasets.items():plot_audit(d,output/mode/ck,mode+' / '+ck)
        a,b=datasets['direct'],datasets['time_factor']
        np.testing.assert_array_equal(a['scene_seeds'],b['scene_seeds'])
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for mode,d in datasets.items():
            for ax,key in zip(axes.flat,['coverage','discovery_fraction','disconnection_ratio',
                                        'actual_speed_mean','uav_nearest_neighbor_mean','uav_close_neighbor_fraction']):
                ax.plot(np.arange(1,len(d['horizon_factors'])+1),
                        d['diagnostics'][...,list(d['diagnostic_names']).index(key)].mean(0),label=mode)
        for ax,key in zip(axes.flat,['Service coverage','Sensing fraction','Disconnected fraction',
                                    'Actual speed (m/s)','Nearest UAV (m)','Close-neighbor fraction']):
            ax.set(title=key,xlabel='Post-action episode step');ax.legend()
        save_figure(fig,output/'figures',f'deployment_{ck}')
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for mode,d in datasets.items():
            for i,ax in enumerate(axes.flat[:4]):ax.plot(d['contributions'][...,i].mean(0),label=mode)
            axes[1,1].plot(d['rewards'].mean(0),label=mode)
            axes[1,2].plot(d['entropy_returns'].mean(0),label=mode)
        for ax,key in zip(axes.flat,['Coverage contribution','Throughput contribution','Robustness contribution',
                                    'Disconnection contribution','Total environment reward','Future entropy MC']):
            ax.set(title=key,xlabel='Pre-action state time');ax.legend()
        save_figure(fig,output/'figures',f'reward_components_{ck}')
        for key in ['reward','coverage','robustness','disconnection_ratio','last100_coverage','final_coverage','soft_q_rmse']:
            values=[]
            for d in [a,b]:
                if key=='reward':x=d['rewards'].mean(1)
                elif key=='soft_q_rmse':x=d['q_values'].min(-1)-d['soft_returns']
                else:
                    name='coverage' if 'coverage' in key else key
                    z=d['diagnostics'][...,list(d['diagnostic_names']).index(name)]
                    x=z[:,-100:].mean(1) if key.startswith('last100') else z[:,-1] if key.startswith('final') else z.mean(1)
                values.append(x)
            paired.append(dict(checkpoint=ck,metric=key,**bootstrap_difference(*values,'rmse' if key=='soft_q_rmse' else 'mean')))
        for d in datasets.values():d.close()
    summary['paired_scene_bootstrap']=paired;atomic_json(output/'comparison.json',summary)
    with (output/'paired_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)
    lines=['# SAC完整奖励：direct / 时间因子对照','',
           '各模型在自身冻结actor、最终温度下诊断。Q的MC标签包含未来动作熵，不包含当前动作熵。'
           '策略表现使用完整环境奖励；best按验证完整奖励选择，last是同预算主要比较。'
           '场景数以保存配置为准（正式实验32个），一个训练种子的场景区间不能代表训练种子置信区间。','',
           '| checkpoint | critic | 步数 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 平均奖励 | min-Q RMSE | NRMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for ck in ['best','last']:
        for mode in runs:
            x=summary['groups'][mode]['checkpoints'][ck]['audit'];p=x['policy'];m=x['min_q']['value']['stages']['all']
            normalized=f"{m['normalized_rmse']:.4f}" if m['normalized_rmse'] is not None else 'N/A'
            lines.append(f"| {ck} | {mode} | {x['checkpoint_env_steps']} | {p['all']['coverage']:.2%} | {p['last100']['coverage']:.2%} | {p['last1']['coverage']:.2%} | {p['all']['reward']:.4f} | {m['rmse']:.4f} | {normalized} |")
    for ck in ['best','last']:
        lines+=['',f'![部署维持 {ck}](figures/deployment_{ck}.png)',
                f'![奖励分解 {ck}](figures/reward_components_{ck}.png)']
    lines+=['','完整原始Q/动作/logp/四项奖励/MC标签/条件动作矩在各组best、last的audit.npz。'
            '终态指执行最后动作后的服务指标，Q最后一步指最后动作之前。没有强制预测或标签位于[0,1]。','',
            '![验证](figures/validation_comparison.png)','![Loss](figures/sac_loss_diagnostics.png)',
            '![墙钟](figures/wall_time_comparison.png)','',
            '## 配对场景差值：time_factor减direct','',
            '| checkpoint | 指标 | 差值 | 场景95%区间 |','|---|---|---:|---|']
    for x in paired:lines.append(f"| {x['checkpoint']} | {x['metric']} | {x['delta']:.6f} | [{x['low']:.6f}, {x['high']:.6f}] |")
    for mode in runs:
        for ck in ['best','last']:
            lines+=['',f'## {mode} / {ck}',f'![soft-Q]({mode}/{ck}/soft_q_diagnostics.png)',
                    f'![随机性]({mode}/{ck}/policy_randomness.png)']
            group=summary['groups'][mode]['checkpoints'][ck]
            for trace in group['traces']:
                seed=Path(trace).stem.split('_')[-1]
                lines.append(f'![拓扑]({mode}/figures/topology_{ck}_{seed}.png)')
    lines+=['','best使用同一验证集选模，其场景区间含选择效应，不能视为独立测试结果。'
            '各模型自己的轨迹和MC采样噪声也会影响误差；冻结最终温度与在线温度变化有所区别。'
            '固定时间尺度Q=(H/H0)f与MAPPO动态PopArt实现不同；梯度尺度变化属于本次参数化的一部分。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def build(runs,output):
    runs={k:Path(v).resolve() for k,v in runs.items()};output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    configs={k:load_config(v/'config.yaml') for k,v in runs.items()}
    a,b=(copy.deepcopy(configs[k].as_dict()) for k in ['direct','time_factor'])
    for k in configs:
        if configs[k].algo.name!='sac' or configs[k].algo.critic_parameterization!=k:raise ValueError('Wrong paired run')
    b['algo']['critic_parameterization']='direct'
    if a!=b:raise ValueError('Paired configs differ beyond critic parameterization')
    summary=dict(groups={},scene_bootstrap_seed=950001,scene_bootstrap_draws=2000,training_seeds=1)
    for mode,run in runs.items():
        cfg=configs[mode];configure_runtime(cfg)
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest['status']!='completed' or manifest['env_steps']!=cfg.run.total_steps:raise ValueError('Run is incomplete')
        hashes={name:digest(run/name) for name in ['config.yaml','manifest.json','metrics.csv','validation.csv','metadata.json','best.pt','last.pt']}
        group=dict(run=str(run),manifest=manifest,source_sha256=hashes,checkpoints={})
        target=output/mode;(target/'figures').mkdir(parents=True,exist_ok=True)
        metrics=read_csv(run/'metrics.csv');history=read_csv(run/'validation.csv')
        component_plots(run,target/'figures',cfg,metrics,history)
        for ck in ['best','last']:
            pcfg,policy=policy_from_checkpoint(run/f'{ck}.pt','cpu')
            _,ev=evaluate(pcfg,policy,pcfg.run.validation_seeds,target/f'validation_{ck}',trace_count=2,reward_curves=True)
            step=policy.provenance['checkpoint_env_steps']
            old=[r for r in history if int(r['env_steps'])==step]
            if len(old)!=1:raise ValueError('No unique checkpoint evaluation')
            for key in ['return','coverage','robustness','disconnection_ratio']:
                np.testing.assert_allclose(ev['mean'][key],float(old[0][key]),atol=1e-9,rtol=1e-9)
            traces=[]
            for seed in pcfg.run.validation_seeds[:2]:
                trace=target/f'validation_{ck}'/f'trace_{seed}.npz';traces.append(str(trace))
                topology(trace,target/'figures',ck,step,pcfg)
            deterministic=copy.deepcopy(pcfg);deterministic.run.evaluation_policy='deterministic'
            _,det=evaluate(deterministic,policy,pcfg.run.validation_seeds,target/f'deterministic_{ck}',trace_count=0)
            data,aud=audit(run,ck,target/ck)
            with np.load(target/f'validation_{ck}'/'reward_curves.npz') as reference:
                np.testing.assert_array_equal(data['rewards'],reference['reward'])
            group['checkpoints'][ck]=dict(evaluation=ev,deterministic_auxiliary=det,audit=aud,traces=traces)
            print(mode,ck,'coverage',ev['mean']['coverage'],'Q RMSE',aud['min_q']['value']['stages']['all']['rmse'],flush=True)
        if any(digest(run/n)!=h for n,h in hashes.items()):raise ValueError('Run changed during report')
        summary['groups'][mode]=group
    atomic_json(output/'comparison.json',summary);publish(output)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=2,default=['runs/sac_full_reward_direct_250k_seed0','runs/sac_full_reward_time_factor_250k_seed0'])
    parser.add_argument('--output',default='analysis/sac_full_reward_time_factor_250k_seed0')
    parser.add_argument('--rebuild-only',action='store_true')
    args=parser.parse_args()
    if args.rebuild_only:publish(Path(args.output))
    else:build(dict(zip(['direct','time_factor'],args.runs)),Path(args.output))
