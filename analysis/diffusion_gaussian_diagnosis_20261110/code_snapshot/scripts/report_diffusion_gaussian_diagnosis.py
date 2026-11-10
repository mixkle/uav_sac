"""Saved-data audit/report for frozen motion-score and DPPO diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from uav_research.diffusion_dppo.diagnosis import read_csv, write_csv, scene_bootstrap
from uav_research.diffusion_dppo.storage import load
from uav_research.runtime import atomic_json
from uav_research.plotting import style, save_figure


GROUPS = ['diffusion', 'gaussian']
POLICIES = ['D0', 'Db', 'Dl', 'Gb', 'Gl']
COLORS = {'D0':'#777777', 'Db':'#009E73', 'Dl':'#009E73', 'Gb':'#0072B2', 'Gl':'#0072B2'}
STYLES = {'D0':':', 'Db':'-', 'Dl':'--', 'Gb':'-', 'Gl':'--'}


def number(row, key):
    return float(row[key]) if key in row and row[key] != '' else np.nan


def audit(args):
    import matplotlib.pyplot as plt
    out = args.output/'exp3_dppo_update';out.mkdir(parents=True, exist_ok=True)
    tables, stages = {}, []
    windows = [('0-1M', 0, 1e6), ('1-5M', 1e6, 5e6), ('5-7.5M', 5e6, 7.5e6), ('7.5-10M', 7.5e6, 1e7)]
    fields = []
    for group in GROUPS:
        rows = read_csv(args.inputs/'snapshots'/group/'metrics.csv')
        keys = [k for k in rows[0] if any(x in k for x in ('kl', 'clip', 'grad', 'actor_loss', 'value_loss', 'normalizer', 'candidate'))]
        tables[group] = rows
        fields += [f'### {group}', '', '`'+'`, `'.join(keys)+'`', '']
        for name, low, high in windows:
            chosen = [r for r in rows if low < int(r['env_steps']) <= high]
            for key in keys:
                values = np.array([number(r, key) for r in chosen])
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite historical metric: {group}/{key}')
                stages.append(dict(group=group, stage=name, metric=key, updates=len(chosen),
                    mean=float(values.mean()), median=float(np.median(values)),
                    p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95))))
    write_csv(out/'training_stage_metrics.csv', stages)
    # Labels reflect reverse DDPM time: 8 first/high-noise; 1 last.
    d = tables['diffusion'];steps=np.array([int(x['env_steps']) for x in d])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, suffix in zip(axes, ['kl', 'clip']):
        image = np.array([[float(r[f'denoise_{i}_{suffix}']) for r in d] for i in range(1, 9)])
        handle = ax.imshow(image, origin='lower', aspect='auto', cmap='cividis',
                           extent=[steps[0], steps[-1], .5, 8.5])
        ax.axvline(7501824, color='white', ls='--', lw=1)
        ax.set(ylabel='DDPM step (8 first, 1 last)', title=f'Conditional-transition {suffix}')
        fig.colorbar(handle, ax=ax)
    axes[-1].set_xlabel('Online joint transitions');save_figure(fig, out, 'ppo_stepwise_kl_clip')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for col, group in enumerate(GROUPS):
        rows=tables[group];x=np.array([int(r['env_steps']) for r in rows])
        for key, color, ls in [('movement_head_grad_norm','#009E73','-'), ('score_head_grad_norm','#0072B2','--')]:
            y=np.array([float(r[key]) for r in rows]);smooth=np.convolve(y,np.ones(50)/50,'valid')
            axes[0,col].plot(x[49:], smooth,color=color,ls=ls,label=key)
        y=np.array([float(r['actor_grad_norm']) for r in rows]);smooth=np.convolve(y,np.ones(50)/50,'valid')
        axes[1,col].plot(x[49:], smooth,color='#0072B2')
        axes[0,col].set(title=group,ylabel='Head gradient norm AFTER global clipping')
        axes[1,col].set(xlabel='Joint transitions',ylabel='Total actor norm BEFORE clipping')
        axes[0,col].legend(fontsize=7)
    save_figure(fig,out,'motion_vs_score_gradient')
    saved=load(args.inputs/'snapshots/diffusion/best.pt',kinds={'dppo_model'})
    state=saved['learner']['actor']
    sensitivity=[]
    for k in range(8):
        gain=float(state['coef_x0'][k]*state['sqrt_one_minus_abar'][k]/state['sqrt_abar'][k])
        std=float(state['transition_std'][k])
        sensitivity.append(dict(denoise_step=k+1, reverse_order=8-k,
            transition_std=std, epsilon_to_mean_gain=gain, standardized_epsilon_gain=gain/std))
    write_csv(out/'schedule_sensitivity.csv',sensitivity)
    write_csv(out/'available_fields.csv',[dict(group=g,field=k) for g,rows in tables.items() for k in rows[0]])
    missing=['历史逐minibatch advantage均值/方差', '历史未裁剪ratio分位数',
             '历史运动/评分log密度拆分', '裁剪前分支梯度及共享编码器的分支贡献', '历史同一观察在更新前后的最终动作分布']
    text=['# 历史日志字段审计','',
        '每条metrics.csv记录为一次rollout的所有PPO minibatch/epoch统计均值，不是逐minibatch原始数据。',
        'KL=(ratio-1)-log(ratio)在旧策略样本上平均；扩散按每UAV、每去噪转移计算，Gaussian按每UAV联合latent计算，二者不是同一个最终动作KL。',
        'denoise_8为第一次反向转移z8→z7，denoise_1为最后z1→z0。',
        '**actor_grad_norm在全局裁剪前记录；运动头、评分头和编码器梯度在裁剪后记录。** 分支间比例可比较，不能拿分支范数直接除以日志的裁剪前总范数。','',
        *fields,'## 缺失的历史明细','',*['- '+x for x in missing],'',
        '新的likelihood_audit_*.csv是冻结模型在新采样链上的密度拆分与更新前重算核查，不是补造历史优化器更新明细。']
    (out/'metrics_field_audit.md').write_text('\n'.join(text)+'\n')
    late = [r for r in d if int(r['env_steps'])>7500000]
    local_kl=np.mean([float(r['denoise_8_kl']) for r in late])
    local_clip=np.mean([float(r['denoise_8_clip']) for r in late])
    middle=np.mean([float(r[f'denoise_{i}_kl']) for r in late for i in range(2,8)])
    result=dict(late_denoise8_kl=float(local_kl),late_denoise8_clip=float(local_clip),
        late_middle_kl=float(middle),late_kl_ratio=float(local_kl/middle),
        late_motion_gradient=float(np.mean([float(r['movement_head_grad_norm']) for r in late])),
        late_score_gradient=float(np.mean([float(r['score_head_grad_norm']) for r in late])),
        nonfinite_metrics=0,interpretation='Localized update imbalance; not proof of causal performance loss')
    atomic_json(out/'audit_summary.json',result)
    (out/'summary.md').write_text('# DPPO更新日志诊断\n\n'+
        f"最后2.5M中，第一次反向去噪转移的KL均值为{local_kl:.6f}、clip比例{local_clip:.2%}；中间2–7步的KL均值为{middle:.6f}，比值{local_kl/middle:.1f}。\n\n"+
        '这支持局部去噪步骤更新失衡的候选解释，尚不能证明它导致部署差距。整体平均KL会掩盖分步差异。评分头裁剪后梯度更大也不是评分任务错误的充分证据，需要结合固定状态评分干预。\n')
    print(json.dumps(result),flush=True)


def paired_interventions(rows, key, left, right, fixed_only=True):
    state = {}
    for r in rows:
        if fixed_only and int(r['is_fixed']) != 1:
            continue
        k=(int(r['scene']),r['state_id'],r['combination'])
        state.setdefault(k,[]).append(float(r[key]))
    scene={}
    for seed,sid,combo in state:
        if combo != left or (seed,sid,right) not in state:continue
        value=np.mean(state[(seed,sid,left)])-np.mean(state[(seed,sid,right)])
        scene.setdefault(seed,[]).append(value)
    return scene_bootstrap([np.mean(x) for x in scene.values()])


def report(args):
    import matplotlib.pyplot as plt
    out=args.output
    exp1=out/'exp1_motion_vs_scoring';exp2=out/'exp2_action_distribution';exp3=out/'exp3_dppo_update'
    contrasts=[];counter_data={};condition_data={}
    for kind in ['best','last']:
        counter_data[kind]=read_csv(exp1/f'paired_one_step_{kind}.csv')
        condition_data[kind]=read_csv(exp2/f'conditional_uav_{kind}.csv')
        for key in ['coverage','tau','reward','disconnection_ratio']:
            for left,right in [('G/G','D/D'),('G/D','D/D'),('D/G','D/D'),('G/G','G/D'),('G/G','D/G')]:
                for fixed in [True,False]:
                    contrasts.append(dict(kind=kind,metric=key,left=left,right=right,
                        cohort='fixed' if fixed else 'fixed_plus_selected_events',
                        **paired_interventions(counter_data[kind],key,left,right,fixed)))
    write_csv(exp1/'paired_contrasts.csv',contrasts)
    atomic_json(exp1/'paired_contrasts.json',contrasts)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['best','last']):
        for j,combo in enumerate(['G/G','G/D','D/G','D/D']):
            chosen=[r for r in counter_data[kind] if r['combination']==combo and int(r['is_fixed'])]
            scene={}
            for r in chosen:scene.setdefault(r['scene'],[]).append(float(r['coverage']))
            v=scene_bootstrap([np.mean(x) for x in scene.values()])
            ax.errorbar(j,100*v['mean'],yerr=np.array([[v['mean']-v['low']],[v['high']-v['mean']]])*100,
                        fmt='o',color='#0072B2' if combo[0]=='G' else '#009E73',capsize=4)
        ax.set(xticks=range(4),xticklabels=['G/G','G/D','D/G','D/D'],ylabel='One-step coverage (%)',title=kind)
    save_figure(fig,exp1,'motion_score_factorial')
    attribution=[]
    for kind in ['best','last']:
        for group in GROUPS:
            rows=read_csv(exp1/f'drop_attribution_{kind}_{group}.csv')
            summary=dict(kind=kind,group=group,events=len(rows))
            for key in ['lost_backhaul','lost_radio','lost_candidate_timing','lost_topk','new_users',
                        'coverage_drop','protocol_gap_users','physical_max_served','newly_disconnected']:
                summary[key]=float(np.mean([float(r[key]) for r in rows])) if rows else None
            summary['events_with_new_disconnect']=float(np.mean([float(r['newly_disconnected'])>0 for r in rows])) if rows else None
            attribution.append(summary)
    write_csv(exp1/'drop_attribution_summary.csv',attribution)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['best','last']):
        chosen=[r for r in attribution if r['kind']==kind]
        base=np.zeros(2)
        for metric,color,label in [('lost_backhaul','#009E73','Backhaul'),('lost_radio','#0072B2','Radio'),
                                   ('lost_candidate_timing','#aaaaaa','Timing'),('lost_topk','#666666','Top-K')]:
            values=np.array([r[metric] for r in chosen]);ax.bar(range(2),values,bottom=base,color=color,label=label);base+=values
        ax.set(xticks=range(2),xticklabels=GROUPS,ylabel='Lost old users per >=10pp event',title=kind)
    axes[0].legend(fontsize=7);save_figure(fig,exp1,'drop_attribution')
    # Conditional summaries are first averaged over UAVs/states, then over scenes.
    distributions=[]
    metrics=['command_mean_norm','command_variance_trace','effective_mean_norm','effective_variance_trace',
             'effective_near_hover','actual_mean_norm','actual_variance_trace','actual_near_hover',
             'effective_direction_resultant','topk_jaccard','score_latent_variance','ue_owner_consistency']
    for kind in ['best','last']:
        for policy in POLICIES:
            rows=[r for r in condition_data[kind] if r['policy']==policy and int(r['is_fixed'])]
            for bucket in ['all','hop1','hop2_3','hop4plus','disconnected','high_coverage','over_capacity']:
                chosen=[r for r in rows if bucket=='all' or
                    (bucket=='hop1' and int(r['hop'])==1) or
                    (bucket=='hop2_3' and 2<=int(r['hop'])<=3) or
                    (bucket=='hop4plus' and 4<=int(r['hop'])<=18) or
                    (bucket=='disconnected' and int(r['hop'])==19) or
                    (bucket=='high_coverage' and float(r['coverage_before'])>=.65) or
                    (bucket=='over_capacity' and int(r['candidate_count'])>10)]
                for metric in metrics:
                    scene={}
                    for r in chosen:
                        value=number(r,metric)
                        if np.isfinite(value):scene.setdefault(r['scene'],[]).append(value)
                    distributions.append(dict(kind=kind,policy=policy,bucket=bucket,metric=metric,
                        **scene_bootstrap([np.mean(x) for x in scene.values()])))
    write_csv(exp2/'conditional_summary.csv',distributions)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['effective_mean_norm','effective_variance_trace','effective_near_hover'],
        ['Conditional mean executed displacement (m)','Executed covariance trace (m²)','Executed near-hover probability']):
        for policy in POLICIES:
            chosen=[next(r for r in distributions if r['kind']=='best' and r['policy']==policy and r['metric']==key and r['bucket']==b)
                    for b in ['hop1','hop2_3','hop4plus','disconnected']]
            ax.plot(range(4),[r['mean'] for r in chosen],label=policy,color=COLORS[policy],ls=STYLES[policy])
        ax.set(xticks=range(4),xticklabels=['1','2–3','4–18','off'],xlabel='Pre-action hop',title=title)
    axes[0].legend(fontsize=7);save_figure(fig,exp2,'motion_magnitude_by_hop')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['topk_jaccard','ue_owner_consistency']):
        for policy in POLICIES:
            chosen=[next(r for r in distributions if r['kind']==kind and r['policy']==policy and r['metric']==key and r['bucket']=='over_capacity') for kind in ['best','last']]
            ax.plot(range(2),[r['mean'] for r in chosen],label=policy,color=COLORS[policy],ls=STYLES[policy],marker='o')
        ax.set(xticks=range(2),xticklabels=['Best-source pool','Last-source pool'],ylabel=key,title='Capacity-saturated UAV cohort')
    axes[0].legend(fontsize=7);save_figure(fig,exp2,'topk_consistency')
    # Optional short forks preserve same physical starts and UE random streams.
    fork_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['best','last']):
        path=exp1/f'forks_{kind}.npz'
        if not path.exists():continue
        with np.load(path) as z:
            values=z['values'];names=z['names'].tolist();combos=z['combinations'].tolist()
            ids=[json.loads(x) for x in z['identities_json']]
        for i,combo in enumerate(combos):
            curve=values[:,i,:,names.index('coverage')]
            ax.plot(np.arange(1,51),curve.mean(0),label=combo,color='#0072B2' if combo[0]=='G' else '#009E73',ls='-' if combo[0]==combo[-1] else '--')
            for horizon in [10,25,50]:
                for metric in ['coverage','reward','tau']:
                    per_state=values[:,i,:horizon,names.index(metric)].mean(1)
                    scene={}
                    for identity,value in zip(ids,per_state):scene.setdefault(identity['scene'],[]).append(value)
                    fork_rows.append(dict(kind=kind,combination=combo,horizon=horizon,metric=metric,
                        **scene_bootstrap([np.mean(x) for x in scene.values()])))
        ax.set(title=kind,xlabel='Fork steps',ylabel='Coverage');ax.legend(fontsize=7)
    if fork_rows:save_figure(fig,exp1,'fork_coverage_curve')
    else:plt.close(fig)
    write_csv(exp1/'fork_summary.csv',fork_rows)
    summary=dict(contrasts=contrasts,drop_attribution=attribution,conditional=distributions,forks=fork_rows)
    atomic_json(out/'diagnostic_summary.json',summary)
    # All exact numerical conclusions remain available for the written interpretation.
    lines=['# Diffusion与Gaussian：同状态定位实验','',
        '原10M结果只读保护。阶段A为同一完整快照的配对动作干预，UE移动随机流完全相同；五个固定时刻为主队列，逐场景最严重事件前状态为另外的有选择诊断队列。','',
        '区间是2000次配对场景bootstrap；先平均状态与动作重复，再在场景层重采样。不是训练种子区间，也不使用保留测试集。','',
        '## 单步反事实：固定时刻主队列','',
        '| 权重 | 比较 | 覆盖差 pp | 场景95%区间 pp |','|---|---|---:|---:|']
    for r in contrasts:
        if r['metric']=='coverage' and r['cohort']=='fixed':
            lines.append(f"| {r['kind']} | {r['left']} − {r['right']} | {100*r['mean']:+.3f} | [{100*r['low']:+.3f}, {100*r['high']:+.3f}] |")
    lines += ['','## 大幅骤降：实际失去的原服务用户','',
        '| 模型 | 事件数 | 回传失效 | 无线失效 | 时序 | Top-K | 同时出现新增断联的事件比例 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in attribution:
        lines.append(f"| {r['group']} {r['kind']} | {r['events']} | {r['lost_backhaul']:.2f} | {r['lost_radio']:.2f} | {r['lost_candidate_timing']:.2f} | {r['lost_topk']:.2f} | {r['events_with_new_disconnect']:.2%} |")
    lines += ['', '以上按回传→无线→候选→Top-K固定次序归因，不代表来源相互独立；原失效连接被其他UAV救回和新获服务的UE均单独记录，净下降分解已逐事件校验。','',
        '## 条件采样：同一固定状态队列，64次/模型','',
        '| 状态池 | 策略 | 缩放位移均值模长 m | 缩放位移方差迹 m² | 缩放后近悬停 | 饱和候选Top-K Jaccard |','|---|---|---:|---:|---:|---:|']
    for kind in ['best','last']:
        for policy in POLICIES:
            selected={r['metric']:r for r in distributions if r['kind']==kind and r['policy']==policy and r['bucket']=='all'}
            j=next(r for r in distributions if r['kind']==kind and r['policy']==policy and r['bucket']=='over_capacity' and r['metric']=='topk_jaccard')
            lines.append(f"| {kind} | {policy} | {selected['effective_mean_norm']['mean']:.3f} | {selected['effective_variance_trace']['mean']:.3f} | {selected['effective_near_hover']['mean']:.3%} | {j['mean']:.4f} |")
    lines += ['', 'D0为扩散预训练后step0，Db/Dl为扩散best/last，Gb/Gl为Gaussian best/last。Best与Last状态池均包含两种源策略访问的状态。混合动作存在分布外限制，不能当作新的公平可部署算法。','',
        '## 数据入口','', '- exp1_motion_vs_scoring：单步干预、全骤降事件、短程分叉与场景区间。',
        '- exp2_action_distribution：完整条件运动/评分样本、真实UE映射、分桶统计。',
        '- exp3_dppo_update：历史字段审计、分步KL/clip、梯度与新冻结链重算审计。',
        '- states：含UE随机流、关联与历史的完整快照，原报告中只有坐标的轨迹未被误用为完整状态。','']
    (out/'report.md').write_text('\n'.join(lines))
    (exp1/'summary.md').write_text('\n'.join(lines[:next(i for i,x in enumerate(lines) if x.startswith('## 条件采样'))])+'\n')
    (exp2/'summary.md').write_text('# 条件分布诊断\n\n完整统计见conditional_summary.csv和总报告，原始64次运动与评分样本在conditional_samples_*.npz，未将空候选填零作为真实评分。\n')
    density=[]
    for kind in ['best','last']:density+=read_csv(exp3/f'likelihood_audit_{kind}.csv')
    write_csv(exp3/'likelihood_audit.csv',density)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for policy in POLICIES:
        rows=[r for r in density if r['policy']==policy]
        for ax,key in zip(axes,['movement_logp_std','score_logp_std']):
            x=[number(r,'candidate_count') for r in rows];y=[number(r,key) for r in rows]
            ax.scatter(x,y,s=6,alpha=.2,color=COLORS[policy],label=policy)
            ax.set(xlabel='Mean candidate count in bucket',ylabel=key)
    axes[0].legend(fontsize=7);save_figure(fig,exp3,'logprob_by_candidate_count')
    print('REPORT_COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','report'])
    p.add_argument('--inputs',type=Path,default=Path('analysis/hop_scaling_joint_service_10m_seed0'))
    p.add_argument('--output',type=Path,default=Path('analysis/diffusion_gaussian_diagnosis_20261110'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);style()
    (audit if a.stage=='audit' else report)(a)


if __name__=='__main__':main()
