"""Read-only online time-factor versus original C comparison and MC audit."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
from compare_value_clipping import audit
from uav_research.config import load_config
from uav_research.critic_time_factor import score, horizon_factors
from uav_research.plotting import plt, style, save_figure, read_csv
from uav_research.runtime import atomic_json


RUNS={'direct':Path('runs/mappo_ablation_c_mobile_coverage_250k_seed0'),
      'time_factor':Path('runs/mappo_time_factor_250k_seed0')}
REPORTS={'direct':Path('analysis/ue_reward_ablations_250k_seed0/c_mobile_coverage'),
         'time_factor':Path('analysis/mappo_time_factor_250k_seed0/policy')}
OUT=Path('analysis/mappo_time_factor_250k_seed0')


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def bootstrap_difference(left,right,metric,seed=950001):
    draws=np.random.default_rng(seed).integers(len(left),size=(2000,len(left)))
    if metric=='rmse':
        a,b=(left**2).mean(1),(right**2).mean(1)
        delta=np.sqrt(b.mean())-np.sqrt(a.mean())
        samples=np.sqrt(b[draws].mean(1))-np.sqrt(a[draws].mean(1))
    else:
        a,b=left,right
        delta=b.mean()-a.mean();samples=(b-a)[draws].mean(1)
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(delta=float(delta),low=float(lo),high=float(hi))


def plot_results(out, summary):
    figs=out/'figures';style()
    for xkey,name in [('env_steps','training_comparison'),('elapsed_seconds','wall_time_comparison')]:
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for label,run in RUNS.items():
            val=read_csv(run/'validation.csv')
            x=np.array([float(r[xkey]) for r in val])/(1000 if xkey=='env_steps' else 60)
            for ax,key in zip(axes,['coverage','discovery_fraction','disconnection_ratio']):
                ax.plot(x,[float(r[key]) for r in val],label=label)
        for ax,title in zip(axes,['Validation coverage','Sensing fraction','Disconnected UAV fraction']):
            ax.set(xlabel='Joint transitions (k)' if xkey=='env_steps' else 'Wall time (min)',title=title,ylabel='Fraction');ax.legend()
        save_figure(fig,figs,name)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for label,run in RUNS.items():
        rows=list({r['env_steps']:r for r in read_csv(run/'metrics.csv')}.values())
        x=np.array([int(r['env_steps']) for r in rows])/1000
        for ax,key in zip(axes.flat,['actor_loss','value_loss','categorical_entropy','ppo_approx_kl','ppo_clip_fraction','max_action_probability']):
            ax.plot(x,[float(r[key]) for r in rows],label=label,alpha=.8)
    for ax,key in zip(axes.flat,['actor_loss','value_loss','categorical_entropy','ppo_approx_kl','ppo_clip_fraction','max_action_probability']):
        ax.set(xlabel='Joint transitions (k)',title=key);ax.legend(fontsize=7)
    save_figure(fig,figs,'loss_and_policy_diagnostics')
    rows=list({r['env_steps']:r for r in read_csv(RUNS['time_factor']/'metrics.csv')}.values())
    x=np.array([int(r['env_steps']) for r in rows])/1000
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    fields=[['critic_value_clip_fraction','critic_clipped_loss_fraction'],
            ['critic_average_prediction_mean','critic_average_target_mean'],
            ['critic_average_prediction_outside','critic_average_target_outside'],
            ['critic_return_normalizer_mean','critic_return_normalizer_std'],
            ['critic_average_normalizer_mean','critic_average_normalizer_std'],
            ['value_return_mean','q_grad_norm']]
    for ax,keys in zip(axes.flat,fields):
        for key in keys:ax.plot(x,[float(r[key]) for r in rows],label=key)
        ax.set(xlabel='Joint transitions (k)');ax.legend(fontsize=6)
    save_figure(fig,figs,'online_critic_diagnostics')
    for ck in ['best','last']:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for label,path in REPORTS.items():
            with np.load(path/f'validation_{ck}'/'reward_curves.npz') as data:
                names=list(data['diagnostic_names']);d=data['diagnostics'].mean(0)
                for ax,key in zip(axes.flat,['coverage','discovery_fraction','uav_nearest_neighbor_mean','uav_close_neighbor_fraction']):
                    ax.plot(data['steps'],d[:,names.index(key)],label=label)
        for ax,key in zip(axes.flat,['coverage','discovery_fraction','uav_nearest_neighbor_mean','uav_close_neighbor_fraction']):
            ax.set(title=key,xlabel='Episode step');ax.legend()
        save_figure(fig,figs,f'deployment_maintenance_{ck}')
        fig,axes=plt.subplots(1,2,figsize=(11,3.8))
        with np.load(out/f'time_factor_{ck}_critic.npz') as data:
            h=horizon_factors(np.arange(400,0,-1),.995)
            c,mc=data['values']/h,data['returns']/h
            axes[0].plot(c.mean(0),label='Predicted average coverage')
            axes[0].plot(mc.mean(0),label='Sampled MC average coverage')
            axes[0].set(xlabel='State time before action',ylabel='Average coverage',title=f'time_factor / {ck}');axes[0].legend(fontsize=7)
            axes[1].scatter(mc.ravel()[::4],c.ravel()[::4],s=3,alpha=.15)
            axes[1].plot([0,1],[0,1],'--',color='gray')
            axes[1].set(xlabel='MC average-coverage label',ylabel='Prediction',title='Every fourth state, fixed order')
        save_figure(fig,figs,f'average_coverage_detail_{ck}')
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for label,path in REPORTS.items():
            with np.load(path/f'validation_{ck}'/'reward_curves.npz') as data:
                names=list(data['diagnostic_names']);d=data['diagnostics'].mean(0)
                for ax,key in zip(axes[0],['coverage','discovery_fraction','disconnection_ratio']):
                    ax.plot(data['steps'],d[:,names.index(key)],label=label)
            with np.load(out/f'{label}_{ck}_critic.npz') as data:
                v,g=data['values'],data['returns'];h=horizon_factors(np.arange(400,0,-1),.995)
                for ax in axes[1,:2]:
                    ax.plot(v.mean(0),label=label+' V');ax.plot(g.mean(0),'--',label=label+' MC')
                axes[1,2].plot((v/h).mean(0),label=label+' avg prediction')
                axes[1,2].plot((g/h).mean(0),'--',label=label+' avg MC')
        for ax,key in zip(axes[0],['coverage','discovery_fraction','disconnection_ratio']):ax.set(title=key)
        axes[1,0].set(title='Raw cumulative value')
        axes[1,1].set(title='Final 20 states',xlim=(380,399))
        axes[1,2].set(title='Discounted average coverage')
        for ax in axes.flat:ax.set(xlabel='Episode time');ax.legend(fontsize=6)
        save_figure(fig,figs,f'episode_and_critic_{ck}')
        fig,axes=plt.subplots(1,2,figsize=(11,3.8))
        for label in RUNS:
            with np.load(out/f'{label}_{ck}_critic.npz') as data:
                e=data['values']-data['returns']
                axes[0].plot(np.sqrt((e**2).mean(0)),label=label)
                axes[1].plot(e.mean(0),label=label)
        for ax,title in zip(axes,['Per-state value RMSE','Mean signed error']):
            ax.set(title=title,xlabel='State time before action');ax.legend()
        save_figure(fig,figs,f'critic_error_{ck}')


def publish(out, summary):
    plot_results(out,summary)
    lines=['# 原C组与在线时间因子critic：250k对照','',
           '两组种子0、250000条联合转移、123轮PPO。原C复用历史结果，新组从头训练actor和critic。'
           '唯一配置变量为critic_parameterization；保留Value Clipping、GAE、网络宽度、学习率与纯覆盖奖励。'
           '初始actor与随机流一致，step0验证覆盖均为34.8227%。时间因子输出层初始化除H0，初始价值函数因此有H/H0差异。','',
           '## 策略表现','',
           '| 方法 | 模型 | 步数 | 全回合覆盖 | 后100步覆盖 | 终态覆盖 | 平均断联 | 悬停率 | 实际速度m/s |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['policy_rows']:
        lines.append(f'| {row["method"]} | {row["checkpoint"]} | {row["step"]} | '+
                     ' | '.join(f'{100*row[k]:.2f}%' for k in ['coverage','last100_coverage','final_coverage','disconnection_ratio','hover_fraction'])+
                     f' | {row["actual_speed_mean"]:.2f} |')
    lines+=['','主要比较同预算last；best按全回合回报选择，不能解释为最优终态。验证32场景、采样种子880001、16环境批次。测试集未使用。','',
            '![训练验证](figures/training_comparison.png)','','![墙钟](figures/wall_time_comparison.png)','',
            '![Loss与PPO](figures/loss_and_policy_diagnostics.png)','','![在线critic诊断](figures/online_critic_diagnostics.png)','',
            '训练日志的reward是日志时刻最后采集步，不是整轮平均；正式验证指标来自完整400步。'
            '历史direct没有记录新增的平均覆盖归一化与价值裁剪频率，不伪造这些日志。','',
            '## 冻结模型的MC预测准确性','',
            '每个模型固定actor与critic，完整采集32回合，G_400=0、gamma0.995反向计算MC。'
            'V取动作前的原始累计价值，平均覆盖为V/H；MC相应为G/H。末步指t=399，价值上界为1。'
            '对照组各用自己的on-policy轨迹，不是同一状态上的网络比较，MC包含采样噪声。','',
            '| 方法 | 模型 | 全程价值RMSE | 后20步价值RMSE | 平均覆盖RMSE | 末步V均值 | 末步MC均值 | 去时间均值相关性 | 价值范围违规 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,m in summary['critic'].items():
        label,ck=key.rsplit('_',1);v=m['value']['stages'];corr=m['value']['time_centered_correlation']
        corr_text = '未定义' if corr is None else f'{corr:.3f}'
        lines.append(f'| {label} | {ck} | {v["all"]["rmse"]:.3f} | {v["last20"]["rmse"]:.3f} | '
                     f'{m["average"]["stages"]["all"]["rmse"]:.4f} | {v["last1"]["prediction_mean"]:.3f} | '
                     f'{v["last1"]["label_mean"]:.3f} | {corr_text} | {m["value_range_violation_fraction"]:.2%} |')
    lines+=['','![Best回合与critic](figures/episode_and_critic_best.png)','','![Last回合与critic](figures/episode_and_critic_last.png)','',
            '![平均覆盖预测放大](figures/average_coverage_detail_last.png)','',
            '![部署维持与聚集](figures/deployment_maintenance_last.png)','',
            '## 配对场景差异','',
            '以下均为time_factor减direct。覆盖率差正值有利，RMSE差负值有利。2000次场景bootstrap，seed950001；'
            '不是训练种子区间，只有一个训练种子。','',
            '| 模型 | 指标 | 差值 | 95%场景区间 |','|---|---|---:|---:|']
    for row in summary['paired']:
        scale=100 if row['metric']!='value_rmse' else 1
        suffix=' pp' if scale==100 else ''
        lines.append(f'| {row["checkpoint"]} | {row["metric"]} | {scale*row["delta"]:+.3f}{suffix} | '
                     f'[{scale*row["low"]:+.3f}, {scale*row["high"]:+.3f}] |')
    policy={(r['method'],r['checkpoint']):r for r in summary['policy_rows']}
    direct,factor=policy[('direct','last')],policy[('time_factor','last')]
    lines += ['', '## 本次结果解读','',
              f'同预算last覆盖从{direct["coverage"]:.2%}提高到{factor["coverage"]:.2%}，'
              f'增量{100*(factor["coverage"]-direct["coverage"]):.2f}个百分点。'
              '全回合覆盖的场景配对区间支持本次场景集合上的正增益；后100步与终态覆盖的区间仍跨零，'
              '不能声称已经稳定提高终态部署。best到last也有回退。','',
              'critic的末端巨大高估明显修正，预测随剩余时长下降；这发生在保留GAE和Value Clipping的条件下。'
              '因此结果不支持必须去掉价值裁剪或完全放弃GAE才能改善预测。'
              '但参数化、输出尺度、初始函数和优化条件同时发生结构性变化，不能据此断言某一个细节是唯一原因。','',
              '覆盖提高并没有伴随感知范围扩大。下表是last的平均每步人数，属于互斥的分阶段淘汰统计：','',
              '| 阶段 | 原C | 时间因子 |','|---|---:|---:|']
    gate_means={k:json.loads((p/'report.json').read_text())['evaluations']['last']['summary']['mean'] for k,p in REPORTS.items()}
    for key in ['sensing_rejected_users','downlink_rejected_users','backhaul_rejected_users','capacity_rejected_users']:
        lines.append(f'| {key} | {gate_means["direct"][key]:.3f} | {gate_means["time_factor"][key]:.3f} |')
    lines += ['', '新组的主要人数改善体现在回传淘汰减少，下行淘汰也略减；感知失败人数反而略增。'
              '这与联网服务资格改善更一致，还不能称为网络已经学会扩大空间覆盖。','',
              f'last的全回合悬停率从{direct["hover_fraction"]:.2%}升至{factor["hover_fraction"]:.2%}，'
              f'实际速度从{direct["actual_speed_mean"]:.2f}降至{factor["actual_speed_mean"]:.2f}m/s，策略仍大量移动。'
              f'终态近距离聚集比例从{direct["final_uav_close_neighbor_fraction"]:.2%}变为{factor["final_uav_close_neighbor_fraction"]:.2%}，'
              '没有消除聚集。','']
    for ck in ['best','last']:
        with np.load(REPORTS['time_factor']/f'validation_{ck}'/'reward_curves.npz') as curves:
            y=curves['reward'].mean(0)
        lines.append(f'- 新组{ck}的四个100步阶段覆盖：'+' → '.join(f'{y[i:i+100].mean():.2%}' for i in range(0,400,100))+'。')
    lines += ['', '在线日志显示前期价值裁剪很活跃，后期下降；归一化累计统计平滑变化，没有出现GAE平均覆盖标签越界。'
              '这不证明每个状态的GAE目标准确，只说明本次记录未出现数值范围异常。'
              '离线预测改善已部分转移到在线训练，但稳定展开仍需单独解决；本轮没有继续修改熵、奖励、观测或协作机制。','']
    lines+=['','## 固定验证场景拓扑','',
            '固定展示10000和10001，未按表现筛选。原C的拓扑使用已有独立验证结果；新组best/last的独立报告同时保存轨迹。','']
    for ck in ['best','last']:
        for seed in [10000,10001]:
            lines.extend([f'### {ck} / 场景{seed}','',
                f'[原C拓扑](../ue_reward_ablations_250k_seed0/c_mobile_coverage/figures/topology_{ck}_{seed}.png)','',
                f'![时间因子拓扑](policy/figures/topology_{ck}_{seed}.png)',''])
    lines+=['## 可复现文件','',
            '- policy/report.md：新组独立验证、loss、奖励与拓扑。',
            '- policy_summary.csv、critic_metrics.csv、paired_differences.csv：结果表。',
            '- *_critic.npz：逐场景V、MC、奖励、时刻；全部复核平均覆盖与正式验证一致。',
            '- comparison.json：配置差异、源文件SHA256和完整指标。',
            '- figures/：PNG/PDF/SVG图像。','',
            'critic改善不自动证明部署能维持；覆盖变化也不能直接归因于唯一机制。'
            '本轮不延长250k、不追加种子或baseline训练。', '',
            '复现与数学约定见[实验说明](../../docs/online_time_factor_critic_experiment.md)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def build(out, report_only=False):
    if report_only:
        publish(out,json.loads((out/'comparison.json').read_text()));return
    out.mkdir(parents=True,exist_ok=True)
    configs={k:load_config(p/'config.yaml') for k,p in RUNS.items()}
    changed=configs['time_factor'].as_dict()
    changed['algo']['critic_parameterization']='direct'
    assert changed==configs['direct'].as_dict(), 'Additional configuration change'
    files=[p/n for p in RUNS.values() for n in ['config.yaml','best.pt','last.pt','metrics.csv','validation.csv','manifest.json']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    reports={k:json.loads((p/'report.json').read_text()) for k,p in REPORTS.items()}
    rows=[];critics={};metrics=[];paired=[]
    for label,run in RUNS.items():
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['status']=='completed' and manifest['env_steps']==250000 and manifest['updates']==123
        first=read_csv(run/'validation.csv')[0]
        np.testing.assert_allclose(float(first['coverage']),.3482265625,atol=1e-12,rtol=0)
        for ck in ['best','last']:
            item=reports[label]['evaluations'][ck]
            assert hashes[str(run/f'{ck}.pt')]==item['checkpoint_sha256']
            m=item['summary']['mean']
            keys=['coverage','last100_coverage','final_coverage','discovery_fraction','disconnection_ratio',
                  'hover_fraction','actual_speed_mean','uav_nearest_neighbor_mean','uav_close_neighbor_fraction',
                  'final_uav_nearest_neighbor_mean','final_uav_close_neighbor_fraction','boundary_reflection_fraction']
            rows.append(dict(method=label,checkpoint=ck,step=item['checkpoint_env_steps'],**{k:m[k] for k in keys}))
            path=out/f'{label}_{ck}_critic.npz'
            audit(run,ck,path,m)
            with np.load(path) as data:
                v,g,r=data['values'],data['returns'],data['rewards'];h=horizon_factors(np.arange(400,0,-1),.995)
                np.testing.assert_array_equal(data['scene_seeds'],configs[label].run.validation_seeds)
                with np.load(REPORTS[label]/f'validation_{ck}'/'reward_curves.npz') as curves:
                    np.testing.assert_array_equal(r,curves['reward'])
                critics[f'{label}_{ck}']=score(dict(value=v,average=v/h),g,.995)
            for scale in ['value','average']:
                for stage,values in critics[f'{label}_{ck}'][scale]['stages'].items():
                    metrics.append(dict(method=label,checkpoint=ck,scale=scale,stage=stage,**values))
            print(label,ck,'critic RMSE',critics[f'{label}_{ck}']['value']['stages']['all']['rmse'],flush=True)
    for ck in ['best','last']:
        episodes={k:read_csv(p/f'validation_{ck}'/'episodes.csv') for k,p in REPORTS.items()}
        assert [r['scene_seed'] for r in episodes['direct']]==[r['scene_seed'] for r in episodes['time_factor']]
        for metric in ['coverage','last100_coverage','final_coverage','disconnection_ratio']:
            a,b=[np.array([float(r[metric]) for r in episodes[k]]) for k in ['direct','time_factor']]
            paired.append(dict(checkpoint=ck,metric=metric,**bootstrap_difference(a,b,metric)))
        with np.load(out/f'direct_{ck}_critic.npz') as a,np.load(out/f'time_factor_{ck}_critic.npz') as b:
            paired.append(dict(checkpoint=ck,metric='value_rmse',**bootstrap_difference(a['values']-a['returns'],b['values']-b['returns'],'rmse')))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files)
    summary=dict(policy_rows=rows,critic=critics,paired=paired,source_sha256=hashes,
                 config_difference={'algo.critic_parameterization':['direct','time_factor']},
                 evaluation_action_seed=880001,validation_scenes=32,training_seed_count=1)
    atomic_json(out/'comparison.json',summary)
    write_csv(out/'policy_summary.csv',rows);write_csv(out/'critic_metrics.csv',metrics);write_csv(out/'paired_differences.csv',paired)
    publish(out,summary)
    print('Saved',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();build(OUT,args.report_only)


if __name__=='__main__':main()
