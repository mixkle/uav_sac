"""Rebuild the fixed-layout association report entirely from saved arrays."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .plotting import COLORS, plt, save_figure, style
from .runtime import atomic_json
from .topk_diagnostic import AUX_NAMES, METRICS, SCORE_NAMES, SCHEMES, DEFAULT_REWARD_WEIGHTS, atomic_npz


WINDOWS = ('all', 'first100', 'last100', 'terminal')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, indices):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite summary statistic')
    mean = float(values.mean())
    if len(values) < 2:
        return dict(mean=mean, low=None, high=None)
    low, high = np.quantile(values[indices].mean(1), [.025, .975])
    return dict(mean=mean, low=float(low), high=float(high))


def window_slices(steps):
    return (slice(None), slice(0, min(100, steps)),
            slice(max(0, steps-100), steps), slice(steps-1, steps))


def _group_average(metrics):
    # Input [time, teacher/constant/random repeats, scheme, metric].
    return np.stack([metrics[:, 0], metrics[:, 1], metrics[:, 2:].mean(1)]).transpose(0, 2, 1, 3)


def trajectory_directory(out, provenance=None):
    provenance = provenance or json.loads((Path(out)/'provenance.json').read_text())
    return Path(provenance['trajectory_source'])/'trajectories' if 'trajectory_source' in provenance else Path(out)/'trajectories'


def build_report(output):
    out = Path(output)
    provenance = json.loads((out/'provenance.json').read_text())
    cfg = provenance.get('audit_config', provenance['source_config'])['env']
    capacity, num_users = cfg['capacity'], cfg['num_users']
    reward_weights = np.asarray(provenance.get('audit_reward_weights', DEFAULT_REWARD_WEIGHTS), np.float64)
    seeds = provenance['settings']['scenes']
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    scene_rows, aux_rows, effect_rows, conditioned = [], [], [], []
    scene_values, scene_effects, curves, aux_curves = [], [], [], []
    loss_samples = {name: [] for name in SCORE_NAMES}
    solver_times = []
    severe = None
    for scene in seeds:
        with np.load(out/'audit'/f'scene_{scene}.npz') as d:
            if 'audit_capacity' in d and int(d['audit_capacity'])!=capacity:
                raise ValueError('Saved audit capacity differs from report config')
            metrics = d['metrics']
            saved_weights = d['reward_weights'] if 'reward_weights' in d else DEFAULT_REWARD_WEIGHTS
            np.testing.assert_array_equal(saved_weights, reward_weights)
            raw = metrics[..., [METRICS.index(k) for k in ('coverage','tau','robustness','disconnection')]]
            contributions = metrics[..., 6:10]
            np.testing.assert_allclose(contributions, raw * reward_weights, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(contributions.sum(-1), metrics[..., METRICS.index('task_reward')], rtol=1e-12, atol=1e-12)
            if list(d['metric_names']) != list(METRICS):
                raise ValueError('Unknown audit metric schema')
            group = _group_average(metrics)  # [group,scheme,time,metric]
            steps = group.shape[2]
            slices = window_slices(steps)
            windows = np.stack([group[:, :, sl].mean(2) for sl in slices], axis=2)
            scene_values.append(windows)
            curves.append(group)
            auxiliaries = d['aux']
            aux_curves.append(auxiliaries)
            solver_times.append(d['solver_seconds'])
            for g, name in enumerate(SCORE_NAMES):
                selections = [0] if g == 0 else ([1] if g == 1 else list(range(2, metrics.shape[1])))
                loss = metrics[:, selections, 0, METRICS.index('protocol_loss')]
                loss_samples[name].append(loss.ravel())
                for k, scheme in enumerate(SCHEMES):
                    for w, window in enumerate(WINDOWS):
                        row = dict(scene_seed=scene, scoring=name, scheme=scheme, window=window)
                        row.update({key: float(windows[g,k,w,i]) for i,key in enumerate(METRICS)})
                        scene_rows.append(row)
                # Absolute-probability effect, not just net signed differences.
            owners = d['owners']
            effects = np.empty((3, 2, steps, 4))
            for g, choices in enumerate(([0], [1], list(range(2, metrics.shape[1])))):
                for k in range(2):
                    teacher = metrics[:, 0, k]
                    own = owners[:, choices, k]
                    effects[g,k,:,0] = (own != owners[:,0,k,None,:]).any(-1).mean(1)
                    effects[g,k,:,1] = (metrics[:,choices,k,0] != teacher[:,0,None]).mean(1)
                    reward_delta = metrics[:,choices,k,METRICS.index('task_reward')] - teacher[:,None,METRICS.index('task_reward')]
                    effects[g,k,:,2] = (np.abs(reward_delta) > 1e-12).mean(1)
                    effects[g,k,:,3] = np.abs(reward_delta).mean(1)
            scene_effects.append(np.stack([effects[:,:,sl].mean(2) for sl in slices], axis=2))
            for w, sl in zip(WINDOWS, slices):
                row = dict(scene_seed=scene, window=w)
                row.update({name: float(auxiliaries[sl,i].mean()) for i,name in enumerate(AUX_NAMES)})
                row['max_per_uav_candidates'] = float(d['pre_counts'][sl].max())
                aux_rows.append(row)
            strata = {
                f'pre_candidates_max_0_to_{capacity}': d['pre_counts'].max(1) <= capacity,
                f'pre_candidates_max_{capacity+1}_to_{2*capacity}': ((d['pre_counts'].max(1)>capacity) & (d['pre_counts'].max(1)<=2*capacity)),
                f'pre_candidates_max_above_{2*capacity}': d['pre_counts'].max(1) > 2*capacity,
                'no_multi_candidate_ue': d['ue_counts'].max(1) <= 1,
                'has_multi_candidate_ue': d['ue_counts'].max(1) >= 2,
            }
            for stratum, mask in strata.items():
                for g, name in enumerate(SCORE_NAMES):
                    row = dict(scene_seed=scene, stratum=stratum, scoring=name,
                               transitions=int(mask.sum()), protocol_loss_sum=0., fill_gain_sum=0., timing_loss_sum=0.)
                    for key in ('protocol_loss','fill_gain','timing_loss'):
                        row[key+'_sum'] = float(group[g,0,mask,METRICS.index(key)].sum())
                    conditioned.append(row)
            losses = metrics[:,:,0,METRICS.index('protocol_loss')]
            idx = np.unravel_index(losses.argmax(), losses.shape)
            if severe is None or losses[idx] > severe['protocol_loss']:
                severe = dict(scene_seed=scene, step=int(idx[0]+1), scoring_index=int(idx[1]),
                              protocol_loss=float(losses[idx]))
    scene_values = np.stack(scene_values)  # [scene,group,scheme,window,metric]
    scene_effects = np.stack(scene_effects)
    curve_data = np.stack(curves)
    auxiliary_data = np.stack(aux_curves)
    indices = np.random.default_rng(950001).integers(0,len(seeds),(2000,len(seeds)))
    summary = dict(groups={}, differences={}, scoring_effects={},
                   bootstrap=dict(unit='scenario, after averaging time and random-score repeats',
                                  repeats=2000, seed=950001, training_seed_interval=False),
                   scenarios=len(seeds), transitions=len(seeds)*steps,
                   score_random_repeats=provenance['settings']['random_repeats'],
                   source_coverage=float(auxiliary_data[:,:,AUX_NAMES.index('source_served')].mean()/num_users),
                   distributions={}, severe_case=severe, auxiliary={},
                   audit_capacity=capacity, trajectory_capacity=provenance['source_config']['env']['capacity'],
                   link_rate_scale=provenance['source_config']['env']['capacity']/capacity)
    summary['audit_reward_weights'] = reward_weights.tolist()
    stat_rows = []
    for g, name in enumerate(SCORE_NAMES):
        summary['groups'][name] = {}
        summary['scoring_effects'][name] = {}
        for k, scheme in enumerate(SCHEMES):
            summary['groups'][name][scheme] = {}
            summary['scoring_effects'][name][scheme] = {}
            for w, window in enumerate(WINDOWS):
                result = {key:interval(scene_values[:,g,k,w,j],indices) for j,key in enumerate(METRICS)}
                summary['groups'][name][scheme][window] = result
                for metric, value in result.items():
                    stat_rows.append(dict(scoring=name,scheme=scheme,window=window,metric=metric,**value))
                effect_names = ('owner_change_fraction','served_count_change_fraction',
                                'task_reward_change_fraction','task_reward_absolute_delta')
                summary['scoring_effects'][name][scheme][window] = {
                    key: interval(scene_effects[:,g,k,w,j],indices) for j,key in enumerate(effect_names)}
        loss = np.concatenate(loss_samples[name])
        summary['distributions'][name] = dict(
            positive_fraction=float((loss>0).mean()), mean=float(loss.mean()),
            median=float(np.median(loss)), p95=float(np.quantile(loss,.95)), max=float(loss.max()),
            histogram={str(int(k)):int(v) for k,v in zip(*np.unique(loss.astype(int),return_counts=True))})
    comparisons = [
        ('teacher_fill_minus_one',0,1,0,0), ('constant_fill_minus_one',1,1,1,0),
        ('random_fill_minus_one',2,1,2,0), ('constant_minus_teacher_P0',1,0,0,0),
        ('random_minus_teacher_P0',2,0,0,0),
    ]
    for name, a, ka, b, kb in comparisons:
        summary['differences'][name] = {}
        for w, window in enumerate(WINDOWS):
            summary['differences'][name][window] = {
                key:interval(scene_values[:,a,ka,w,j]-scene_values[:,b,kb,w,j],indices)
                for j,key in enumerate(METRICS)}
    for j, name in enumerate(AUX_NAMES):
        summary['auxiliary'][name] = interval(auxiliary_data[:,:,j].mean(1),indices)
    durations = np.concatenate(solver_times)
    summary['solver_seconds'] = dict(zip(('O1','O2','all_scores_and_schemes'),map(float,durations.sum(0))))
    atomic_json(out/'summary.json',summary)
    write_csv(out/'per_scene.csv',scene_rows)
    write_csv(out/'per_scene_auxiliary.csv',aux_rows)
    write_csv(out/'statistics.csv',stat_rows)
    write_csv(out/'conditioned_losses.csv',conditioned)
    effect_rows=[]
    for g,name in enumerate(SCORE_NAMES):
        for k,scheme in enumerate(SCHEMES):
            for w,window in enumerate(WINDOWS):
                for metric,stat in summary['scoring_effects'][name][scheme][window].items():
                    effect_rows.append(dict(scoring=name,scheme=scheme,window=window,metric=metric,**stat))
    write_csv(out/'score_effects.csv',effect_rows)
    atomic_npz(out/'figure_data.npz',scene_seeds=np.array(seeds),
               curves=curve_data, auxiliary=auxiliary_data, scene_statistics=scene_values,
               metrics=np.array(METRICS), scores=np.array(SCORE_NAMES), schemes=np.array(SCHEMES),
               windows=np.array(WINDOWS), scoring_effects=scene_effects)
    plot_overview(figures,curve_data,auxiliary_data,summary,num_users,capacity)
    for scene in (10000,10001):
        if scene in seeds:
            topology(out,scene,0,[t for t in (150,200,400) if t<=steps],f'topology_{scene}',cfg)
    if severe and severe['protocol_loss']>0:
        topology(out,severe['scene_seed'],severe['scoring_index'],[severe['step']],
                 'topology_selected_worst_protocol_loss',cfg)
    write_report(out,provenance,summary,num_users)
    if 'comparison_reference' in provenance:
        compare_capacities(Path(provenance['comparison_reference']),out)
    return summary


def compare_capacities(reference,out):
    """Scenario-paired capacity contrast on exactly the same recorded positions."""
    with np.load(reference/'figure_data.npz') as a, np.load(out/'figure_data.npz') as b:
        np.testing.assert_array_equal(a['scene_seeds'],b['scene_seeds'])
        np.testing.assert_array_equal(a['metrics'],b['metrics'])
        baseline=a['scene_statistics'].copy();changed=b['scene_statistics'].copy()
        source_curve=a['curves'].copy();new_curve=b['curves'].copy()
    old=json.loads((reference/'summary.json').read_text())
    new=json.loads((out/'summary.json').read_text())
    p=json.loads((out/'provenance.json').read_text())
    cfg=p['audit_config']['env'];m=cfg['num_users']
    n=len(baseline)
    indices=np.random.default_rng(950001).integers(0,n,(2000,n))
    idx=METRICS.index
    comparison=dict(reference=str(reference),same_physical_trajectories=True,
                    old_capacity=p['source_config']['env']['capacity'],new_capacity=cfg['capacity'],
                    link_rate_scale=new['link_rate_scale'],groups={},
                    scope='fixed capacity-20 teacher trajectories; not a capacity-10 trained policy')
    rows=[]
    for g,name in enumerate(SCORE_NAMES):
        comparison['groups'][name]={}
        for k,scheme in enumerate(('P0','P1','O1','O2')):
            comparison['groups'][name][scheme]={}
            for w,window in enumerate(WINDOWS):
                if k<2:
                    deltas={key:changed[:,g,k,w,idx(key)]-baseline[:,g,k,w,idx(key)]
                            for key in ('coverage','task_reward','throughput_bps','robustness',
                                        'protocol_loss','fill_gain','timing_loss','feasibility_loss')}
                else:
                    keys=['coverage','protocol_loss']+(['timing_loss'] if k==3 else [])
                    x=lambda v: v[:,g,0,w,idx('coverage')]+sum(v[:,g,0,w,idx(key)]/m for key in keys[1:])
                    deltas={'coverage':x(changed)-x(baseline)}
                statistics={key:interval(value,indices) for key,value in deltas.items()}
                comparison['groups'][name][scheme][window]=statistics
                for key,stat in statistics.items():
                    rows.append(dict(scoring=name,scheme=scheme,window=window,metric=key,**stat))
    atomic_json(out/'comparison_capacity20.json',comparison)
    write_csv(out/'comparison_capacity20.csv',rows)
    def coverage(s,g,k):
        stat=s['groups'][g]['P0']['all']
        if k<2:return s['groups'][g][SCHEMES[k]]['all']['coverage']['mean']
        return stat['coverage']['mean']+stat['protocol_loss']['mean']/m+(stat['timing_loss']['mean']/m if k==3 else 0)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for g,name in enumerate(SCORE_NAMES):
        for d,s,label,color in ((-.17,old,'Capacity 20',COLORS[0]),(.17,new,'Capacity 10',COLORS[1])):
            axes[g].bar(np.arange(4)+d,[coverage(s,name,k)*100 for k in range(4)],width=.34,label=label,color=color)
        axes[g].set(xticks=range(4),xticklabels=('P0','P1','O1','O2'),title=name)
    axes[0].set_ylabel('Mean coverage (%)');axes[-1].legend(fontsize=7)
    save_figure(fig,out/'figures','capacity20_vs_capacity10')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for g,name in enumerate(SCORE_NAMES):
        for curves,ls,label in ((source_curve,'--','N=20'),(new_curve,'-','N=10')):
            x=curves[:,g,0].mean(0)
            axes[0].plot(np.arange(1,len(x)+1),x[:,idx('protocol_loss')]/m*100,
                         color=COLORS[g],ls=ls,label=f'{name} {label}')
            axes[1].plot(np.arange(1,len(x)+1),x[:,idx('fill_gain')]/m*100,color=COLORS[g],ls=ls)
    axes[0].set(title='Protocol loss',xlabel='Post-action step',ylabel='Coverage gap (pp)');axes[0].legend(fontsize=6)
    axes[1].set(title='Refill recovery',xlabel='Post-action step',ylabel='Coverage gain (pp)')
    save_figure(fig,out/'figures','capacity_protocol_loss_comparison')
    lines=['# 容量10与容量20：同一物理轨迹的配对诊断','',
           '两组使用完全相同的已保存位置、运动、SINR、发现与回传状态，随机评分也逐边配对。容量10不执行actor，原容量20教师的历史与关联只作为轨迹来源；本结果不是容量10训练结果。','',
           '## 覆盖与补位','',
           '| 评分 | 容量 | P0覆盖 | P1覆盖 | O1上界 | O2上界 | 协议损失（人/步） | 补位恢复（人/步） |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in SCORE_NAMES:
        for cap,s in ((20,old),(cfg['capacity'],new)):
            values=[coverage(s,name,k)*100 for k in range(4)]
            stat=s['groups'][name]['P0']['all']
            lines.append(f"| {name} | {cap} | "+' | '.join(f'{v:.5f}%' for v in values)+f" | {stat['protocol_loss']['mean']:.5f} | {stat['fill_gain']['mean']:.5f} |")
    lines+=['','## 竞争与评分作用','']
    for cap,s in ((20,old),(cfg['capacity'],new)):
        e=s['scoring_effects']['random']['P0']['all']
        lines.append(f"- 容量{cap}：每机候选超容量比例{s['auxiliary']['over_capacity_fraction']['mean']*100:.5f}%；随机评分改变服务人数的转移比例{e['served_count_change_fraction']['mean']*100:.5f}%；随机评分与教师评分完整奖励平均绝对差{e['task_reward_absolute_delta']['mean']:.7f}。")
    lines+=['','## 配对场景差值（容量10减容量20）','',
            '| 评分 | P0覆盖差（百分点） | 95%场景区间 | 补位覆盖收益：容量20 / 容量10（百分点） |',
            '|---|---:|---:|---:|']
    for name in SCORE_NAMES:
        v=comparison['groups'][name]['P0']['all']['coverage']
        ci='不适用' if v['low'] is None else f"[{v['low']*100:.5f}, {v['high']*100:.5f}]"
        recover=lambda s:s['groups'][name]['P0']['all']['fill_gain']['mean']/m*100
        lines.append(f"| {name} | {v['mean']*100:.5f} | {ci} | {recover(old):.5f} / {recover(new):.5f} |")
    teacher=new['groups']['teacher']['P0']['all']
    lines+=['','## 解释边界','',
            f"容量10的教师评分协议损失为{teacher['protocol_loss']['mean']:.6f}人/步，补位恢复{teacher['fill_gain']['mean']:.6f}人/步；评分时序缺口为{teacher['timing_loss']['mean']:.6f}人/步，物理与容量残余缺口为{teacher['feasibility_loss']['mean']:.6f}人/步。应同时比较绝对规模，不能只看相对容量20的倍数。",'',
            '固定总带宽20 MHz、总功率30 dBm时，容量从20降为10使每UE带宽、功率与噪声同步翻倍。因此同布局SINR保持不变，候选图和随机评分可严格配对，但合法链路速率翻倍。完整任务回报差同时包含资源与接纳容量的变化，不能全部归因于关联协议；覆盖指标不受这一速率缩放直接影响。','',
            'O2是当前物理图与容量下的集中式覆盖人数上界，不是完整奖励最优值。这里没有评估容量10重新训练后的布局分布，也没有提供闭环补位收益证据。','',
            '场景区间采用2000次配对bootstrap、种子950001，不是训练种子置信区间。原容量20诊断文件与轨迹保持不变，完整重建使用本目录provenance中的来源路径。']
    (out/'comparison_capacity20.md').write_text('\n'.join(lines)+'\n')


def plot_overview(figures, curves, aux, summary, m, capacity=20):
    style()
    mean = curves.mean(0)
    time = np.arange(1,mean.shape[2]+1)
    idx = METRICS.index
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for g,name in enumerate(SCORE_NAMES):
        for k,label in enumerate(('P0: one-shot','P1: refill')):
            axes[g].plot(time,mean[g,k,:,idx('coverage')]*100,label=label,color=COLORS[k])
        p0=mean[g,0]
        known=p0[:,idx('served')]+p0[:,idx('protocol_loss')]
        full=known+p0[:,idx('timing_loss')]
        axes[g].plot(time,known/m*100,label='O1: known-edge maximum',color=COLORS[2],ls='--')
        axes[g].plot(time,full/m*100,label='O2: physical maximum',color=COLORS[3],ls=':')
        axes[g].set(title=name,xlabel='Post-action step')
    axes[0].set_ylabel('Service coverage (%)');axes[-1].legend(fontsize=6)
    save_figure(fig,figures,'coverage_and_upper_bounds')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    p0=mean[0,0]
    keys=('protocol_loss','timing_loss','feasibility_loss')
    axes[0].stackplot(time,*[p0[:,idx(k)]/m*100 for k in keys],labels=keys,colors=COLORS[:3])
    axes[0].set(xlabel='Post-action step',ylabel='Unserved UE (% of all UE)',title='Teacher scores: deficit decomposition')
    axes[0].legend(fontsize=7)
    for g,name in enumerate(SCORE_NAMES):
        axes[1].plot(time,mean[g,0,:,idx('protocol_loss')]/m*100,color=COLORS[g],label=name+' protocol')
        axes[1].plot(time,mean[g,0,:,idx('fill_gain')]/m*100,color=COLORS[g],ls='--',label=name+' recovered')
    axes[1].set(xlabel='Post-action step',ylabel='Coverage difference (pp)',title='Protocol loss and refill recovery')
    axes[1].legend(fontsize=6)
    save_figure(fig,figures,'coverage_deficit_decomposition')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for g,name in enumerate(SCORE_NAMES):
        dist=summary['distributions'][name]
        x=np.array([int(k) for k in dist['histogram']]);y=np.array(list(dist['histogram'].values()))
        axes[0].plot(x,y/y.sum()*100,'o-',label=name,ms=3,color=COLORS[g])
        stat=summary['groups'][name]['P0']['all']
        total=stat['protocol_loss']['mean'];recovered=stat['fill_gain']['mean']
        axes[1].bar(g,recovered,color=COLORS[2])
        axes[1].bar(g,total-recovered,bottom=recovered,color=COLORS[1])
        effects=summary['scoring_effects'][name]['P0']['all']
        for j,key in enumerate(('owner_change_fraction','served_count_change_fraction','task_reward_change_fraction')):
            axes[2].bar(g+(j-1)*.22,effects[key]['mean']*100,width=.22,color=COLORS[j],
                        label=key.replace('_fraction','') if g==0 else None)
    axes[0].set(xlabel='Avoidable unserved UE',ylabel='Frequency (%)',title='P0 protocol loss distribution');axes[0].legend(fontsize=7)
    axes[1].set(xticks=range(3),xticklabels=SCORE_NAMES,ylabel='UE per transition',title='Recovered (green) / remaining (red)')
    axes[2].set(xticks=range(3),xticklabels=SCORE_NAMES,ylabel='Transitions (%)',title='Score changes relative to teacher');axes[2].legend(fontsize=6)
    save_figure(fig,figures,'protocol_loss_and_scoring_effects')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for j,key in enumerate(('throughput_bps','robustness','task_reward')):
        scale=1e-6 if key=='throughput_bps' else 1
        for g,name in enumerate(SCORE_NAMES):
            for k,ls in enumerate(('-','--')):
                axes[j].plot(time,mean[g,k,:,idx(key)]*scale,color=COLORS[g],ls=ls,label=f'{name} {SCHEMES[k]}')
        axes[j].set(xlabel='Post-action step',ylabel='Mbps' if j==0 else key)
    axes[-1].legend(fontsize=6)
    save_figure(fig,figures,'task_metrics_and_refill')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for key in ('over_capacity_fraction','multi_candidate_fraction','one_candidate_fraction','no_candidate_fraction'):
        axes[0].plot(time,aux[:,:,AUX_NAMES.index(key)].mean(0)*100,label=key)
    axes[0].set(xlabel='Post-action step',ylabel='Fraction (%)');axes[0].legend(fontsize=6)
    rows=[]
    with (figures.parent/'conditioned_losses.csv').open() as f:
        rows=list(csv.DictReader(f))
    labels=(f'pre_candidates_max_0_to_{capacity}',f'pre_candidates_max_{capacity+1}_to_{2*capacity}',f'pre_candidates_max_above_{2*capacity}',
            'no_multi_candidate_ue','has_multi_candidate_ue')
    for g,name in enumerate(SCORE_NAMES):
        heights=[]
        for label in labels:
            selected=[r for r in rows if r['scoring']==name and r['stratum']==label]
            n=sum(int(r['transitions']) for r in selected)
            heights.append(sum(float(r['protocol_loss_sum']) for r in selected)/n if n else np.nan)
        axes[1].bar(np.arange(5)+(g-1)*.24,heights,width=.24,label=name,color=COLORS[g])
    axes[1].set(xticks=range(5),xticklabels=(f'max<={capacity}',f'{capacity}<max<={2*capacity}',f'max>{2*capacity}','no competition','competition'),
                ylabel='Avoidable unserved UE',title='Conditional protocol loss')
    axes[1].tick_params(axis='x',labelrotation=25);axes[1].legend(fontsize=7)
    save_figure(fig,figures,'candidate_competition_and_conditioned_losses')


def topology(out,scene,score_index,times,name,cfg):
    if not times:
        return
    with np.load(trajectory_directory(out)/f'scene_{scene}.npz') as f:
        d={k:f[k] for k in ('uav_xy','user_xy','A','B','sensed','Y')}
    with np.load(out/'audit'/f'scene_{scene}.npz') as f:
        owners=f['owners'][:,score_index].copy();optima=f['optima'].copy()
        proposals=f['proposals'][:,score_index].copy()
        label=str(f['score_names'][score_index])
    style()
    fig,axes=plt.subplots(len(times),4,figsize=(14,3.4*len(times)),squeeze=False)
    bs=np.asarray(cfg['bs_positions'])[:,:2]
    for row,t in enumerate(times):
        xy,users=d['uav_xy'][t],d['user_xy'][t]
        before=d['sensed'][t-1]&(d['Y'][t-1]>=cfg['sinr_threshold_db'])&(d['B'][t-1].min(-1)[:,None]<=cfg['num_uavs'])
        after=d['sensed'][t]&(d['Y'][t]>=cfg['sinr_threshold_db'])&(d['B'][t].min(-1)[:,None]<=cfg['num_uavs'])
        known=before&after
        for col,title in enumerate(('P0: one-shot','P1: refill','O1: known maximum','O2: physical maximum')):
            ax=axes[row,col];own=owners[t-1,col] if col<2 else optima[t-1,col-2]
            for i,j in zip(*np.where(np.triu(d['A'][t],1))):
                ax.plot(*xy[[i,j]].T,color='#939393',lw=.65,zorder=1)
            for i in range(len(xy)):
                for g,p in enumerate(bs):
                    if np.linalg.norm(np.r_[xy[i]-p,cfg['altitude']])<=cfg['ug_range']:
                        ax.plot([xy[i,0],p[0]],[xy[i,1],p[1]],color='#009E73',lw=.6,ls='--',zorder=1)
            # Show the actual valid proposal edges in P0; other views show the
            # candidate pool they can draw from. All plotted at the same positions.
            edges=(proposals[t-1]&after) if col==0 else (known if col<3 else after)
            for i,j in zip(*np.where(edges)):
                ax.plot([xy[i,0],users[j,0]],[xy[i,1],users[j,1]],color='#cccccc',lw=.35,alpha=.5,zorder=0)
            served=own>=0
            ax.scatter(*users[~served].T,s=9,c='#d55e00',marker='x',label='Unserved UE',zorder=3)
            ax.scatter(*users[served].T,s=8,c='#56b4e9',label='Served UE',zorder=3)
            for j in np.flatnonzero(served):
                i=own[j]
                ax.plot([xy[i,0],users[j,0]],[xy[i,1],users[j,1]],color=COLORS[i%len(COLORS)],lw=.5,alpha=.65)
            reach=d['B'][t].min(-1)<=cfg['num_uavs']
            ax.scatter(*xy[reach].T,s=28,c='#0072b2',marker='^',edgecolors='k',linewidths=.3,zorder=4)
            ax.scatter(*xy[~reach].T,s=28,c='#aaaaaa',marker='^',edgecolors='k',linewidths=.3,zorder=4)
            ax.scatter(*bs.T,s=65,c='black',marker='s',zorder=5)
            counts=np.bincount(own[served],minlength=len(xy))
            for i,p in enumerate(xy):
                ax.text(p[0]+15,p[1]+20,f'{i+1}:{counts[i]}',fontsize=5)
            ax.set(xlim=(-70,cfg['width']+70),ylim=(-70,cfg['height']+70),aspect='equal',
                   title=f'{title} | t={t} | served={served.sum()}')
            if row==len(times)-1:ax.set_xlabel('x (m)')
            if col==0:ax.set_ylabel('y (m)')
    fig.suptitle(f'Scene {scene}, scores={label}; UAV label: ID:served; selected-case figures are illustrative',fontsize=9)
    save_figure(fig,out/'figures',name)


def write_report(out,provenance,s,m):
    audit_cfg=provenance.get('audit_config',provenance['source_config'])['env']
    capacity=audit_cfg['capacity']
    total_capacity=audit_cfg['num_uavs']*capacity
    idle_floor=max(0,1-m/total_capacity)*100
    source_env=provenance['source_config']['env']
    source_weights=(1., source_env['throughput_weight'], source_env['robustness_weight'], -source_env['disconnection_weight'])
    def reward_text(weights):
        terms=[]
        for weight,label in zip(weights,('覆盖','归一化数据率','鲁棒性','断联比例')):
            if weight:
                terms.append(('+' if weight>0 else '−')+(f'{abs(weight):g}×' if abs(weight)!=1 else '')+label)
        return ''.join(terms).lstrip('+') or '0'
    training_reward=reward_text(source_weights)
    audit_reward=reward_text(provenance.get('audit_reward_weights',DEFAULT_REWARD_WEIGHTS))
    source_budget=provenance.get('source_run_env_steps')
    budget_text=f'源运行完成{source_budget:,}步' if source_budget is not None else '源运行预算以来源记录为准'
    def stat(g,k,key,w='all'):
        return s['groups'][g][k][w][key]['mean']
    def fmt_interval(x,scale=1):
        value=f"{x['mean']*scale:.5f}"
        return value if x['low'] is None else value+f" [{x['low']*scale:.5f}, {x['high']*scale:.5f}]"
    lines=[
        '# 固定布局关联诊断报告', '',
        '本实验仅运行冻结策略和固定转移的影子关联，无网络训练、无补位闭环反馈。', '',
        f"源模型：`{provenance['checkpoint']}`；best实际训练步数 **{provenance['checkpoint_env_steps']:,}**，{budget_text}，训练奖励为 **{training_reward}**。", '',
        f"共{s['scenarios']}个验证/诊断场景、{s['transitions']:,}条联合转移；随机评分每转移重复{s['score_random_repeats']}次。教师实际轨迹平均覆盖 **{s['source_coverage']*100:.6f}%**。", '',
        f"本次影子容量 **{capacity} UE/UAV**，原轨迹采集容量 **{s['trajectory_capacity']}**，链路速率按现有资源公式乘以 **{s['link_rate_scale']:g}**。"+
        ('原教师覆盖只用于核对记录来源，不代表新容量策略的表现。' if capacity!=s['trajectory_capacity'] else '轨迹来自该容量真实训练的策略，本轮没有临时更改容量或缩放链路速率。'), '',
        f'使用源模型的原观察、历史、集中关联生成轨迹；影子任务奖励按 **{audit_reward}** 计算。鲁棒性和断联仍单独记录，不会在零权重时计入收益。', '',
        '## 全程结果', '',
        '| 评分 | P0覆盖 | P1覆盖 | O1上界 | O2上界 | 协议损失（人/步） | 补位恢复（人/步） | 损失出现比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for g in SCORE_NAMES:
        p=stat(g,'P0','served');loss=stat(g,'P0','protocol_loss');timing=stat(g,'P0','timing_loss')
        lines.append(f"| {g} | {p/m*100:.4f}% | {stat(g,'P1','coverage')*100:.4f}% | {(p+loss)/m*100:.4f}% | {(p+loss+timing)/m*100:.4f}% | {loss:.6f} | {stat(g,'P0','fill_gain'):.6f} | {s['distributions'][g]['positive_fraction']*100:.4f}% |")
    lines += ['', '## 吞吐与任务收益', '',
              '| 评分 | 协议 | 吞吐Mbps | 覆盖贡献 | 数据率贡献 | 鲁棒性贡献 | 断联贡献 | 每步任务奖励 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in SCORE_NAMES:
        for scheme in SCHEMES:
            values=[stat(g,scheme,'throughput_bps')/1e6]+[stat(g,scheme,k) for k in
                    ('reward_coverage','reward_throughput','reward_robustness','reward_disconnection','task_reward')]
            lines.append(f'| {g} | {scheme} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', f"其中random指标先在每个转移内平均{s['score_random_repeats']}次评分，再按场景/时间汇总；损失分布另外覆盖全部随机重复。O1/O2是覆盖人数上界，不是任务奖励上界。", '',
              '## 配对场景区间', '', '| 比较 | 覆盖变化（百分点；95%场景区间） | 任务奖励变化（每步） |', '|---|---:|---:|']
    for name,windows in s['differences'].items():
        lines.append(f"| {name} | {fmt_interval(windows['all']['coverage'],100)} | {fmt_interval(windows['all']['task_reward'])} |")
    lines += ['', '区间由2000次场景级bootstrap得到（种子950001），不是训练种子置信区间；随机评分重复与时间步不当作独立场景。', '',
              '## 评分是否能改变结果', '', '| 相对教师P0 | 关联归属改变比例 | 服务人数改变比例 | 任务奖励改变比例 | 奖励绝对差（每步） |', '|---|---:|---:|---:|---:|']
    for g in ('constant','random'):
        e=s['scoring_effects'][g]['P0']['all']
        lines.append(f"| {g} | {e['owner_change_fraction']['mean']*100:.4f}% | {e['served_count_change_fraction']['mean']*100:.4f}% | {e['task_reward_change_fraction']['mean']*100:.4f}% | {e['task_reward_absolute_delta']['mean']:.7f} |")
    aux=s['auxiliary']
    lines += ['', '## 候选与容量背景', '',
              f"每机动作前候选人数超过容量{capacity}的概率：**{aux['over_capacity_fraction']['mean']*100:.4f}%**。", '',
              f"同信息候选图中，UE具有至少两个候选UAV的比例：**{aux['multi_candidate_fraction']['mean']*100:.4f}%**；无候选比例：**{aux['no_candidate_fraction']['mean']*100:.4f}%**。", '',
              f'总容量{total_capacity}、UE{m}，即使全覆盖也至少有{idle_floor:.2f}%名额空闲；总空闲率不是协议浪费证据。', '',
              '## 缺口与结论', '']
    for g in SCORE_NAMES:
        p=stat(g,'P0','protocol_loss');f=stat(g,'P0','fill_gain')
        ratio=f/p*100 if p>0 else None
        recovery='不适用（没有协议缺口）' if ratio is None else f'{ratio:.2f}%'
        lines += [f"- **{g}**：协议可避免损失{p:.6f}人/步，补位恢复{f:.6f}，回收比例{recovery}；剩余重分配缺口{stat(g,'P0','reassignment_loss'):.6f}，评分时序缺口{stat(g,'P0','timing_loss'):.6f}，物理与容量残余缺口{stat(g,'P0','feasibility_loss'):.6f}。"]
    lines += ['',
              '协议缺口是同一已评分候选图下的可避免损失；时序缺口来自运动后新增但尚未评分的合法边；残余缺口综合感知、下行、回传、布局及容量，不据此认定某一个物理因素为唯一根因。', '',
              '评分造成的服务归属变化不一定改变服务人数，但可能改变吞吐和故障保留；完整任务判断应结合绝对收益变化，不只看“发生改变”的频率。', '',
              '本结果只适用于冻结教师访问的布局分布；不能推断扩散模型最终性能，也不能把影子补位收益当作闭环部署收益。', '',
              '## 图表与重建', '',
              '- `figures/coverage_and_upper_bounds.png`：三种评分的P0/P1与两个覆盖上界。',
              '- `figures/coverage_deficit_decomposition.png`：未覆盖缺口与补位恢复。',
              '- `figures/protocol_loss_and_scoring_effects.png`：损失分布及评分影响。',
              '- `figures/task_metrics_and_refill.png`：吞吐、鲁棒性诊断及本次任务奖励。',
              '- `figures/candidate_competition_and_conditioned_losses.png`：竞争程度及条件损失。',
              '- `figures/topology_10000.png`、`topology_10001.png`：预先固定场景，动作后150/200/400步。',
              '- `figures/topology_selected_worst_protocol_loss.png`：仅在观察到协议损失时生成，明确为挑选的严重案例。', '',
              '所有图输出PDF/SVG/PNG，数值来源保存在逐场景audit、CSV、summary.json及figure_data.npz。`--rebuild`只读取保存数据，不加载策略或重新采样。', '',
              'P0/P1的protocol_loss字段分别指各自相对O1的缺口；fill_gain与reassignment_loss是从原P0出发的转移级诊断，两行保持相同，避免解释为额外执行。', '',
              '## 验收与来源', '',
              '完整来源、哈希、运行耗时、内存与验证状态分别见 `provenance.json`、`execution.json`。数学约束在每条转移及每次评分上检查；求解结果按场景原子保存，可从已保存轨迹继续审计。',
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
