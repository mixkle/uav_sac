"""Rebuild reward/precision details of saved one-action value counterfactuals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from check_mappo_action_values import FIELDS, MODES, SOURCE, evidence, paired_interval
from check_mappo_generalization import interval
from compare_online_time_factor import write_csv


def main(output):
    root=Path(output)
    meta=json.loads((root/'experiment.json').read_text())
    result=json.loads((root/'summary.json').read_text())
    assert meta['status']=='completed'
    details,components,baselines,persistence,state_intervals=[],[],[],[],[]
    sources=[Path(__file__),root/'experiment.json',root/'summary.json']
    old_meta=json.loads((SOURCE/'experiment.json').read_text())
    sources.append(SOURCE/'experiment.json')
    old_checks={(r['time'],r['start']):r['identical_ue_path_sha256']
                for r in old_meta['checks'] if r['repeat']==0}
    for check in meta['checks']:
        if check['repeat']==0:
            assert check['ue_path_sha256']==old_checks[(check['time'],check['start'])]
    for label in meta['groups']:
        for at,records in meta['records'].items():
            path=root/label/f't{at}/curves.npz'; sources.append(path)
            mask=np.array([r['good'] for r in records])
            with np.load(path) as data:
                r=data['diagnostics'][...,FIELDS.index('reward')]
                cov=data['diagnostics'][...,FIELDS.index('coverage')]
                discount=float(data['discount'])
                scores=evidence(r,data['values'],discount,float(data['gae_lambda']))
                pred=scores['initial_value'][0].mean(0)[mask]
                label_mc=scores['mc_q'][0].mean(0)[mask]
                baselines.append(dict(model=label,time=int(at),scenes=int(mask.sum()),
                    initial_value_prediction_mean=float(pred.mean()),initial_mc_mean=float(label_mc.mean()),
                    initial_mc_mean_rmse=float(np.sqrt(np.mean((pred-label_mc)**2)))))
                # Reuse the exact first future path of the old all-hover branch.
                # Hover physics/reward do not depend on actor/history: every actual
                # command is zero. UE paths are hash-matched above for both actors.
                old_path=SOURCE/f't{at}/curves.npz';sources.append(old_path)
                with np.load(old_path) as old:
                    old_reward=old['hover_all'][0,...,list(old['fields']).index('task_reward')]
                    if label=='M1-C':
                        np.testing.assert_array_equal(r[0,0],old['policy'][0,...,list(old['fields']).index('task_reward')])
                weights=discount**np.arange(r.shape[-1])
                for mode,rew in [('one_step_hover',r[2,0]),('persistent_hover',old_reward)]:
                    delta=((rew-r[0,0])*weights).sum(-1)[mask]
                    persistence.append(dict(model=label,time=int(at),intervention=mode,
                                            future_repeat=0,scenes=int(mask.sum()),**interval(delta)))
                for mi,mode in enumerate(MODES[1:],1):
                    changes={key:scores[key][mi]-scores[key][0] for key in ['mc_q','td1','gae128']}
                    repeat_count=changes['mc_q'].shape[0]
                    draws=np.random.default_rng(950001).integers(repeat_count,size=(2000,repeat_count))
                    bounds={key:np.quantile(value[draws].mean(1),[.025,.975],axis=0)
                            for key,value in changes.items()}
                    for i,record in enumerate(records):
                        entry=dict(model=label,time=int(at),candidate=mode,scene_seed=record['seed'],good=record['good'])
                        for key,value in changes.items():
                            entry.update({key:float(value[:,i].mean()),key+'_low':float(bounds[key][0,i]),
                                          key+'_high':float(bounds[key][1,i])})
                        for score in ['td1','gae128']:
                            entry[score+'_opposite_nonzero_intervals']=bool(
                                (entry['mc_q_low']>0 and entry[score+'_high']<0) or
                                (entry['mc_q_high']<0 and entry[score+'_low']>0))
                        state_intervals.append(entry)
                    delta_mc=(scores['mc_q'][mi]-scores['mc_q'][0])[:,mask]
                    delta_td=(scores['td1'][mi]-scores['td1'][0])[:,mask]
                    first_coverage=(cov[mi,...,0]-cov[0,...,0])[:,mask]
                    first_reward=(r[mi,...,0]-r[0,...,0])[:,mask]
                    details.append(dict(model=label,time=int(at),candidate=mode,scenes=int(mask.sum()),
                        equal_first_coverage_fraction=float((np.abs(first_coverage)<1e-12).mean()),
                        first_coverage_change_pp=float(first_coverage.mean()*100),
                        first_coverage_absolute_change_pp=float(np.abs(first_coverage).mean()*100),
                        first_reward_change=float(first_reward.mean()),
                        absolute_first_reward_change=float(np.abs(first_reward).mean()),
                        mc_delta_mean=float(delta_mc.mean()),td_delta_mean=float(delta_td.mean()),
                        mean_mc_delta_standard_error=float((delta_mc.std(0,ddof=1)/np.sqrt(len(delta_mc))).mean()),
                        median_absolute_scene_mc_delta=float(np.median(np.abs(delta_mc.mean(0)))),
                        paired_next_prediction_error_rmse=float(np.sqrt(np.mean((delta_td.mean(0)-delta_mc.mean(0))**2)))))
                    for j,name in enumerate(data['contribution_names']):
                        c=data['contributions']
                        delta=((c[mi,...,j]-c[0,...,j])*discount**np.arange(r.shape[-1])).sum(-1)
                        components.append(dict(model=label,time=int(at),candidate=mode,component=str(name),
                                               **paired_interval(delta[:,mask])))
    for filename,rows in [('signal_precision.csv',details),('reward_component_values.csv',components),
                          ('initial_value_accuracy.csv',baselines),('paired_persistence_context.csv',persistence),
                          ('state_repeat_intervals.csv',state_intervals)]:
        write_csv(root/filename,rows)
    (root/'interpretation_sources.json').write_text(json.dumps(
        {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    lines=['# 一步动作价值诊断：信号精度与奖励分解','',
        '全部结果来自已保存的轨迹，不新增采样。首先看MC区间是否足以确定动作排序，再判断TD/GAE是否同向。有限8次配对延续不能当作精确Q标签。','',
        '## 完整任务回报与训练优势差值','',
        '|模型|起点|候选|MC差值[nested区间]|TD1差值|GAE128差值|',
        '|---|---:|---|---:|---:|---:|']
    for label in meta['groups']:
        for at in meta['records']:
            for mode in MODES[1:]:
                selected={r['metric']:r for r in result['paired_scores'] if
                    (r['model'],r['time'],r['subset'],r['candidate'])==(label,int(at),'good',mode)}
                q=selected['mc_q']
                lines.append(f'|{label}|{at}|{mode}|{q["mean"]:+.3f} [{q["nested_low"]:+.3f},{q["nested_high"]:+.3f}]|{selected["td1"]["mean"]:+.3f}|{selected["gae128"]["mean"]:+.3f}|')
    lines+=['','区间同时重采场景与每场景未来重复，是本次单训练种子的诊断区间。只改变第一步，随后恢复原策略；不得直接与持续悬停的回报增益等同。','',
        '## 单步信号与MC精度','',
        '|模型|起点|候选|覆盖完全不变的重复比例|平均单步覆盖差(pp)|MC差值每状态平均标准误|每状态MC差值绝对值中位数|',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in details:
        lines.append(f'|{r["model"]}|{r["time"]}|{r["candidate"]}|{r["equal_first_coverage_fraction"]:.1%}|{r["first_coverage_change_pp"]:+.3f}|{r["mean_mc_delta_standard_error"]:.3f}|{r["median_absolute_scene_mc_delta"]:.3f}|')
    lines+=['','标准误与差值量级接近时，不能仅凭每个状态MC均值的正负判断critic是否正确。覆盖人数不变也不代表完整奖励无反馈：吞吐、鲁棒性和断联贡献仍需分别检查。','',
        '## 逐状态动作排序','',
        '|模型|起点|候选|信号|全部good状态符号一致率|重复区间不跨零的状态数|这些状态的一致率|',
        '|---|---:|---|---|---:|---:|---:|']
    for r in result['ranking']:
        if r['subset']!='good' or r['score']=='gae_full':continue
        reliable=r['sign_agreement_interval_excludes_zero']
        lines.append(f'|{r["model"]}|{r["time"]}|{r["candidate"]}|{r["score"]}|{r["sign_agreement"]:.1%}|{r["repeat_interval_excludes_zero_scenes"]}/{r["scenes"]}|{reliable:.1%}|' if reliable is not None else
                     f'|{r["model"]}|{r["time"]}|{r["candidate"]}|{r["score"]}|{r["sign_agreement"]:.1%}|0/{r["scenes"]}|不适用|')
    lines+=['','区间筛选后的子集有选择偏差；这个描述既不是无偏总体准确率，也不能排除剩余MC噪声。','',
        '## 完整奖励分量的MC差值','',
        '|模型|起点|候选|覆盖贡献|吞吐贡献|鲁棒性贡献|断联贡献|',
        '|---|---:|---|---:|---:|---:|---:|']
    for label in meta['groups']:
        for at in meta['records']:
            for mode in MODES[1:]:
                selected={r['component']:r['mean'] for r in components if
                    (r['model'],r['time'],r['candidate'])==(label,int(at),mode)}
                lines.append(f'|{label}|{at}|{mode}|'+ '|'.join(f'{selected[k]:+.3f}' for k in
                    ['coverage','throughput','robustness','disconnection'])+'|')
    lines+=['','组件均为实际加权后的折扣奖励贡献，其和等于完整MC差值；区间保存在reward_component_values.csv。',
            '', '## 同一首套未来路径：一步与持续悬停','',
            '复用旧实验的全悬停轨迹，仅取与本次第0次重复完全相同的UE路径。全悬停的物理状态/奖励不依赖actor或历史，因此可从相同物理快照对两个模型复用；原组policy轨迹也逐元素核对。这里的区间只重采场景，不涵盖未来重复，不能替代上面的8次MC估计。','',
            '|模型|起点|干预|折扣回报差[场景区间]|','|---|---:|---|---:|']
    for r in persistence:
        lines.append(f'|{r["model"]}|{r["time"]}|{r["intervention"]}|{r["mean"]:+.3f} [{r["low"]:+.3f},{r["high"]:+.3f}]|')
    lines+=['','持续悬停对应不同后续策略，不能用它的收益直接要求当前一步TD或GAE给出相同分数。',
            '', '## 个别相反排序及初始价值误差','',
            '下面按MC与TD1各自的重复区间均不跨零且方向相反筛选案例。这是事后探索性筛选，未做多重比较校正；只能作为复查线索，不代表总体错误率。GAE的区间另存，不将TD1错误等同于真实PPO更新必然错误。','',
            '|模型|起点|场景|候选|MC差值[重复区间]|TD1差值[重复区间]|GAE128差值|',
            '|---|---:|---:|---|---:|---:|---:|']
    for r in state_intervals:
        if not r['good'] or not r['td1_opposite_nonzero_intervals']:continue
        lines.append(f'|{r["model"]}|{r["time"]}|{r["scene_seed"]}|{r["candidate"]}|{r["mc_q"]:+.3f} [{r["mc_q_low"]:+.3f},{r["mc_q_high"]:+.3f}]|{r["td1"]:+.3f} [{r["td1_low"]:+.3f},{r["td1_high"]:+.3f}]|{r["gae128"]:+.3f}|')
    lines+=['','|模型|起点|初始V均值|8次MC均值|跨状态RMSE（对8次MC均值）|','|---|---:|---:|---:|---:|']
    for r in baselines:
        lines.append(f'|{r["model"]}|{r["time"]}|{r["initial_value_prediction_mean"]:.2f}|{r["initial_mc_mean"]:.2f}|{r["initial_mc_mean_rmse"]:.2f}|')
    lines+=['','良好布局子集的critic仍低估累计回报，但共同的V(s)偏差在候选差值中抵消，不能直接用低估幅度解释动作排序。历史组这些布局与历史来自原策略前缀，并非自身on-policy分布。',
            '', '## 本次可支持的结论','',
            '候选方案在全组的MC差值区间均跨零，尚不能证明单步悬停或半幅在好部署上普遍更优，也不能证明critic普遍反向指导。局部存在相反排序的复查线索；GAE与TD1的表现不同，不能跳过GAE及实际PPO更新把全部问题归结为单步V误差。',
            '持续保持的收益已在配对旧轨迹中看到，而仅停一步随后恢复旧策略的收益弱且不确定。这支持下一步检查持续5/20步等短块保持，并用对应K步bootstrap与MC比较；不把持续干预收益错误归到单步判断。',
            '', '图表、定义、来源核对及进一步限制见[主报告](report.md)。']
    (root/'signal_precision.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(details,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='analysis/mappo_central_action_values_100k_seed0')
    main(parser.parse_args().output)
