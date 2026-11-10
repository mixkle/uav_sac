"""Saved-data interpretation of matched K-step bootstrap and holding benefits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_mappo_action_blocks import FIELDS
from check_mappo_action_values import paired_interval
from compare_online_time_factor import write_csv
from uav_research.plotting import COLORS,plt,save_figure,style


def main(output):
    out=Path(output);meta=json.loads((out/'experiment.json').read_text())
    result=json.loads((out/'summary.json').read_text());assert meta['status']=='completed'
    rows=result['paired_scores'];errors=[];release=[]
    sources=[Path(__file__),out/'experiment.json',out/'summary.json']
    def choose(model,t,length,mode,metric):
        return next(r for r in rows if (r['model'],r['time'],r['length'],r['mode'],r['subset'],r['metric'])==
                    (model,t,length,mode,'good',metric))
    for model in meta['groups']:
        gamma=meta['groups'][model]['config']['algo']['discount']
        for at,records in meta['records'].items():
            t=int(at);mask=np.array([r['good'] for r in records])
            for length in [1,5,20]:
                path=Path(meta['baseline_source'])/model/f't{t}/curves.npz' if length==1 else out/model/f'K{length}/t{t}/curves.npz'
                sources.append(path)
                with np.load(path) as data:
                    diagnostics=data['diagnostics']
                    for i,mode in [(1,'half'),(2,'hover')]:
                        error=choose(model,t,length,mode,'boundary_error')
                        mc=choose(model,t,length,mode,'mc');boot=choose(model,t,length,mode,'bootstrap')
                        errors.append(dict(model=model,time=t,length=length,mode=mode,
                            bootstrap_minus_mc=error['mean']*gamma**length,
                            low=error['low']*gamma**length,high=error['high']*gamma**length,
                            nested_low=error['nested_low']*gamma**length,nested_high=error['nested_high']*gamma**length,
                            mc_gain_clear=bool(mc['nested_low']>0),
                            mc_loss_clear=bool(mc['nested_high']<0),
                            bootstrap_gain_clear=bool(boot['nested_low']>0),
                            bootstrap_loss_clear=bool(boot['nested_high']<0)))
                        delta=(diagnostics[i,...,FIELDS.index('actual_speed_mean')]-
                               diagnostics[0,...,FIELDS.index('actual_speed_mean')])[...,length:length+20].mean(-1)
                        values=paired_interval(delta[:,mask])
                        release.append(dict(model=model,time=t,length=length,mode=mode,
                            mean_speed_after_release=float(diagnostics[i,...,length:length+20,FIELDS.index('actual_speed_mean')][:,mask].mean()),
                            speed_difference=values['mean'],speed_difference_low=values['nested_low'],speed_difference_high=values['nested_high'],
                            release_coverage_delta_pp=100*choose(model,t,length,mode,'release_coverage')['mean'],
                            next20_coverage_delta_pp=100*choose(model,t,length,mode,'post_release20_coverage')['mean'],
                            last100_coverage_delta_pp=100*choose(model,t,length,mode,'last100_coverage')['mean'],
                            final_coverage_delta_pp=100*choose(model,t,length,mode,'final_coverage')['mean']))
                        np.testing.assert_allclose(mc['mean'],choose(model,t,length,mode,'prefix')['mean']+
                                                   choose(model,t,length,mode,'tail_mc')['mean'],atol=1e-10)
                        np.testing.assert_allclose(boot['mean'],choose(model,t,length,mode,'prefix')['mean']+
                                                   choose(model,t,length,mode,'tail_prediction')['mean'],atol=1e-10)
    write_csv(out/'bootstrap_errors.csv',errors);write_csv(out/'release_behavior.csv',release)
    style();fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ri,model in enumerate(meta['groups']):
        for ci,at in enumerate(meta['records']):
            t=int(at);ax=axes[ri,ci]
            for j,(key,label) in enumerate([('prefix','Observed during hold'),('tail_mc','MC after release'),('tail_prediction','Critic after release')]):
                selected=[choose(model,t,k,'hover',key) for k in [1,5,20]]
                y=np.array([r['mean'] for r in selected]);lo=np.array([r['nested_low'] for r in selected]);hi=np.array([r['nested_high'] for r in selected])
                ax.errorbar(np.arange(3)+(j-1)*.1,y,yerr=np.stack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),
                            marker='o',capsize=3,label=label,color=COLORS[j])
            ax.axhline(0,color='grey',lw=.8);ax.legend(fontsize=8)
            ax.set(xticks=np.arange(3),xticklabels=['1','5','20'],xlabel='Hold length K (steps)',
                   ylabel='Discounted contribution difference vs policy',title=f'{model}, t{t}; good states')
    save_figure(fig,out/'figures','holding_and_continuation_value')
    lines=['# 连续保持5／20步：结果解读','',
        '来源是固定两个100k模型、原t150/t200布局与8套完全配对的未来随机流。以下主表使用原先预定义的good子集24/28，不按本次结果筛选；模型和训练代码没有更新。','',
        '## 20步悬停：收益与critic判断','',
        '|模型|起点|MC收益[nested区间]|匹配20步bootstrap收益[nested区间]|bootstrap减MC[nested区间]|',
        '|---|---:|---:|---:|---:|']
    for e in errors:
        if e['length']!=20 or e['mode']!='hover':continue
        q=choose(e['model'],e['time'],20,'hover','mc');b=choose(e['model'],e['time'],20,'hover','bootstrap')
        lines.append(f'|{e["model"]}|{e["time"]}|{q["mean"]:+.3f} [{q["nested_low"]:+.3f},{q["nested_high"]:+.3f}]|{b["mean"]:+.3f} [{b["nested_low"]:+.3f},{b["nested_high"]:+.3f}]|{e["bootstrap_minus_mc"]:+.3f} [{e["nested_low"]:+.3f},{e["nested_high"]:+.3f}]|')
    lines+=['','误差区间来自配对残差，而非把两个区间相减。收益明确但预测区间跨零，称为未充分捕捉收益；只有收益与预测各自区间均不跨零且反向时，才有更强的反向排序线索。','',
        '## 保持期间与恢复后的收益','',
        '|模型|起点|K|保持期间实际收益|恢复后实际折扣收益|恢复后critic预测收益|',
        '|---|---:|---:|---:|---:|---:|']
    for model in meta['groups']:
        for at in meta['records']:
            for k in [1,5,20]:
                values=[choose(model,int(at),k,'hover',key)['mean'] for key in ['prefix','tail_mc','tail_prediction']]
                lines.append(f'|{model}|{at}|{k}|'+ '|'.join(f'{v:+.3f}' for v in values)+'|')
    lines+=['','![保持和延续价值](figures/holding_and_continuation_value.png)','',
        '## 时长的直接配对对照','',
        '|模型|起点|悬停时长比较|完整MC差值[nested区间]|','|---|---:|---|---:|']
    for r in result['duration_differences']:
        if r['subset']=='good' and r['mode']=='hover' and r['metric']=='mc':
            lines.append(f'|{r["model"]}|{r["time"]}|{r["comparison"]}|{r["mean"]:+.3f} [{r["nested_low"]:+.3f},{r["nested_high"]:+.3f}]|')
    lines+=['','某个时长显著、另一个不显著，不等于时长差异本身显著；判断时长作用应读取本表的直接配对差值。','',
        '## 解除保持后的部署与运动','',
        '|模型|起点|K|解除后20步实际速度(m/s)|解除时覆盖差(pp)|随后20步覆盖差(pp)|最后100步覆盖差(pp)|终态覆盖差(pp)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in release:
        if r['mode']!='hover' or r['length']==1:continue
        lines.append(f'|{r["model"]}|{r["time"]}|{r["length"]}|{r["mean_speed_after_release"]:.2f}|{r["release_coverage_delta_pp"]:+.2f}|{r["next20_coverage_delta_pp"]:+.2f}|{r["last100_coverage_delta_pp"]:+.2f}|{r["final_coverage_delta_pp"]:+.2f}|')
    lines+=['','本表部署差值为点估计，区间见paired_scores.csv，不因覆盖点估计为正就认定长期改善。解除后是原冻结actor，实验不会让它自动学会悬停。',
            '', '## 结论边界','',
            '匹配的K步量在保持结束处才用原策略V。保持期间记录真实奖励，因此误差可定位到结束状态之后的相对价值，而不是共同初始V偏差。',
            'MC有限、单个训练种子、场景来自研究诊断池。bootstrap区间同时重采场景和未来重复，不是训练种子区间。历史组的快照前缀来自原策略，状态与历史可能偏离自身on-policy分布。',
            'K增大同时改变干预行为及bootstrap时刻；本实验不能把时长效应全部归因于估计器，更不能直接将联合动作方案的优势替换成标准MAPPO单agent优势。',
            '全体32状态、半幅方案、奖励分量、逐状态排序和完整可重建图表见[原始报告](report.md)。']
    (out/'interpretation.md').write_text('\n'.join(lines)+'\n')
    (out/'interpretation_sources.json').write_text(json.dumps(
        {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps([r for r in errors if r['length']==20 and r['mode']=='hover'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='analysis/mappo_central_action_blocks_100k_seed0')
    main(parser.parse_args().output)
