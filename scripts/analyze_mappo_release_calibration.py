"""Rebuild the focused release-state calibration interpretation from predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from calibrate_mappo_release_values import GROUPS, scene_split
from check_mappo_action_values import RUNS, paired_interval
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from uav_research.runtime import atomic_json
from uav_research.plotting import COLORS, plt, save_figure, style


def analyze(out):
    meta=json.loads((out/'experiment.json').read_text());assert meta['status']=='completed'
    rows=[];learning={};signal=[];style()
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    absolute,aa=plt.subplots(2,3,figsize=(14,8))
    relative,ra=plt.subplots(2,2,figsize=(11,8))
    for li,label in enumerate(RUNS):
        folder=out/label
        with np.load(folder/'labels.npz') as d:data={k:d[k].copy() for k in d.files}
        with np.load(folder/'predictions.npz') as d:preds={k:d[k].copy() for k in d.files}
        with (folder/'learning.csv').open() as f:learning[label]=list(csv.DictReader(f))
        check=scene_split(data['scene_seeds'],meta['protocol'])['check']
        target=data['returns'];truth=(target[:,1:]-target[:,:1])[...,check].mean(2).ravel()
        for ci,ck in enumerate(['source','best','last']):
            delta=(preds[ck][:,1:]-preds[ck][:,:1])[...,check].mean(2).ravel()
            signal.append(dict(model=label,checkpoint=ck,scenes=len(check),
                scene_mean_pair_rmse=float(np.sqrt(np.mean((delta-truth)**2))),
                correlation=float(np.corrcoef(delta,truth)[0,1]),slope=float(np.polyfit(truth,delta,1)[0])))
            x=target[...,check].mean(2).ravel();y=preds[ck][...,check].mean(2).ravel()
            aa[li,ci].scatter(x,y,s=11,alpha=.6)
            lo=min(x.min(),y.min());hi=max(x.max(),y.max())
            aa[li,ci].plot([lo,hi],[lo,hi],'--',color='gray')
            aa[li,ci].set(title=f'{label}: {ck}',xlabel='MC release value (8 prefixes mean)',
                          ylabel='Predicted release value')
            if ck!='best':
                col=0 if ck=='source' else 1
                ra[li,col].scatter(truth,delta,s=15,alpha=.65)
                lo=min(truth.min(),delta.min());hi=max(truth.max(),delta.max())
                ra[li,col].plot([lo,hi],[lo,hi],'--',color='gray')
                ra[li,col].set(title=f'{label}: {ck}',xlabel='MC release-value difference (8 prefixes mean)',
                              ylabel='Predicted release-value difference')
        # Primary diagnostic: K20 hover versus policy, on the unchanged good subset.
        for col,at in enumerate([150,200]):
            gi=GROUPS.index((at,20));ix=check[data['good'][gi,check]]
            tail=data['returns'][gi];prefix=data['prefix'][gi];gk=float(data['gamma'])**20
            truth=gk*(tail[2]-tail[0])[:,ix]
            immediate=(prefix[2]-prefix[0])[:,ix]
            full=truth+immediate
            for ck in ['source','best','last']:
                pred=gk*(preds[ck][gi,2]-preds[ck][gi,0])[:,ix]
                boot=immediate+pred;error=boot-full
                result=dict(model=label,time=at,length=20,mode='hover',subset='check_good',checkpoint=ck,
                    scenes=len(ix),mc=float(full.mean()),bootstrap=float(boot.mean()),
                    tail_mc=float(truth.mean()),tail_prediction=float(pred.mean()))
                for metric,values in [('mc',full),('bootstrap',boot),('bootstrap_error',error)]:
                    interval=paired_interval(values)
                    result.update({f'{metric}_{key}':value for key,value in interval.items()})
                rows.append(result)
            ax=axes[li,col]
            for ci,ck in enumerate(['source','best','last']):
                error=(gk*(preds[ck][gi,2]-preds[ck][gi,0])[:,ix]-truth).mean(0)
                ax.plot(data['scene_seeds'][ix]-610000,error,'o-',color=COLORS[ci],label=ck)
            ax.axhline(0,color='black',lw=.8)
            ax.set(title=f'{label}, t{at}, K20 hover (check good n={len(ix)})',
                   xlabel='Diagnostic scene index',ylabel='K-step bootstrap minus MC')
            ax.legend(fontsize=7)
    save_figure(fig,out/'figures','check_hover_bootstrap_errors')
    save_figure(absolute,out/'figures','check_absolute_release_values')
    save_figure(relative,out/'figures','check_relative_source_last')
    write_csv(out/'check_hover_intervals.csv',rows)
    write_csv(out/'check_relative_signal.csv',signal)
    with (out/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (out/'check_changes.csv').open() as f:changes=list(csv.DictReader(f))
    lines=['# 校准结果解读','',
        '本轮不是重新训练策略。原M1-C与历史M1-C-H的actor均冻结，只校准各自的时间因子critic。来源文件哈希与源模型均未改变。',
        f'总耗时约{meta["elapsed_seconds"]:.1f}秒，完成{meta["reconstruction_checks"]}条向量前缀核对、每模型3072个端点、各300轮/1800次critic更新。','',
        '## 绝对价值与相对价值要分开看','',
        '|模型|版本|拟合绝对RMSE|检查绝对RMSE|拟合配对RMSE|检查配对RMSE|','|---|---|---:|---:|---:|---:|']
    for label in RUNS:
        for ck in ['source','fit_bias_only','best','last']:
            get=lambda split:next(r for r in summary if r['model']==label and r['checkpoint']==ck and r['split']==split)
            fit,check=get('fit'),get('check')
            lines.append(f'|{label}|{ck}|{float(fit["rmse"]):.3f}|{float(check["rmse"]):.3f}|{float(fit["pair_rmse"]):.3f}|{float(check["pair_rmse"]):.3f}|')
    lines+=['','仅增加一个拟合偏差常数也可能显著降低绝对RMSE，但不改变同一起点的价值差。这正是不能只凭平均MC拟合曲线判断动作指导能力的原因。',
        '此处epoch0指已训练100k的源critic、尚未做本次离线校准；不是未训练的随机策略。','',
        '![绝对释放价值](figures/check_absolute_release_values.png)',
        '![相对价值：原模型与校准末轮](figures/check_relative_source_last.png)','',
        '将每个场景/时刻/干预的8次前缀平均后，再比较价值差：','',
        '|模型|版本|场景平均配对RMSE|预测差值与MC差值相关性|线性斜率|','|---|---|---:|---:|---:|']
    for row in signal:
        lines.append(f'|{row["model"]}|{row["checkpoint"]}|{row["scene_mean_pair_rmse"]:.3f}|{row["correlation"]:.3f}|{row["slope"]:.3f}|')
    lines+=['','斜率接近0意味着预测差值相对于MC差值变化很小；相关性仅作描述，同一场景的多个比较不是独立样本。8次前缀会形成不同的结束状态，不是对同一物理端点进行8次未来采样，因此这里仍不能视为无噪声的真实价值。','']
    for label in RUNS:
        logs=learning[label];best=meta['models'][label]['best_epoch']
        first,last=logs[0],logs[-1];minimum=min(logs,key=lambda r:float(r['selection_rmse']))
        lines += [f'**{label}**：按选模集配对误差选出的best为epoch {best}。选模集绝对RMSE最低出现在epoch {minimum["epoch"]}（{float(minimum["selection_rmse"]):.3f}），该时的配对RMSE为{float(minimum["selection_pair_rmse"]):.3f}，原模型为{float(first["selection_pair_rmse"]):.3f}。',
            f'拟合集配对RMSE从{float(first["fit_pair_rmse"]):.3f}变为{float(last["fit_pair_rmse"]):.3f}；选模集从{float(first["selection_pair_rmse"]):.3f}变为{float(last["selection_pair_rmse"]):.3f}。', '']
    lines += ['## 原先最关心的20步悬停','',
        '下面只用本次未参与校准的check场景，并保留原good判定。与上一轮全32场景的均值不能直接混比。误差为bootstrap−MC。区间同时重采场景和场景内的未来重复，仅用于探索性诊断。','',
        '|模型|起点|版本|场景数|MC收益差|bootstrap收益差|误差及95%区间|','|---|---:|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'|{row["model"]}|{row["time"]}|{row["checkpoint"]}|{row["scenes"]}|{row["mc"]:+.3f}|{row["bootstrap"]:+.3f}|{row["bootstrap_error_mean"]:+.3f} [{row["bootstrap_error_nested_low"]:+.3f}, {row["bootstrap_error_nested_high"]:+.3f}]|')
    lines+=['','![检查集20步悬停的逐场景误差](figures/check_hover_bootstrap_errors.png)','',
        '## 这次排除了什么，以及还没排除什么','',
        '- 已排除端点时序或MC标签漏算前后缀的明显实现错误：重放动作/无线诊断/原价值逐项一致，且逐样本满足匹配bootstrap误差恒等式。',
        '- 没有GAE或bootstrap标签污染，也没有value clipping，但是否改善相对价值必须看检查集，不能只看绝对偏差或拟合集。',
        '- 如果拟合集配对误差下降而检查不降，说明当前MLP至少能调整这些差值；问题进一步指向跨场景泛化、可用数据量和带噪MC监督，而非仅仅共同尺度偏差。',
        '- 这不是“已证明GAE没有问题”或“MLP表达能力不够”。MC标签每个端点只有一个未来实现；数据少且高维，旧数据分布和优化设置也在影响结果。历史版critic缺少actor历史的限制仍存在。',
        '- 仅做端点MC监督无法检验PPO如何把联合动作收益分配给单架UAV，也无法证明校准后在线部署会改善。','',
        '下一步应优先扩大独立场景并直接审计相对价值，而不是因为绝对RMSE下降就把这个离线critic接回在线训练。可先做同物理端点的多未来采样来估计MC噪声，再比较等量的普通MC监督与显式配对差值监督；这些都未在本轮自动运行。','',
        '完整指标见[报告](report.md)，配对差值明细见`paired_details.csv`，统计变化见`check_changes.csv`。','']
    (out/'interpretation.md').write_text('\n'.join(lines))
    paths=[Path(__file__),out/'experiment.json',out/'summary.csv',out/'check_changes.csv']
    paths += [out/label/name for label in RUNS for name in ['labels.npz','predictions.npz','learning.csv']]
    atomic_json(out/'interpretation_sources.json',{str(p):digest(p) for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='analysis/mappo_release_mc_calibration_100k_seed0')
    args=parser.parse_args();analyze(Path(args.output))
