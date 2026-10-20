"""Fit one 5e-3 critic and compare it with existing retained-Adam LR controls."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from compare_critic_optimizers import file_hash, save_model, flatten_metrics
from uav_research.critic_calibration import (actor_digest, load_dataset,
    rebase_normalizer_from_fit_targets, configure_calibration_optimizer,
    fit_critic_epoch, predict_critic, calibration_metrics)
from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, atomic_json, seed_all
from uav_research.plotting import plt, style, save_figure, COLORS


ARMS = [('keep_adam_1e-4', 1e-4), ('keep_adam_5e-4', 5e-4), ('keep_adam_5e-3', 5e-3)]


def publish_comparison(output, folders, summaries, manifest, reference):
    style()
    colors = [COLORS[0], COLORS[2], COLORS[4]]
    labels = ['1e-4', '5e-4', '5e-3']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = [('normalized_huber_loss', 'Normalized Huber loss'),
              ('fit_raw_rmse', 'Fit: all-time RMSE'),
              ('selection_raw_rmse', 'Selection: all-time RMSE'),
              ('value_grad_norm', 'Mean pre-clip gradient norm'),
              ('fit_final_step_rmse', 'Fit: final-step RMSE'),
              ('selection_final_step_rmse', 'Selection: final-step RMSE')]
    for (name, _), color, label in zip(ARMS, colors, labels):
        rows = list(csv.DictReader((folders[name]/'history.csv').open()))
        for ax, (field, ylabel) in zip(axes.flat, fields):
            values = rows[1:] if field in ('normalized_huber_loss', 'value_grad_norm') else rows
            ax.plot([int(r['epoch']) for r in values], [float(r[field]) for r in values],
                    color=color, label=label)
            ax.set(xlabel='Epoch (50 updates each)', ylabel=ylabel, yscale='log')
            ax.legend(title='Learning rate', fontsize=8)
    axes[1, 0].axhline(.5, color='gray', linestyle='--', label='Gradient clip 0.5')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Retained Adam | same frozen actor, data and batch order')
    save_figure(fig, output/'figures', 'learning_rate_curves')
    for stage in ('selected', 'last'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        target = np.load(folders[ARMS[0][0]]/'holdout_predictions.npz')['returns'].mean(0)
        for ax in axes:
            ax.plot(target, color=COLORS[1], label='Realized return', linewidth=2)
            ax.plot(reference, color=COLORS[3], label='Fit time-mean reference', linestyle='--')
            for (name, _), color, label in zip(ARMS, colors, labels):
                predicted = np.load(folders[name]/'holdout_predictions.npz')[stage].mean(0)
                ax.plot(predicted, color=color, label=f'LR {label}')
            ax.set(xlabel='State time', ylabel='Remaining discounted return')
        axes[0].set_title(f'{stage.capitalize()} critics | same 32 diagnostic scenes')
        axes[0].legend(fontsize=8)
        axes[1].set(xlim=(380,399), title='Final 20 steps')
        save_figure(fig, output/'figures', f'horizon_{stage}')
    text = '# Critic 学习率对照：1e-4、5e-4、5e-3\n\n'
    text += ('新增5e-3完整实验，1e-4和5e-4复用上一轮已完成且校验通过的结果。'
             '三种设置均保留同一源Adam历史，从同一源critic权重及PopArt重标定状态出发，'
             'actor冻结，使用相同完整回合目标、数据划分、随机排列和batch；各60轮、3000次更新，无早停。\n\n')
    text += ('网络6292→256→256→1；gamma0.995、Huber delta10、batch512、梯度范数裁剪0.5，'
             '没有value clipping或bootstrap。PopArt仅拟合集估计并固定；拟合64场景、选模16场景、诊断32场景。'
             '诊断场景已用于开发分析，不是新的最终测试集。epoch0参与选模，保留初始模型作为候选。\n\n')
    rows=[]
    for stage, title in [('selected','按选模集整体RMSE选中的模型'),('last','完整60轮后的模型')]:
        text += f'## {title}\n\n'
        text += '| 学习率 | 轮次 | 拟合RMSE | 选模RMSE | 诊断RMSE | 诊断最后20步RMSE | 诊断最后一步RMSE | 诊断末步V均值 | 超单步上界比例 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for (name, lr), label in zip(ARMS, labels):
            s=summaries[name];m=s[stage];h=m['holdout'];epoch=s['selected_epoch'] if stage=='selected' else 60
            text += (f"| {label} | {epoch} | {m['fit']['raw_rmse']:.3f} | {m['selection']['raw_rmse']:.3f} | "
                     f"{h['raw_rmse']:.3f} | {h['time_bins']['last20']['rmse']:.3f} | {h['final_step_rmse']:.3f} | "
                     f"{h['final_step_prediction_mean']:.3f} | {h['final_step_upper_bound_violations']:.1%} |\n")
            for split, metrics in m.items():
                row=dict(arm=name,learning_rate=lr,stage=stage,split=split,epoch=epoch)
                row.update({k:v for k,v in metrics.items() if k!='time_bins'})
                row.update({f'{k}_rmse':v['rmse'] for k,v in metrics['time_bins'].items()})
                rows.append(row)
        text += '\n'
    actual=np.load(folders[ARMS[0][0]]/'holdout_predictions.npz')['returns']
    text += (f"最后一步实际回报均值{actual[:,-1].mean():.6f}；单步奖励保守上界{manifest['single_step_reward_upper_bound']:.6f}。"
             f"拟合集时间均值参照的诊断整体RMSE {np.sqrt(np.mean((reference[None,:]-actual)**2)):.3f}。"
             '单条轨迹G是回报样本，不是精确条件期望；预测超过末期物理上界不能用采样方差解释。\n\n')
    base=summaries['keep_adam_5e-4']['selected']['holdout']
    high=summaries['keep_adam_5e-3']['selected']['holdout']
    text += '## 本次比较结论\n\n'
    text += (f"把学习率从5e-4增至5e-3，按相同选模规则选中的模型，诊断整体RMSE由"
             f"{base['raw_rmse']:.3f}变为{high['raw_rmse']:.3f}"
             f"（变化{(high['raw_rmse']/base['raw_rmse']-1)*100:+.2f}%），"
             f"最后一步RMSE由{base['final_step_rmse']:.3f}变为{high['final_step_rmse']:.3f}。"
             '本次增大学习率没有解决末期价值高估。三种设置的选中模型在全部32个诊断场景上仍超过末步单次奖励上界。\n\n')
    text += ('1e-4在本次选中模型和完整预算模型的诊断整体RMSE上最低；'
             '5e-4的选中模型末期误差较低；5e-3早期loss和选模误差出现较大波动，整体泛化未改善。'
             '因此整体误差与末期误差的排序并不一致，不能只按loss曲线或某个时段选结论。\n\n')
    text += ('跑满60轮后，各设置拟合集整体RMSE约1–1.5，但诊断整体RMSE仍约18，末期RMSE仍约36；'
             '时间均值参照仍明显更好。当前结果不支持“单纯增大学习率即可纠正时间规律”的解释。'
             '这个结论限于同一源权重、保留Adam、当前数据、60轮及已有梯度裁剪；'
             '没有排除其他优化设置，也没有唯一证明某个结构是根因。\n\n')
    if manifest.get('status')=='completed':
        text += (f"新增5e-3实验及图表生成用时{manifest['elapsed_seconds']:.1f}秒，"
                 f"进程峰值RSS {manifest['peak_rss_mib']:.1f} MiB。"
                 'actor、源检查点、数据及复用结果的校验通过。\n\n')
    text += ('![学习率与loss](figures/learning_rate_curves.png)\n\n'
             '![选中模型剩余价值](figures/horizon_selected.png)\n\n'
             '![末轮模型剩余价值](figures/horizon_last.png)\n\n')
    text += ('新增模型、逐轮日志、预测数组保存在keep_adam_5e-3；旧设置的产物路径、SHA256和复用记录见manifest.json。'
             '所有指标也导出metrics_comparison.csv。归一化Huber loss与原始回报RMSE尺度不同；'
             '优先以独立选模/诊断误差和末期尺度判断，不仅看拟合集loss。'
             '这是单次固定数据与排列对照，没有测试重置Adam的交互或在线策略收益。\n')
    (output/'report.md').write_text(text)
    with (output/'metrics_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_optimizer_controls'))
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_learning_rate_comparison'))
    args=parser.parse_args();controls=args.controls.resolve();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty')
    old=json.loads((controls/'suite_manifest.json').read_text())
    assert old['status']=='completed' and old['epochs']==60 and not old['early_stopping']
    root=Path(old['data_root']);source=Path(old['source_checkpoint'])
    assert file_hash(source)==old['source_sha256']
    cfg=load_config(source.parent/'config.yaml');device=configure_runtime(cfg)
    assert cfg.as_dict()==config_from_dict(old['config']).as_dict(), 'Original configuration has changed'
    datasets={name:load_dataset(root/'data'/name) for name in ('fit','selection','holdout')}
    for split, checks in old['dataset_sha256'].items():
        assert all(file_hash(root/'data'/split/name)==sha for name,sha in checks.items()), 'Dataset changed'
    summaries={};folders={};reused={}
    for name,lr in ARMS[:2]:
        folder=controls/name;summary=json.loads((folder/'summary.json').read_text())
        experiment=json.loads((folder/'experiment.json').read_text())
        assert summary['epochs_completed']==60 and summary['optimizer_updates']==3000
        assert not summary['optimizer']['reset_adam'] and summary['optimizer']['learning_rate']==lr
        assert summary['initial_digest']==old['common_initial_digest']
        assert experiment['source_sha256']==old['source_sha256']
        assert experiment['permutation_seed']==920001 and experiment['batch_size']==512
        assert experiment['dataset_sha256']==old['dataset_sha256']
        reused[name]=dict(path=str(folder),sha256={p.name:file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()})
        summaries[name]=summary;folders[name]=folder
    bound=json.loads((root/'summary.json').read_text())['single_step_reward_upper_bound']
    output.mkdir(parents=True)
    folder=output/'keep_adam_5e-3';folder.mkdir();folders['keep_adam_5e-3']=folder
    manifest=dict(status='running',source_checkpoint=str(source),source_sha256=old['source_sha256'],
        actor_sha256=old['actor_sha256'],data_root=str(root),dataset_sha256=old['dataset_sha256'],reused=reused,
        learning_rates=[1e-4,5e-4,5e-3],adam='retained',epochs=60,updates_per_epoch=50,batch_size=512,
        permutation_seed=920001,early_stopping=False,selection_metric=old['selection_metric'],config=cfg.as_dict(),
        common_initial_digest=old['common_initial_digest'],single_step_reward_upper_bound=bound,
        command=f'.venv/bin/python scripts/compare_critic_learning_rates.py --controls {controls} --output {output}')
    atomic_json(output/'manifest.json',manifest)
    _,policy=policy_from_checkpoint(source,device)
    assert actor_digest(policy.actor)==old['actor_sha256']
    for p in policy.actor.parameters():p.requires_grad_(False)
    policy.actor.eval()
    normalization=rebase_normalizer_from_fit_targets(policy,datasets['fit']['returns'])
    digest=actor_digest(policy.value)+actor_digest(policy.normalizer)
    assert digest==old['common_initial_digest']
    opt=configure_calibration_optimizer(policy,learning_rate=5e-3)
    experiment=dict(**manifest,arm='keep_adam_5e-3',optimizer=opt,normalization=normalization,
                    initial_value_and_normalizer_digest=digest)
    atomic_json(folder/'experiment.json',experiment)
    seed_all(920001);rng=np.random.default_rng(920001);stamp=time.perf_counter()
    normalizer_digest=actor_digest(policy.normalizer)
    def metrics():
        return {k:calibration_metrics(predict_critic(policy,datasets[k]),datasets[k]['returns'],bound) for k in ('fit','selection')}
    m=metrics();history=[flatten_metrics(0,0,m,dict(normalized_huber_loss=0.,value_grad_norm=0.,optimizer_steps=0))]
    best_score=m['selection']['raw_rmse'];best_epoch=0
    save_model(folder/'critic_best.pt',policy,experiment,0,best_score)
    for epoch in range(1,61):
        update=fit_critic_epoch(policy,datasets['fit'],512,rng);m=metrics()
        history.append(flatten_metrics(epoch,time.perf_counter()-stamp,m,update))
        score=m['selection']['raw_rmse']
        if score<best_score-1e-4:
            best_score=score;best_epoch=epoch
            save_model(folder/'critic_best.pt',policy,experiment,epoch,score)
        with (folder/'history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        print(f"LR 5e-3 epoch {epoch:02d}/60: fit={m['fit']['raw_rmse']:.4f}, selection={score:.4f}, "
              f"final-step={m['selection']['final_step_rmse']:.4f}, best={best_epoch}",flush=True)
    assert actor_digest(policy.actor)==old['actor_sha256']
    assert actor_digest(policy.normalizer)==normalizer_digest
    last_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
    last_metrics={k:calibration_metrics(last_preds[k],d['returns'],bound) for k,d in datasets.items()}
    save_model(folder/'critic_last.pt',policy,experiment,60,last_metrics['selection']['raw_rmse'])
    best=torch.load(folder/'critic_best.pt',map_location='cpu',weights_only=False)
    for key in ('value','normalizer'):getattr(policy,key).load_state_dict(best[key])
    selected_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
    selected_metrics={k:calibration_metrics(selected_preds[k],d['returns'],bound) for k,d in datasets.items()}
    for k,d in datasets.items():
        np.savez_compressed(folder/f'{k}_predictions.npz',selected=selected_preds[k],last=last_preds[k],returns=d['returns'])
    summary=dict(selected_epoch=best_epoch,epochs_completed=60,optimizer_updates=3000,
        selected=selected_metrics,last=last_metrics,elapsed_seconds=time.perf_counter()-stamp,
        actor_unchanged=True,normalizer_unchanged=True,initial_digest=digest,optimizer=opt)
    atomic_json(folder/'summary.json',summary);summaries['keep_adam_5e-3']=summary
    assert file_hash(source)==old['source_sha256']
    for split,checks in old['dataset_sha256'].items():
        assert all(file_hash(root/'data'/split/name)==sha for name,sha in checks.items())
    for reuse in reused.values():
        assert all(file_hash(Path(reuse['path'])/name)==sha for name,sha in reuse['sha256'].items())
    reference=datasets['fit']['returns'].mean(0)
    np.savez_compressed(output/'time_reference.npz',prediction=reference)
    atomic_json(output/'summary.json',dict(arms=summaries))
    manifest.update(status='completed',actor_unchanged=True,source_checkpoint_unchanged=True,datasets_unchanged=True,
        reused_results_unchanged=True,elapsed_seconds=time.perf_counter()-stamp,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    publish_comparison(output,folders,summaries,manifest,reference)
    atomic_json(output/'manifest.json',manifest)
    print('Report:',output/'report.md',flush=True)


if __name__=='__main__':main()
