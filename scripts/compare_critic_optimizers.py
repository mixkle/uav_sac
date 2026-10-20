"""Two controlled offline critic experiments using existing complete episodes."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from uav_research.critic_calibration import (actor_digest, load_dataset,
    rebase_normalizer_from_fit_targets, configure_calibration_optimizer,
    fit_critic_epoch, predict_critic, calibration_metrics)
from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, atomic_json, atomic_save, seed_all, rng_state
from uav_research.plotting import plt, style, save_figure, COLORS


ARMS = [('keep_adam_5e-4', False, 5e-4), ('reset_adam_5e-4', True, 5e-4),
        ('keep_adam_1e-4', False, 1e-4)]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_state(policy):
    return {name: {k: v.detach().cpu().clone() for k, v in getattr(policy, name).state_dict().items()}
            for name in ('value', 'normalizer')}


def save_model(path, policy, experiment, epoch, metric):
    atomic_save(path, dict(kind='critic_calibration_optimizer_control', epoch=epoch,
                          experiment=experiment, selection_rmse=metric, **model_state(policy),
                          value_optimizer=copy.deepcopy(policy.value_opt.state_dict()), rng=rng_state()))


def flatten_metrics(epoch, elapsed, metrics, updates=None):
    row = dict(epoch=epoch, elapsed_seconds=elapsed)
    if updates:
        row.update(updates)
    for split, m in metrics.items():
        for key in ('raw_rmse', 'raw_mae', 'raw_bias', 'final_step_rmse',
                    'final_step_prediction_mean', 'final_step_upper_bound_violations'):
            row[f'{split}_{key}'] = m[key]
        for name, values in m['time_bins'].items():
            row[f'{split}_{name}_rmse'] = values['rmse']
    return row


def draw(output, summaries):
    style()
    groups = [('Adam history', ARMS[:2]), ('Learning rate (Adam retained)', [ARMS[0], ARMS[2]])]
    for title, arms in groups:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, (name, _, _) in enumerate(arms):
            rows = list(csv.DictReader((output/name/'history.csv').open()))
            for ax, key, ylabel in zip(axes,
                    ['fit_raw_rmse', 'selection_raw_rmse', 'selection_final_step_rmse'],
                    ['Fit: all-time RMSE', 'Selection: all-time RMSE', 'Selection: final-step RMSE']):
                ax.plot([int(r['epoch']) for r in rows], [float(r[key]) for r in rows],
                        color=COLORS[i], label=name)
                ax.set(xlabel='Epoch (50 updates each)', ylabel=ylabel, yscale='log')
                ax.legend(fontsize=8)
        fig.suptitle(title)
        save_figure(fig, output/'figures', 'adam_history' if title=='Adam history' else 'learning_rate')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    reference = np.load(output/'time_reference.npz')['prediction']
    target = np.load(output/ARMS[0][0]/'holdout_predictions.npz')['returns'].mean(0)
    for ax in axes:
        ax.plot(target, label='Realized return', color=COLORS[1], linewidth=2)
        ax.plot(reference, label='Fit time-mean reference', color=COLORS[3], linestyle='--')
        for i, (name, _, _) in enumerate(ARMS):
            pred = np.load(output/name/'holdout_predictions.npz')['selected'].mean(0)
            ax.plot(pred, label=name, color=[COLORS[0], COLORS[2], COLORS[4]][i])
        ax.set(xlabel='State time', ylabel='Remaining discounted return')
    axes[0].set_title('Selected critics | same 32 diagnostic scenes')
    axes[0].legend(fontsize=8)
    axes[1].set(xlim=(380,399), title='Final 20 steps')
    save_figure(fig, output/'figures', 'horizon_comparison')


def report(output, manifest, summaries, reference):
    text = '# Critic：Adam 历史与学习率的两组受控实验\n\n'
    text += ('actor 冻结，复用上轮完整400步回合、奖励和折扣回报。第一组固定5e-4，'
             '仅比较保留/重置Adam；第二组固定保留Adam，仅比较5e-4/1e-4。共享对照，合计三个设置。\n\n')
    text += ('所有设置从同一源critic权重开始，使用相同PopArt重标定、数据排列和batch。'
             f"每个设置完整运行{manifest['epochs']}轮，每轮50次更新，无早停；按独立选模集全时段原始RMSE选择模型（包含epoch0），另报告最后模型。\n\n")
    text += ('网络6292→256→256→1；Huber delta10、batch512、梯度裁剪0.5；无value clipping或bootstrap。'
             '归一化只用拟合集目标，更新中固定。拟合64场景、选模16场景、诊断32场景，种子和动作流均与上轮相同。'
             '这里的32场景已用于之前分析，属于开发诊断集，不能再作为全新的最终测试集。\n\n')
    text += '| 设置 | 选中轮次 | 拟合RMSE | 选模RMSE | 诊断RMSE | 最后20步RMSE | 最后一步RMSE | 最后一步V均值 | 超单步上界比例 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for name, _, _ in ARMS:
        s=summaries[name];h=s['selected']['holdout']
        text += (f"| {name} | {s['selected_epoch']} | {s['selected']['fit']['raw_rmse']:.3f} | "
                 f"{s['selected']['selection']['raw_rmse']:.3f} | {h['raw_rmse']:.3f} | "
                 f"{h['time_bins']['last20']['rmse']:.3f} | {h['final_step_rmse']:.3f} | "
                 f"{h['final_step_prediction_mean']:.3f} | {h['final_step_upper_bound_violations']:.1%} |\n")
    text += f"\n| 第{manifest['epochs']}轮模型 | 拟合RMSE | 选模RMSE | 诊断RMSE | 诊断最后一步RMSE |\n|---|---:|---:|---:|---:|\n"
    for name, _, _ in ARMS:
        s=summaries[name]['last']
        text += f"| {name} | {s['fit']['raw_rmse']:.3f} | {s['selection']['raw_rmse']:.3f} | {s['holdout']['raw_rmse']:.3f} | {s['holdout']['final_step_rmse']:.3f} |\n"
    text += (f"\n诊断集最后一步实际回报均值：{reference['actual_last_reward_mean']:.6f}。"
             f"只按时间取拟合集平均回报的参照：整体RMSE {reference['metrics']['raw_rmse']:.3f}，"
             f"最后一步RMSE {reference['metrics']['final_step_rmse']:.3f}。此参照不使用诊断标签拟合。\n\n")
    base=summaries['keep_adam_5e-4']['selected']['holdout']
    reset=summaries['reset_adam_5e-4']['selected']['holdout']
    lower=summaries['keep_adam_1e-4']['selected']['holdout']
    text += '## 本次结果解释\n\n'
    text += (f"重置Adam：按选模集选中的模型，诊断整体RMSE由{base['raw_rmse']:.3f}变为{reset['raw_rmse']:.3f}，"
             f"变化{(reset['raw_rmse']/base['raw_rmse']-1)*100:+.2f}%；"
             f"但最后一步RMSE由{base['final_step_rmse']:.3f}变为{reset['final_step_rmse']:.3f}。"
             '本次重置历史没有解决末期校准，因此“只要清除旧Adam状态就能修复”的解释不成立；不能据此说Adam历史完全没有影响。\n\n')
    text += (f"降低学习率（保留Adam）：整体RMSE由{base['raw_rmse']:.3f}变为{lower['raw_rmse']:.3f}，"
             f"变化{(lower['raw_rmse']/base['raw_rmse']-1)*100:+.2f}%；"
             f"最后一步RMSE却由{base['final_step_rmse']:.3f}变为{lower['final_step_rmse']:.3f}。"
             '因此在本次数据上，较小学习率有有限的整体改善，并未解决有限回合的末期形状。整体选模指标与末期正确性必须同时检查。\n\n')
    text += '各设置完整预算后的拟合/泛化检查：\n\n| 设置 | 末轮拟合最后一步RMSE | 末轮诊断最后一步RMSE | 后30轮拟合整体RMSE的标准差 |\n|---|---:|---:|---:|\n'
    for name, _, _ in ARMS:
        s=summaries[name]
        rows=list(csv.DictReader((output/name/'history.csv').open()))
        spread=float(np.std([float(r['fit_raw_rmse']) for r in rows[1:][-30:]]))
        text += f"| {name} | {s['last']['fit']['final_step_rmse']:.3f} | {s['last']['holdout']['final_step_rmse']:.3f} | {spread:.3f} |\n"
    text += ('\n这里的标准差描述各轮误差波动，不是训练种子置信区间。'
             '拟合集整体误差降得很低、末期拟合也比上轮13轮实验更好，但独立场景末期仍有几十的误差；'
             '在这组固定数据下，证据更集中于跨场景泛化与可复用时间规律的学习，而非仅有更新次数不足。'
             '时间均值参照仍显著优于三个网络；输入已有t/400，问题不是缺少时间列。\n\n')
    text += ('这组实验未证明某一网络结构是唯一根因，未检验其他学习率或“重置Adam+较小学习率”的组合。'
             '下一步建议继续固定actor、奖励和数据，单独比较原网络与显式时间分支/剩余时长参数化，'
             '同时报告拟合集和选模集的全时段、末期误差，再考虑分时段采样或损失权重。'
             '本次没有将校准权重用于在线续训，部署和覆盖表现不会因critic单独更新而改变。\n\n')
    text += ('![Adam对照](figures/adam_history.png)\n\n![学习率对照](figures/learning_rate.png)\n\n'
             '![剩余价值](figures/horizon_comparison.png)\n\n')
    text += ('仅比较一次固定数据/排列下的离线拟合，不能推导多种子统计优势或在线部署收益；'
             '学习率结论条件为“保留Adam”，本次未评估重置Adam与较小学习率的交互。'
             '完整指标见各设置summary.json、每轮指标history.csv；选中和末轮权重分别保存。'
             'actor、源检查点、数据哈希和共同初始权重的校验见suite_manifest.json及各experiment.json。\n')
    if manifest.get('status')=='completed':
        text += (f"\n总用时{manifest['elapsed_seconds']/60:.2f}分钟，进程峰值RSS {manifest['peak_rss_mib']:.1f} MiB。"
                 f"共享对照与上轮拟合历史最大RMSE差{manifest['shared_baseline_max_history_error']:.2e}。"
                 'actor、源检查点及原始数据的校验通过。\n')
    (output/'report.md').write_text(text)
    comparison=[]
    for name, _, _ in ARMS:
        for stage in ('selected','last'):
            s=summaries[name]
            for split, metrics in s[stage].items():
                row=dict(arm=name,stage=stage,split=split,
                         epoch=s['selected_epoch'] if stage=='selected' else s['epochs_completed'])
                row.update({k:v for k,v in metrics.items() if k!='time_bins'})
                row.update({f'{key}_rmse':v['rmse'] for key,v in metrics['time_bins'].items()})
                comparison.append(row)
    with (output/'metrics_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_calibration'))
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_optimizer_controls'))
    parser.add_argument('--epochs',type=int,default=60)
    args=parser.parse_args()
    if args.epochs<=0: raise ValueError('epochs must be positive')
    root=args.data_root.resolve();output=args.output.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty; existing results will not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    exp=json.loads((root/'experiment.json').read_text());old=json.loads((root/'summary.json').read_text())
    source=Path(exp['source_checkpoint']);cfg=load_config(source.parent/'config.yaml');device=configure_runtime(cfg)
    assert file_hash(source)==exp['source_sha256']
    datasets={k:load_dataset(root/'data'/k) for k in exp['splits']}
    hashes={}
    for name,d in datasets.items():
        assert d['metadata']['scene_seeds']==exp['splits'][name]['scene_seeds']
        assert d['metadata']['action_seed']==exp['splits'][name]['action_seed']
        assert d['metadata']['actor_sha256']==exp['actor_sha256']
        assert d['metadata']['discount']==cfg.algo.discount
        assert d['metadata']['horizon']==cfg.env.horizon
        assert d['metadata']['state_dim']==cfg.env.state_dim
        hashes[name]={p.name:file_hash(p) for p in sorted((root/'data'/name).iterdir()) if p.is_file()}
    assert cfg.env.__dict__==config_from_dict(exp['config']).env.__dict__, 'Physical config changed since collection'
    bound=old['single_step_reward_upper_bound']
    manifest=dict(status='running',data_root=str(root),source_checkpoint=str(source),
        source_sha256=exp['source_sha256'],actor_sha256=exp['actor_sha256'],dataset_sha256=hashes,
        groups={'adam_history':['keep_adam_5e-4','reset_adam_5e-4'],
                'learning_rate':['keep_adam_5e-4','keep_adam_1e-4']},
        epochs=args.epochs,batch_size=512,updates_per_epoch=50,early_stopping=False,
        selection_metric='selection all-time raw RMSE; epoch 0 included',permutation_seed=920001,
        learning_rate_comparison_adam='retained',config=cfg.as_dict(),
        command=f'.venv/bin/python scripts/compare_critic_optimizers.py --data-root {root} --output {output} --epochs {args.epochs}')
    atomic_json(output/'suite_manifest.json',manifest)
    summaries={};common_initial=None;stamp=time.perf_counter()
    for name,reset,lr in ARMS:
        folder=output/name;folder.mkdir()
        _,policy=policy_from_checkpoint(source,device)
        assert actor_digest(policy.actor)==exp['actor_sha256']
        for p in policy.actor.parameters():p.requires_grad_(False)
        policy.actor.eval()
        normalization=rebase_normalizer_from_fit_targets(policy,datasets['fit']['returns'])
        initial_sha=actor_digest(policy.value)+actor_digest(policy.normalizer)
        if common_initial is None:common_initial=initial_sha
        assert initial_sha==common_initial, 'Unequal initial weights/statistics'
        opt=configure_calibration_optimizer(policy,reset_adam=reset,learning_rate=lr)
        assert (opt['initialized_parameter_states']==0)==reset
        arm_exp=dict(**manifest,arm=name,optimizer=opt,normalization=normalization,initial_value_and_normalizer_digest=initial_sha)
        atomic_json(folder/'experiment.json',arm_exp)
        seed_all(920001);rng=np.random.default_rng(920001);arm_start=time.perf_counter()
        metrics={k:calibration_metrics(predict_critic(policy,datasets[k]),datasets[k]['returns'],bound) for k in ('fit','selection')}
        history=[flatten_metrics(0,0,metrics,dict(normalized_huber_loss=0.,value_grad_norm=0.,optimizer_steps=0))]
        best_score=metrics['selection']['raw_rmse'];best_epoch=0
        save_model(folder/'critic_best.pt',policy,arm_exp,0,best_score)
        for epoch in range(1,args.epochs+1):
            updates=fit_critic_epoch(policy,datasets['fit'],512,rng)
            metrics={k:calibration_metrics(predict_critic(policy,datasets[k]),datasets[k]['returns'],bound) for k in ('fit','selection')}
            history.append(flatten_metrics(epoch,time.perf_counter()-arm_start,metrics,updates))
            score=metrics['selection']['raw_rmse']
            if score<best_score-1e-4:
                best_score=score;best_epoch=epoch
                save_model(folder/'critic_best.pt',policy,arm_exp,epoch,score)
            with (folder/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            print(f"{name} epoch {epoch:02d}/{args.epochs}: fit={metrics['fit']['raw_rmse']:.4f}, "
                  f"selection={score:.4f}, last-step={metrics['selection']['final_step_rmse']:.4f}, best={best_epoch}",flush=True)
        assert actor_digest(policy.actor)==exp['actor_sha256']
        last_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
        last_metrics={k:calibration_metrics(last_preds[k],d['returns'],bound) for k,d in datasets.items()}
        save_model(folder/'critic_last.pt',policy,arm_exp,args.epochs,last_metrics['selection']['raw_rmse'])
        best=torch.load(folder/'critic_best.pt',map_location='cpu',weights_only=False)
        for key in ('value','normalizer'):getattr(policy,key).load_state_dict(best[key])
        best_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
        selected={k:calibration_metrics(best_preds[k],d['returns'],bound) for k,d in datasets.items()}
        for k,d in datasets.items():
            np.savez_compressed(folder/f'{k}_predictions.npz',selected=best_preds[k],last=last_preds[k],returns=d['returns'])
        summary=dict(selected_epoch=best_epoch,epochs_completed=args.epochs,optimizer_updates=args.epochs*50,
                     selected=selected,last=last_metrics,elapsed_seconds=time.perf_counter()-arm_start,
                     actor_unchanged=True,initial_digest=initial_sha,optimizer=opt)
        atomic_json(folder/'summary.json',summary);summaries[name]=summary
        del policy,best,last_preds,best_preds
    # The common arm must reproduce the earlier calibration over its shared prefix.
    prior=list(csv.DictReader((root/'fit_history.csv').open()))
    current=list(csv.DictReader((output/ARMS[0][0]/'history.csv').open()))
    errors=[abs(float(a['selection_rmse'])-float(b['selection_raw_rmse'])) for a,b in zip(prior,current)]
    assert max(errors)<1e-5, 'Shared baseline no longer reproduces previous fitting history'
    assert file_hash(source)==exp['source_sha256']
    for name,checks in hashes.items():
        assert all(file_hash(root/'data'/name/p)==sha for p,sha in checks.items())
    curve=datasets['fit']['returns'].mean(0)
    ref=calibration_metrics(np.broadcast_to(curve,datasets['holdout']['returns'].shape),datasets['holdout']['returns'],bound)
    np.savez_compressed(output/'time_reference.npz',prediction=curve)
    reference=dict(metrics=ref,actual_last_reward_mean=float(datasets['holdout']['returns'][:,-1].mean()))
    atomic_json(output/'summary.json',dict(arms=summaries,time_reference=reference))
    draw(output,summaries);report(output,manifest,summaries,reference)
    manifest.update(status='completed',actor_unchanged=True,source_checkpoint_unchanged=True,datasets_unchanged=True,
                    common_initial_digest=common_initial,shared_baseline_max_history_error=max(errors),
                    elapsed_seconds=time.perf_counter()-stamp,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(output/'suite_manifest.json',manifest)
    print('Report:',output/'report.md',flush=True)


if __name__=='__main__':main()
