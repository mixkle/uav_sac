"""Continue an offline critic from its final checkpoint, preserving Adam and RNG."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import torch
from compare_critic_optimizers import file_hash, model_state, flatten_metrics
from uav_research.critic_calibration import (actor_digest, load_dataset,
    restore_calibration_checkpoint, fit_critic_epoch, predict_critic, calibration_metrics)
from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, atomic_json, atomic_save, rng_state
from uav_research.plotting import plt, style, save_figure, COLORS


def checkpoint(path, policy, experiment, epoch, score, generator, best_epoch, best_score):
    atomic_save(path, dict(kind='critic_calibration_extension', epoch=epoch,
        experiment=experiment, selection_rmse=score, **model_state(policy),
        value_optimizer=copy.deepcopy(policy.value_opt.state_dict()), rng=rng_state(),
        permutation_rng=copy.deepcopy(generator.bit_generator.state),
        historical_best_epoch=best_epoch, historical_best_score=best_score))


def publish(output, history, summary, predictions):
    style()
    start, stop = summary['from_epoch'], summary['to_epoch']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, field, title in [(axes[0,0],'raw_rmse','All-time raw RMSE'),
                            (axes[1,0],'final_step_rmse','Final-step raw RMSE')]:
        for split,color in [('fit',COLORS[0]),('selection',COLORS[1])]:
            ax.plot([int(r['epoch']) for r in history], [float(r[f'{split}_{field}']) for r in history],
                    color=color,label=split)
        ax.set(xlabel='Total critic fit epochs',ylabel=title,yscale='log')
        ax.legend()
    axes[0,1].plot([int(r['epoch']) for r in history[1:]],
                   [float(r['normalized_huber_loss']) for r in history[1:]],color=COLORS[0])
    axes[0,1].set(xlabel='Total critic fit epochs',ylabel='Normalized Huber loss',yscale='log')
    axes[1,1].plot([int(r['epoch']) for r in history],
                   [float(r['selection_raw_rmse']) for r in history],color=COLORS[1],label='Selection all-time RMSE')
    axes[1,1].axhline(float(history[start]['selection_raw_rmse']),color='gray',linestyle=':',label=f'Epoch {start}')
    axes[1,1].set(xlabel='Total critic fit epochs',ylabel='Selection raw RMSE',xlim=(start,stop))
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.axvline(start,color='gray',linestyle='--',alpha=.6)
    fig.suptitle('LR 1e-4 | same Adam, weights, targets and permutation stream')
    save_figure(fig,output/'figures','learning_curves')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,split,title in zip(axes,('fit','holdout','holdout'),('Fitting scenes','Diagnostic scenes','Diagnostic: final 20 steps')):
        d=predictions[split]
        ax.plot(d['returns'].mean(0),color=COLORS[1],label='Realized return')
        ax.plot(d['before'].mean(0),color=COLORS[0],label=f'Epoch {start}')
        ax.plot(d['last'].mean(0),color=COLORS[2],label=f'Epoch {stop}')
        ax.plot(d['selected'].mean(0),color=COLORS[3],linestyle='--',
                label=f"Selected epoch {summary['selected_epoch']}")
        ax.set(xlabel='State time',ylabel='Remaining discounted return',title=title)
        ax.legend(fontsize=8)
    axes[2].set_xlim(380,399)
    save_figure(fig,output/'figures','horizon_comparison')
    text='# Critic 1e-4延长拟合实验\n\n'
    text+=(f"从第{start}轮critic_last.pt继续到第{stop}轮，新增{summary['added_optimizer_updates']}次更新。"
           '完整恢复权重、Adam一阶/二阶矩及步数、固定PopArt统计和随机状态；不重置Adam，不重估归一化。'
           'actor冻结，奖励与完整回合回报不变。\n\n')
    text+=('早期检查点未保存独立NumPy排列器状态；按原种子920001和每轮一次完整排列的规则，'
           f'重放前{start}次排列以恢复后续批次序列。新检查点直接保存permutation_rng。'
           '续训前全部三个集合的预测与第60轮保存数组逐项一致。\n\n')
    text+=('学习率1e-4、batch512，每轮50次更新；Huber delta10，梯度范数裁剪0.5；'
           '无value clipping、无bootstrap、无早停。每轮计算拟合与选模指标，以保持原有选模规则；'
           '每20轮记录进度和保存恢复检查点，诊断集仅在延长拟合前后计算。'
           '选模继续保留前60轮的历史最佳，按选模集整体原始RMSE选择，不依据诊断集选模。\n\n')
    text+='| 模型 | 轮次 | 拟合RMSE | 选模RMSE | 诊断RMSE | 诊断最后20步RMSE | 诊断最后一步RMSE | 诊断末步V均值 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for stage,label,epoch in [('before','续训起点',start),('last','末轮',stop),('selected','全程选中模型',summary['selected_epoch'])]:
        m=summary[stage];h=m['holdout']
        text+=(f"| {label} | {epoch} | {m['fit']['raw_rmse']:.4f} | {m['selection']['raw_rmse']:.4f} | "
               f"{h['raw_rmse']:.4f} | {h['time_bins']['last20']['rmse']:.4f} | {h['final_step_rmse']:.4f} | {h['final_step_prediction_mean']:.4f} |\n")
    text+=(f"\n诊断末步实际奖励均值{summary['last']['holdout']['final_step_return_mean']:.6f}，"
           f"单步奖励保守上界{summary['single_step_reward_upper_bound']:.6f}。"
           f"第{stop}轮末步预测超上界比例{summary['last']['holdout']['final_step_upper_bound_violations']:.1%}。\n\n")
    text+='## 结果解释\n\n'
    improvements={k:100*(1-summary['last'][k]['raw_rmse']/summary['before'][k]['raw_rmse'])
                  for k in ('fit','selection','holdout')}
    text+=(f"第{start}至{stop}轮，拟合集整体RMSE降低{improvements['fit']:.2f}%，"
           f"选模集降低{improvements['selection']:.2f}%，诊断集降低{improvements['holdout']:.2f}%。"
           '继续优化有收益，但独立场景的改善远小于训练场景，末期物理尺度仍未校准。'
           f"拟合集末步RMSE从{summary['before']['fit']['final_step_rmse']:.3f}降至"
           f"{summary['last']['fit']['final_step_rmse']:.3f}；诊断末步RMSE仍为"
           f"{summary['last']['holdout']['final_step_rmse']:.3f}。\n\n")
    for lo,hi in ((stop-59,stop-40),(stop-19,stop)):
        block=[r for r in history if lo<=int(r['epoch'])<=hi]
        if block:
            mean=lambda key:float(np.mean([float(r[key]) for r in block]))
            text+=(f"第{lo}–{hi}轮平均：训练Loss {mean('normalized_huber_loss'):.6f}，"
                   f"选模整体RMSE {mean('selection_raw_rmse'):.4f}，"
                   f"选模末步RMSE {mean('selection_final_step_rmse'):.4f}。\n\n")
    text+=(f"原规则选中的全程最佳仍为第{summary['selected_epoch']}轮。"
           '从后期窗口看，独立场景误差改善已经较慢，末期误差没有同步显著下降。'
           '因此本次延长训练没有解决价值高估；不能把大误差只归因于最初60轮训练不够，'
           '也不能宣称已证明严格收敛或无限延长一定无效。'
           '下一步可优先对照显式剩余时间表达或增加独立拟合场景，分别控制变量；无需据此改变奖励。\n\n')
    text+='| 轮次 | 归一化训练Loss | 拟合RMSE | 选模RMSE | 选模末步RMSE |\n|---|---:|---:|---:|---:|\n'
    for r in history:
        epoch=int(r['epoch'])
        if epoch>=start and (epoch%20==0 or epoch==stop):
            text+=(f"| {epoch} | {float(r['normalized_huber_loss']):.6f} | {float(r['fit_raw_rmse']):.4f} | "
                   f"{float(r['selection_raw_rmse']):.4f} | {float(r['selection_final_step_rmse']):.4f} |\n")
    text+='\n![拟合和选模曲线](figures/learning_curves.png)\n\n![价值形状](figures/horizon_comparison.png)\n\n'
    text+=('该诊断集已用于前期分析，只作为开发诊断；不能据此宣称新的独立测试或多种子统计结论。'
           '300轮是否足够需看误差趋势；固定预算末轮仍有误差，不能推出无限延长也必然无效。'
           '逐轮完整日志见history.csv，各阶段分时段指标见summary.json，预测数组包含before/last/selected/returns。\n\n')
    text+=(f"本次新增运行耗时{summary['elapsed_seconds']/60:.2f}分钟，峰值RSS {summary['peak_rss_mib']:.1f} MiB。"
           f"Adam步数从{summary['initial_adam_step']}推进至{summary['final_adam_step']}。"
           '原actor、检查点、训练数据和归一化统计的校验均通过。\n')
    (output/'report.md').write_text(text)
    rows=[]
    for stage,epoch in [('before',start),('last',stop),('selected',summary['selected_epoch'])]:
        for split,metrics in summary[stage].items():
            row=dict(stage=stage,epoch=epoch,split=split)
            row.update({k:v for k,v in metrics.items() if k!='time_bins'})
            row.update({f'{key}_rmse':v['rmse'] for key,v in metrics['time_bins'].items()})
            rows.append(row)
    with (output/'metrics_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_optimizer_controls/keep_adam_1e-4'))
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_1e4_extended_300'))
    parser.add_argument('--to-epoch',type=int,default=300)
    args=parser.parse_args();previous=args.input.resolve();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use an empty output directory')
    exp=json.loads((previous/'experiment.json').read_text())
    source=Path(exp['source_checkpoint']);data_root=Path(exp['data_root'])
    input_files={p.name:file_hash(p) for p in previous.iterdir() if p.is_file()}
    assert file_hash(source)==exp['source_sha256']
    cfg=load_config(source.parent/'config.yaml');device=configure_runtime(cfg)
    assert cfg.as_dict()==config_from_dict(exp['config']).as_dict()
    datasets={k:load_dataset(data_root/'data'/k) for k in ('fit','selection','holdout')}
    for split,checks in exp['dataset_sha256'].items():
        assert all(file_hash(data_root/'data'/split/name)==sha for name,sha in checks.items())
    loaded=torch.load(previous/'critic_last.pt',map_location='cpu',weights_only=False)
    start=int(loaded['epoch']);stop=args.to_epoch
    if not stop>start:raise ValueError('Target epoch must exceed resumed epoch')
    _,policy=policy_from_checkpoint(source,device)
    for p in policy.actor.parameters():p.requires_grad_(False)
    policy.actor.eval()
    generator=restore_calibration_checkpoint(policy,loaded,datasets['fit']['returns'].size)
    assert policy.value_opt.param_groups[0]['lr']==1e-4
    initial_step=int(next(iter(policy.value_opt.state.values()))['step'].item())
    actor_sha=actor_digest(policy.actor);normalizer_sha=actor_digest(policy.normalizer)
    assert actor_sha==exp['actor_sha256']
    for state in policy.value_opt.state.values():assert int(state['step'].item())==initial_step
    # Verify the complete imported Adam state, not just its step counter.
    actual_opt=policy.value_opt.state_dict()
    for key,state in loaded['value_optimizer']['state'].items():
        for field,value in state.items():assert torch.equal(actual_opt['state'][key][field],value)
    before_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
    for k,p in before_preds.items():
        with np.load(previous/f'{k}_predictions.npz') as saved:np.testing.assert_array_equal(p,saved['last'])
    bound=json.loads((data_root/'summary.json').read_text())['single_step_reward_upper_bound']
    before={k:calibration_metrics(before_preds[k],d['returns'],bound) for k,d in datasets.items()}
    old_best=torch.load(previous/'critic_best.pt',map_location='cpu',weights_only=False)
    best_epoch=int(old_best['epoch']);best_score=float(old_best['selection_rmse'])
    history=list(csv.DictReader((previous/'history.csv').open()))
    assert len(history)==start+1 and int(history[-1]['epoch'])==start
    offset=float(history[-1]['elapsed_seconds'])
    output.mkdir(parents=True)
    experiment=dict(exp,continuation_from=str(previous/'critic_last.pt'),continuation_sha256=input_files['critic_last.pt'],
                    from_epoch=start,to_epoch=stop,epochs=stop,permutation_restore='replay completed permutations' if 'permutation_rng' not in loaded else 'saved state',
                    command=f'.venv/bin/python scripts/extend_critic_calibration.py --input {previous} --output {output} --to-epoch {stop}')
    atomic_json(output/'experiment.json',experiment)
    manifest=dict(experiment,status='running',initial_adam_step=initial_step,input_file_sha256=input_files)
    atomic_json(output/'manifest.json',manifest)
    shutil.copyfile(previous/'critic_best.pt',output/'critic_best.pt')
    restored_digest=actor_digest(policy.value)
    atomic_json(output/'resume_audit.json',dict(all_predictions_match=True,all_adam_tensors_match=True,
        value_sha256=restored_digest,normalizer_sha256=normalizer_sha,initial_adam_step=initial_step,
        permutation_rng=copy.deepcopy(generator.bit_generator.state),learning_rate=1e-4))
    del loaded,old_best
    stamp=time.perf_counter()
    for epoch in range(start+1,stop+1):
        update=fit_critic_epoch(policy,datasets['fit'],512,generator)
        assert update['optimizer_steps']==50
        metrics={k:calibration_metrics(predict_critic(policy,datasets[k]),datasets[k]['returns'],bound) for k in ('fit','selection')}
        score=metrics['selection']['raw_rmse']
        history.append(flatten_metrics(epoch,offset+time.perf_counter()-stamp,metrics,update))
        if score<best_score-1e-4:
            best_score=score;best_epoch=epoch
            checkpoint(output/'critic_best.pt',policy,experiment,epoch,score,generator,best_epoch,best_score)
        with (output/'history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        if epoch==start+1 or epoch%20==0 or epoch==stop:
            print(f"Epoch {epoch}/{stop}: loss={update['normalized_huber_loss']:.6f}, fit={metrics['fit']['raw_rmse']:.4f}, "
                  f"selection={score:.4f}, selection last-step={metrics['selection']['final_step_rmse']:.4f}, global best={best_epoch}",flush=True)
            checkpoint(output/'critic_last.pt',policy,experiment,epoch,score,generator,best_epoch,best_score)
            atomic_json(output/'progress.json',dict(epoch=epoch,metrics=metrics,selected_epoch=best_epoch,
                        elapsed_seconds=time.perf_counter()-stamp))
    final_step=int(next(iter(policy.value_opt.state.values()))['step'].item())
    assert final_step==initial_step+50*(stop-start)
    assert actor_digest(policy.actor)==actor_sha and actor_digest(policy.normalizer)==normalizer_sha
    last_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
    last={k:calibration_metrics(last_preds[k],d['returns'],bound) for k,d in datasets.items()}
    selected_checkpoint=torch.load(output/'critic_best.pt',map_location='cpu',weights_only=False)
    for key in ('value','normalizer'):getattr(policy,key).load_state_dict(selected_checkpoint[key])
    selected_preds={k:predict_critic(policy,d) for k,d in datasets.items()}
    selected={k:calibration_metrics(selected_preds[k],d['returns'],bound) for k,d in datasets.items()}
    predictions={}
    for k,d in datasets.items():
        predictions[k]=dict(before=before_preds[k],last=last_preds[k],selected=selected_preds[k],returns=d['returns'])
        np.savez_compressed(output/f'{k}_predictions.npz',**predictions[k])
    assert file_hash(source)==exp['source_sha256']
    assert all(file_hash(previous/name)==sha for name,sha in input_files.items())
    for split,checks in exp['dataset_sha256'].items():
        assert all(file_hash(data_root/'data'/split/name)==sha for name,sha in checks.items())
    summary=dict(from_epoch=start,to_epoch=stop,added_optimizer_updates=(stop-start)*50,
        selected_epoch=best_epoch,before=before,last=last,selected=selected,
        initial_adam_step=initial_step,final_adam_step=final_step,
        single_step_reward_upper_bound=bound,elapsed_seconds=time.perf_counter()-stamp,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        actor_unchanged=True,normalizer_unchanged=True,source_files_unchanged=True,datasets_unchanged=True)
    atomic_json(output/'summary.json',summary)
    publish(output,history,summary,predictions)
    atomic_json(output/'manifest.json',dict(manifest,**summary,status='completed'))
    print('Report:',output/'report.md',flush=True)


if __name__=='__main__':main()
