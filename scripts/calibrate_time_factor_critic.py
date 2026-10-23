"""Paired offline C-group critic experiment; supports resume and report rebuild."""
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
import yaml

from uav_research.config import load_config
from uav_research.critic_calibration import actor_digest, load_dataset, discounted_episode_returns
from uav_research.critic_time_factor import (
    ARMS, OfflineCritic, horizon_factors, target_statistics, make_optimizer,
    fit_epoch, predict, score, save_offline, restore_offline, paired_bootstrap,
)
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, atomic_json, seed_all
from uav_research.plotting import plt, style, save_figure


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def epoch_row(epoch, elapsed, scores, update=None):
    row = dict(epoch=epoch, elapsed_seconds=elapsed, loss=update['loss'] if update else '',
               gradient_norm=update['gradient_norm'] if update else '',
               optimizer_steps=update['optimizer_steps'] if update else 0,
               permutation_sha256=update['permutation_sha256'] if update else '')
    for split, m in scores.items():
        for scale in ['value','average']:
            for stage in ['all','last20','last1']:
                row[f'{split}_{scale}_{stage}_rmse'] = m[scale]['stages'][stage]['rmse']
    return row


def validate_data(datasets, cfg, actor_sha):
    all_seeds = []
    for name, d in datasets.items():
        meta, states = d['metadata'], d['states']
        assert meta['actor_sha256'] == actor_sha
        assert meta['horizon'] == 400 and meta['state_dim'] == 6292
        assert meta['discount'] == cfg.algo.discount
        assert len(states) == {'fit':64,'selection':16,'holdout':32}[name]
        assert np.asarray(meta['scene_seeds']).tolist() == list(range(
            {'fit':61000,'selection':62000,'holdout':63000}[name],
            {'fit':61000,'selection':62000,'holdout':63000}[name]+len(states)))
        np.testing.assert_allclose(states[:,:,-1], np.broadcast_to(np.arange(400)/400,states.shape[:2]),atol=1e-7)
        for start in range(0,len(states),8):
            assert np.isfinite(states[start:start+8]).all()
        all_seeds.extend(meta['scene_seeds'])
    assert len(all_seeds) == len(set(all_seeds))
    assert not set(all_seeds) & set(cfg.run.validation_seeds+cfg.run.test_seeds)


def publish(out):
    summary = json.loads((out/'summary.json').read_text())
    # Fixed-time references have no between-scene variation at a given time.
    for m in summary['references'].values():
        for scale in ['value','average']:
            m[scale]['time_centered_correlation'] = None
    atomic_json(out/'summary.json',summary)
    figs = out/'figures'
    style()
    fig, axes = plt.subplots(1,3,figsize=(14,3.8))
    for arm in ARMS:
        with (out/arm/'history.csv').open() as f:
            rows = list(csv.DictReader(f))
        epoch = [int(r['epoch']) for r in rows]
        for ax, key in zip(axes, ['fit_value_all_rmse','selection_value_all_rmse','selection_average_all_rmse']):
            ax.semilogy(epoch,[float(r[key]) for r in rows],label=arm)
    for ax, title in zip(axes,['Fit return RMSE','Selection return RMSE','Selection average-coverage RMSE']):
        ax.set(xlabel='Epoch',title=title);ax.legend(fontsize=7)
    save_figure(fig,figs,'learning_curves')
    for split in ['fit','selection','holdout']:
        with np.load(out/f'{split}_predictions.npz') as data:
            t = np.arange(data['returns'].shape[1])
            for stage in ['selected','last']:
                fig,axes=plt.subplots(1,3,figsize=(14,3.8))
                for ax in axes[:2]:ax.plot(t,data['returns'].mean(0),color='black',label='MC return')
                axes[2].plot(t,data['average_labels'].mean(0),color='black',label='MC average coverage')
                for arm in ARMS:
                    v,c=data[f'{arm}_{stage}_value'],data[f'{arm}_{stage}_average']
                    for ax in axes[:2]:ax.plot(t,v.mean(0),label=arm)
                    axes[2].plot(t,c.mean(0),label=arm)
                for key in ['factor_constant','time_profile']:
                    for ax in axes[:2]:ax.plot(t,data[f'{key}_value'].mean(0),'--',alpha=.7,label=key)
                    axes[2].plot(t,data[f'{key}_average'].mean(0),'--',alpha=.7,label=key)
                axes[1].set(xlim=(380,399))
                for ax,title in zip(axes,['Return, full episode','Return, final 20 actions','Discounted average coverage']):
                    ax.set(xlabel='State time before action',title=title);ax.legend(fontsize=6)
                fig.suptitle(f'{split} / {stage}',fontsize=11)
                save_figure(fig,figs,f'horizon_{split}_{stage}')
    with np.load(out/'holdout_predictions.npz') as data:
        fig,axes=plt.subplots(1,2,figsize=(11,3.8))
        labels=data['average_labels']
        axes[0].plot(labels.mean(0),color='black',label='MC average coverage')
        for arm in ARMS[1:]:
            p=data[f'{arm}_selected_average']
            axes[0].plot(p.mean(0),label=arm)
            axes[1].scatter(labels.ravel()[::4],p.ravel()[::4],s=3,alpha=.15,label=arm)
        for name in ['factor_constant','time_profile']:
            axes[0].plot(data[f'{name}_average'].mean(0),'--',label=name)
        axes[0].set(xlabel='State time before action',ylabel='Average coverage',title='Best factor models: expanded scale')
        axes[1].plot([0,1],[0,1],'--',color='gray')
        axes[1].set(xlabel='Sampled average-coverage label',ylabel='Predicted average coverage',title='Every fourth state, fixed order')
        for ax in axes:ax.legend(fontsize=7)
        save_figure(fig,figs,'holdout_average_detail')
        fig,axes=plt.subplots(1,2,figsize=(11,3.8))
        for ax,i in zip(axes,[0,1]):
            ax.plot(data['returns'][i],color='black',label='MC return')
            for arm in ARMS:ax.plot(data[f'{arm}_selected_value'][i],label=arm)
            ax.set(xlabel='State time before action',title=f'Holdout scene {int(data["scene_seeds"][i])}');ax.legend(fontsize=7)
        save_figure(fig,figs,'fixed_holdout_scenes')
        fig,axes=plt.subplots(1,2,figsize=(11,3.8))
        for arm in ARMS:
            err=data[f'{arm}_selected_value']-data['returns']
            axes[0].plot(np.sqrt((err**2).mean(0)),label=arm)
            axes[1].plot(err.mean(0),label=arm)
        for ax,title in zip(axes,['Per-state RMSE','Signed mean error']):
            ax.set(xlabel='State time before action',title=title);ax.legend(fontsize=7)
        save_figure(fig,figs,'holdout_errors')
    metric_rows=[]
    for arm, stages in summary['arms'].items():
        for checkpoint in ['initial','selected','last']:
            for split,m in stages[checkpoint].items():
                for scale in ['value','average']:
                    for stage, stats in m[scale]['stages'].items():
                        metric_rows.append(dict(arm=arm,checkpoint=checkpoint,split=split,scale=scale,stage=stage,
                                               **stats,time_centered_correlation=m[scale]['time_centered_correlation']))
    for reference,m in summary['references'].items():
        for scale in ['value','average']:
            for stage,stats in m[scale]['stages'].items():
                metric_rows.append(dict(arm=reference,checkpoint='fixed',split='holdout',scale=scale,stage=stage,
                                       **stats,time_centered_correlation=m[scale]['time_centered_correlation']))
    write_csv(out/'metrics.csv',metric_rows)
    lines=['# 原C组：时间因子与平均覆盖率critic校准','',
           '原C组250k last、6292→256→256→1，冻结actor。三组继承同一主干，输出层权重/偏置归零、Adam重置。'
           '复用原64/16/32个MC场景，不重新采样。累计价值G与平均覆盖标签G/H由同一回报解析转换。','',
           'A直接预测累计价值；B预测平均覆盖、按恢复的累计价值计算损失；C预测平均覆盖、按平均覆盖计算损失。'
           '固定拟合集总体均值/标准差，Huber δ=10；无sigmoid、预测裁剪、GAE或bootstrap。','',
           '三组300轮、batch512，每轮50次更新，LR5e−4、Adam eps1e−5、梯度裁剪0.5。'
           '逐轮样本排列完全一致。best按选模集累计价值RMSE选出（含epoch0）；检查集仅最终评估。','',
           '| 模型 | 选中轮数 | 拟合RMSE | 检查RMSE | 检查平均覆盖RMSE | 检查末20步RMSE | 末步预测 | 末步MC | 时间去均值相关性 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,result in summary['arms'].items():
        for checkpoint in ['initial','selected','last']:
            m=result[checkpoint]['holdout'];v=m['value']['stages']; corr=m['value']['time_centered_correlation']
            fit=result[checkpoint]['fit']['value']['stages']['all']['rmse']
            epoch={'initial':0,'selected':result['selected_epoch'],'last':summary['epochs']}[checkpoint]
            lines.append(f'| {arm} {checkpoint} | {epoch} | {fit:.3f} | {v["all"]["rmse"]:.3f} | '
                f'{m["average"]["stages"]["all"]["rmse"]:.4f} | {v["last20"]["rmse"]:.3f} | '
                f'{v["last1"]["prediction_mean"]:.3f} | {v["last1"]["label_mean"]:.3f} | {corr:.3f} |'
                if corr is not None else f'| {arm} {checkpoint} | {epoch} | {fit:.3f} | {v["all"]["rmse"]:.3f} | '
                f'{m["average"]["stages"]["all"]["rmse"]:.4f} | {v["last20"]["rmse"]:.3f} | '
                f'{v["last1"]["prediction_mean"]:.3f} | {v["last1"]["label_mean"]:.3f} | 未定义 |')
    lines += ['', '两个固定参考只使用拟合集统计；它们没有学习场景信息：','']
    for name,m in summary['references'].items():
        lines.append(f'- {name}：检查集累计价值RMSE {m["value"]["stages"]["all"]["rmse"]:.3f}，'
                     f'平均覆盖RMSE {m["average"]["stages"]["all"]["rmse"]:.4f}。')
    lines += ['', '![学习曲线](figures/learning_curves.png)','',
              '![检查集best](figures/horizon_holdout_selected.png)','',
              '![检查集last](figures/horizon_holdout_last.png)','',
              '![逐时误差](figures/holdout_errors.png)','',
              '![平均覆盖展开尺度](figures/holdout_average_detail.png)','',
              '![固定场景](figures/fixed_holdout_scenes.png)','',
              '## 配对误差与解释边界','',
              '配对区间按32个检查场景重采样2000次，种子950001；不是训练种子置信区间。'
              '差值为右组减左组，负值表示右组误差较小。','',
              '| 检查点 | 比较 | 全程累计价值RMSE差 | 95%场景bootstrap区间 |','|---|---|---:|---:|']
    for row in summary['paired']:
        if row['scale']=='value' and row['stage']=='all' and row['metric']=='rmse' and row['left'] in ARMS and row['right'] in ARMS:
            lines.append(f'| {row["checkpoint"]} | {row["right"]} − {row["left"]} | {row["delta"]:+.3f} | [{row["low"]:+.3f}, {row["high"]:+.3f}] |')
    lines += ['', '## 本次证据如何定位问题','',
              'A能将拟合集误差降得很低，检查集仍有明显末端高估；B保留同一个累计价值损失后显著改善，'
              '因此结果不支持“必须改变Huber损失或改成平均覆盖损失才能修复”的解释。'
              '更符合证据的解释是：现有高维输入MLP未能自行学出可泛化的剩余时长作用，解析参数化提供了有效的结构约束。'
              'B与A还存在输出尺度、初始函数和优化条件差异，不能将其进一步分解为唯一机制。','',
              '固定H×平均覆盖的零训练参考已经具有正确的大部分时间形状。因此需要把这部分解析收益与网络训练增益分开。'
              'B的场景去时间均值相关性高于A，说明它并非只会输出一条公共时间曲线；'
              '但相对于更强的每时刻均值参考，32场景bootstrap区间仍需保守解读。','',
              '下表补充比较参考模型的区间；差值为右组减左组，正值表示左组误差较小。','',
              '| 比较（best） | 累计价值RMSE差 | 95%场景区间 |','|---|---:|---:|']
    for row in summary['paired']:
        if (row['checkpoint']=='selected' and row['scale']=='value' and row['stage']=='all'
                and row['metric']=='rmse' and row['left'] in ARMS[1:] and row['right'] in summary['references']):
            lines.append(f'| {row["right"]} − {row["left"]} | {row["delta"]:+.3f} | [{row["low"]:+.3f}, {row["high"]:+.3f}] |')
    lines += ['', 'C相对B没有显示额外改善；其点估计较差，但检查场景区间包含零，不能声称两者存在稳定差异。'
              '两种时间因子组的拟合集误差随训练下降，检查误差却从早期best回升，说明仍有过拟合或泛化余量，'
              '不能把后续优化方向简单归结为增加训练轮数。','',
              '全部分阶段RMSE/MAE/偏差见metrics.csv；范围违规、相关性和来源哈希见summary.json、experiment.json。'
              '同一时刻减去检查集跨场景均值只用于诊断，不用于拟合；参考模型在这个诊断中是常量，相关性未定义。','',
              'B/C的epoch0相同且已经具有时间形状；仅后期回报自然缩小不能证明网络学会了场景价值。'
              '必须同时比较epoch0、两个固定时间参考、拟合与检查差距，以及场景残差相关性。'
              'C还改变了跨时刻损失权重，在Huber二次区间对应累计误差权重约1/H²，另有固定标准差因子。','',
              'MC标签是采样实现值而非精确条件期望；三组比较始终使用同一批标签。'
              '独立校准不更新actor，所以不能从critic指标直接声称覆盖会改善，也不能单独证明GAE是唯一根因。','',
              '在线MAPPO、原C配置和源权重保持原样；这些离线检查点不能用于在线续训。'
              '没有自动追加轮数、超参数扫描或多种子。','',
              '重建图表：`.venv/bin/python scripts/calibrate_time_factor_critic.py --report-only`。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def run_experiment(options, resume=False):
    out=Path(options['output']).resolve(); data_root=Path(options['dataset']).resolve()
    source=Path(options['checkpoint']).resolve()
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError('Use an empty output directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(source.parent/'config.yaml'); configure_runtime(cfg)
    cfg,policy=policy_from_checkpoint(source,'cpu')
    assert cfg.env.state_dim==6292 and cfg.env.user_initialization=='uniform'
    assert cfg.env.reward_primary=='coverage' and cfg.env.user_speed_std==1
    assert all(getattr(cfg.env,k)==0 for k in ['throughput_weight','robustness_weight','disconnection_weight'])
    assert cfg.env.horizon==400 and cfg.algo.hidden==256 and cfg.algo.name=='mappo'
    assert policy.provenance['checkpoint_env_steps']==250000
    assert options['epochs']==300 and options['batch_size']==512
    policy.actor.eval()
    for p in policy.actor.parameters():p.requires_grad_(False)
    actor_sha=actor_digest(policy.actor)
    datasets={k:load_dataset(data_root/k) for k in ['fit','selection','holdout']}
    validate_data(datasets,cfg,actor_sha)
    source_files=[source.parent/n for n in ['last.pt','best.pt','config.yaml','metrics.csv','validation.csv']]
    files=source_files+[p for p in sorted(data_root.rglob('*')) if p.is_file()]
    hashes={str(p):file_hash(p) for p in files}
    old=json.loads((data_root.parent/'experiment.json').read_text())
    assert hashes[str(source)]==old['source_sha256']
    for name,sha in old['dataset_sha256'].items():
        assert file_hash(data_root.parent/name)==sha
    for split,d in datasets.items():
        with np.load(data_root/split/'episodes.npz') as ep:
            np.testing.assert_array_equal(ep['rewards'],ep['coverage'])
            np.testing.assert_array_equal(d['returns'],discounted_episode_returns(ep['rewards'],cfg.algo.discount))
    statistics=target_statistics(datasets['fit']['returns'],cfg.algo.discount)
    experiment=dict(options=options,source_checkpoint=str(source),source_hashes=hashes,
                    actor_sha256=actor_sha,statistics=statistics,config=cfg.as_dict(),
                    architecture='6292 -> 256 ReLU -> 256 ReLU -> 1 linear',
                    initialization='inherited trunk; zero output layer; fresh Adam; fixed fit-only statistics')
    if resume:
        assert json.loads((out/'experiment.json').read_text())==experiment, 'Resume provenance/configuration mismatch'
    else:atomic_json(out/'experiment.json',experiment)
    seed_all(options['permutation_seed'])
    prior_elapsed = json.loads((out/'manifest.json').read_text()).get('elapsed_seconds',0.) if resume and (out/'manifest.json').exists() else 0.
    start=time.perf_counter()-prior_elapsed; arms={}; all_predictions={k:{} for k in datasets}
    permutations=None
    initial_state=None
    for arm in ARMS:
        dest=out/arm;dest.mkdir(exist_ok=True)
        model=OfflineCritic(policy.value,arm,statistics,400,cfg.algo.discount)
        if initial_state is None:initial_state=copy.deepcopy(model.network.state_dict())
        assert all(torch.equal(p,initial_state[n]) for n,p in model.network.state_dict().items())
        optimizer=make_optimizer(model,options['learning_rate']);assert not optimizer.state
        generator=np.random.default_rng(options['permutation_seed'])
        initial_predictions={k:predict(model,datasets[k]) for k in ['fit','selection']}
        initial_scores={k:score(p,datasets[k]['returns'],cfg.algo.discount) for k,p in initial_predictions.items()}
        initial_score=initial_scores['selection']['value']['stages']['all']['rmse']
        best,best_epoch=initial_score,0;first_epoch=1
        history=[epoch_row(0,time.perf_counter()-start,initial_scores)]
        if resume and (dest/'critic_last.pt').exists():
            saved=restore_offline(dest/'critic_last.pt',model,optimizer,generator)
            assert saved['experiment']==experiment
            first_epoch=saved['epoch']+1
            with (dest/'history.csv').open() as f:previous=list(csv.DictReader(f))
            # Published history is limited to the last complete checkpoint.
            history=[{k:(int(v) if k in ['epoch','optimizer_steps'] else v if k=='permutation_sha256' or v=='' else float(v)) for k,v in r.items()}
                     for r in previous if int(r['epoch'])<first_epoch]
            candidates=[r for r in history]
            selected=min(candidates,key=lambda r:float(r['selection_value_all_rmse']))
            best,best_epoch=float(selected['selection_value_all_rmse']),int(selected['epoch'])
            best_saved=torch.load(dest/'critic_best.pt',map_location='cpu',weights_only=False)
            if best_saved['epoch']>=first_epoch:
                raise ValueError('Best checkpoint is ahead of last checkpoint; resume requires a complete safe point')
        else:
            save_offline(dest/'critic_best.pt',model,optimizer,generator,0,experiment)
            save_offline(dest/'critic_last.pt',model,optimizer,generator,0,experiment)
            write_csv(dest/'history.csv',history)
        try:
            for epoch in range(first_epoch,options['epochs']+1):
                update=fit_epoch(model,optimizer,datasets['fit'],options['batch_size'],generator,options['max_grad_norm'])
                scores={k:score(predict(model,datasets[k]),datasets[k]['returns'],cfg.algo.discount) for k in ['fit','selection']}
                selection=scores['selection']['value']['stages']['all']['rmse']
                history.append(epoch_row(epoch,time.perf_counter()-start,scores,update))
                if selection<best:
                    best,best_epoch=selection,epoch
                    save_offline(dest/'critic_best.pt',model,optimizer,generator,epoch,experiment)
                write_csv(dest/'history.csv',history)
                # Small offline model/Adam checkpoint, no state/trajectory copies.
                save_offline(dest/'critic_last.pt',model,optimizer,generator,epoch,experiment)
                atomic_json(out/'manifest.json',dict(status='training',arm=arm,epoch=epoch,selected_epoch=best_epoch,
                                                     elapsed_seconds=time.perf_counter()-start))
                if epoch==1 or epoch%25==0:
                    print(f'{arm} epoch={epoch} fit_RMSE={scores["fit"]["value"]["stages"]["all"]["rmse"]:.4f} '
                          f'selection_RMSE={selection:.4f} best={best_epoch}',flush=True)
        except BaseException as exc:
            atomic_json(out/'manifest.json',dict(status='failed',arm=arm,error=repr(exc),elapsed_seconds=time.perf_counter()-start))
            raise
        order_hashes=[r['permutation_sha256'] for r in history[1:]]
        if permutations is None:permutations=order_hashes
        else:assert order_hashes==permutations, 'Pairing lost: sample permutations differ'
        assert sum(int(r['optimizer_steps']) for r in history)==15000
        result=dict(selected_epoch=best_epoch,optimizer_steps=15000)
        # Holdout evaluation occurs only after all 300 epochs for this arm.
        initial_model=OfflineCritic(policy.value,arm,statistics,400,cfg.algo.discount)
        last_predictions={k:predict(model,d) for k,d in datasets.items()}
        for checkpoint,predictions in [('initial',{k:predict(initial_model,d) for k,d in datasets.items()}),('last',last_predictions)]:
            result[checkpoint]={k:score(p,datasets[k]['returns'],cfg.algo.discount) for k,p in predictions.items()}
            for k,p in predictions.items():
                for scale,array in p.items():all_predictions[k][f'{arm}_{checkpoint}_{scale}']=array
        restore_offline(dest/'critic_best.pt',model,optimizer,generator)
        selected_predictions={k:predict(model,d) for k,d in datasets.items()}
        result['selected']={k:score(p,datasets[k]['returns'],cfg.algo.discount) for k,p in selected_predictions.items()}
        for k,p in selected_predictions.items():
            for scale,array in p.items():all_predictions[k][f'{arm}_selected_{scale}']=array
        arms[arm]=result
        atomic_json(dest/'summary.json',result)
    h=horizon_factors(np.arange(400,0,-1),cfg.algo.discount)
    references={}
    for k,d in datasets.items():
        all_predictions[k]['returns']=d['returns']
        all_predictions[k]['average_labels']=d['returns']/h
        all_predictions[k]['scene_seeds']=np.asarray(d['metadata']['scene_seeds'])
        for name,curve in [('factor_constant',h*statistics['average_mean']),('time_profile',datasets['fit']['returns'].mean(0))]:
            p=dict(value=np.broadcast_to(curve,d['returns'].shape),average=np.broadcast_to(curve/h,d['returns'].shape))
            for scale,array in p.items():all_predictions[k][f'{name}_{scale}']=array
            if k=='holdout':references[name]=score(p,d['returns'],cfg.algo.discount)
        np.savez_compressed(out/f'{k}_predictions.npz',**all_predictions[k])
    for k in datasets:
        for scale in ['value','average']:
            np.testing.assert_array_equal(all_predictions[k][f'b_factor_return_loss_initial_{scale}'],
                                          all_predictions[k][f'c_factor_average_loss_initial_{scale}'])
    pairs=[]
    for checkpoint in ['selected','last']:
        for scale in ['value','average']:
            predictions={arm:all_predictions['holdout'][f'{arm}_{checkpoint}_{scale}'] for arm in ARMS}
            predictions.update({ref:all_predictions['holdout'][f'{ref}_{scale}'] for ref in references})
            labels=all_predictions['holdout']['returns' if scale=='value' else 'average_labels']
            pairs.extend(dict(checkpoint=checkpoint,scale=scale,**row) for row in paired_bootstrap(
                predictions,labels,options['bootstrap_replicates'],options['bootstrap_seed']))
    assert actor_digest(policy.actor)==actor_sha and all(p.grad is None for p in policy.actor.parameters())
    assert all(file_hash(p)==hashes[str(p)] for p in files)
    write_csv(out/'paired_differences.csv',pairs)
    summary=dict(epochs=options['epochs'],arms=arms,references=references,paired=pairs,
                 actor_unchanged=True,data_and_source_unchanged=True,paired_sample_orders=True,
                 elapsed_seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(out/'summary.json',summary)
    publish(out)
    atomic_json(out/'manifest.json',dict(status='completed',elapsed_seconds=summary['elapsed_seconds'],
                 peak_rss_mib=summary['peak_rss_mib'],total_optimizer_steps=45000))
    print('Saved:',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/critic_time_factor_coverage.yaml'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    options=yaml.safe_load(args.config.read_text())
    if args.report_only:publish(Path(options['output']))
    else:run_experiment(options,args.resume)


if __name__=='__main__':main()
