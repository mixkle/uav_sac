"""Frozen C actor, unchanged MLP critic, full Monte Carlo supervised targets."""
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
from uav_research.algorithms import gae
from uav_research.critic_calibration import (
    actor_digest, collect_complete_episodes, discounted_episode_returns,
    rebase_normalizer_from_fit_targets, fit_critic_epoch, predict_critic,
    calibration_metrics,
)
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime, atomic_json, atomic_save, seed_all, rng_state
from uav_research.plotting import plt, style, save_figure


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(predicted, returns, gamma):
    result = calibration_metrics(predicted, returns, 1.0)
    error = predicted[:, 150:250] - returns[:, 150:250]
    result['time_bins']['middle100'] = dict(rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()))
    bound = (1-gamma**np.arange(returns.shape[1], 0, -1))/(1-gamma)
    result['above_horizon_bound_fraction'] = float((predicted > bound + 1e-6).mean())
    return result


def save_checkpoint(path, policy, experiment, epoch, score, generator):
    atomic_save(path, dict(kind='critic_calibration_mc', epoch=epoch, experiment=experiment,
        selection_rmse=score, value=copy.deepcopy(policy.value.state_dict()),
        normalizer=copy.deepcopy(policy.normalizer.state_dict()),
        value_optimizer=copy.deepcopy(policy.value_opt.state_dict()), rng=rng_state(),
        permutation_rng=copy.deepcopy(generator.bit_generator.state)))


def audit_targets(out):
    """Read-only frozen-policy label comparison, not historical rollout replay."""
    experiment = json.loads((out/'experiment.json').read_text())
    a = experiment['config']['algo']
    with np.load(out/'holdout_predictions.npz') as data:
        v = np.array(data['before'], dtype=np.float64)
        mc = np.array(data['returns'], dtype=np.float64)
    with np.load(out/'data'/'holdout'/'episodes.npz') as episodes:
        rewards = np.array(episodes['rewards'], dtype=np.float64)
    vt, rt = torch.tensor(v.T), torch.tensor(rewards.T)
    next_values = torch.cat([vt[1:], torch.zeros_like(vt[:1])])
    done = torch.zeros_like(rt, dtype=torch.bool); done[-1] = True
    _, full = gae(rt, vt, next_values, done, a['discount'], a['gae_lambda'])
    _, exact = gae(rt, vt, next_values, done, a['discount'], 1.0)
    np.testing.assert_allclose(exact.numpy().T, mc, rtol=2e-6, atol=2e-4)
    fragments = torch.empty_like(vt)
    for start in range(0, len(vt), a['rollout_steps']):
        sl = slice(start, start+a['rollout_steps'])
        _, fragments[sl] = gae(rt[sl], vt[sl], next_values[sl], done[sl], a['discount'], a['gae_lambda'])
    full, fragments = full.numpy().T, fragments.numpy().T
    for target in (full, fragments):
        np.testing.assert_allclose(target[:, -1], rewards[:, -1], atol=1e-12)
    scores = {k: metrics(p, mc, a['discount']) for k,p in
              [('original_value', v), ('full_episode_gae_target', full), ('fragment_gae_target', fragments)]}
    atomic_json(out/'target_audit.json', dict(scores=scores, gae_lambda=a['gae_lambda'],
        rollout_steps=a['rollout_steps'], lambda_one_matches_mc=True,
        caveat='Frozen source critic and actor; fragments start at t=0,128,...; not historical online targets.'))
    np.savez_compressed(out/'target_audit.npz', mc=mc, full_gae=full, fragment_gae=fragments, original_value=v)
    style()
    fig, axes = plt.subplots(1,2,figsize=(11,3.8))
    for ax in axes:
        for label, p in [('MC label',mc),('Frozen original V',v),('Full-episode GAE target',full),('128-step fragment GAE target',fragments)]:
            ax.plot(p.mean(0), label=label)
        ax.set(xlabel='State time before action',ylabel='Raw return units'); ax.legend(fontsize=8)
    axes[0].set_title('Frozen-policy label diagnostic: holdout')
    axes[1].set(xlim=(380,399),title='Final 20 states')
    save_figure(fig,out/'figures','target_comparison')


def row_for(epoch, elapsed, scores, update=None):
    row = dict(epoch=epoch, elapsed_seconds=elapsed,
        normalized_huber_loss=update['normalized_huber_loss'] if update else '',
        value_grad_norm=update['value_grad_norm'] if update else '',
        optimizer_steps=update['optimizer_steps'] if update else 0)
    for split, m in scores.items():
        for key in ('raw_rmse', 'raw_mae', 'raw_bias', 'final_step_rmse', 'final_step_prediction_mean'):
            row[f'{split}_{key}'] = m[key]
        for stage, v in m['time_bins'].items():
            row[f'{split}_{stage}_rmse'] = v['rmse']
    return row


def publish(out, history, summary, predictions):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    x = [r['epoch'] for r in history]
    axes[0].semilogy(x[1:], [r['normalized_huber_loss'] for r in history[1:]])
    axes[0].set(title='Training normalized Huber loss', xlabel='Epoch')
    for split in ('fit', 'selection'):
        axes[1].semilogy(x, [r[f'{split}_raw_rmse'] for r in history], label=split)
        axes[2].semilogy(x, [r[f'{split}_final_step_rmse'] for r in history], label=split)
    for ax, title in zip(axes[1:], ['Raw all-time RMSE', 'Raw last-step RMSE']):
        ax.set(title=title, xlabel='Epoch'); ax.legend()
    save_figure(fig, out/'figures', 'learning_curves')
    for split, p in predictions.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax in axes:
            for key in ('returns', 'before', 'last', 'selected', 'time_reference'):
                ax.plot(p[key].mean(0), label=key, alpha=.85)
            ax.set(xlabel='State time before action', ylabel='Raw discounted return'); ax.legend(fontsize=8)
        axes[0].set_title(f'{split}: frozen C actor')
        axes[1].set(xlim=(380, 399), title='Final 20 states')
        save_figure(fig, out/'figures', f'horizon_{split}')
    lines = ['# C组：冻结actor、真实MC标签的critic拟合', '',
        f"源：`{summary['source_checkpoint']}`。网络6292→256→256→1，两层ReLU；奖励仅覆盖率。",
        'actor参数和源检查点哈希不变。保留critic初始权重及Adam，学习率5e-4、梯度裁剪0.5、Huber δ=10；无价值裁剪、GAE标签或bootstrap。', '',
        '64拟合场景61000–61063，16选模场景62000–62015，32检查场景63000–63031；动作种子分别930001/930002/930003。每个场景完整400步，γ=0.995，终止回报0。',
        '目标归一化只用拟合集MC标签，PopArt重标定保持初始原始尺度预测，随后固定统计量。batch512，每轮50次更新，完整300轮、不早停；选模集全时段RMSE选best（含epoch0）。检查集不参与训练或选模。', '',
        '| 阶段 | 数据集 | 全程RMSE | 前100步RMSE | 中间100步RMSE | 最后20步RMSE | 末步RMSE | 末步V均值 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('before', 'last', 'selected'):
        for split, m in summary[stage].items():
            lines.append(f"| {stage} | {split} | {m['raw_rmse']:.3f} | {m['time_bins']['first100']['rmse']:.3f} | {m['time_bins']['middle100']['rmse']:.3f} | {m['time_bins']['last20']['rmse']:.3f} | {m['final_step_rmse']:.3f} | {m['final_step_prediction_mean']:.3f} |")
    lines += ['', f"选中epoch：{summary['selected_epoch']}；完整训练{summary['epochs']}轮，共{summary['optimizer_steps']}次critic更新，耗时{summary['elapsed_seconds']/60:.2f}分钟。",
        f"检查集最后一步实际回报均值：{summary['before']['holdout']['final_step_return_mean']:.4f}。只有一个奖励剩余，真实价值上界为1。",
        f"仅用拟合集各时刻平均回报作为参照，其检查集全程RMSE为{summary['time_reference']['raw_rmse']:.3f}、末步RMSE为{summary['time_reference']['final_step_rmse']:.3f}。该参照不修改网络，不读取检查标签拟合。", '',
        '## 本次结果', '',
        f"末轮拟合集RMSE从{summary['before']['fit']['raw_rmse']:.3f}降至{summary['last']['fit']['raw_rmse']:.3f}，"
        f"拟合集末步RMSE从{summary['before']['fit']['final_step_rmse']:.3f}降至{summary['last']['fit']['final_step_rmse']:.3f}。"
        '这直接检验了同一MLP在原始状态和正确MC标签上的拟合情况，不涉及actor策略改善。', '',
        f"独立检查集：原始RMSE {summary['before']['holdout']['raw_rmse']:.3f}，"
        f"末轮RMSE {summary['last']['holdout']['raw_rmse']:.3f}，选中模型RMSE {summary['selected']['holdout']['raw_rmse']:.3f}。"
        f"选中模型末步预测均值{summary['selected']['holdout']['final_step_prediction_mean']:.3f}，"
        f"真实末步回报{summary['selected']['holdout']['final_step_return_mean']:.3f}。"
        '需同时查看拟合集与检查集曲线，避免以全程平均误差掩盖末端问题。', '',
        '本次拟合集曲线可以贴近弯曲的MC回报，并在末期下降；独立检查集仍然前期低估、后期高估。'
        '证据支持现有MLP能拟合已见场景的标签，但没有学到足够可泛化的价值规律，不能归结为“MLP只能输出近似直线”。'
        '一种与结果一致的解释是：高维位置、关联和SINR特征使网络容易利用场景特征拟合轨迹，'
        '而未充分利用原有时间标量学习跨场景共用的剩余时长关系；这仍需单独对照验证。', '',
        '![学习曲线](figures/learning_curves.png)', '',
        '![拟合集价值曲线](figures/horizon_fit.png)', '',
        '![独立检查集价值曲线](figures/horizon_holdout.png)', '',
        '## 如何解释', '',
        '训练集能拟合但检查集不好，支持跨场景泛化或可复用时间规律学习不足；不能说MLP连正确标签都学不会。训练与检查都好，则支持现有网络在固定策略下有预测能力，在线目标/数据/更新过程值得继续排查。两边都不好，只能说明当前优化设置和预算未拟合成功，不能证明MLP不存在可用参数。', '',
        '本实验冻结策略、使用固定数据并增加离线更新，还去除了bootstrap和价值裁剪；MC成功不等于单独证明GAE是唯一原因。反之，MC失败也不能证明GAE无关。每个场景一条MC轨迹有采样噪声，不是精确条件期望V；末步上界违规和系统性的时间曲线错误仍可直接识别。', '',
        '## 冻结源critic的GAE标签诊断', '',
        '![标签比较](figures/target_comparison.png)', '',
        '另用同批检查轨迹和冻结的源critic计算GAE标签，比较完整400步GAE与从0/128/256/384开始的片段GAE。'
        'λ=1的完整回合结果与MC一致；λ=0.95仍会混入源critic估计。该图是固定模型下的标签诊断，不是历史在线rollout重放，不能据此断言GAE实现错误。', '',
        '不更新actor，因此这次校准不会改变部署覆盖率；校准权重是独立产物，不能作为完整MAPPO续训检查点。', '',
        '配置与复现说明：[实验文档](../../docs/critic_mc_coverage_experiment.md)。逐轮记录见`history.csv`，完整指标见`summary.json`，分时预测/标签见`*_predictions.npz`，数据与来源哈希见`experiment.json`。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/critic_mc_coverage.yaml'))
    args = parser.parse_args()
    options = yaml.safe_load(args.config.read_text())
    out, source = Path(options['output']).resolve(), Path(options['checkpoint']).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory; results are not overwritten')
    if options['epochs'] <= 0 or options['batch_size'] <= 0:
        raise ValueError('Epochs and batch size must be positive')
    cfg = load_config(source.parent/'config.yaml')
    configure_runtime(cfg)
    cfg, policy = policy_from_checkpoint(source, 'cpu')
    if not (cfg.algo.name == 'mappo' and cfg.env.action_space == 'discrete9'
            and cfg.env.reward_primary == 'coverage'
            and all(getattr(cfg.env, k) == 0 for k in ('throughput_weight','robustness_weight','disconnection_weight'))):
        raise ValueError('Requires pure-coverage discrete MAPPO')
    if cfg.env.horizon != 400 or cfg.env.state_dim != 6292:
        raise ValueError('Requires current C scene and state structure')
    out.mkdir(parents=True, exist_ok=True)
    original_sha, actor_sha = file_hash(source), actor_digest(policy.actor)
    for p in policy.actor.parameters():
        p.requires_grad_(False)
    policy.actor.eval()
    splits = {k: (list(range(v['seed_start'], v['seed_start']+v['count'])), v['action_seed']) for k,v in options['splits'].items()}
    flat = [s for seeds, _ in splits.values() for s in seeds]
    assert len(flat) == len(set(flat)) and not set(flat) & set(cfg.run.validation_seeds+cfg.run.test_seeds)
    experiment = dict(options, source_checkpoint=str(source), source_sha256=original_sha,
        source_env_steps=policy.provenance['checkpoint_env_steps'], config=cfg.as_dict(), actor_sha256=actor_sha,
        normalizer='fit MC targets only; PopArt rebase; fixed during fitting', optimizer='inherited Adam',
        objective='full-episode MC return; normalized Huber delta10; no clipping or bootstrap',
        baseline_value_clipping=cfg.algo.value_clipping, calibration_value_clipping=False)
    atomic_json(out/'experiment.json', experiment)
    atomic_json(out/'manifest.json', dict(status='collecting', **experiment))
    stamp = time.perf_counter()
    datasets = {k: collect_complete_episodes(cfg, policy, seeds, seed, out/'data'/k) for k,(seeds,seed) in splits.items()}
    for k, data in datasets.items():
        with np.load(out/'data'/k/'episodes.npz') as episodes:
            np.testing.assert_array_equal(episodes['rewards'], episodes['coverage'])
            np.testing.assert_array_equal(data['returns'], discounted_episode_returns(episodes['rewards'], cfg.algo.discount))
        assert data['metadata']['actor_sha256'] == actor_sha
    data_hashes = {str(p.relative_to(out)): file_hash(p) for p in (out/'data').rglob('*') if p.is_file()}
    experiment['dataset_sha256'] = data_hashes
    before = {k: predict_critic(policy, d) for k,d in datasets.items()}
    before_metrics = {k: metrics(before[k], d['returns'], cfg.algo.discount) for k,d in datasets.items()}
    normalization = rebase_normalizer_from_fit_targets(policy, datasets['fit']['returns'])
    for k,d in datasets.items():
        np.testing.assert_allclose(predict_critic(policy,d), before[k], atol=3e-5, rtol=1e-5)
    experiment['normalization'] = normalization
    atomic_json(out/'experiment.json', experiment)
    norm_sha = actor_digest(policy.normalizer)
    seed_all(options['permutation_seed'])
    generator = np.random.default_rng(options['permutation_seed'])
    best, best_epoch = before_metrics['selection']['raw_rmse'], 0
    history = [row_for(0, time.perf_counter()-stamp, {k: before_metrics[k] for k in ('fit','selection')})]
    save_checkpoint(out/'critic_best.pt', policy, experiment, 0, best, generator)
    total_updates = 0
    for epoch in range(1, options['epochs']+1):
        update = fit_critic_epoch(policy, datasets['fit'], options['batch_size'], generator)
        total_updates += update['optimizer_steps']
        scores = {k: metrics(predict_critic(policy,datasets[k]), datasets[k]['returns'], cfg.algo.discount) for k in ('fit','selection')}
        history.append(row_for(epoch, time.perf_counter()-stamp, scores, update))
        score = scores['selection']['raw_rmse']
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(out/'critic_best.pt', policy, experiment, epoch, score, generator)
        with (out/'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        if epoch == 1 or epoch % 25 == 0 or epoch == options['epochs']:
            save_checkpoint(out/'critic_last.pt', policy, experiment, epoch, score, generator)
            atomic_json(out/'progress.json', dict(epoch=epoch, scores=scores, selected_epoch=best_epoch, elapsed_seconds=time.perf_counter()-stamp))
            print(f"Epoch {epoch}: fit={scores['fit']['raw_rmse']:.4f}, selection={score:.4f}, selection last={scores['selection']['final_step_rmse']:.4f}; best={best_epoch}", flush=True)
    last = {k: predict_critic(policy, d) for k,d in datasets.items()}
    selected = torch.load(out/'critic_best.pt', map_location='cpu', weights_only=False)
    for name in ('value','normalizer'):
        getattr(policy,name).load_state_dict(selected[name])
    chosen = {k: predict_critic(policy,d) for k,d in datasets.items()}
    time_mean = datasets['fit']['returns'].mean(0)
    predictions = {}
    for k,d in datasets.items():
        predictions[k] = dict(before=before[k], last=last[k], selected=chosen[k], returns=d['returns'],
                              time_reference=np.broadcast_to(time_mean,d['returns'].shape))
        np.savez_compressed(out/f'{k}_predictions.npz', **predictions[k])
    assert actor_digest(policy.actor) == actor_sha and all(p.grad is None for p in policy.actor.parameters())
    assert actor_digest(policy.normalizer) == norm_sha and file_hash(source) == original_sha
    assert all(file_hash(out/p) == sha for p,sha in data_hashes.items())
    summary = dict(source_checkpoint=str(source), selected_epoch=best_epoch, epochs=options['epochs'],
        optimizer_steps=total_updates, before=before_metrics,
        last={k: metrics(last[k],d['returns'],cfg.algo.discount) for k,d in datasets.items()},
        selected={k: metrics(chosen[k],d['returns'],cfg.algo.discount) for k,d in datasets.items()},
        time_reference=metrics(predictions['holdout']['time_reference'],datasets['holdout']['returns'],cfg.algo.discount),
        actor_unchanged=True, normalizer_fixed=True, source_unchanged=True, data_unchanged=True,
        elapsed_seconds=time.perf_counter()-stamp, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(out/'summary.json',summary)
    audit_targets(out)
    publish(out,history,summary,predictions)
    atomic_json(out/'manifest.json',dict(status='completed',**summary))
    print('Report:',out/'report.md', flush=True)


if __name__ == '__main__':
    main()
