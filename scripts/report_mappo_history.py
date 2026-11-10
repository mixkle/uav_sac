"""Frozen M1-C / M1-C-H comparison, MC audit and identical-deployment branches."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from check_mappo_deployment_branches import run_branch, stage_metrics, ALL_FIELDS
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest, log_plots, SOURCE_FILES
from report_mappo_training import topology
from uav_research.algorithms import make_learner
from uav_research.bridge_diagnostics import audit_policy
from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import history_specification, stack_observations
from uav_research.plotting import plt, style, save_figure, read_csv
from uav_research.runtime import configure_runtime, seed_all, isolated_rng, atomic_json
from uav_research.vector import VectorEnvironment

LABELS = ['M1-C', 'M1-C-H']
BASE = 'continuous_mappo_full_reward_time_factor_relative_central'
DEFAULT_RUNS = [f'runs/{BASE}{suffix}_100k_seed0' for suffix in ['', '_history']]
DEFAULT_OUTPUT = 'analysis/mappo_central_history_100k_seed0'
DEFAULT_BRANCH = Path('analysis/mappo_central_best_deployment_branches_100k_seed0')


def validate_configs(configs):
    a, b = [copy.deepcopy(c) for c in configs]
    if a.env.observation_history != 'none' or b.env.observation_history != 'one_step':
        raise ValueError('Expected no-history baseline, then one_step experiment')
    b.env.observation_history = 'none'
    if a.as_dict() != b.as_dict():
        raise ValueError('History comparison may change only observation_history')
    if not (a.algo.name == 'mappo' and a.env.action_space == 'continuous'
            and a.algo.mappo_state_dependent_std and a.algo.critic_parameterization == 'time_factor'
            and a.env.user_initialization == 'uniform_interior' and a.env.boundary_penalty_weight == 0
            and a.run.evaluation_policy == 'sample'):
        raise ValueError('Requires continuous, central-UE, time_factor M1-C configuration')


def per_scene(data):
    """All values retain the scene axis; actions are averaged later."""
    diag, names = data['diagnostics'], list(data['diagnostic_names'])
    cov = diag[..., names.index('coverage')]
    h = cov.shape[1]
    early = slice(50, 150) if h >= 150 else slice(0, max(1, h // 2))
    result = dict(coverage=cov.mean(1), steps51_150_coverage=cov[:, early].mean(1),
                  last100_coverage=cov[:, -100:].mean(1), final_coverage=cov[:, -1],
                  maintenance=cov[:, -100:].mean(1)-cov[:, early].mean(1),
                  task_return=data['task_rewards'].sum(1), train_return=data['rewards'].sum(1))
    stages = dict(all=slice(None), first100=slice(0, 100), middle100=slice(150, 250),
                  last100=slice(-100, None), last20=slice(-20, None), last1=slice(-1, None))
    for stage, sl in stages.items():
        if cov[:, sl].shape[1] == 0:
            continue
        for i, name in enumerate(names):
            result[f'{stage}_{name}'] = diag[:, sl, i].mean(1)
        for i, name in enumerate(data['contribution_names']):
            result[f'{stage}_reward_{name}'] = data['contributions'][:, sl, i].mean(1)
        for i, name in enumerate(data['movement_names']):
            result[f'{stage}_{name}'] = data['movement'][:, sl, i].mean(1)
        pred, label, factor = data['values'][..., 0], data['value_labels'], data['horizon_factors']
        for scale, error in [('value', pred-label), ('average', (pred-label)/factor)]:
            result[f'{stage}_{scale}_mse'] = (error[:, sl]**2).mean(1)
            result[f'{stage}_{scale}_mae'] = np.abs(error[:, sl]).mean(1)
            result[f'{stage}_{scale}_bias'] = error[:, sl].mean(1)
    for i, name in enumerate(data['moment_names']):
        result[str(name)] = data['moments'][..., i].mean((1, 2))
    if 'deployment_retention' in data:
        for i, at in enumerate(data['deployment_times']):
            for j, name in enumerate(['links', 'connected_nodes']):
                result[f't{at}_{name}_retention'] = data['deployment_retention'][:, at:, i*2+j].mean(1)
    return result


def summarize_files(paths):
    samples = []
    seeds = None
    for path in paths:
        with np.load(path) as data:
            if seeds is None:
                seeds = data['scene_seeds'].copy()
            np.testing.assert_array_equal(seeds, data['scene_seeds'])
            samples.append(per_scene(data))
    values = {k: np.stack([s[k] for s in samples]).mean(0) for k in samples[0]}
    rows = [dict(scene_seed=int(seed), **{k: float(v[i]) for k, v in values.items()})
            for i, seed in enumerate(seeds)]
    return rows


def reconstruct_histories(cfg, policy, cases, scenes):
    """Replay the original prefix, preserving actual previous state, not current."""
    result = {}
    by_start = {}
    for case in cases:
        by_start.setdefault(case['start'], []).append(case)
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(880001)
        for begin, batch_cases in sorted(by_start.items()):
            count = len(batch_cases[0]['records'])
            with VectorEnvironment(cfg.env, count, 'batch') as vector:
                vector.restore([s['snapshot'] for s in scenes[begin:begin+count]])
                obs = stack_observations([e.obs for e in vector.envs])
                for t in range(max(c['time'] for c in batch_cases) + 1):
                    case = next((c for c in batch_cases if c['time'] == t), None)
                    if case is not None:
                        for e, snap in zip(vector.envs, case['snapshots']):
                            assert e.t == snap['t'] == t
                            for key in ['uav_xy', 'user_xy', 'last_action']:
                                np.testing.assert_array_equal(getattr(e, key), snap['action' if key == 'last_action' else key])
                            assert e.movement_rng.bit_generator.state == snap['movement_rng']
                        torch.testing.assert_close(torch.get_rng_state(), case['torch_rng'], rtol=0, atol=0)
                        result[(begin, t)] = previous.copy()
                    if t == max(c['time'] for c in batch_cases):
                        break
                    action = policy.act(obs, deterministic=False)
                    previous = np.stack([np.column_stack([a, e.B.min(1)/(cfg.env.num_uavs+1),
                                         e.sensed.sum(1)/cfg.env.capacity])
                                         for e, a in zip(vector.envs, action)]).astype(np.float32)
                    obs, *_ = vector.step(action)
                # Advance the remainder to preserve batch-to-batch action RNG order.
                for _ in range(t, cfg.env.horizon):
                    action = policy.act(obs, deterministic=False)
                    obs, *_ = vector.step(action)
    return result


def compare_branches(runs, out, source):
    source = Path(source)
    meta = json.loads((source/'experiment.json').read_text())
    parent = Path(meta['parent'])
    cases = torch.load(source/'branch_states.pt', map_location='cpu', weights_only=False)
    scenes = torch.load(parent/'initial_scenes.pt', map_location='cpu', weights_only=False)['unseen'][:32]
    cfg, original = policy_from_checkpoint(Path(runs[0])/'best.pt', 'cpu')
    expected = next((v for k, v in meta['source_hashes'].items()
                     if Path(k).resolve() == (Path(runs[0])/'best.pt').resolve()), None)
    if expected is None or digest(Path(runs[0])/'best.pt') != expected:
        raise ValueError('Branch prefix must use the original unchanged M1-C best checkpoint')
    histories = reconstruct_histories(cfg, original, cases, scenes)
    torch.save(histories, out/'branch_histories.pt')
    rows, audits = [], []
    all_arrays = {}
    for ck in ['best', 'last']:
        arrays = {t: {label: [] for label in LABELS} for t in sorted({c['time'] for c in cases})}
        for case in cases:
            t, begin = case['time'], case['start']
            paired_paths = {r: [] for r in range(3)}
            for label, run in zip(LABELS, runs):
                pcfg, policy = policy_from_checkpoint(Path(run)/f'{ck}.pt', 'cpu')
                policy.actor.requires_grad_(False)
                adapted = copy.deepcopy(case)
                if label == 'M1-C-H':
                    for i, snap in enumerate(adapted['snapshots']):
                        snap['observation_history'] = 'one_step'
                        snap['history_specification'] = history_specification(pcfg.env)
                        snap['history'] = histories[(begin, t)][i].copy()
                repeated = []
                for repeat in range(3):
                    if repeat == 0:
                        state = case['torch_rng']
                    else:
                        seed = int(np.random.SeedSequence([971001, t, begin, repeat]).generate_state(1)[0])
                        with isolated_rng():
                            torch.manual_seed(seed); state = torch.get_rng_state().clone()
                    data, audit = run_branch(pcfg, policy, adapted, 'policy', state)
                    repeated.append(data); paired_paths[repeat].append(audit)
                arrays[t][label].append(np.stack(repeated))
            for repeat, pair in paired_paths.items():
                assert pair[0]['ue_path_sha256'] == pair[1]['ue_path_sha256']
                assert torch.equal(pair[0]['torch_rng'], pair[1]['torch_rng'])
                audits.append(dict(checkpoint=ck, time=t, start=begin, repeat=repeat,
                                   identical_ue_path_sha256=pair[0]['ue_path_sha256'], identical_action_rng=True))
        for t, groups in arrays.items():
            records = [r for c in cases if c['time'] == t for r in c['records']]
            merged = {label: np.concatenate(parts, axis=1) for label, parts in groups.items()}
            target = out/'branches'/ck/f't{t}'
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'curves.npz', **merged, fields=np.array(ALL_FIELDS),
                                episode_steps=np.arange(t+1, cfg.env.horizon+1))
            all_arrays[f'{ck}_t{t}'] = str(target/'curves.npz')
            vals = {label: stage_metrics(a, records, cfg.algo.discount) for label, a in merged.items()}
            for subset, mask in [('all', np.ones(len(records), bool)),
                                 ('good', np.array([r['good'] for r in records]))]:
                if not mask.any():
                    continue
                for metric in vals[LABELS[0]]:
                    delta = vals[LABELS[1]][metric][mask]-vals[LABELS[0]][metric][mask]
                    rows.append(dict(checkpoint=ck, time=t, subset=subset, scenes=int(mask.sum()),
                                     metric=metric,
                                     baseline_mean=float(vals[LABELS[0]][metric][mask].mean()),
                                     history_mean=float(vals[LABELS[1]][metric][mask].mean()),
                                     **interval(delta)))
    atomic_json(out/'branches.json', dict(differences=rows, checks=audits, arrays=all_arrays,
        sources={str(p): digest(p) for p in [source/'branch_states.pt', source/'experiment.json', parent/'initial_scenes.pt']},
        records={str(t): [r for c in cases if c['time'] == t for r in c['records']]
                 for t in sorted({c['time'] for c in cases})}))
    write_csv(out/'branch_differences.csv', rows)
    return rows


def build(runs, output, samples=64, diagnostic_seeds=None, action_seeds=None, branch_source=DEFAULT_BRANCH):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    runs = [Path(p) for p in runs]
    configs = [load_config(p/'config.yaml') for p in runs]
    validate_configs(configs)
    diagnostic_seeds = list(range(610000, 610064)) if diagnostic_seeds is None else list(diagnostic_seeds)
    action_seeds = [880001, 880002, 880003] if action_seeds is None else list(action_seeds)
    if any(set(diagnostic_seeds) & set(c.run.test_seeds) for c in configs):
        raise ValueError('Reserved test set must remain untouched')
    sources = [p/name for p in runs for name in SOURCE_FILES]
    sources += [Path(__file__)] + list(Path('src/uav_research').glob('*.py'))
    sources += [Path('scripts')/name for name in [
        'check_mappo_generalization.py', 'check_mappo_deployment_branches.py',
        'compare_online_time_factor.py', 'report_mappo_training.py', 'report_mappo_bridge.py']]
    hashes = {str(p): digest(p) for p in sources}
    groups = {}
    for label, run, cfg in zip(LABELS, runs, configs):
        configure_runtime(cfg)
        manifest = json.loads((run/'manifest.json').read_text())
        if manifest['status'] != 'completed' or manifest['env_steps'] != cfg.run.total_steps:
            raise ValueError(f'Not a completed configured run: {run}')
        selection = json.loads((run/'best_selection.json').read_text())
        g = dict(config=cfg.as_dict(), logs={f: read_csv(run/f)
                for f in ['metrics.csv', 'validation.csv']}, cohorts={})
        for cohort, seeds in [('validation', cfg.run.validation_seeds), ('diagnostic', diagnostic_seeds)]:
            g['cohorts'][cohort] = {}
            for ck in ['initial', 'best', 'last']:
                if ck == 'initial':
                    pcfg = copy.deepcopy(cfg)
                    seed_all(cfg.run.seed); policy = make_learner(cfg, torch.device('cpu')); step = 0
                else:
                    pcfg, policy = policy_from_checkpoint(run/f'{ck}.pt', 'cpu')
                    step = policy.provenance['checkpoint_env_steps']
                    if pcfg.env != cfg.env or pcfg.algo != cfg.algo:
                        raise ValueError('Checkpoint/config mismatch')
                    if step != (selection['env_steps'] if ck == 'best' else cfg.run.total_steps):
                        raise ValueError('Checkpoint step mismatch')
                policy.actor.eval().requires_grad_(False)
                before = actor_digest(policy.actor)
                paths = []
                # Initial uses the training evaluation action seed; trained models use all repeats.
                for action_seed in ([cfg.run.evaluation_action_seed] if ck == 'initial' else action_seeds):
                    pcfg.run.validation_seeds = seeds
                    pcfg.run.evaluation_action_seed = action_seed
                    folder = out/label/cohort/ck/str(action_seed)
                    times = tuple(t for t in [150, 200] if t < cfg.env.horizon)
                    data, metrics = audit_policy(pcfg, policy, folder, samples,
                        deployment_times=times,
                        trace_seeds=diagnostic_seeds[:2] if cohort == 'diagnostic' and action_seed == action_seeds[0] else ())
                    paths.append(str(folder/'audit.npz'))
                    if cohort == 'validation' and action_seed == cfg.run.evaluation_action_seed:
                        match = [r for r in g['logs']['validation.csv'] if int(r['env_steps']) == step]
                        if not match:
                            raise ValueError('No validation log for checkpoint')
                        for key, actual in [('coverage', data['diagnostics'][..., 0].mean()),
                                            ('return', data['rewards'].sum(1).mean())]:
                            np.testing.assert_allclose(actual, float(match[-1][key]), rtol=1e-8, atol=1e-8)
                        if ck == 'best':
                            score = data['task_rewards' if cfg.run.best_metric == 'task_return' else 'rewards'].sum(1).mean()
                            np.testing.assert_allclose(score, selection.get('selection_score', selection['validation_return']), rtol=1e-8, atol=1e-8)
                    print(f'{label}/{cohort}/{ck}/{action_seed}: coverage {data["diagnostics"][..., 0].mean():.2%}', flush=True)
                assert actor_digest(policy.actor) == before and all(p.grad is None for p in policy.actor.parameters())
                g['cohorts'][cohort][ck] = dict(step=step, files=paths, actor_sha256=before)
        groups[label] = g
    result = dict(groups=groups, action_seeds=action_seeds, diagnostic_seeds=diagnostic_seeds,
                  source_hashes=hashes, training_seed_count=1, formal_training_started=False,
                  branch_differences=compare_branches(runs, out, branch_source) if branch_source else [])
    if hashes != {p: digest(p) for p in hashes}:
        raise ValueError('Source changed while reporting')
    atomic_json(out/'comparison.json', result)
    publish(out, result)
    return result


def publish(output, comparison):
    out = Path(output); groups = comparison['groups']; style()
    log_plots(groups, out/'figures')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    history_logs = groups['M1-C-H']['logs']['metrics.csv']
    for ax, key in zip(axes, ['history_input_grad_norm', 'history_input_weight_norm']):
        ax.plot([int(r['env_steps']) for r in history_logs],
                [float(r.get(key) or np.nan) for r in history_logs])
        ax.set(title=key, xlabel='Joint transitions')
    save_figure(fig, out/'figures', 'history_input_learning')
    rows, diffs = [], []
    for cohort in ['validation', 'diagnostic']:
        for ck in ['initial', 'best', 'last']:
            paired = {}
            for label, g in groups.items():
                item = g['cohorts'][cohort][ck]
                values = summarize_files(item['files'])
                for row in values:
                    rows.append(dict(group=label, cohort=cohort, checkpoint=ck, checkpoint_step=item['step'], **row))
                paired[label] = values
            assert [r['scene_seed'] for r in paired[LABELS[0]]] == [r['scene_seed'] for r in paired[LABELS[1]]]
            for key in paired[LABELS[0]][0]:
                if key != 'scene_seed':
                    delta = np.array([r[key] for r in paired[LABELS[1]]])-np.array([r[key] for r in paired[LABELS[0]]])
                    diffs.append(dict(cohort=cohort, checkpoint=ck, metric=key, **interval(delta)))
            if cohort != 'diagnostic':
                continue
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            fields = ['coverage', 'discovery_fraction', 'disconnection_ratio',
                      'actual_speed_mean', 'uav_nearest_neighbor_mean', 'boundary_reflection_fraction']
            vfig, vaxes = plt.subplots(1, 2, figsize=(11, 4))
            mfig, maxes = plt.subplots(2, 2, figsize=(11, 7))
            for label, g in groups.items():
                item = g['cohorts'][cohort][ck]
                arrays = []
                vp, vt, moments, retention = [], [], [], []
                for file in item['files']:
                    with np.load(file) as data:
                        arrays.append(data['diagnostics'].copy()); names = list(data['diagnostic_names'])
                        vp.append(data['values'][..., 0].mean(0)); vt.append(data['value_labels'].mean(0))
                        h = data['horizon_factors'].copy()
                        moments.append(data['moments'].copy())
                        moment_t = data['moment_times'].copy()
                        if 'deployment_retention' in data:
                            retention.append(data['deployment_retention'].copy())
                            deployment_times = data['deployment_times'].copy()
                for ax, key in zip(axes.flat, fields):
                    ax.plot(np.arange(1, len(h)+1), np.mean(arrays, axis=(0, 1))[:, names.index(key)], label=label)
                for ax, factor in zip(vaxes, [1, h]):
                    ax.plot(np.mean(vp, 0)/factor, label=f'{label}: prediction')
                    ax.plot(np.mean(vt, 0)/factor, '--', label=f'{label}: MC')
                for ax, j in zip(maxes[0], [0, 1]):
                    ax.plot(moment_t, np.mean(moments, axis=(0, 1, 3))[:, j], label=label)
                if retention:
                    mean_retention = np.mean(retention, axis=(0, 1))
                    for j, at in enumerate(deployment_times):
                        for ax, k in zip(maxes[1], [0, 1]):
                            ax.plot(np.arange(at+1, len(h)+1), mean_retention[at:, j*2+k],
                                    label=f'{label}, anchor {at}')
                trace_dir = Path(item['files'][0]).parent
                cfg = config_from_dict(g['config'])
                frames = [0, 150, 200, cfg.env.horizon] if cfg.env.horizon > 200 else [0, cfg.env.horizon//2, cfg.env.horizon]
                for scene in comparison['diagnostic_seeds'][:2]:
                    topology(trace_dir/f'trace_{scene}.npz', out/label/'figures', ck, item['step'], cfg,
                             boundary_overlay=True, frame_steps=frames)
            for ax, field in zip(axes.flat, fields):
                ax.set(title=field, xlabel='Episode step'); ax.legend(fontsize=7)
            for ax, title in zip(vaxes, ['Cumulative value', 'Discounted average reward']):
                ax.set(title=title, xlabel='Pre-action time'); ax.legend(fontsize=7)
            save_figure(fig, out/'figures', f'coverage_{ck}')
            save_figure(vfig, out/'figures', f'critic_{ck}')
            for ax, title in zip(maxes.flat, ['Conditional displacement variance (m²)', 'Probability of command <1 m',
                                               'Initial UU link retention', 'Initial connected-node retention']):
                ax.set(title=title, xlabel='Episode time')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6)
            save_figure(mfig, out/'figures', f'actions_and_retention_{ck}')
    write_csv(out/'scene_metrics.csv', rows); write_csv(out/'paired_differences.csv', diffs)
    critic_rows = []
    for cohort in ['validation', 'diagnostic']:
        for ck in ['initial', 'best', 'last']:
            for label in LABELS:
                selected = [r for r in rows if (r['group'], r['cohort'], r['checkpoint']) == (label, cohort, ck)]
                for stage in ['all', 'first100', 'middle100', 'last100', 'last20', 'last1']:
                    for scale in ['value', 'average']:
                        prefix = f'{stage}_{scale}'
                        if prefix+'_mse' in selected[0]:
                            critic_rows.append(dict(group=label, cohort=cohort, checkpoint=ck, stage=stage, scale=scale,
                                rmse=float(np.sqrt(np.mean([r[prefix+'_mse'] for r in selected]))),
                                mae=float(np.mean([r[prefix+'_mae'] for r in selected])),
                                bias=float(np.mean([r[prefix+'_bias'] for r in selected]))))
    write_csv(out/'critic_errors.csv', critic_rows)
    state_dim = config_from_dict(groups['M1-C']['config']).env.state_dim
    lines = ['# M1-C 一步历史观测对照', '',
             f'只改变actor的一步历史输入；时间因子critic仍读取{state_dim}维当前全局状态。主要比较同预算last，best作补充。',
             '验证场景用于选模；610000–610063已用于研究诊断，不是最终测试集。初始策略只使用原动作种子，best/last先平均各动作重复，再按场景bootstrap 2000次（950001）。', '',
             '| 场景组 | 模型 | 检查点 | 步数 | 平均覆盖 | 51–150步 | 后100步 | 终态 | 维持差 | 任务回报 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort in ['validation', 'diagnostic']:
        for ck in ['initial', 'best', 'last']:
            for label in LABELS:
                selected = [r for r in rows if r['group'] == label and r['cohort'] == cohort and r['checkpoint'] == ck]
                mean = {k: float(np.mean([r[k] for r in selected])) for k in selected[0] if isinstance(selected[0][k], (int, float))}
                lines.append(f'|{cohort}|{label}|{ck}|{int(mean["checkpoint_step"])}|{mean["coverage"]:.2%}|'
                             f'{mean["steps51_150_coverage"]:.2%}|{mean["last100_coverage"]:.2%}|{mean["final_coverage"]:.2%}|'
                             f'{mean["maintenance"]:+.2%}|{mean["task_return"]:.3f}|')
    lines += ['', '![last部署](figures/coverage_last.png)', '![last critic](figures/critic_last.png)', '',
              '训练指标是rollout末步快照，不能当作固定策略完整回合均值。近悬停或移动减少本身不构成成功。'
              '维持差改善还须检查前期与后期的绝对覆盖，避免以丧失探索换取平坦曲线。', '',
              'MC使用各模型自身轨迹的完整环境奖励，不包含actor熵正则；两模型不是同状态分布上的critic比较。'
              '价值误差的逐场景MSE、MAE、偏差同时报告累计价值与平均奖励尺度。场景区间不是训练种子置信区间。', '',
              '链路保留仅统计t150/t200当时已有的非自环UU链路；联网保留仅统计当时联网节点，无初始元素时记为1。', '',
              '## 同物理部署状态继续执行', '',
              '原策略t150/t200快照对两模型共用；历史来自精确重放的原前缀。场景由原策略产生，属于干预诊断。']
    branch = comparison.get('branch_differences', [])
    if branch:
        lines += ['', '| 检查点 | 时刻 | 子集 | 原覆盖 | 历史覆盖 | 覆盖差（历史−原） | 95%场景区间 |', '|---|---:|---|---:|---:|---:|---|']
        for r in branch:
            if r['metric'] == 'post_coverage':
                lines.append(f'|{r["checkpoint"]}|{r["time"]}|{r["subset"]}|{r["baseline_mean"]:.2%}|{r["history_mean"]:.2%}|{r["mean"]*100:+.2f} pp|[{r["low"]*100:+.2f},{r["high"]*100:+.2f}] pp|')
    else:
        lines += ['', '本次为小配置验收，未接入历史主场景分支数据。正式报告缺少分支来源时应停止，不能静默省略。']
    lines += ['', '数据：scene_metrics.csv、paired_differences.csv及各模型/cohort/checkpoint/动作种子/audit.npz。'
              'comparison.json保存配置、来源哈希与actor哈希；--rebuild仅从保存数据重建图表。', '',
              '历史可能改善动作对局部变化的响应，但并不直接提供中继责任；当前critic也未输入历史，需结合MC误差判断限制。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return rows, diffs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs=2, default=DEFAULT_RUNS)
    p.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--branch-source', type=Path, default=DEFAULT_BRANCH)
    p.add_argument('--rebuild', action='store_true')
    args = p.parse_args()
    if args.rebuild:
        publish(args.output, json.loads((args.output/'comparison.json').read_text()))
    else:
        build(args.runs, args.output, args.samples, branch_source=args.branch_source)


if __name__ == '__main__':
    main()
