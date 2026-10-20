"""Evaluate a frozen discrete MAPPO actor under sharper action distributions.

Only diagnostic files are written; training code, weights and config are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from analyze_ablation_topology import stats, write_csv
from uav_research.actions import DISCRETE_DIRECTIONS
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import as_tensor_observation
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


MODES = [('t1', 1.0), ('t07', .7), ('t04', .4), ('t02', .2), ('argmax', None)]
DIAGNOSTICS = ['entropy', 'max_probability', 'hover_probability', 'displacement_variance_m2']


def distribution(logits, temperature):
    if temperature is None:
        probabilities = torch.nn.functional.one_hot(logits.argmax(-1), 9).to(logits.dtype)
        return Categorical(probs=probabilities, validate_args=False)
    return Categorical(logits=logits / temperature, validate_args=False)


def diagnostics(dist, step_size):
    p = dist.probs
    directions = torch.as_tensor(DISCRETE_DIRECTIONS.copy(), device=p.device)
    mean = p @ directions
    variance = (p * directions.square().sum(-1)).sum(-1) - mean.square().sum(-1)
    return torch.stack((dist.entropy().mean(-1), p.max(-1).values.mean(-1),
                        p[..., 8].mean(-1), variance.clamp_min(0).mean(-1)*step_size**2), -1)


def label(mode):
    return 'argmax' if mode == 'argmax' else f'T={dict(MODES)[mode]:g}'


def metrics(values, fields, action_counts):
    result = {}
    for name, part in [('initial', values[0]), ('midpoint', values[200]),
                       ('final', values[-1]), ('mean', values[1:].mean(0)),
                       ('last100', values[-100:].mean(0))]:
        result.update({f'{name}_{key}': float(v) for key, v in zip(fields, part)})
    result['episode_return'] = float(values[1:, fields.index('reward')].sum())
    result['coverage_last100_minus_mid100'] = float(
        values[-100:, fields.index('coverage')].mean()-values[151:251, fields.index('coverage')].mean())
    for a, count in enumerate(action_counts):
        result[f'action_{a}_frequency'] = float(count/action_counts.sum())
    return result


def evaluate(run, reference, output, action_seeds):
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run/'config.yaml')
    device = configure_runtime(cfg)
    checkpoint = run/'best.pt'
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg, policy = policy_from_checkpoint(checkpoint, device)
    assert cfg.env.action_space == 'discrete9' and cfg.algo.name == 'mappo'
    assert cfg.env.user_speed_std == 0 and cfg.env.horizon == 400
    assert not set(cfg.run.validation_seeds) & set(cfg.run.test_seeds)
    if not action_seeds or action_seeds[0] != cfg.run.evaluation_action_seed:
        raise ValueError('The first action seed must match the original evaluation seed for replay verification')
    if len(set(action_seeds)) != len(action_seeds):
        raise ValueError('Action seeds must be distinct')
    saved = json.loads((reference/'report.json').read_text())['evaluations']['best']
    assert saved['checkpoint_sha256'] == before_hash
    policy.actor.eval()
    for p in policy.actor.parameters():
        p.requires_grad_(False)
    rows, curve_rows, probes = [], [], []
    data_by_mode, fields = {}, None
    started = time.monotonic()
    verification = dict(policy_updates=0, reserved_test_scenes_used=0, matched_original_frames=0)
    for mode, temperature in MODES:
        mode_data = []
        repeats = action_seeds[:1] if temperature is None else action_seeds
        for action_seed in repeats:
            batches = []
            with isolated_rng(), torch.inference_mode():
                torch.manual_seed(action_seed)
                for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
                    seeds = cfg.run.validation_seeds[start:start+cfg.run.num_envs]
                    with VectorEnvironment(cfg.env, len(seeds), 'batch') as vector:
                        obs, _, _ = vector.reset(seeds)
                        logs, trace = [], []
                        counts = np.zeros((len(seeds), 9), np.int64)
                        old_traces = {}
                        if mode == 't1' and action_seed == cfg.run.evaluation_action_seed:
                            old_traces = {i: np.load(reference/'validation_best'/f'trace_{seed}.npz')
                                          for i, seed in enumerate(seeds) if seed in cfg.run.validation_seeds[:2]}
                        for t in range(cfg.env.horizon+1):
                            raw = [stats(env) for env in vector.envs]
                            base_fields = list(raw[0])
                            logits = policy.actor.distribution(as_tensor_observation(obs, device)).logits
                            dist = distribution(logits, temperature)
                            diag = diagnostics(dist, cfg.env.max_speed*cfg.env.dt).cpu().numpy()
                            fields = base_fields + DIAGNOSTICS
                            logs.append(np.concatenate((np.array([[r[k] for k in base_fields] for r in raw]), diag), axis=-1))
                            if action_seed == action_seeds[0] and start == 0:
                                trace.append({k: np.stack([getattr(env, k).copy() for env in vector.envs[:2]])
                                              for k in ['uav_xy', 'user_xy', 'A', 'B', 'C', 'sensed']})
                            if mode == 't1' and t < cfg.env.horizon:
                                for probe_mode, probe_temp in MODES:
                                    pd = diagnostics(distribution(logits, probe_temp), cfg.env.max_speed*cfg.env.dt).cpu().numpy()
                                    probes.append(dict(mode=probe_mode, action_seed=action_seed, step=t,
                                                       **dict(zip(DIAGNOSTICS, pd.mean(0).tolist()))))
                            for i, old in old_traces.items():
                                for key in ['uav_xy', 'user_xy', 'B']:
                                    np.testing.assert_array_equal(getattr(vector.envs[i], key), old[key][t])
                                verification['matched_original_frames'] += 1
                            if t < cfg.env.horizon:
                                actions = logits.argmax(-1) if temperature is None else dist.sample()
                                actions = actions.cpu().numpy()
                                for i in range(len(seeds)):
                                    counts[i] += np.bincount(actions[i], minlength=9)
                                obs, _, _, _, _, _ = vector.step(actions)
                        for old in old_traces.values():
                            old.close()
                        arr = np.stack(logs)
                        if trace:
                            np.savez_compressed(output/f'trace_{mode}.npz',
                                                **{k: np.stack([s[k] for s in trace]) for k in trace[0]},
                                                scene_seeds=seeds[:2], action_seed=action_seed)
                        for i, seed in enumerate(seeds):
                            rows.append(dict(mode=mode, action_seed=action_seed, scene_seed=seed,
                                             **metrics(arr[:, i], fields, counts[i])))
                        batches.append(arr)
            data = np.concatenate(batches, axis=1)
            mode_data.append(data)
            print(f'{mode} seed={action_seed}: coverage={data[1:,:,0].mean():.4f} '
                  f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        data_by_mode[mode] = np.stack(mode_data)
        np.savez_compressed(output/f'metrics_{mode}.npz', values=data_by_mode[mode], fields=fields,
                            scene_seeds=cfg.run.validation_seeds, action_seeds=repeats)
        averaged = data_by_mode[mode].mean((0, 2))
        curve_rows.extend(dict(mode=mode, step=t, **dict(zip(fields, vals.tolist()))) for t, vals in enumerate(averaged))
    # The original seed and batching must reproduce the already-published evaluation.
    original = data_by_mode['t1'][0]
    np.testing.assert_allclose(original[1:,:,fields.index('coverage')].mean(), saved['summary']['mean']['coverage'], atol=1e-12)
    np.testing.assert_allclose(original[1:,:,fields.index('reward')].sum(0).mean(), saved['summary']['mean']['return'], atol=1e-10)
    with (reference/'validation_best'/'episodes.csv').open() as stream:
        old_rows = {int(r['scene_seed']): r for r in __import__('csv').DictReader(stream)}
    for i, seed in enumerate(cfg.run.validation_seeds):
        np.testing.assert_allclose(original[1:,i,fields.index('coverage')].mean(), float(old_rows[seed]['coverage']), atol=1e-12)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
    write_csv(output/'episodes.csv', rows)
    write_csv(output/'curves.csv', curve_rows)
    write_csv(output/'fixed_reference_states.csv', probes)
    verification.update(checkpoint_sha256=before_hash, checkpoint_env_steps=saved['checkpoint_env_steps'],
                        validated_episodes=len(rows), action_seeds=action_seeds,
                        scene_seeds=cfg.run.validation_seeds, wall_seconds=time.monotonic()-started,
                        checkpoint_unchanged=True, original_validation_reproduced=True,
                        argmax_repeats=1, training_seeds=1)
    atomic_json(output/'verification.json', verification)
    summarize(rows, output)
    figures(data_by_mode, fields, cfg, output)
    write_report(output)


def summarize(rows, output):
    summary, comparisons = {}, []
    keys = [k for k in rows[0] if k not in ['mode','action_seed','scene_seed']]
    scenes = sorted({r['scene_seed'] for r in rows})
    rng = np.random.default_rng(73117)
    bootstrap = rng.integers(len(scenes), size=(5000, len(scenes)))
    scene_means = {}
    for mode, _ in MODES:
        selected = [r for r in rows if r['mode']==mode]
        summary[mode] = {k: float(np.mean([r[k] for r in selected])) for k in keys}
        scene_means[mode] = {k: np.array([np.mean([r[k] for r in selected if r['scene_seed']==s]) for s in scenes]) for k in keys}
        for key in ['mean_coverage','last100_coverage','final_coverage','episode_return']:
            seeds = sorted({r['action_seed'] for r in selected})
            means = [np.mean([r[key] for r in selected if r['action_seed']==s]) for s in seeds]
            summary[mode][key+'_action_seed_std'] = float(np.std(means, ddof=1)) if len(means)>1 else None
    compare_keys = ['mean_coverage','last100_coverage','final_coverage','mean_connected_count',
                    'last100_connected_sensing','last100_connected_edge30','coverage_last100_minus_mid100','episode_return']
    for mode, _ in MODES[1:]:
        for key in compare_keys:
            delta = scene_means[mode][key]-scene_means['t1'][key]
            low, high = np.quantile(delta[bootstrap].mean(1), [.025,.975])
            comparisons.append(dict(mode=mode, metric=key, delta=float(delta.mean()),
                                    paired_scene_ci_low=float(low), paired_scene_ci_high=float(high),
                                    improved_scenes=int((delta>0).sum()), scene_count=len(scenes)))
    atomic_json(output/'summary.json', summary)
    write_csv(output/'paired_comparisons.csv', comparisons)


def figures(data, fields, cfg, output):
    style()
    panels = [('coverage','Actual service coverage',100,'% of UE'),
              ('connected_sensing','UE sensed by connected UAVs',100,'% of UE'),
              ('connected_count','BS-reachable UAVs',1,'UAV count'),
              ('connected_edge30','Connected UAVs near boundary (<30 m)',100,'%'),
              ('entropy','Action entropy',1,'nats'),('hover_probability','Hover probability',100,'%')]
    fig, axes = plt.subplots(2,3,figsize=(14,7.5))
    for (mode,_),color in zip(MODES,COLORS):
        mean = data[mode].mean((0,2))
        for ax,(key,title,scale,unit) in zip(axes.flat,panels):
            ax.plot(np.arange(len(mean)),mean[:,fields.index(key)]*scale,label=label(mode),color=color)
            ax.set(title=title,xlabel='Episode step',ylabel=unit)
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Frozen B best (200,704) | 32 scenes × 5 action seeds; argmax once per scene')
    save_figure(fig,output/'figures','temperature_evolution')
    summary=json.loads((output/'summary.json').read_text())
    fig, axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,key,title in zip(axes,['mean_coverage','last100_coverage','final_coverage'],['Whole episode','Last 100 steps','Final state']):
        ax.bar([label(m) for m,_ in MODES],[summary[m][key]*100 for m,_ in MODES],color=COLORS[:5])
        ax.set(ylabel='Coverage (%)',title=title,ylim=(0,55))
    save_figure(fig,output/'figures','coverage_summary')
    from matplotlib.patches import Circle
    for i,seed in enumerate(cfg.run.validation_seeds[:2]):
        fig,axes=plt.subplots(5,3,figsize=(11,16))
        for row,(mode,_) in enumerate(MODES):
            with np.load(output/f'trace_{mode}.npz') as tr:
                for col,t in enumerate([0,200,400]):
                    ax=axes[row,col];p=tr['uav_xy'][t,i];users=tr['user_xy'][t,i]
                    hops=tr['B'][t,i].min(1);connected=hops<=cfg.env.num_uavs;direct=hops==1
                    served=tr['C'][t,i].any(0);sensed=tr['sensed'][t,i].any(0)
                    ax.scatter(users[~sensed,0],users[~sensed,1],s=5,c='#bbbbbb')
                    ax.scatter(users[sensed&~served,0],users[sensed&~served,1],s=7,c=COLORS[3])
                    ax.scatter(users[served,0],users[served,1],s=9,c=COLORS[2])
                    for u in range(len(p)):
                        for v in range(u):
                            if tr['A'][t,i,u,v]:ax.plot(p[[u,v],0],p[[u,v],1],c=COLORS[0] if connected[u] else '#999999',lw=.6,alpha=.6)
                    for g,bs in enumerate(np.asarray(cfg.env.bs_positions)[:,:2]):
                        ax.scatter(*bs,marker='*',c=COLORS[1],s=65,clip_on=False)
                        ax.add_patch(Circle(bs,np.sqrt(cfg.env.ug_range**2-cfg.env.altitude**2),fill=False,ls=':',color='#999999',lw=.5))
                        for u in np.flatnonzero(tr['B'][t,i,:,g]==1):
                            ax.plot([bs[0],p[u,0]],[bs[1],p[u,1]],c=COLORS[1],ls='--',lw=.4,alpha=.5)
                    for mask,color,marker in [(direct,COLORS[0],'^'),(connected&~direct,COLORS[2],'s'),(~connected,COLORS[4],'^')]:
                        ax.scatter(p[mask,0],p[mask,1],marker=marker,c=color,s=25,zorder=4)
                    ax.set(xlim=(0,cfg.env.width),ylim=(0,cfg.env.height),aspect='equal',
                           title=f'{label(mode)} | step {t}\nServed {served.mean():.1%}, connected {connected.sum()}/18')
                    ax.title.set_fontsize(9)
                    if row==4: ax.set_xlabel('x (m)')
                    if col==0: ax.set_ylabel('y (m)')
        fig.suptitle(f'Scene {seed}, action seed 880001\nBlue: direct, green squares: via relay, orange: disconnected',fontsize=11)
        save_figure(fig,output/'figures',f'topology_{seed}')


def write_report(output):
    import csv
    summary = json.loads((output/'summary.json').read_text())
    verification = json.loads((output/'verification.json').read_text())
    comparisons = list(csv.DictReader((output/'paired_comparisons.csv').open()))
    probes = list(csv.DictReader((output/'fixed_reference_states.csv').open()))
    text = [
        '# B best：收紧动作分布能否维持部署', '',
        '冻结B的best（200704条联合转移），只改变评估动作分布。T=1为原策略；'
        'T=0.7/0.4/0.2使用softmax(logits/T)采样；argmax使用最大logit动作，同值取最小编号。'
        '环境、奖励、网络及actor权重不变，没有梯度更新。', '',
        '32个原验证场景10000–10031；四种温度各使用动作种子880001–880005，argmax每场景只跑一次，'
        '共672个完整400步回合。不是多训练种子实验，不使用保留测试集。', '',
        '## 覆盖与网络维持', '',
        '| 策略 | 平均覆盖 | 最后100步覆盖 | 终态覆盖 | 终态联网数 | 终态多跳接入数 | 终态联网UAV贴边 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for mode, _ in MODES:
        s=summary[mode]
        text.append(f'| {label(mode)} | {s["mean_coverage"]:.2%} | {s["last100_coverage"]:.2%} | '
                    f'{s["final_coverage"]:.2%} | {s["final_connected_count"]:.2f} | {s["final_relay_count"]:.2f} | '
                    f'{s["final_connected_edge30"]:.2%} |')
    text += ['', '贴边指距任意边界不足30 m；先按场景计算联网节点中的比例再平均。多跳接入数不是承担转发的中继数。', '',
             '![覆盖与分布随时间变化](figures/temperature_evolution.png)', '',
             '## 相对T=1的配对变化', '',
             '| 策略 | 平均覆盖变化，百分点（95%场景bootstrap区间） | 最后100步覆盖变化，百分点（95%区间） |',
             '|---|---:|---:|']
    for mode,_ in MODES[1:]:
        cells=[]
        for key in ['mean_coverage','last100_coverage']:
            r=next(r for r in comparisons if r['mode']==mode and r['metric']==key)
            cells.append(f'{float(r["delta"])*100:+.2f} [{float(r["paired_scene_ci_low"])*100:+.2f}, {float(r["paired_scene_ci_high"])*100:+.2f}]')
        text.append(f'| {label(mode)} | '+ ' | '.join(cells)+' |')
    text += ['', '每个场景先对五个动作种子平均，再配对32个场景做5000次bootstrap。'
             '区间条件于当前一个训练模型及五个动作种子，不是训练种子置信区间；也没有校正多重比较。'
             'argmax没有动作采样随机性，未将重复执行算作独立样本。', '',
             '## 相同状态上的概率诊断', '',
             '下表全部在T=1的同一批访问状态上计算，各策略使用相同的原logits。'
             '因此这里的差别来自概率变换本身，不来自飞到不同位置。统计执行前的400个动作状态，'
             '二维指令位移方差为协方差迹，尚未应用边界反射。', '',
             '| 策略 | 熵/nat | 最大动作概率 | 悬停概率 | 指令位移方差/m² |',
             '|---|---:|---:|---:|---:|']
    for mode,_ in MODES:
        selected=[r for r in probes if r['mode']==mode]
        d={k:np.mean([float(r[k]) for r in selected]) for k in DIAGNOSTICS}
        text.append(f'| {label(mode)} | {d["entropy"]:.3f} | {d["max_probability"]:.2%} | '
                    f'{d["hover_probability"]:.2%} | {d["displacement_variance_m2"]:.2f} |')
    text += ['', '各温度沿自身轨迹的概率指标保存在summary.json/curves.csv中；它们还包含状态分布变化。', '',
             '## 动作采样重复', '', '| 策略 | 五个动作种子的平均覆盖标准差，百分点 |', '|---|---:|']
    for mode,_ in MODES:
        sd=summary[mode]['mean_coverage_action_seed_std']
        text.append(f'| {label(mode)} | '+('无采样随机性' if sd is None else f'{sd*100:.2f}')+' |')
    if all(summary[m]['last100_coverage']<summary['t1']['last100_coverage'] for m,_ in MODES[1:]):
        text += ['', '## 解释', '',
                 '**本实验未发现收紧当前策略能改善部署维持；所有更尖锐的分布，其最后100步覆盖均低于T=1。** '
                 '概率诊断确认熵和指令位移方差确实降低，因此不能把无改善解释为温度没有生效。', '',
                 '三个有限低温度的最后100步覆盖配对区间均低于0。argmax的最后100步均值也更低，'
                 '但该项场景区间跨0，其劣化证据弱于三个低温度。argmax后期覆盖下降较慢，'
                 '却稳定在较低覆盖水平；不能将下降较慢等同于维持了高覆盖部署。', '',
                 '结合联网数、联网感知范围及贴边情况，应优先检查动作排序、持续方向偏置和中继协作。'
                 '随机采样可能缓和部分不利偏好，而非单纯破坏正确部署。'
                 '这不等价于每个最高概率动作都错误：改变联合动作的节奏及后续状态分布也会改变效果。'
                 '这并不证明训练时降低熵系数必然无效：重新训练可能学出不同logits；'
                 '也不排除训练量不足或critic误差，本实验只检验已有actor的评估分布。', '']
    text += ['## 固定场景拓扑', '',
             '下面固定使用10000/10001和动作种子880001，按初始、第200步、第400步对比，未按表现选择场景。'
             '蓝三角直接接入，绿方块多跳接入，橙三角断联；绿色UE已服务，紫色UE仅感知，灰色UE未感知。', '',
             '![场景10000](figures/topology_10000.png)', '', '![场景10001](figures/topology_10001.png)', '',
             '## 复核与文件', '',
             f'- 原T=1、880001复现原验证均值及逐场景覆盖；前两个场景位置/B矩阵逐帧匹配，共{verification["matched_original_frames"]}帧。',
             f'- 检查点执行前后SHA256一致：`{verification["checkpoint_sha256"]}`。',
             '- 温度概率、熵随温度收紧、argmax以及位移方差的数值检查通过；原策略没有更新。',
             '- episodes.csv：逐策略/动作种子/场景的阶段指标及动作频率；metrics_*.npz：完整逐时刻指标。',
             '- paired_comparisons.csv：配对差值；fixed_reference_states.csv：同状态概率诊断。',
             '- figures/：PNG、PDF、SVG；trace_*.npz：固定两个场景的三阶段可重建完整轨迹。',
             '- 复现：项目根目录执行`.venv/bin/python scripts/evaluate_policy_temperature.py`。', '']
    (output/'report.md').write_text('\n'.join(text))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/mappo_ablation_b_static_connected_250k_seed0'))
    parser.add_argument('--reference',type=Path,default=Path('analysis/ue_reward_ablations_250k_seed0/b_static_connected'))
    parser.add_argument('--output',type=Path,default=Path('analysis/ue_reward_ablations_250k_seed0/b_best_temperature'))
    parser.add_argument('--action-seeds',type=int,nargs='+',default=[880001,880002,880003,880004,880005])
    args=parser.parse_args()
    evaluate(args.run.resolve(),args.reference.resolve(),args.output.resolve(),args.action_seeds)
