"""Frozen M1-C / M1-C-H speed intervention; never trains or queries a critic.

Only environment max_speed changes. UE paths, initial scenes, normalized action
noise, episode duration, and the checkpoint are paired across 15/30/45 m/s.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from uav_research.bridge_diagnostics import DIAGNOSTICS
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


BASE = 'continuous_mappo_full_reward_time_factor_relative_central'
RUNS = [f'runs/{BASE}{suffix}_100k_seed0' for suffix in ['', '_history']]
LABELS = ['M1-C', 'M1-C-H']
DEFAULT_OUTPUT = 'analysis/mappo_central_motion_scale_100k_seed0'
EXTRA_FIELDS = ['cumulative_actual_distance_m', 'cumulative_command_distance_m',
                'net_displacement_mean_m', 'spatial_radius_rms_m',
                'command_norm_mean', 'centroid_x_m', 'centroid_y_m']
FIELDS = DIAGNOSTICS + EXTRA_FIELDS


def moving_average(x, window=21):
    """Centered window; peak time denotes center of action-result indices 1..400."""
    if window % 2 != 1 or window > x.shape[-1]:
        raise ValueError('An odd window no larger than the horizon is required')
    padded = np.concatenate([np.zeros((*x.shape[:-1], 1)), np.cumsum(x, axis=-1)], -1)
    return (padded[..., window:] - padded[..., :-window]) / window


def peak_times(curves, window=21):
    return moving_average(curves, window).argmax(-1) + window // 2 + 1


def rollout(cfg, policy, speed, scene_seeds, action_seed):
    started = time.perf_counter()
    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.max_speed = float(speed)
    unchanged = copy.deepcopy(env_cfg)
    unchanged.max_speed = cfg.env.max_speed
    assert unchanged == cfg.env and env_cfg.dt == 1 and env_cfg.horizon == 400
    chunks, positions, checks = [], [], []
    with isolated_rng(), torch.inference_mode():
        torch.manual_seed(int(action_seed))
        for begin in range(0, len(scene_seeds), cfg.run.num_envs):
            seeds = scene_seeds[begin:begin + cfg.run.num_envs]
            with VectorEnvironment(env_cfg, len(seeds), 'batch') as vector:
                obs, _, _ = vector.reset(seeds)
                initial_uav = np.stack([e.uav_xy for e in vector.envs])
                initial_ue = np.stack([e.user_xy for e in vector.envs])
                initial_h = hashlib.sha256(initial_uav.tobytes() + initial_ue.tobytes()).hexdigest()
                ue_hash = hashlib.sha256(initial_ue.tobytes())
                batch_positions = [initial_uav.astype(np.float32)]
                cumulative_actual = np.zeros(len(seeds))
                cumulative_command = np.zeros(len(seeds))
                rows = []
                for t in range(env_cfg.horizon):
                    action = policy.act(obs, deterministic=False)
                    if t == 0:
                        first_action_hash = hashlib.sha256(action.tobytes()).hexdigest()
                    command = np.linalg.norm(action, axis=-1).mean(-1)
                    cumulative_command += command * speed * env_cfg.dt
                    obs, _, _, done, truncated, infos = vector.step(action)
                    np.testing.assert_array_equal(done, np.full(len(seeds), t == env_cfg.horizon-1))
                    assert not truncated.any()
                    xy = np.stack([e.uav_xy for e in vector.envs])
                    ue_hash.update(np.stack([e.user_xy for e in vector.envs]).tobytes())
                    actual = np.array([i['actual_speed_mean'] for i in infos]) * env_cfg.dt
                    assert np.all(actual <= speed * env_cfg.dt + 1e-6)
                    cumulative_actual += actual
                    center = xy.mean(1)
                    extra = np.column_stack([cumulative_actual, cumulative_command,
                        np.linalg.norm(xy-initial_uav, axis=-1).mean(1),
                        np.sqrt(((xy-center[:, None])**2).sum(-1).mean(1)),
                        command, center])
                    diagnostics = np.array([[i[k] for k in DIAGNOSTICS] for i in infos])
                    np.testing.assert_allclose(diagnostics[:, DIAGNOSTICS.index('coverage')],
                        diagnostics[:, DIAGNOSTICS.index('discovery_fraction')] -
                        sum(diagnostics[:, DIAGNOSTICS.index(k)] for k in
                            ['downlink_rejected_users', 'backhaul_rejected_users', 'capacity_rejected_users'])
                        / env_cfg.num_users, atol=1e-12, rtol=1e-12)
                    rows.append(np.column_stack([diagnostics, extra]))
                    batch_positions.append(xy.astype(np.float32))
                chunks.append(np.swapaxes(np.stack(rows), 0, 1))
                positions.append(np.swapaxes(np.stack(batch_positions), 0, 1))
                checks.append(dict(begin=begin, initial_sha256=initial_h,
                                   ue_path_sha256=ue_hash.hexdigest(),
                                   first_action_sha256=first_action_hash,
                                   final_action_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
                print(f'  speed={speed}, action_seed={action_seed}, scenes={begin+len(seeds)}/{len(scene_seeds)}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    return np.concatenate(chunks), np.concatenate(positions), checks


def build(output, scene_count=64, action_seeds=(880001, 880002, 880003)):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    if scene_count not in [16,32,48,64]:
        raise ValueError('Use 16/32/48/64 diagnostic scenes, preserving the reference batch sizes')
    runtime_cfg = load_config(Path(RUNS[0])/'config.yaml')
    runtime_cfg.run.cpu_threads = 8
    runtime_cfg.run.numba_threads = 4
    configure_runtime(runtime_cfg)
    scenes = list(range(610000, 610000+scene_count))
    sources = [Path(r)/name for r in RUNS for name in ['config.yaml', 'last.pt']]
    sources += [Path(__file__)] + list(Path('src/uav_research').glob('*.py'))
    hashes = {str(p): digest(p) for p in sources}
    meta = dict(status='running', frozen=True, actor_updates=0, critic_queries=0,
                scene_seeds=scenes, action_seeds=list(action_seeds), speeds=[15,30,45],
                cpu_threads=8, numba_threads=4, batch_size=16, groups={},
                source_sha256=hashes, scope='Dynamics intervention, not retraining or a policy-quality benchmark')
    atomic_json(out/'experiment.json', meta)
    paired_ue_paths = {}
    for label, run in zip(LABELS, RUNS):
        configure_runtime(runtime_cfg)
        cfg, policy = policy_from_checkpoint(Path(run)/'last.pt', 'cpu')
        cfg.run.cpu_threads = 8
        cfg.run.numba_threads = 4
        cfg.run.num_envs = 16
        configure_runtime(cfg)
        assert not set(scenes).intersection(cfg.run.test_seeds)
        assert cfg.env.max_speed == 30 and cfg.env.boundary_penalty_weight == 0
        assert policy.provenance['checkpoint_env_steps'] == 100000
        policy.actor.eval().requires_grad_(False)
        before = actor_digest(policy.actor)
        meta['groups'][label] = dict(run=run, actor_sha256=before, checkpoint_steps=100000,
                                     training_config=cfg.as_dict(), speeds={})
        initial_actions = {}
        for speed in [15,30,45]:
            started = time.perf_counter()
            arrays, positions, checks = [], [], []
            for action_seed in action_seeds:
                values, xy, audit = rollout(cfg, policy, speed, scenes, action_seed)
                for b in audit:
                    key = (int(action_seed), b['begin'])
                    physical = (b['initial_sha256'], b['ue_path_sha256'], b['final_action_rng_sha256'])
                    if key in paired_ue_paths:
                        assert physical == paired_ue_paths[key], 'Physical layout, UE path or action noise differs'
                    else:
                        paired_ue_paths[key] = physical
                    if key in initial_actions:
                        assert b['first_action_sha256'] == initial_actions[key]
                    else:
                        initial_actions[key] = b['first_action_sha256']
                checks.append(dict(action_seed=int(action_seed), batches=audit))
                arrays.append(values); positions.append(xy)
                print(f'{label} speed={speed} action_seed={action_seed}: {time.perf_counter()-started:.1f}s', flush=True)
            data = np.stack(arrays)
            # Reproduce all original diagnostics, not just aggregate coverage.
            matched = []
            if speed == 30:
                for i, seed in enumerate(action_seeds):
                    original = Path('analysis/mappo_central_history_100k_seed0')/label/f'diagnostic/last/{seed}/audit.npz'
                    if original.exists():
                        with np.load(original) as saved:
                            np.testing.assert_array_equal(saved['scene_seeds'][:scene_count], scenes)
                            assert list(saved['diagnostic_names']) == DIAGNOSTICS
                            np.testing.assert_allclose(data[i, ..., :len(DIAGNOSTICS)],
                                saved['diagnostics'][:scene_count], rtol=0, atol=0)
                        matched.append(str(original))
            target = out/label/f'speed_{speed}'
            target.mkdir(parents=True)
            np.savez_compressed(target/'curves.npz', diagnostics=data, fields=np.array(FIELDS),
                uav_xy=np.stack(positions), scene_seeds=np.array(scenes), action_seeds=np.array(action_seeds),
                speed=speed, episode_steps=np.arange(1,401))
            assert actor_digest(policy.actor) == before
            meta['groups'][label]['speeds'][str(speed)] = dict(seconds=time.perf_counter()-started,
                checks=checks, exact_original_30m_reproductions=matched)
            atomic_json(out/'experiment.json', meta)
    assert all(digest(Path(p)) == h for p,h in hashes.items())
    meta['status'] = 'completed'
    meta['all_initial_layouts_ue_paths_action_noise_paired'] = True
    meta['source_hashes_unchanged'] = True
    atomic_json(out/'experiment.json', meta)
    publish(out)


def publish(output):
    out = Path(output)
    meta = json.loads((out/'experiment.json').read_text())
    if meta['status'] != 'completed':
        raise ValueError('Cannot publish an incomplete experiment')
    style()
    summary, sensitivity, differences, scene_rows = [], [], [], []
    peak_boot, datasets = {}, {}
    count = len(meta['scene_seeds'])
    draws = np.random.default_rng(950001).integers(count, size=(2000,count))
    weights = np.stack([np.bincount(x, minlength=count) for x in draws])/count
    for label in LABELS:
        datasets[label] = {}
        for speed in meta['speeds']:
            with np.load(out/label/f'speed_{speed}/curves.npz') as saved:
                assert list(saved['fields']) == FIELDS
                data = saved['diagnostics'].mean(0)  # average action repeats within scene first
            datasets[label][speed] = data
            def field(k): return data[..., FIELDS.index(k)]
            cov = field('coverage')
            mean = cov.mean(0)
            at = int(peak_times(mean))
            boot_curves = weights @ cov
            peaks = peak_times(boot_curves)
            peak_boot[(label,speed)] = peaks
            smooth = moving_average(mean)
            plateau = np.flatnonzero(smooth >= .99*smooth.max())+11
            row = dict(model=label, speed=speed, peak_step=at,
                peak_ci_low=float(np.quantile(peaks,.025)), peak_ci_high=float(np.quantile(peaks,.975)),
                peak_coverage=float(smooth.max()), coverage=float(mean.mean()),
                last100_coverage=float(mean[-100:].mean()), final_coverage=float(mean[-1]),
                steps51_150_coverage=float(mean[50:150].mean()),
                maintenance=float(mean[-100:].mean()-mean[50:150].mean()),
                mean_actual_speed=float(field('actual_speed_mean').mean()),
                peak_cumulative_actual_m=float(field('cumulative_actual_distance_m')[:,at-1].mean()),
                peak_net_displacement_m=float(field('net_displacement_mean_m')[:,at-1].mean()),
                peak_equivalent_30m_step=at*speed/30,
                plateau99_first=int(plateau[0]), plateau99_last=int(plateau[-1]),
                scene_peak_median=float(np.median(peak_times(cov))),
                late_peak_fraction=float((peak_times(cov)>=380).mean()),
                final_sensing=float(field('discovery_fraction')[:,-1].mean()),
                final_disconnection=float(field('disconnection_ratio')[:,-1].mean()),
                final_nearest_m=float(field('uav_nearest_neighbor_mean')[:,-1].mean()),
                final_boundary150=float(field('boundary_within_150_fraction')[:,-1].mean()))
            summary.append(row)
            for window in [11,21,41]:
                sensitivity.append(dict(model=label,speed=speed,window=window,
                    peak_step=int(peak_times(mean,window))))
            for i, seed in enumerate(meta['scene_seeds']):
                scene_rows.append(dict(model=label,speed=speed,scene_seed=seed,
                    peak_step=int(peak_times(cov[i])),coverage=float(cov[i].mean()),
                    last100_coverage=float(cov[i,-100:].mean()), final_coverage=float(cov[i,-1])))
        ref = datasets[label][30]
        for speed in [15,45]:
            current = datasets[label][speed]
            measures = dict(
                coverage=(current[...,FIELDS.index('coverage')]-ref[...,FIELDS.index('coverage')]).mean(1),
                last100_coverage=(current[:,-100:,FIELDS.index('coverage')]-ref[:,-100:,FIELDS.index('coverage')]).mean(1),
                final_coverage=current[:,-1,FIELDS.index('coverage')]-ref[:,-1,FIELDS.index('coverage')])
            for name, values in measures.items():
                differences.append(dict(model=label,speed=speed,reference_speed=30,metric=name,**interval(values)))
            delta = peak_boot[(label,speed)]-peak_boot[(label,30)]
            own = next(r for r in summary if r['model']==label and r['speed']==speed)
            base = next(r for r in summary if r['model']==label and r['speed']==30)
            differences.append(dict(model=label,speed=speed,reference_speed=30,metric='mean_curve_peak_step',
                mean=own['peak_step']-base['peak_step'],low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975))))
    for name, rows in [('summary.csv',summary),('paired_differences.csv',differences),
                       ('peak_window_sensitivity.csv',sensitivity),('scene_metrics.csv',scene_rows)]:
        write_csv(out/name, rows)
    figures = out/'figures'
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ri,label in enumerate(LABELS):
        for j,speed in enumerate(meta['speeds']):
            data=datasets[label][speed]; cov=data[...,FIELDS.index('coverage')].mean(0)
            sm=moving_average(cov); times=np.arange(11,391)
            at=int(peak_times(cov))
            axes[ri,0].plot(times,sm*100,color=COLORS[j],label=f'{speed} m/s; peak {at}')
            axes[ri,0].scatter(at,sm.max()*100,color=COLORS[j],s=20)
            axes[ri,1].plot(times*speed/30,sm*100,color=COLORS[j],label=f'{speed} m/s')
            path=data[...,FIELDS.index('cumulative_actual_distance_m')].mean(0)
            axes[ri,2].plot(path[10:390],sm*100,color=COLORS[j],label=f'{speed} m/s')
        for ax in axes[ri]: ax.legend(fontsize=8); ax.set(ylabel=f'{label}: coverage (%)')
        axes[ri,0].set(xlabel='Episode step',title='Frozen policy; 21-step centered mean')
        axes[ri,1].set(xlabel='Step x speed / 30',title='Nominal movement-normalized time')
        axes[ri,2].set(xlabel='Mean cumulative actual path per UAV (m)',title='Observed movement scale')
    save_figure(fig,figures,'coverage_motion_scale')
    keys=['discovery_fraction','disconnection_ratio','uav_nearest_neighbor_mean',
          'boundary_within_150_fraction','net_displacement_mean_m','spatial_radius_rms_m']
    titles=['Sensing fraction','Disconnected UAV fraction','Nearest neighbor distance (m)',
            'Within 150 m of boundary','Mean displacement from start (m)','Spatial RMS radius (m)']
    for label in LABELS:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,key,title in zip(axes.flat,keys,titles):
            for j,speed in enumerate(meta['speeds']):
                y=datasets[label][speed][...,FIELDS.index(key)].mean(0)
                ax.plot(np.arange(1,401),y,color=COLORS[j],label=f'{speed} m/s')
            ax.set(xlabel='Episode step',title=title); ax.legend(fontsize=8)
        fig.suptitle(f'{label}; 64 paired diagnostic scenes x 3 action seeds' if count==64
                     else f'{label}; {count} paired diagnostic scenes')
        save_figure(fig,figures,f'{label}_deployment_diagnostics')
    atomic_json(out/'summary.json',dict(summary=summary,paired_differences=differences,
                                       smoothing_sensitivity=sensitivity))
    lines=['# 冻结策略的运动尺度诊断','',
        '仅改变环境UAV最大速度15/30/45 m/s，dt=1秒、400步回合、UE运动、通信和归一化策略保持原样。两模型均使用100k last；不训练，不调用critic。',
        '64个场景610000–610063为已经用于研究的诊断场景，不是保留测试集。动作种子880001/880002/880003，每场景先平均动作重复，再进行2000次配对场景bootstrap（950001）。',
        '峰值为21步居中移动平均的最大点，时间取窗口中心；区间是对场景重采样后重新求平均曲线峰值，不是训练种子区间，也不是单场景峰值区间。相近多个峰会导致区间很宽。另保存11/41步平滑敏感性，避免把单个最大点当作精确转折。','',
        '|模型|速度(m/s)|峰值步[场景区间]|平滑峰值覆盖|全程覆盖|后100步|峰值累计实际路程(m/UAV)|峰值平均净位移(m/UAV)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["model"]}|{r["speed"]}|{r["peak_step"]} [{r["peak_ci_low"]:.0f}, {r["peak_ci_high"]:.0f}]|{r["peak_coverage"]:.2%}|{r["coverage"]:.2%}|{r["last100_coverage"]:.2%}|{r["peak_cumulative_actual_m"]:.0f}|{r["peak_net_displacement_m"]:.0f}|')
    lines+=['','![覆盖与运动尺度](figures/coverage_motion_scale.png)','',
            '![原策略诊断](figures/M1-C_deployment_diagnostics.png)','',
            '![历史策略诊断](figures/M1-C-H_deployment_diagnostics.png)','',
            '## 解释边界','',
            '这是对冻结策略的动力学干预；15/45不是各自重新训练后的表现。相同高斯随机流不意味着状态演化后动作相同，只有初始动作相同；尤其一步历史作用的时间尺度也随物理速度改变。UE仍按原真实时间移动，不能期待轨迹在缩放后完全重合。',
            '若速度增加时覆盖峰值系统性提前，支持空间部署演化影响转折；若累计路程或净位移不能对齐，不能宣称存在统一临界距离或精确1/v规律。速度干预不直接检验奖励、信用分配或actor优化哪个机制最初造成了持续运动。',
            '30 m/s必须逐元素复现既有诊断数据；配对初始布局、完整UE路径、动作随机流消耗以及各模型相同的初始动作跨速度严格核对。源码、源配置、checkpoint及actor哈希前后保持不变。',
            '', '## 重建','', '```bash',
            f'.venv/bin/python scripts/check_mappo_motion_scale.py --output {out} --rebuild', '```']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    parser.add_argument('--scenes',type=int,default=64)
    parser.add_argument('--action-seeds',type=int,nargs='+',default=[880001,880002,880003])
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if args.rebuild: publish(args.output)
    else: build(args.output,args.scenes,args.action_seeds)
