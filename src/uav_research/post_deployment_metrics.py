"""Trajectory-first late-deployment metrics; missing evidence stays missing."""
from pathlib import Path
import numpy as np


METRIC_LABELS = {
    'tail100_coverage_pct': '后100覆盖 %',
    'tail50_coverage_pct': '后50覆盖 %',
    'final_coverage_pct': '终态覆盖 %',
    'tail100_throughput_mbps': '后100吞吐 Mbps',
    'tail50_throughput_mbps': '后50吞吐 Mbps',
    'final_throughput_mbps': '终态吞吐 Mbps',
    'tail100_coverage_std_pp': '后100平均σ pp',
    'tail50_coverage_std_pp': '后50平均σ pp',
    'episode_return': '场景平均回合回报',
    'tail100_connected_pct': '后100联网UAV %',
    'tail50_connected_pct': '后50联网UAV %',
    'final_connected_pct': '终态联网UAV %',
    'tail100_drop_count': '后100骤降 次/回合',
    'tail50_drop_count': '后50骤降 次/回合',
    'tail100_max_drop_pp': '后100最大跌幅 pp',
    'tail50_max_drop_pp': '后50最大跌幅 pp',
    'tail100_coverage_p05_pct': '后100覆盖P05 %',
    'tail50_coverage_p05_pct': '后50覆盖P05 %',
    'formation_step': '首次达标时间 步',
}
METRICS = tuple(METRIC_LABELS)


def trajectory_metrics(coverage, scenes, *, throughput=None, rewards=None,
                       disconnection=None):
    """Return one scalar per scene, before averaging action repetitions.

    All times are post-action. A final-N window includes the transition into
    its first state. At least N+1 states are required to measure that drop.
    Optional fields return None, never a fabricated zero.
    """
    c = np.asarray(coverage, dtype=np.float64)
    scenes = np.asarray(scenes, dtype=np.int64)
    if c.ndim != 2 or c.shape[0] != len(scenes) or c.shape[1] < 101:
        raise ValueError('Expected >=101 post-action steps and explicit scene IDs')
    if len(np.unique(scenes)) != len(scenes):
        raise ValueError('Duplicate scene ID within a repetition')
    if not np.isfinite(c).all() or np.any((c < -1e-9) | (c > 1+1e-9)):
        raise ValueError('Invalid coverage')
    checked = {}
    for name, value in [('throughput', throughput), ('rewards', rewards),
                        ('disconnection', disconnection)]:
        if value is None:
            checked[name] = None
            continue
        a = np.asarray(value, dtype=np.float64)
        if a.shape != c.shape or not np.isfinite(a).all():
            raise ValueError('Nonfinite or misaligned '+name)
        if name == 'throughput' and np.any(a < -1e-6):
            raise ValueError('Negative throughput')
        if name == 'disconnection' and np.any((a < -1e-9) | (a > 1+1e-9)):
            raise ValueError('Invalid disconnection fraction')
        checked[name] = a
    r, reward, d = (checked[k] for k in ['throughput', 'rewards', 'disconnection'])
    out = dict(scene_seed=scenes)
    drops = np.maximum(0, c[:, :-1] - c[:, 1:])
    for n in [100, 50]:
        tail = c[:, -n:]
        out[f'tail{n}_coverage_pct'] = tail.mean(1)*100
        out[f'tail{n}_coverage_std_pp'] = tail.std(1, ddof=0)*100
        out[f'tail{n}_coverage_p05_pct'] = np.quantile(tail, .05, axis=1, method='linear')*100
        out[f'tail{n}_drop_count'] = (drops[:, -n:] >= .1-1e-12).sum(1).astype(float)
        out[f'tail{n}_max_drop_pp'] = drops[:, -n:].max(1)*100
        out[f'tail{n}_throughput_mbps'] = None if r is None else r[:, -n:].mean(1)/1e6
        out[f'tail{n}_connected_pct'] = None if d is None else (1-d[:, -n:]).mean(1)*100
    out['final_coverage_pct'] = c[:, -1]*100
    out['final_throughput_mbps'] = None if r is None else r[:, -1]/1e6
    out['final_connected_pct'] = None if d is None else (1-d[:, -1])*100
    out['episode_return'] = None if reward is None else reward.sum(1)
    target = c[:, -100:].mean(1)
    crossed = c >= target[:, None]-1e-12
    if not crossed.any(1).all():
        raise ValueError('Last100 mean must have a crossing in its own trajectory')
    out['formation_step'] = crossed.argmax(1).astype(float)+1
    return out


def summarize(repetitions, *, bootstrap_repeats=2000, seed=950001):
    """First average per-trajectory statistics within a scene, then scenes."""
    if not repetitions:
        raise ValueError('No trajectories')
    scenes = repetitions[0]['scene_seed']
    for rep in repetitions:
        np.testing.assert_array_equal(rep['scene_seed'], scenes)
    idx = np.random.default_rng(seed).integers(len(scenes), size=(bootstrap_repeats, len(scenes)))
    result, intervals, values = {}, {}, {}
    for key in METRICS:
        if any(rep[key] is None for rep in repetitions):
            values[key] = result[key] = intervals[key] = None
        else:
            a = np.mean([rep[key] for rep in repetitions], axis=0)
            values[key] = a
            result[key] = float(a.mean())
            intervals[key] = np.quantile(a[idx].mean(1), [.025, .975]).tolist()
    result.update(scene_count=len(scenes), action_repetitions=len(repetitions))
    return result, intervals, values


def load_trajectory(path, *, throughput_scale_bps=None, coverage_name='coverage',
                    throughput_name='throughput_bps', reward_name='reward', expected_steps=400):
    """Load current and historical formats with explicit physical units."""
    with np.load(Path(path), allow_pickle=False) as z:
        fields = {}
        if 'raw' in z and 'raw_names' in z:
            fields.update(zip(z['raw_names'].tolist(), np.moveaxis(z['raw'], -1, 0)))
        if 'metrics' in z and 'names' in z:
            fields.update(zip(z['names'].tolist(), np.moveaxis(z['metrics'], -1, 0)))
        if 'diagnostics' in z:
            fields.update(zip(z['diagnostic_names'].tolist(), np.moveaxis(z['diagnostics'], -1, 0)))
        coverage = fields[coverage_name]
        if coverage.ndim != 2 or coverage.shape[1] != expected_steps:
            raise ValueError('Expected a complete '+str(expected_steps)+'-step episode')
        throughput = fields.get(throughput_name)
        if throughput is None and 'tau' in fields and throughput_scale_bps is not None:
            throughput = fields['tau']*throughput_scale_bps
        reward = fields.get(reward_name)
        if reward_name == 'reward':
            for key in ['reward', 'rewards']:
                if key in z:
                    reward = np.asarray(z[key], dtype=np.float64)
        scenes_key = 'scene_seeds' if 'scene_seeds' in z else 'seeds'
        scenes = z[scenes_key]
        result = trajectory_metrics(coverage, scenes, throughput=throughput,
                                    rewards=reward, disconnection=fields.get('disconnection_ratio'))
    return result
