"""Coverage-drop diagnostics and fixed reward ablation definitions.

This module does not alter environment dynamics, rewards, PPO or model loading.
"""
import copy
import numpy as np


REWARD_ABLATIONS = {
    'disconnection': dict(disconnection_weight=.1, robustness_weight=0.),
    'robustness': dict(disconnection_weight=0., robustness_weight=.5),
    'combined': dict(disconnection_weight=.1, robustness_weight=.5),
}
# Keep the historical three-group report matrix unchanged.
DISCONNECTION_SWEEP = {
    'disconnection_mu05': dict(disconnection_weight=.5, robustness_weight=0.),
    'disconnection_mu2': dict(disconnection_weight=2., robustness_weight=0.),
    'disconnection_mu5': dict(disconnection_weight=5., robustness_weight=0.),
}
BASELINE_CONFIG = 'configs/mappo_coverage_throughput_capacity10_1m.yaml'
BASELINE_RUN = 'runs/mappo_coverage_throughput_capacity10_1m_seed0'


def config_path(group):
    if group not in {**REWARD_ABLATIONS, **DISCONNECTION_SWEEP}:
        raise ValueError(f'Unknown reward group: {group}')
    return f'configs/mappo_capacity10_stability_{group}_1m_server.yaml'


def run_path(group):
    if group not in {**REWARD_ABLATIONS, **DISCONNECTION_SWEEP}:
        raise ValueError(f'Unknown reward group: {group}')
    return f'runs/mappo_capacity10_stability_{group}_1m_seed0'


def ablation_config(baseline, group, device='cuda:0'):
    cfg = copy.deepcopy(baseline)
    for key, value in {**REWARD_ABLATIONS, **DISCONNECTION_SWEEP}[group].items():
        setattr(cfg.env, key, value)
    cfg.run.device = device
    return cfg.validate()


def coverage_drop_metrics(coverage):
    """One full episode of post-action coverage; excludes reset t0 -> t1.

    A 10 pp fall is an absolute difference of .1. Five-step falls compare
    endpoints and do not sum overlapping single-step losses. Plateau ties
    are included when computing maximum peak-to-later-state drawdown.
    """
    c = np.asarray(coverage, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all():
        raise ValueError('Expected a nonempty finite episode coverage vector')
    if (c < 0).any() or (c > 1).any():
        raise ValueError('Coverage must lie in [0,1]')
    one = np.maximum(0., c[:-1] - c[1:])
    five = np.maximum(0., c[:-5] - c[5:]) if len(c) > 5 else np.empty(0)
    result = dict(coverage=float(c.mean()), final_coverage=float(c[-1]),
        last100_coverage=float(c[-100:].mean()),
        max_drop_1=float(one.max(initial=0.)),
        max_drop_5=float(five.max(initial=0.)),
        max_drawdown=float((np.maximum.accumulate(c) - c).max()),
        peak_coverage=float(c.max()), peak_step=int(c.argmax()+1),
        min20_coverage=float(np.convolve(c, np.ones(20)/20, 'valid').min())
                       if len(c) >= 20 else float(c.mean()))
    for threshold in (.05, .10, .20):
        code = int(round(threshold*100))
        count = int((one >= threshold - 1e-12).sum())
        result[f'drop_events_{code}pp'] = count
        result[f'any_drop_{code}pp'] = int(count > 0)
    if len(c) >= 150:
        result['early51_150_coverage'] = float(c[50:150].mean())
        result['maintain_coverage'] = result['last100_coverage'] - result['early51_150_coverage']
    return result
