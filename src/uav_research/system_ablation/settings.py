"""Frozen choices for the 2026-11-12 A--F screening experiment."""
import math

DEFAULTS = dict(suite='diffusion-gaussian-1m-v1', run_id='G0', motion_rule='exponential',
    constant_ratio=.5, ue_context='full_set', context_limit=10,
    association='learned_topk', score_actions=True, observation_scope='local',
    initialization='random', initial_checkpoint='', selection_tolerance=1e-12,
    bc_objective='denoising')
FOLLOWUP_DEFAULTS=dict(disable_noisy_score_to_motion=False,diffusion_init_mode='epsilon')


def options(cfg):
    raw = cfg.diffusion.ablation if hasattr(cfg, 'diffusion') else getattr(cfg, '_service_ablation', {})
    return {**DEFAULTS, **FOLLOWUP_DEFAULTS, **raw}


def validate_settings(cfg):
    raw = cfg.diffusion.ablation
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS)-set(FOLLOWUP_DEFAULTS):
        raise ValueError('Unknown system ablation settings')
    o = options(cfg)
    if cfg.run.best_metric != 'last100_coverage_then_throughput':
        raise ValueError('System suite selects best by last100 coverage then throughput')
    if o['suite'] != DEFAULTS['suite'] or not isinstance(o['run_id'], str):
        raise ValueError('Unknown system ablation suite')
    for key, allowed in dict(motion_rule=('exponential','linear','none','constant'),
            ue_context=('full_set','top10_set','top10_flat','statistics'),
            association=('learned_topk','sinr_topk','centralized_greedy'),
            observation_scope=('local','global'), initialization=('random','bc','actor'),
            bc_objective=('denoising','nll')).items():
        if o[key] not in allowed:
            raise ValueError('Invalid ablation '+key)
    if type(o['score_actions']) is not bool or o['context_limit'] != 10:
        raise ValueError('Expected boolean score_actions and 10 context slots')
    if type(o['disable_noisy_score_to_motion']) is not bool or o['diffusion_init_mode'] not in ('epsilon','normal_reference_residual'):
        raise ValueError('Invalid follow-up diffusion parameterization')
    if o['disable_noisy_score_to_motion'] and (
            cfg.algo.name!='diffusion_dppo' or not o['score_actions']):
        raise ValueError('Noisy-score ablation requires joint diffusion actor')
    if o['diffusion_init_mode']!='epsilon' and cfg.algo.name!='diffusion_dppo':
        raise ValueError('Reference-residual initialization requires diffusion actor')
    if o['diffusion_init_mode']!='epsilon' and o['initialization']!='random':
        raise ValueError('Reference-residual experiment must start without BC or actor loading')
    if not 0 < o['constant_ratio'] <= 1 or not math.isfinite(o['constant_ratio']):
        raise ValueError('Invalid constant speed ratio')
    if o['selection_tolerance'] != 1e-12:
        raise ValueError('Freeze lexicographic selection tolerance at 1e-12')
    if not o['score_actions'] and o['association'] == 'learned_topk':
        raise ValueError('Movement-only actor needs rule association')
    if o['motion_rule'] == 'none' and cfg.env.motion_scaling['enabled']:
        raise ValueError('No scaling must disable env.motion_scaling')
    if o['motion_rule'] != 'none' and not cfg.env.motion_scaling['enabled']:
        raise ValueError('Scaling rule requires enabled motion_scaling')
    if cfg.algo.name == 'gaussian_service_mappo' and o['bc_objective'] != 'nll':
        raise ValueError('Gaussian BC uses masked joint-action NLL')
    if o['initialization'] != 'random' and not o['initial_checkpoint']:
        raise ValueError('Initialization requires a source checkpoint')


def speed_factor(hops, cfg):
    import numpy as np
    from ..motion import hop_speed_factor
    o, ms = options(cfg), cfg.motion_scaling
    h = np.asarray(hops)
    if o['motion_rule'] == 'none': return np.ones(h.shape)
    if o['motion_rule'] == 'constant': return np.full(h.shape, o['constant_ratio'], dtype=np.float64)
    if o['motion_rule'] == 'linear':
        return np.where(h <= cfg.num_uavs,
            ms['min_ratio']+(1-ms['min_ratio'])*(h-1)/cfg.num_uavs, 1.)
    return hop_speed_factor(h, cfg.num_uavs, ms['min_ratio'], ms['hop_growth'])


def better(summary, incumbent, tolerance=1e-12):
    """Coverage first; ONLY numerical equality invokes throughput. No tie drift."""
    point = (float(summary['last100_coverage']), float(summary['last100_throughput_bps']))
    if not all(math.isfinite(x) for x in point):
        raise FloatingPointError('Nonfinite selection metrics')
    if incumbent is None: return True, point
    return (point[0] > incumbent[0]+tolerance or
            (abs(point[0]-incumbent[0]) <= tolerance and point[1] > incumbent[1])), point
