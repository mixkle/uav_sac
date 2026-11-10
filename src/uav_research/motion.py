"""Deterministic execution transforms shared by training and diagnostics."""
import numpy as np

def hop_speed_factor(hops, num_uavs, alpha_min=.25, k=.4):
    hops = np.asarray(hops)
    if not (np.isfinite(alpha_min) and 0 < alpha_min <= 1 and np.isfinite(k) and k > 0):
        raise ValueError("Invalid hop scaling parameters")
    if num_uavs < 1 or not np.isfinite(hops).all() or np.any(hops != np.floor(hops)) or np.any(hops < 1):
        raise ValueError("Require positive integer hop counts")
    connected = hops <= num_uavs
    result = np.ones(hops.shape, np.float64)
    result[connected] = alpha_min+(1-alpha_min)*(-np.expm1(-k*(hops[connected]-1)))/(-np.expm1(-k*num_uavs))
    return result

def motion_contract(cfg):
    return dict(motion_scaling=dict(cfg.motion_scaling), history_motion_source=cfg.history_motion_source)
