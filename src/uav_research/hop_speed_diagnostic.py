"""Read-only policy diagnostic: scale physical motion using pre-action BS hops.

Never selected by the production environment factory. Policy commands and their
one-step history remain unscaled; only the commanded physical displacement is
scaled. Wireless calculations, UE motion, association and rewards are unchanged.
"""
import numpy as np
from .environment import UAVEnvironment, reflect
from .vector import VectorEnvironment


from .motion import hop_speed_factor


class HopSpeedEnvironment(UAVEnvironment):
    def __init__(self, cfg, *, enabled=True, alpha_min=.25, k=.4):
        if cfg.action_space != 'continuous':
            raise ValueError('This diagnostic supports continuous policies only')
        self.enabled, self.alpha_min, self.k = enabled, alpha_min, k
        hop_speed_factor([1], cfg.num_uavs, alpha_min, k)
        self.speed_factor = np.ones(cfg.num_uavs)
        self.pre_action_hops = np.full(cfg.num_uavs, cfg.num_uavs+1, np.int64)
        self.effective_displacement = np.zeros((cfg.num_uavs,2))
        super().__init__(cfg)

    def reset(self, seed=None):
        self.speed_factor.fill(1)
        self.effective_displacement.fill(0)
        self.pre_action_hops.fill(self.cfg.num_uavs+1)
        return super().reset(seed)

    def _move(self, action):
        self.pre_action_hops = self.B.min(axis=1).copy()
        self.speed_factor = (hop_speed_factor(self.pre_action_hops, self.cfg.num_uavs, self.alpha_min, self.k)
                             if self.enabled else np.ones(self.cfg.num_uavs))
        factors = self.speed_factor.copy()
        hops = self.pre_action_hops.copy()
        super()._move(action)
        self.speed_factor, self.pre_action_hops = factors, hops
        # The parent advances UE RNG/time/history exactly once. Recompute only
        # UAV position before the vector environment evaluates any new links.
        nominal = self.last_action*self.cfg.max_speed*self.cfg.dt
        self.effective_displacement = nominal.astype(np.float64)*self.speed_factor[:,None]
        if self.enabled:
            proposed = self.action_origin_xy+self.effective_displacement
            self.uav_xy = (reflect(proposed,[self.cfg.width,self.cfg.height])
                           if self.cfg.uav_boundary=='reflect'
                           else np.clip(proposed,[0,0],[self.cfg.width,self.cfg.height]))
            self.actual_displacement = self.uav_xy-self.action_origin_xy

    def _install(self,*args):
        super()._install(*args)
        changed = np.any(np.abs(self.actual_displacement-self.effective_displacement)>1e-5,axis=1).mean()
        self.info['boundary_reflection_fraction'] = float(self.cfg.uav_boundary=='reflect')*float(changed)
        self.info['boundary_clip_fraction'] = float(self.cfg.uav_boundary=='clip')*float(changed)
        self.info['speed_factor_mean'] = float(self.speed_factor.mean())
        self.info['pre_action_connected_fraction'] = float((self.pre_action_hops<=self.cfg.num_uavs).mean())
        self.info['effective_action_norm_mean'] = float((np.linalg.norm(self.last_action,axis=1)*self.speed_factor).mean())
        self.info['newly_disconnected_fraction'] = float(((self.pre_action_hops<=self.cfg.num_uavs)&(self.B.min(1)>self.cfg.num_uavs)).mean()) if self.t else 0.
        self.info['newly_connected_fraction'] = float(((self.pre_action_hops>self.cfg.num_uavs)&(self.B.min(1)<=self.cfg.num_uavs)).mean()) if self.t else 0.

    def trace_frame(self):
        return {**super().trace_frame(), 'speed_factor':self.speed_factor.copy(),
                'pre_action_hops':self.pre_action_hops.copy(),
                'effective_displacement':self.effective_displacement.copy()}

    def snapshot(self):
        return {**super().snapshot(), 'hop_speed_diagnostic':dict(enabled=self.enabled,alpha_min=self.alpha_min,k=self.k),
                'speed_factor':self.speed_factor.copy(),'pre_action_hops':self.pre_action_hops.copy(),
                'effective_displacement':self.effective_displacement.copy()}

    def restore(self,data):
        if data.get('hop_speed_diagnostic') != dict(enabled=self.enabled,alpha_min=self.alpha_min,k=self.k):
            raise ValueError('Hop-speed diagnostic snapshot mismatch')
        for name in ('speed_factor','pre_action_hops','effective_displacement'):
            setattr(self,name,np.asarray(data[name]).copy())
        super().restore(data)


class HopSpeedVector(VectorEnvironment):
    def __init__(self,cfg,count,backend='batch',**kwargs):
        if backend=='process':
            raise ValueError('Use the local batch diagnostic')
        super().__init__(cfg,count,backend)
        self.envs = [HopSpeedEnvironment(cfg,**kwargs) for _ in range(count)]
