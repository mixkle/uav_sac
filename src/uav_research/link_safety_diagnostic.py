"""Local geometric utilities for a diagnostic scalar link-safety filter.

No production environment or policy factory imports this module. Parent choice
uses only the neighbor records already present in the local observation.
"""
import numpy as np


def maximum_link_scale(relative_xy, actor_displacement, alpha, radius):
    """Maximize s in [0, alpha] subject to ||relative_xy+s*v|| <= radius.

    Returns (s, feasible). When the feasible set is empty, returns (0, False);
    this zero is a fallback and must never be reported as satisfying the bound.
    A point initially outside the ball can still enter it along an inward ray.
    """
    r=np.asarray(relative_xy,np.float64);v=np.asarray(actor_displacement,np.float64)
    if r.shape!=(2,) or v.shape!=(2,) or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError('Require finite 2D vectors')
    if not np.isfinite(alpha) or not 0<=alpha<=1 or not np.isfinite(radius) or radius<=0:
        raise ValueError('Require alpha in [0,1] and positive finite radius')
    if np.linalg.norm(r+alpha*v)<=radius:
        return float(alpha),True
    speed2=float(v@v)
    if speed2==0:
        return (float(alpha),True) if np.linalg.norm(r)<=radius else (0.,False)
    center=-float(r@v)/speed2
    closest=r+center*v
    slack=radius*radius-float(closest@closest)
    tol=32*np.finfo(float).eps*max(radius*radius,float(r@r),1.)
    if slack < -tol:return 0.,False
    half=np.sqrt(max(0.,slack)/speed2)
    lo,hi=max(0.,center-half),min(float(alpha),center+half)
    if lo>hi+1e-12 or hi<0:return 0.,False
    scale=float(np.clip(hi,0,alpha))
    if np.linalg.norm(r+scale*v)>radius+1e-7:return 0.,False
    return scale,True


def visible_parent(fixed, cfg, uav, *, require_lower_hop):
    """Minimum-hop visible connected neighbor; ties: distance, then ID.

    Returns (parent index, observed relative xy, own hop), or (-1,None,h).
    It never scans the global adjacency/positions to reveal a hidden neighbor.
    """
    row=np.asarray(fixed)[uav];u=cfg.num_uavs
    own_h=int(np.rint(row[3]*(u+1)))
    if own_h<=1 or own_h>u:return -1,None,own_h
    start=cfg.self_dim+2*cfg.num_bs
    slots=row[start:start+cfg.neighbor_slots*cfg.neighbor_dim].reshape(cfg.neighbor_slots,cfg.neighbor_dim)
    candidates=[]
    for slot in slots:
        if slot[0]<=0:continue
        j=int(np.rint(slot[0]*u))-1;h=int(np.rint(slot[4]*(u+1)))
        if j<0 or j>=u or j==uav or h>u:continue
        if require_lower_hop and h>=own_h:continue
        offset=np.asarray(slot[1:3],np.float64)*[cfg.width,cfg.height]
        if cfg.observation_coordinates!='relative':offset-=row[:2]*[cfg.width,cfg.height]
        candidates.append((h,float(offset@offset),j,offset))
    if not candidates:return -1,None,own_h
    candidates.sort(key=lambda q:q[:3]);_,_,j,offset=candidates[0]
    return j,offset.copy(),own_h


from .hop_speed_diagnostic import HopSpeedEnvironment
from .environment import reflect
from .vector import VectorEnvironment


class LinkSafetyEnvironment(HopSpeedEnvironment):
    """Independent scalar filters; upstream position is held fixed in the bound.

    No upstream command or future state is read. Infeasibility falls back to
    zero movement, while recording that this does not satisfy the requested ball.
    """
    def __init__(self,cfg,*,margin,alpha_min=.25,k=.4):
        if not np.isfinite(margin) or not 0<margin<cfg.uu_range:
            raise ValueError('Require 0 < margin < UU range')
        self.margin=float(margin)
        self.parents=np.full(cfg.num_uavs,-1,np.int64)
        self.constraint_feasible=np.zeros(cfg.num_uavs,bool)
        self.hop_upper=np.ones(cfg.num_uavs)
        self.parent_pre_xy=np.zeros((cfg.num_uavs,2))
        super().__init__(cfg,enabled=True,alpha_min=alpha_min,k=k)

    def reset(self,seed=None):
        self.parents.fill(-1);self.constraint_feasible.fill(False)
        self.hop_upper.fill(1);self.parent_pre_xy.fill(0)
        return super().reset(seed)

    def set_positions(self,*args,**kwargs):
        self.parents.fill(-1);self.constraint_feasible.fill(False)
        self.hop_upper.fill(1);self.parent_pre_xy.fill(0)
        self.speed_factor.fill(1);self.effective_displacement.fill(0)
        self.pre_action_hops.fill(self.cfg.num_uavs+1)
        return super().set_positions(*args,**kwargs)

    def _move(self,action):
        previous=self.uav_xy.copy()
        self.parents.fill(-1);self.constraint_feasible.fill(False);self.parent_pre_xy.fill(0)
        for i in range(self.cfg.num_uavs):
            j,offset,h=visible_parent(self.obs.fixed,self.cfg,i,require_lower_hop=True)
            if j>=0:
                # The chosen ID is already observed. Use the same visible
                # coordinates at physical float64 precision, not hidden nodes.
                np.testing.assert_allclose(previous[j]-previous[i],offset,atol=.001,rtol=0)
                self.parents[i]=j;self.parent_pre_xy[i]=previous[j]
        super()._move(action)
        self.hop_upper=self.speed_factor.copy()
        nominal=self.last_action*self.cfg.max_speed*self.cfg.dt
        radius=self.cfg.uu_range-self.margin
        for i,j in enumerate(self.parents):
            if j>=0:
                s,feasible=maximum_link_scale(previous[i]-previous[j],nominal[i],self.hop_upper[i],radius)
                self.speed_factor[i]=s;self.constraint_feasible[i]=feasible
        self.effective_displacement=nominal.astype(np.float64)*self.speed_factor[:,None]
        proposed=self.action_origin_xy+self.effective_displacement
        self.uav_xy=(reflect(proposed,[self.cfg.width,self.cfg.height]) if self.cfg.uav_boundary=='reflect'
                     else np.clip(proposed,[0,0],[self.cfg.width,self.cfg.height]))
        self.actual_displacement=self.uav_xy-self.action_origin_xy

    def _install(self,*args):
        super()._install(*args)
        active=self.parents>=0;safe_parents=np.maximum(self.parents,0)
        radius=self.cfg.uu_range-self.margin
        proposed=self.action_origin_xy+self.effective_displacement
        static_distance=np.linalg.norm(proposed-self.parent_pre_xy,axis=1)
        actual_static_distance=np.linalg.norm(self.uav_xy-self.parent_pre_xy,axis=1)
        moving_distance=np.linalg.norm(self.uav_xy-self.uav_xy[safe_parents],axis=1)
        feasible=active&self.constraint_feasible
        eligible=(self.pre_action_hops>1)&(self.pre_action_hops<=self.cfg.num_uavs) if self.t else np.zeros_like(active)
        self.info.update(link_constraint_count=int(active.sum()),
            link_constraint_infeasible_count=int((active&~self.constraint_feasible).sum()),
            link_constraint_limited_count=int((active&(self.speed_factor<self.hop_upper-1e-10)).sum()),
            link_constraint_no_upstream_count=int((eligible&~active).sum()),
            link_constraint_static_violation_count=int((feasible&(static_distance>radius+1e-7)).sum()),
            link_constraint_reflected_static_violation_count=int((feasible&(actual_static_distance>radius+1e-7)).sum()),
            link_constraint_moving_margin_violation_count=int((active&(moving_distance>radius+1e-7)).sum()),
            link_constraint_actual_broken_count=int((active&(moving_distance>self.cfg.uu_range+1e-7)).sum()),
            link_constraint_feasible_but_actual_broken_count=int((feasible&(moving_distance>self.cfg.uu_range+1e-7)).sum()))
        if self.info['link_constraint_static_violation_count']:
            raise AssertionError('Scalar solver violated its feasible stationary-parent bound')

    def trace_frame(self):
        return {**super().trace_frame(),'constraint_parent':self.parents.copy(),
                'constraint_feasible':self.constraint_feasible.copy(),
                'hop_upper':self.hop_upper.copy(),'parent_pre_xy':self.parent_pre_xy.copy()}

    def snapshot(self):
        return {**super().snapshot(),'link_safety_diagnostic':dict(margin=self.margin,upstream='visible lower hop',infeasible='zero'),
                **{n:getattr(self,n).copy() for n in ('parents','constraint_feasible','hop_upper','parent_pre_xy')}}

    def restore(self,data):
        if data.get('link_safety_diagnostic')!=dict(margin=self.margin,upstream='visible lower hop',infeasible='zero'):
            raise ValueError('Link safety diagnostic snapshot mismatch')
        for name in ('parents','constraint_feasible','hop_upper','parent_pre_xy'):
            setattr(self,name,np.asarray(data[name]).copy())
        super().restore(data)


class LinkSafetyVector(VectorEnvironment):
    def __init__(self,cfg,count,backend='batch',*,margin):
        if backend=='process':raise ValueError('Use batch link-safety diagnostic')
        super().__init__(cfg,count,backend)
        self.envs=[LinkSafetyEnvironment(cfg,margin=margin) for _ in range(count)]
