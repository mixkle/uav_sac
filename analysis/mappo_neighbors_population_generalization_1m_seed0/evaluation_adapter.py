"""Read-only population-transfer tools; no training/environment factory changes.

Smaller physical environments exclude removed UAVs from links/interference.
Only the actor input is embedded into the training observation layout.
"""
import numpy as np
import torch

from .environment import UAVEnvironment
from .models import disk_transform
from .observations import RawObservation
from .vector import VectorEnvironment

SUBSET_SEED = 972001
SCENARIOS = {
    'original': (18, 150),
    'ue60': (18, 90),
    'ue80': (18, 120),
    'uav14': (14, 150),
    'uav10': (10, 150),
}


def retained_uav_ids(seed, count, training_count=18):
    """Nested random subsets, fixed before seeing positions or service quality."""
    if not 1 <= count <= training_count:
        raise ValueError('Invalid active UAV count')
    if count == training_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SUBSET_SEED]))
    return np.sort(rng.permutation(training_count)[:count])


def _training_hop(values, active_count, training_count):
    raw = np.rint(np.asarray(values)*(active_count+1))
    return np.where(raw >= active_count+1, 1., raw/(training_count+1)).astype(np.float32)


def embed_observation(obs, active_cfg, training_cfg, active_ids):
    """Map active IDs/adjacencies/hops into frozen training slots, no UE padding.

    active_ids is [U] or [E,U], containing original zero-based training IDs.
    Missing neighbor slots remain all-zero; disconnected hop remains 1.
    """
    u, trained = active_cfg.num_uavs, training_cfg.num_uavs
    if active_cfg.neighbor_slots != training_cfg.neighbor_slots:
        raise ValueError('Neighbor-slot count must not change within a frozen model')
    if active_cfg.observation_history != training_cfg.observation_history:
        raise ValueError('History mode changed')
    ids = np.asarray(active_ids)
    if ids.dtype.kind not in 'iu' or ids.shape[-1] != u:
        raise ValueError('Invalid active ID mapping')
    maps = np.broadcast_to(ids, obs.fixed.shape[:-2]+(u,)).reshape(-1,u)
    if ((maps<0)|(maps>=trained)).any() or any(len(set(row))!=u for row in maps):
        raise ValueError('Active IDs must be distinct training IDs')
    if u == trained and np.array_equal(maps, np.broadcast_to(np.arange(trained), maps.shape)):
        return obs
    src = obs.fixed.reshape(-1,u,active_cfg.fixed_obs_dim)
    dst = np.zeros((len(src),u,training_cfg.fixed_obs_dim),np.float32)
    for batch,mapping in enumerate(maps):
        dst[batch,:,:5] = src[batch,:,:5]
        dst[batch,:,3] = _training_hop(src[batch,:,3],u,trained)
        block = dst[batch,:,5:5+trained]
        block[:,mapping] = src[batch,:,5:5+u]
        old_start,new_start = active_cfg.self_dim,training_cfg.self_dim
        bs_dim = 2*training_cfg.num_bs
        dst[batch,:,new_start:new_start+bs_dim] = src[batch,:,old_start:old_start+bs_dim]
        for slot in range(training_cfg.neighbor_slots):
            a = old_start+bs_dim+slot*active_cfg.neighbor_dim
            b = new_start+bs_dim+slot*training_cfg.neighbor_dim
            row = src[batch,:,a:a+active_cfg.neighbor_dim]
            present = row[:,0]>0
            local_id = np.rint(row[present,0]*u).astype(np.int64)-1
            if ((local_id<0)|(local_id>=u)).any():
                raise ValueError('Malformed neighbor ID')
            dst[batch,present,b] = (mapping[local_id]+1)/trained
            dst[batch,present,b+1:b+6] = row[present,1:6]
            dst[batch,present,b+4] = _training_hop(row[present,4],u,trained)
            # Indexing through a per-UAV view avoids NumPy advanced-index axis moves.
            block = dst[batch,:,b+6:b+6+trained]
            block[:,mapping] = row[:,6:6+u]
        if active_cfg.observation_history == 'one_step':
            dst[batch,:,-4:] = src[batch,:,-4:]
            dst[batch,:,-2] = _training_hop(src[batch,:,-2],u,trained)
    return RawObservation(dst.reshape(*obs.fixed.shape[:-1],training_cfg.fixed_obs_dim),obs.users,obs.offsets)


class PairedMotionRNG:
    """Draw the original population's motion, then retain the same UE identities."""
    def __init__(self, generator, source_users, user_ids):
        self.generator,self.source_users,self.user_ids = generator,source_users,np.asarray(user_ids)

    def normal(self, loc, scale, size):
        if size != len(self.user_ids): raise ValueError('Unexpected motion sample shape')
        return self.generator.normal(loc,scale,self.source_users)[self.user_ids]

    def uniform(self, low, high, size):
        if size != len(self.user_ids): raise ValueError('Unexpected motion sample shape')
        return self.generator.uniform(low,high,self.source_users)[self.user_ids]


class PopulationEnvironment(UAVEnvironment):
    """Diagnostic-only paired reset/motion; inherited physical dynamics unchanged."""
    def __init__(self, cfg, training_cfg):
        super().__init__(cfg)
        self.training_cfg = training_cfg

    def reset(self, seed=None):
        if seed is None: raise ValueError('Diagnostic scenes require an explicit seed')
        original = UAVEnvironment(self.training_cfg)
        original.reset(seed)
        super().reset(seed)
        self.active_ids = retained_uav_ids(seed,self.cfg.num_uavs,self.training_cfg.num_uavs)
        self.user_ids = np.arange(self.cfg.num_users,dtype=np.int64)
        self.movement_rng = PairedMotionRNG(
            np.random.default_rng(np.random.SeedSequence([int(seed),201])),
            self.training_cfg.num_users,self.user_ids)
        return self.set_positions(original.uav_xy[self.active_ids],original.user_xy[self.user_ids])

    def trace_frame(self):
        return dict(**super().trace_frame(),active_uav_ids=self.active_ids.copy(),active_ue_ids=self.user_ids.copy())


class PopulationVector(VectorEnvironment):
    def __init__(self, cfg, training_cfg, count):
        super().__init__(cfg,count,'batch')
        self.envs = [PopulationEnvironment(cfg,training_cfg) for _ in range(count)]

    @property
    def active_ids(self):
        return np.stack([env.active_ids for env in self.envs])


@torch.no_grad()
def paired_action(actor, obs, active_cfg, training_cfg, active_ids):
    """Full-18 Gaussian noise bank preserves retained agents' random numbers."""
    embedded = embed_observation(obs,active_cfg,training_cfg,active_ids)
    device = next(actor.parameters()).device
    distribution = actor.distribution(embedded.to_torch(device))
    batch_shape = distribution.loc.shape[:-2]
    bank = torch.empty((*batch_shape,training_cfg.num_uavs,2),
                       dtype=distribution.loc.dtype,device=device).normal_()
    ids = torch.as_tensor(active_ids,device=device)
    ids = torch.broadcast_to(ids,(*batch_shape,active_cfg.num_uavs))
    noise = torch.gather(bank,-2,ids[...,None].expand(*ids.shape,2))
    latent = distribution.loc+noise*distribution.scale
    action,_ = disk_transform(latent)
    return action.cpu().numpy()
