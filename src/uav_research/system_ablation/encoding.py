"""Context ablations. Candidate scoring eligibility is NEVER truncated."""
import math
import torch
from torch import nn
from ..diffusion_dppo.policy import mean_segments, owner_indices
from .settings import options


def setup(actor, cfg):
    o, d = options(cfg), cfg.diffusion
    if o['ue_context'] == 'top10_flat':
        actor.motion_encoder = nn.Sequential(nn.Linear(60,64),nn.ReLU(),nn.Linear(64,d.ue_width),nn.ReLU())
    elif o['ue_context'] == 'statistics':
        actor.motion_encoder = nn.Sequential(nn.Linear(15,d.ue_width),nn.ReLU())
    if o['observation_scope'] == 'global':
        din = cfg.env.num_uavs*4+cfg.env.num_users*3
        actor.global_encoder = nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        # Additional oracle encoder is an explicit capacity difference.
        actor.global_projection = nn.Linear(64,d.context_width,bias=False)


def pools(actor, obs, h):
    """Return observation pool/count and candidate motion pool, all [agent,...]."""
    o = options(actor.cfg)
    if o['ue_context'] == 'full_set':
        return mean_segments(h,obs.lengths),obs.lengths,mean_segments(h[obs.candidate_indices],obs.candidate_lengths)
    if o['ue_context'] == 'statistics':
        mean = mean_segments(obs.users,obs.lengths)
        second = mean_segments(obs.users.square(),obs.lengths)
        maximum = torch.nan_to_num(torch.segment_reduce(obs.users,'max',lengths=obs.lengths,unsafe=True),neginf=0.)
        pool = actor.motion_encoder(torch.cat([mean,second,maximum],-1))
        # Do not leak the full learned candidate pool into the motion path.
        return pool,obs.lengths,torch.zeros_like(pool)
    n = len(obs.lengths)
    owners = owner_indices(obs.lengths,len(h))
    # Stable sort: group, descending beacon quality, then physical UE ID.
    order = torch.argsort(obs.ue_ids,stable=True)
    order = order[torch.argsort(obs.users[order,2],descending=True,stable=True)]
    order = order[torch.argsort(owners[order],stable=True)]
    starts = obs.lengths.cumsum(0)-obs.lengths
    rank = torch.arange(len(h),device=h.device)-starts[owners[order]]
    selected = order[rank < 10]
    own, slot = owners[selected],rank[rank < 10]
    counts = obs.lengths.clamp_max(10)
    padded = torch.zeros((n,10,5),device=h.device)
    valid = torch.zeros((n,10),device=h.device)
    padded[own,slot] = obs.users[selected]
    valid[own,slot] = 1
    if o['ue_context'] == 'top10_flat':
        pool = actor.motion_encoder(torch.cat([padded.flatten(1),valid],-1))
    else:
        # h retains every UE for the complete scoring branch.
        pool = torch.zeros((n,h.shape[-1]),device=h.device).index_add(0,own,h[selected])/counts.clamp_min(1)[:,None]
    candidate_mask = torch.zeros(len(h),dtype=torch.bool,device=h.device)
    candidate_mask[obs.candidate_indices] = True
    choose = candidate_mask[selected]
    cc = torch.zeros(n,device=h.device).index_add(0,own[choose],torch.ones_like(own[choose],dtype=torch.float32))
    candidate_pool = torch.zeros_like(pool).index_add(0,own[choose],h[selected[choose]])/cc.clamp_min(1)[:,None]
    if o['ue_context'] == 'top10_flat':
        # C2/C3 share the SAME subset; raw candidate membership adds no UE rows.
        # Flatten variant uses its fixed-slot embedding for movement as well.
        candidate_pool = pool
    return pool,counts,candidate_pool


def encode(actor, obs):
    from ..diffusion_dppo.policy import Encoding
    h = actor.ue(obs.users)
    pooled,count,motion_pool = pools(actor,obs,h)
    context = actor.context(torch.cat([obs.fixed,pooled,
        (torch.log1p(count.float())/math.log1p(actor.cfg.env.capacity))[:,None]],-1))
    if options(actor.cfg)['observation_scope'] == 'global':
        if obs.global_context is None: raise ValueError('Oracle input missing')
        context = context+actor.global_projection(actor.global_encoder(obs.global_context))
    enc = Encoding(context,h[obs.candidate_indices],obs.candidate_lengths)
    enc.motion_pool = motion_pool
    return enc
