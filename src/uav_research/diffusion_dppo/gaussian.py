"""Set-conditioned diagonal Gaussian over movement and ragged UE score latents.

PPO treats pre-transform latent samples as augmented actions. The fixed disk and
sigmoid Jacobians cancel in the ratio, including with deterministic hop scaling.
"""
import math
import time
import numpy as np
import torch
from torch import nn
from ..models import disk_transform
from .policy import Encoding, mean_segments, sum_segments, network
from .observations import ServiceAction, DiffusionChain, offsets


class GaussianServiceActor(nn.Module):
    steps = 1  # One joint Gaussian likelihood, never a DDPM transition.

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.diffusion
        self.ue = nn.Sequential(nn.Linear(5, d.ue_width), nn.ReLU(), nn.Linear(d.ue_width, d.ue_width), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(cfg.env.fixed_obs_dim+d.ue_width+1,d.context_width),
            nn.ReLU(), nn.Linear(d.context_width,d.context_width), nn.ReLU())
        self.movement_head = network(d.context_width+d.ue_width,256,4)
        self.score_head = network(d.context_width+2*d.ue_width,256,2)

    def encode(self, obs):
        h = self.ue(obs.users)
        count = torch.log1p(obs.lengths.float())/math.log1p(self.cfg.env.capacity)
        context = self.context(torch.cat([obs.fixed,mean_segments(h,obs.lengths),count[:,None]],-1))
        return Encoding(context,h[obs.candidate_indices],obs.candidate_lengths)

    def distribution(self, enc):
        pooled = mean_segments(enc.candidates,enc.lengths)
        m = self.movement_head(torch.cat([enc.context,pooled],-1))
        s = self.score_head(torch.cat([enc.candidates,enc.context[enc.owners],pooled[enc.owners]],-1))
        a = self.cfg.algo
        return m[:,:2],m[:,2:].clamp(a.log_std_min,a.log_std_max),s[:,0],s[:,1].clamp(a.log_std_min,a.log_std_max)

    @staticmethod
    def joint_logp(enc, params, movement, scores):
        mm,ml,sm,sl = params
        lm = (-.5*((movement-mm)*(-ml).exp()).square()-ml-.5*math.log(2*math.pi)).sum(-1)
        ls = -.5*((scores-sm)*(-sl).exp()).square()-sl-.5*math.log(2*math.pi)
        return lm+sum_segments(ls[:,None],enc.lengths).squeeze(-1)

    @torch.no_grad()
    def sample(self, obs, generator=None, deterministic=False):
        device = next(self.parameters()).device
        start = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if device.type == "cuda" else None
        if events: events[0].record()
        enc = self.encode(obs.tensor(device))
        encoded_at = time.perf_counter()
        if events: events[1].record()
        params = self.distribution(enc)
        mm,ml,sm,sl = params
        m = mm if deterministic else mm+ml.exp()*torch.randn(mm.shape,device=device,generator=generator)
        s = sm if deterministic else sm+sl.exp()*torch.randn(sm.shape,device=device,generator=generator)
        logp = self.joint_logp(enc,params,m,s)
        if not (torch.isfinite(m).all() & torch.isfinite(s).all() & torch.isfinite(logp).all()):
            raise FloatingPointError("Nonfinite Gaussian service action")
        sampled_at = time.perf_counter()
        action = ServiceAction(disk_transform(m)[0].cpu().numpy().reshape(*obs.fixed.shape[:-1],2),
            s.sigmoid().cpu().numpy(),obs.candidate_ids.copy(),offsets(obs.candidate_lengths))
        # Reuse ragged storage with two endpoints; the first is a placeholder,
        # excluded from likelihood. The kind explicitly distinguishes Gaussian.
        chain = DiffusionChain(torch.stack([torch.zeros_like(m),m],1).cpu().numpy(),
            torch.stack([torch.zeros_like(s),s],1).cpu().numpy(),offsets(obs.candidate_lengths),
            logp[:,None].cpu().numpy(),kind="gaussian")
        self.last_sample_timing = dict(encoding_seconds=(events[0].elapsed_time(events[1])/1000 if events else encoded_at-start),denoising_seconds=0.,
            packing_transfer_seconds=time.perf_counter()-sampled_at,inference_seconds=time.perf_counter()-start)
        return action,chain

    def evaluate_chain(self, obs, chain):
        if chain.kind != "gaussian":
            raise ValueError("Gaussian PPO requires original Gaussian latent samples")
        device = next(self.parameters()).device
        enc = self.encode(obs.tensor(device))
        return self.joint_logp(enc,self.distribution(enc),
            torch.as_tensor(chain.movement[:,-1],device=device),
            torch.as_tensor(chain.scores[:,-1],device=device))[:,None]
