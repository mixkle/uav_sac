"""Permutation-equivariant, ragged DDPM; exact Gaussian transition densities."""
from dataclasses import dataclass
import math
import time
import torch
from torch import nn

from ..models import disk_transform
from .observations import ServiceAction, DiffusionChain, offsets


def network(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, dout))


def mean_segments(values, lengths):
    # Offsets/lengths are validated on CPU by ServiceObservation and preserve
    # their sums under expansion. Skip CUDA's repeated host-synchronizing checks.
    return torch.nan_to_num(torch.segment_reduce(values, "mean", lengths=lengths, unsafe=True), nan=0.)


def sum_segments(values, lengths):
    return torch.segment_reduce(values, "sum", lengths=lengths, unsafe=True)


def owner_indices(lengths, size):
    return torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device),
                                   lengths, output_size=size)


@dataclass
class Encoding:
    context: torch.Tensor
    candidates: torch.Tensor
    lengths: torch.Tensor
    owners: torch.Tensor | None = None

    def __post_init__(self):
        if self.owners is None:
            self.owners = owner_indices(self.lengths, len(self.candidates))


class DiffusionActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.diffusion
        self.steps = d.steps
        self.ue = nn.Sequential(nn.Linear(5, d.ue_width), nn.ReLU(),
                                nn.Linear(d.ue_width, d.ue_width), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(cfg.env.fixed_obs_dim+d.ue_width+1, d.context_width),
            nn.ReLU(), nn.Linear(d.context_width, d.context_width), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(d.ue_width+1, 128), nn.ReLU(),
                                  nn.Linear(128, d.pair_width), nn.ReLU())
        self.movement_head = network(2+d.context_width+d.pair_width+d.time_width, 256, 2)
        self.score_head = network(1+d.ue_width+d.context_width+2+d.pair_width+d.time_width, 256, 1)
        grid = torch.linspace(0, 1, d.steps+1, dtype=torch.float64)
        abar = torch.cos((grid+d.cosine_offset)/(1+d.cosine_offset)*math.pi/2).square()
        abar = abar / abar[0]
        beta = (1-abar[1:]/abar[:-1]).clamp(max=d.max_beta)
        alpha = 1-beta
        abar = torch.cumprod(alpha, 0)
        prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        posterior_var = beta*(1-prev)/(1-abar)
        # For z_(k-1)|z_k: epsilon -> x0 -> DDPM posterior mean.
        self.register_buffer("sqrt_abar", abar.sqrt().float())
        self.register_buffer("sqrt_one_minus_abar", (1-abar).sqrt().float())
        self.register_buffer("coef_x0", (beta*prev.sqrt()/(1-abar)).float())
        self.register_buffer("coef_xt", ((1-prev)*alpha.sqrt()/(1-abar)).float())
        self.register_buffer("transition_std", posterior_var.sqrt().clamp_min(d.std_floor).float())
        k = torch.arange(1, d.steps+1, dtype=torch.float32)[:, None]
        frequency = torch.exp(-math.log(10000)*torch.arange(d.time_width//2)/(d.time_width//2))
        self.register_buffer("time_embedding", torch.cat([(k*frequency).sin(), (k*frequency).cos()], -1))

    def encode(self, obs):
        h = self.ue(obs.users)
        pooled = mean_segments(h, obs.lengths)
        count = torch.log1p(obs.lengths.float()) / math.log1p(self.cfg.env.capacity)
        g = self.context(torch.cat([obs.fixed, pooled, count[:, None]], -1))
        # Integer routing indices originate on CPU; boolean CUDA indexing would
        # synchronize to discover its output size on every encoding call.
        return Encoding(g, h[obs.candidate_indices], obs.candidate_lengths)

    def predict_noise(self, encoding, movement, scores, step):
        n = len(movement)
        own = encoding.owners
        pairs = self.pair(torch.cat([encoding.candidates, scores[:, None]], -1))
        pooled = mean_segments(pairs, encoding.lengths)
        time = self.time_embedding[step]
        if time.ndim == 1:
            time = time.expand(n, -1)
        mov = self.movement_head(torch.cat([movement, encoding.context, pooled, time], -1))
        score = self.score_head(torch.cat([scores[:, None], encoding.candidates,
            encoding.context[own], movement[own], pooled[own], time[own]], -1)).squeeze(-1)
        return mov, score

    def transition(self, encoding, movement, scores, step):
        eps_mov, eps_score = self.predict_noise(encoding, movement, scores, step)
        own = encoding.owners
        am, bm = self.sqrt_abar[step], self.sqrt_one_minus_abar[step]
        c0, ct = self.coef_x0[step], self.coef_xt[step]
        if am.ndim:
            ax, bx, c0x, ctx = am[own], bm[own], c0[own], ct[own]
            am, bm, c0, ct = [x[:, None] for x in (am, bm, c0, ct)]
        else:
            ax, bx, c0x, ctx = am, bm, c0, ct
        mu_mov = c0*((movement-bm*eps_mov)/am) + ct*movement
        mu_score = c0x*((scores-bx*eps_score)/ax) + ctx*scores
        return mu_mov, mu_score

    def logp(self, encoding, movement, scores, next_movement, next_scores, step):
        mu_m, mu_s = self.transition(encoding, movement, scores, step)
        std = self.transition_std[step]
        own = encoding.owners
        std_s = std[own] if std.ndim else std
        std_m = std[:, None] if std.ndim else std
        lm = (-.5*((next_movement-mu_m)/std_m).square()-std_m.log()-.5*math.log(2*math.pi)).sum(-1)
        ls = -.5*((next_scores-mu_s)/std_s).square()-std_s.log()-.5*math.log(2*math.pi)
        return lm + sum_segments(ls[:, None], encoding.lengths).squeeze(-1)

    @torch.no_grad()
    def sample(self, obs, generator=None, deterministic=False):
        device = next(self.parameters()).device
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if device.type == "cuda" else None
        if events:
            events[0].record()
        encoded = self.encode(obs.tensor(device))
        encoded_at = time.perf_counter()
        if events:
            events[1].record()
        m = torch.randn((obs.groups, 2), device=device, generator=generator)
        s = torch.randn((len(obs.candidate_ids),), device=device, generator=generator)
        if deterministic:
            m.zero_()
            s.zero_()
        movements, scores, logps = [m], [s], []
        for step in reversed(range(self.steps)):
            mu_m, mu_s = self.transition(encoded, m, s, step)
            if deterministic:
                nm, ns = mu_m, mu_s
            else:
                nm = mu_m + self.transition_std[step]*torch.randn(m.shape, device=device, generator=generator)
                ns = mu_s + self.transition_std[step]*torch.randn(s.shape, device=device, generator=generator)
            logps.append(self.logp_from_mean(encoded.lengths, mu_m, mu_s, nm, ns, step))
            m, s = nm, ns
            movements.append(m)
            scores.append(s)
        mchain, schain, logp = torch.stack(movements, 1), torch.stack(scores, 1), torch.stack(logps, 1)
        if events:
            events[2].record()
        denoised_at = time.perf_counter()
        if not (torch.isfinite(mchain).all() & torch.isfinite(schain).all() & torch.isfinite(logp).all()):
            raise FloatingPointError("Nonfinite DDPM sample/transition probability")
        command = disk_transform(m)[0].cpu().numpy().reshape(*obs.fixed.shape[:-1], 2)
        route = offsets(obs.candidate_lengths)
        action = ServiceAction(command, s.sigmoid().cpu().numpy(), obs.candidate_ids.copy(), route)
        chain = DiffusionChain(mchain.cpu().numpy(), schain.cpu().numpy(), route.copy(), logp.cpu().numpy())
        finished = time.perf_counter()
        self.last_sample_timing = dict(
            encoding_seconds=events[0].elapsed_time(events[1])/1000 if events else encoded_at-started,
            denoising_seconds=events[1].elapsed_time(events[2])/1000 if events else denoised_at-encoded_at,
            packing_transfer_seconds=finished-denoised_at, inference_seconds=finished-started)
        return action, chain

    def logp_from_mean(self, lengths, mu_m, mu_s, nm, ns, step):
        std = self.transition_std[step]
        lm = (-.5*((nm-mu_m)/std).square()-std.log()-.5*math.log(2*math.pi)).sum(-1)
        ls = -.5*((ns-mu_s)/std).square()-std.log()-.5*math.log(2*math.pi)
        return lm+sum_segments(ls[:, None], lengths).squeeze(-1)

    def evaluate_chain(self, obs, chain):
        """Batch all 8 transitions; UE/context encoding computed once per update."""
        # Data-parallel replicas expose parameter tensors as attributes rather
        # than registered Parameters. A schedule buffer gives the same device
        # for both the original module and its differentiable replicas.
        device = self.transition_std.device
        enc = self.encode(obs.tensor(device))
        m = torch.as_tensor(chain.movement, device=device)
        s = torch.as_tensor(chain.scores, device=device)
        n, l = len(m), self.steps
        lengths = enc.lengths.repeat_interleave(l)
        # [agent, denoising-step, candidate] ownership order for segment_reduce.
        total = len(s)*l
        start = torch.cumsum(enc.lengths, 0)-enc.lengths
        base = torch.repeat_interleave(start.repeat_interleave(l), lengths, output_size=total)
        block_start = torch.cumsum(lengths, 0)-lengths
        within = torch.arange(total, device=device)-torch.repeat_interleave(block_start, lengths, output_size=total)
        row = base+within
        position = torch.repeat_interleave(torch.arange(l, device=device).repeat(n), lengths, output_size=total)
        expanded = Encoding(enc.context.repeat_interleave(l, 0), enc.candidates[row], lengths)
        step = torch.arange(l-1, -1, -1, device=device).repeat(n)
        result = self.logp(expanded, m[:, :-1].reshape(-1, 2), s[row, position],
                           m[:, 1:].reshape(-1, 2), s[row, position+1], step)
        return result.reshape(n, l)

    def denoising_loss(self, obs, target_movement, target_scores, *, generator=None, steps=None, noise=None):
        device = next(self.parameters()).device
        enc = self.encode(obs.tensor(device))
        m = torch.as_tensor(target_movement, device=device)
        s = torch.as_tensor(target_scores, device=device)
        own = enc.owners
        step = torch.randint(self.steps, (len(m),), device=device, generator=generator) if steps is None else steps
        if noise is None:
            em = torch.randn(m.shape, device=device, generator=generator)
            es = torch.randn(s.shape, device=device, generator=generator)
        else:
            em, es = noise
        zm = self.sqrt_abar[step, None]*m+self.sqrt_one_minus_abar[step, None]*em
        zs = self.sqrt_abar[step[own]]*s+self.sqrt_one_minus_abar[step[own]]*es
        pm, ps = self.predict_noise(enc, zm, zs, step)
        mov = (em-pm).square().mean(-1)
        score = sum_segments((es-ps).square()[:, None], enc.lengths).squeeze(-1)/enc.lengths.clamp_min(1)
        return mov, score


def make_service_actor(cfg):
    if cfg.algo.name == "gaussian_service_mappo":
        from .gaussian import GaussianServiceActor
        return GaussianServiceActor(cfg)
    return DiffusionActor(cfg)
