"""Team GAE and per-agent, per-denoising-transition clipped PPO."""
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn

from ..algorithms import MAPPO, gae, mappo_value_loss, gradient_norm
from ..models import mlp, ValueNormalizer, DiscountedHorizon
from .config import contract
from .observations import stack, stack_chains
from .policy import make_service_actor
from .parallel import evaluate_chain_parallel


@dataclass
class DppoRollout:
    observations: list = field(default_factory=list)
    chains: list = field(default_factory=list)
    fields: dict = field(default_factory=lambda: {k: [] for k in
        ("state", "reward", "value", "next_value", "terminated", "truncated")})

    def append(self, obs, chain, **fields):
        self.observations.append(obs.copy())
        self.chains.append(chain)
        for k in self.fields:
            self.fields[k].append(np.asarray(fields[k]).copy())

    def packed(self):
        return stack(self.observations), stack_chains(self.chains), {
            k: np.stack(v) for k, v in self.fields.items()}


class DiffusionLearner:
    # Exact existing cumulative-value interfaces; no Gaussian actor is created.
    values = MAPPO.values
    value_components = MAPPO.value_components

    def __init__(self, cfg, device):
        self.cfg, self.device, self.updates, self.optimizer_steps = cfg, torch.device(device), 0, 0
        a, e = cfg.algo, cfg.env
        self.actor_update_devices = [torch.device(x) for x in cfg.diffusion.actor_update_devices]
        for dev in self.actor_update_devices:
            if dev.index >= torch.cuda.device_count():
                raise ValueError(f"Actor update device {dev} is not visible")
        self.actor = make_service_actor(cfg).to(self.device)
        if e.motion_scaling["enabled"]:
            # Paired new algorithms start with an identical team critic despite
            # their different actor parameter counts/random initialization draws.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg.run.seed+714031)
                self.value = mlp(e.state_dim, 1, a.hidden).to(self.device)
        else:
            self.value = mlp(e.state_dim, 1, a.hidden).to(self.device)
        self.normalizer = ValueNormalizer().to(self.device)
        self.average_normalizer = ValueNormalizer().to(self.device)
        self.horizon_factor = DiscountedHorizon(e.horizon, a.discount).to(self.device)
        with torch.no_grad():
            self.value[-1].weight.div_(self.horizon_factor.table[-1])
            self.value[-1].bias.div_(self.horizon_factor.table[-1])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a.learning_rate("actor"), eps=1e-5)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=a.learning_rate("critic"), eps=1e-5)

    def act(self, obs, deterministic=False):
        return self.actor.sample(obs, deterministic=deterministic)[0]

    def update(self, rollout, progress=None):
        obs, chain, raw = rollout.packed()
        a, d, u = self.cfg.algo, self.cfg.diffusion, self.cfg.env.num_uavs
        data = {k: torch.as_tensor(v, device=self.device) for k, v in raw.items()}
        advantages, returns = gae(data["reward"], data["value"], data["next_value"],
            data["terminated"], a.discount, a.gae_lambda, data["truncated"])
        adv = ((advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)).flatten()
        state = data["state"].flatten(0, 1)
        returns, old = returns.flatten(), data["value"].flatten()
        factors = self.horizon_factor(state)
        if (factors <= 0).any():
            raise ValueError("Training observations must precede termination")
        average_target = returns / factors
        if not torch.isfinite(average_target).all():
            raise FloatingPointError("Nonfinite GAE target")
        self.average_normalizer.update(average_target, self.value[-1], self.value_opt)
        self.normalizer.update_statistics(returns)
        target, old_n = self.normalizer.normalize(returns), self.normalizer.normalize(old)
        metrics = []
        n = len(state)
        for epoch in range(a.ppo_epochs):
            permutation = torch.randperm(n, device=self.device)
            for indices in torch.tensor_split(permutation, min(a.ppo_minibatches, n)):
                groups = (indices[:, None]*u+torch.arange(u, device=self.device)).flatten()
                group_np = groups.cpu().numpy()
                all_adv = adv[indices].repeat_interleave(u)
                self.actor_opt.zero_grad(set_to_none=True)
                stats = torch.zeros(4+2*self.actor.steps, device=self.device)
                grouped = torch.zeros((3, 3), device=self.device)
                for lo in range(0, len(groups), d.microbatch_agents):
                    hi = min(lo+d.microbatch_agents, len(groups))
                    selected = group_np[lo:hi]
                    selected_obs = obs.take(selected)
                    chosen_chain = chain.take(selected)
                    newp = evaluate_chain_parallel(self.actor, selected_obs, chosen_chain,
                                                   self.actor_update_devices)
                    oldp = torch.as_tensor(chosen_chain.logp, device=self.device)
                    lr = newp-oldp
                    ratio = lr.exp()
                    if not (torch.isfinite(lr).all() & torch.isfinite(ratio).all()):
                        raise FloatingPointError("Nonfinite per-transition PPO ratio (not clamped)")
                    advantage = all_adv[lo:hi, None]
                    surrogate = torch.minimum(ratio*advantage,
                        ratio.clamp(1-a.ppo_clip, 1+a.ppo_clip)*advantage)
                    weight = (hi-lo)/len(groups)
                    loss = -surrogate.mean()*weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite DPPO actor loss")
                    loss.backward()
                    with torch.no_grad():
                        clip = ((ratio-1).abs()>a.ppo_clip).float()
                        kl = (ratio-1)-lr
                        stats[0] += loss.detach()
                        stats[1] += kl.mean()*weight
                        stats[2] += clip.mean()*weight
                        stats[3] += (oldp-newp).mean()*weight
                        stats[4:4+self.actor.steps] += kl.mean(0)*weight
                        stats[4+self.actor.steps:] += clip.mean(0)*weight
                        count = torch.as_tensor(selected_obs.candidate_lengths, device=self.device)
                        for j, mask in enumerate((count == 0, (count > 0) & (count <= self.cfg.env.capacity),
                                                  count > self.cfg.env.capacity)):
                            grouped[j, 0] += kl[mask].sum()
                            grouped[j, 1] += clip[mask].sum()
                            grouped[j, 2] += mask.sum()*self.actor.steps
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), a.max_grad_norm, error_if_nonfinite=True)
                gradients = dict(encoder_grad_norm=gradient_norm(self.actor.ue),
                    movement_head_grad_norm=gradient_norm(self.actor.movement_head),
                    score_head_grad_norm=gradient_norm(self.actor.score_head))
                self.actor_opt.step()
                raw_value, avg, _ = self.value_components(state[indices])
                predicted = self.normalizer.normalize(raw_value)
                value_loss = mappo_value_loss(predicted, target[indices], old_n[indices],
                                              d.value_clip, a.value_clipping)
                if not torch.isfinite(value_loss):
                    raise FloatingPointError("Nonfinite DPPO critic loss")
                self.value_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                vg = nn.utils.clip_grad_norm_(self.value.parameters(), a.max_grad_norm, error_if_nonfinite=True)
                self.value_opt.step()
                self.optimizer_steps += 1
                row = dict(actor_loss=stats[0], ppo_approx_kl=stats[1], ppo_clip_fraction=stats[2],
                    ppo_sample_kl=stats[3], value_loss=value_loss.detach(), actor_grad_norm=actor_grad,
                    critic_grad_norm=vg, **gradients,
                    critic_value_clip_fraction=((predicted.detach()-old_n[indices]).abs()>d.value_clip).float().mean(),
                    critic_average_prediction_mean=avg.detach().mean())
                for k in range(self.actor.steps):
                    row[f"denoise_{self.actor.steps-k}_kl" if self.cfg.algo.name == "diffusion_dppo" else "gaussian_joint_kl"] = stats[4+k]
                    row[f"denoise_{self.actor.steps-k}_clip" if self.cfg.algo.name == "diffusion_dppo" else "gaussian_joint_clip"] = stats[4+self.actor.steps+k]
                for j, name in enumerate(("empty", "within_capacity", "over_capacity")):
                    row[f"candidates_{name}_kl"] = grouped[j, 0]/grouped[j, 2].clamp_min(1)
                    row[f"candidates_{name}_clip"] = grouped[j, 1]/grouped[j, 2].clamp_min(1)
                keys = list(row)
                vals = torch.stack([row[k].detach() for k in keys]).cpu().tolist()
                metrics.append(dict(zip(keys, vals)))
                if progress:
                    progress(epoch+1, self.optimizer_steps)
        self.updates += 1
        result = {k: float(np.mean([r[k] for r in metrics])) for k in metrics[0]}
        result.update(value_return_mean=float(returns.mean()),
            critic_average_target_mean=float(average_target.mean()),
            critic_average_target_outside=float(((average_target<0)|(average_target>1)).float().mean()),
            critic_return_normalizer_mean=float(self.normalizer.mean),
            critic_return_normalizer_std=float(self.normalizer.std),
            critic_average_normalizer_mean=float(self.average_normalizer.mean),
            critic_average_normalizer_std=float(self.average_normalizer.std))
        return result

    def state_dict(self):
        return dict(contract=contract(self.cfg), updates=self.updates, optimizer_steps=self.optimizer_steps,
            **{k: getattr(self, k).state_dict() for k in ("actor", "value", "normalizer",
              "average_normalizer", "horizon_factor", "actor_opt", "value_opt")})

    def load_state_dict(self, data):
        if data.get("contract") != contract(self.cfg):
            raise ValueError("DPPO experiment/observation/protocol/learning settings changed during resume")
        for k in ("actor", "value", "normalizer", "average_normalizer", "horizon_factor", "actor_opt", "value_opt"):
            getattr(self, k).load_state_dict(data[k])
        self.updates, self.optimizer_steps = data["updates"], data["optimizer_steps"]
