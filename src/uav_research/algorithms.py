from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from .config import Config
from .observations import as_tensor_observation
from .models import (
    GaussianActor,
    CategoricalActor,
    DeterministicActor,
    TwinQ,
    ValueNormalizer,
    mlp,
    project_disk,
)


def gradient_norm(module):
    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    return nn.utils.get_total_norm(gradients) if gradients else torch.tensor(0.0)


def numeric_metrics(values):
    """Transfer CUDA diagnostics once per update rather than once per scalar."""
    if next(iter(values.values())).device.type == "cuda":
        packed = (
            torch.stack([value.detach() for value in values.values()]).cpu().tolist()
        )
        return dict(zip(values, packed))
    return {key: float(value.detach()) for key, value in values.items()}


def mappo_value_loss(predicted, returns, old_value, clip, value_clipping=True):
    """Huber loss in normalized value units; optional PPO value clipping."""
    loss = nn.functional.huber_loss(
        predicted, returns, reduction="none", delta=10.0
    )
    if value_clipping:
        clipped = old_value + (predicted - old_value).clamp(-clip, clip)
        loss = torch.maximum(
            loss,
            nn.functional.huber_loss(
                clipped, returns, reduction="none", delta=10.0
            ),
        )
    return loss.mean()


@torch.no_grad()
def soft_update(target, source, tau):
    for dest, value in zip(target.parameters(), source.parameters()):
        dest.lerp_(value, tau)


def soft_target(reward, terminated, next_q, next_log_prob, discount, alpha):
    return reward + discount * (~terminated).float() * (next_q - alpha * next_log_prob)


def gae(rewards, values, next_values, terminated, discount, lam, truncated=None):
    """next_values are terminal-observation values, before resetting the env."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        boot = (~terminated[t]).float()
        continuation = boot if truncated is None else boot * (~truncated[t]).float()
        delta = rewards[t] + discount * boot * next_values[t] - values[t]
        carry = delta + discount * lam * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


class SAC:
    def __init__(self, cfg: Config, device):
        self.cfg, self.device, self.updates = cfg, device, 0
        e, a = cfg.env, cfg.algo
        self.actor = GaussianActor(
            e.obs_dim,
            a.hidden,
            a.log_std_min,
            a.log_std_max,
            ue_hidden=a.ue_hidden,
            count_scale=e.capacity,
        ).to(device)
        self.critic = TwinQ(e.state_dim, e.num_uavs, a.hidden).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=a.learning_rate("actor")
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=a.learning_rate("critic")
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(a.initial_alpha), device=device)
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=a.temperature_lr)
        self.target_entropy = (
            -2 * e.num_uavs if a.target_entropy is None else a.target_entropy
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        return (
            self.actor.sample(
                as_tensor_observation(obs, self.device),
                deterministic,
            )[0]
            .cpu()
            .numpy()
        )

    def update(self, batch):
        a = self.cfg.algo
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(batch["next_obs"])
            q1, q2 = self.target(batch["next_state"], next_action)
            target = soft_target(
                batch["reward"],
                batch["terminated"],
                torch.minimum(q1, q2),
                next_logp.sum(-1),
                a.discount,
                alpha,
            )
        q1, q2 = self.critic(batch["state"], batch["actions"])
        q_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad = gradient_norm(self.critic)
        self.critic_opt.step()
        self.critic.requires_grad_(False)
        try:
            action, logp, _, std = self.actor.sample(batch["obs"], with_std=True)
            aq1, aq2 = self.critic(batch["state"], action)
            actor_loss = (alpha * logp.sum(-1) - torch.minimum(aq1, aq2)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = gradient_norm(self.actor)
            self.actor_opt.step()
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = -(
            self.log_alpha * (logp.sum(-1).detach() + self.target_entropy)
        ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        soft_update(self.target, self.critic, a.polyak)
        self.updates += 1
        return numeric_metrics(
            dict(
                q_loss=q_loss,
                actor_loss=actor_loss,
                q_grad_norm=q_grad,
                actor_grad_norm=actor_grad,
                encoder_grad_norm=gradient_norm(self.actor.encoder),
                alpha=self.log_alpha.exp(),
                joint_entropy=-logp.sum(-1).mean(),
                q_mean=((q1 + q2) / 2).mean(),
                target_mean=target.mean(),
                td_abs=(q1 - target).abs().mean(),
                policy_std=std,
            )
        )

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target=self.target.state_dict(),
            actor_opt=self.actor_opt.state_dict(),
            critic_opt=self.critic_opt.state_dict(),
            log_alpha=self.log_alpha.detach().clone(),
            alpha_opt=self.alpha_opt.state_dict(),
            updates=self.updates,
        )

    def load_state_dict(self, data):
        for name in (
            "actor",
            "critic",
            "target",
            "actor_opt",
            "critic_opt",
            "alpha_opt",
        ):
            getattr(self, name).load_state_dict(data[name])
        with torch.no_grad():
            self.log_alpha.copy_(data["log_alpha"])
        self.updates = data["updates"]


class MATD3:
    def __init__(self, cfg: Config, device):
        self.cfg, self.device, self.updates = cfg, device, 0
        e, a = cfg.env, cfg.algo
        self.actor = DeterministicActor(
            e.obs_dim, a.hidden, a.ue_hidden, e.capacity
        ).to(device)
        self.critic = TwinQ(e.state_dim, e.num_uavs, a.hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=a.learning_rate("actor")
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=a.learning_rate("critic")
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        action = self.actor(as_tensor_observation(obs, self.device))
        if not deterministic:
            action = project_disk(
                action + torch.randn_like(action) * self.cfg.algo.td3_exploration_noise
            )
        return action.cpu().numpy()

    def update(self, batch):
        a = self.cfg.algo
        with torch.no_grad():
            next_action = self.actor_target(batch["next_obs"])
            noise = (torch.randn_like(next_action) * a.td3_target_noise).clamp(
                -a.td3_noise_clip, a.td3_noise_clip
            )
            tq1, tq2 = self.target(
                batch["next_state"], project_disk(next_action + noise)
            )
            target = batch["reward"] + a.discount * (
                ~batch["terminated"]
            ).float() * torch.minimum(tq1, tq2)
        q1, q2 = self.critic(batch["state"], batch["actions"])
        loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        q_grad = gradient_norm(self.critic)
        self.critic_opt.step()
        self.updates += 1
        metrics = dict(
            q_loss=loss,
            q_grad_norm=q_grad,
            q_mean=q1.mean(),
            target_mean=target.mean(),
            td_abs=(q1 - target).abs().mean(),
        )
        if self.updates % a.td3_policy_delay == 0:
            self.critic.requires_grad_(False)
            try:
                actor_loss = -self.critic(batch["state"], self.actor(batch["obs"]))[
                    0
                ].mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                metrics.update(
                    actor_loss=actor_loss,
                    actor_grad_norm=gradient_norm(self.actor),
                    encoder_grad_norm=gradient_norm(self.actor.encoder),
                )
                self.actor_opt.step()
            finally:
                self.critic.requires_grad_(True)
            soft_update(self.actor_target, self.actor, a.polyak)
            soft_update(self.target, self.critic, a.polyak)
        return numeric_metrics(metrics)

    def state_dict(self):
        return {
            **{
                name: getattr(self, name).state_dict()
                for name in (
                    "actor",
                    "critic",
                    "target",
                    "actor_target",
                    "actor_opt",
                    "critic_opt",
                )
            },
            "updates": self.updates,
        }

    def load_state_dict(self, data):
        for name in (
            "actor",
            "critic",
            "target",
            "actor_target",
            "actor_opt",
            "critic_opt",
        ):
            getattr(self, name).load_state_dict(data[name])
        self.updates = data["updates"]


class MAPPO:
    def __init__(self, cfg: Config, device):
        self.cfg, self.device, self.updates = cfg, device, 0
        e, a = cfg.env, cfg.algo
        if e.action_space == "discrete9":
            self.actor = CategoricalActor(
                e.obs_dim, a.hidden, a.ue_hidden, e.capacity
            ).to(device)
        else:
            self.actor = GaussianActor(
                e.obs_dim,
                a.hidden,
                a.log_std_min,
                a.log_std_max,
                state_dependent_std=False,
                ue_hidden=a.ue_hidden,
                count_scale=e.capacity,
            ).to(device)
        self.value = mlp(e.state_dim, 1, a.hidden).to(device)
        self.normalizer = ValueNormalizer().to(device)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=a.learning_rate("actor"), eps=1e-5
        )
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=a.learning_rate("critic"), eps=1e-5
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        return (
            self.actor.sample(
                as_tensor_observation(obs, self.device),
                deterministic,
            )[0]
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def collect(self, obs, state):
        action, logp, z = self.actor.sample(as_tensor_observation(obs, self.device))
        return (
            action.cpu().numpy(),
            logp.cpu().numpy(),
            z.cpu().numpy(),
            self.values(state).cpu().numpy(),
        )

    @torch.no_grad()
    def values(self, state):
        return self.normalizer.denormalize(
            self.value(
                torch.as_tensor(state, device=self.device, dtype=torch.float32)
            ).squeeze(-1)
        )

    def update(self, rollout):
        a = self.cfg.algo

        def t(x):
            return torch.as_tensor(x, device=self.device)

        rewards, values, next_values, done = map(
            t,
            [
                rollout["reward"],
                rollout["value"],
                rollout["next_value"],
                rollout["terminated"],
            ],
        )
        advantages, returns = gae(
            rewards,
            values,
            next_values,
            done,
            a.discount,
            a.gae_lambda,
            t(rollout["truncated"]),
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-5
        )
        obs = rollout["obs"].flatten_batch()
        action_key = "actions" if self.cfg.env.action_space == "discrete9" else "latent"
        state, latent, old_logp = [
            t(rollout[k]).flatten(0, 1) for k in ("state", action_key, "logp")
        ]
        old_value, adv, returns = (
            values.flatten(),
            advantages.flatten(),
            returns.flatten(),
        )
        self.normalizer.update(returns, self.value[-1], self.value_opt)
        normalized_returns, normalized_old = (
            self.normalizer.normalize(returns),
            self.normalizer.normalize(old_value),
        )
        n = len(state)
        metrics = []
        for _ in range(a.ppo_epochs):
            permutation = torch.randperm(n, device=self.device)
            for indices in torch.tensor_split(permutation, min(a.ppo_minibatches, n)):
                selected_obs = obs[indices.cpu().numpy()].to_torch(self.device)
                if self.cfg.env.action_space == "discrete9":
                    logp, entropy, max_probability = self.actor.evaluate_actions(
                        selected_obs, latent[indices].long()
                    )
                else:
                    logp, entropy = self.actor.evaluate_latent(
                        selected_obs, latent[indices]
                    )
                ratio = (
                    logp - old_logp[indices]
                ).exp()  # [batch,U], never product over UAVs.
                surrogate = torch.minimum(
                    ratio * adv[indices, None],
                    ratio.clamp(1 - a.ppo_clip, 1 + a.ppo_clip) * adv[indices, None],
                )
                actor_loss = -surrogate.mean() - a.entropy_coef * entropy.mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), a.max_grad_norm
                )
                self.actor_opt.step()
                predicted = self.value(state[indices]).squeeze(-1)
                v_loss = mappo_value_loss(
                    predicted, normalized_returns[indices],
                    normalized_old[indices], a.ppo_clip, a.value_clipping,
                )
                self.value_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                value_grad = nn.utils.clip_grad_norm_(
                    self.value.parameters(), a.max_grad_norm
                )
                self.value_opt.step()
                metrics.append(
                    numeric_metrics(
                        dict(
                            actor_loss=actor_loss,
                            value_loss=v_loss,
                            actor_grad_norm=actor_grad,
                            encoder_grad_norm=gradient_norm(self.actor.encoder),
                            q_grad_norm=value_grad,
                            **(
                                dict(
                                    categorical_entropy=entropy.mean(),
                                    max_action_probability=max_probability.mean(),
                                )
                                if self.cfg.env.action_space == "discrete9"
                                else {}
                            ),
                            joint_entropy=entropy.sum(-1).mean(),
                            ppo_approx_kl=(old_logp[indices] - logp).mean(),
                            ppo_clip_fraction=((ratio - 1).abs() > a.ppo_clip)
                            .float()
                            .mean(),
                            value_return_mean=returns.mean(),
                        )
                    )
                )
        self.updates += 1
        return {k: float(np.mean([row[k] for row in metrics])) for k in metrics[0]}

    def state_dict(self):
        return {
            **{
                name: getattr(self, name).state_dict()
                for name in ("actor", "value", "normalizer", "actor_opt", "value_opt")
            },
            "updates": self.updates,
        }

    def load_state_dict(self, data):
        for name in ("actor", "value", "normalizer", "actor_opt", "value_opt"):
            getattr(self, name).load_state_dict(data[name])
        self.updates = data["updates"]


def make_learner(cfg, device):
    return {"sac": SAC, "mappo": MAPPO, "matd3": MATD3}[cfg.algo.name](cfg, device)
