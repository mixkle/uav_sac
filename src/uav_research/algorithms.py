from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from .config import Config
from .observations import as_tensor_observation, history_specification
from .models import (
    GaussianActor,
    IndependentGaussianActors,
    policy_actor_modules,
    CategoricalActor,
    DeterministicActor,
    TwinQ,
    TimeFactorTwinQ,
    ValueNormalizer,
    DiscountedHorizon,
    mlp,
    add_history_inputs,
    add_neighbor_inputs,
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
            e.current_obs_dim if e.observation_history == "one_step" else e.obs_dim,
            a.hidden,
            a.log_std_min,
            a.log_std_max,
            ue_hidden=a.ue_hidden,
            count_scale=e.capacity,
        ).to(device)
        if e.observation_history == "one_step":
            add_history_inputs(self.actor, e.current_obs_dim)
        self.critic = (
            TimeFactorTwinQ(e.state_dim, e.num_uavs, a.hidden, e.horizon, a.discount)
            if a.critic_parameterization == "time_factor"
            else TwinQ(e.state_dim, e.num_uavs, a.hidden)
        ).to(device)
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
        q_kwargs, next_q_kwargs = {}, {}
        if a.critic_parameterization == "time_factor":
            h = self.critic.horizon_factor(batch["state"])
            next_h = self.target.horizon_factor(batch["next_state"])
            q_kwargs, next_q_kwargs = {"factors": h}, {"factors": next_h}
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(batch["next_obs"])
            q1, q2 = self.target(batch["next_state"], next_action, **next_q_kwargs)
            target = soft_target(
                batch["reward"],
                batch["terminated"],
                torch.minimum(q1, q2),
                next_logp.sum(-1),
                a.discount,
                alpha,
            )
        q1, q2 = self.critic(batch["state"], batch["actions"], **q_kwargs)
        q_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad = gradient_norm(self.critic)
        self.critic_opt.step()
        self.critic.requires_grad_(False)
        try:
            action, logp, _, std = self.actor.sample(batch["obs"], with_std=True)
            aq1, aq2 = self.critic(batch["state"], action, **q_kwargs)
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
                **(dict(
                    sac_q1_mean=q1.mean(), sac_q2_mean=q2.mean(),
                    sac_q_disagreement=(q1-q2).abs().mean(),
                    sac_average_q_mean=((q1+q2)/2/h).mean(),
                    sac_average_target_mean=(target/h).mean(),
                    sac_target_entropy_term=(a.discount*(~batch["terminated"]).float()*(-alpha*next_logp.sum(-1))).mean(),
                ) if a.critic_parameterization == "time_factor" else {}),
            )
        )

    def state_dict(self):
        return dict(
            observation_coordinates=self.cfg.env.observation_coordinates,
            observation_history=history_specification(self.cfg.env),
            critic_parameterization=self.cfg.algo.critic_parameterization,
            critic_time_scaling=("H_over_H0-v1" if self.cfg.algo.critic_parameterization == "time_factor" else None),
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
        if data.get("observation_coordinates", "absolute") != self.cfg.env.observation_coordinates:
            raise ValueError("SAC observation coordinates changed during loading")
        if data.get("observation_history", {"mode": "none"}) != history_specification(self.cfg.env):
            raise ValueError("SAC observation_history changed during loading")
        if data.get("critic_parameterization", "direct") != self.cfg.algo.critic_parameterization:
            raise ValueError("SAC critic parameterization changed during loading")
        if self.cfg.algo.critic_parameterization == "time_factor" and data.get("critic_time_scaling") != "H_over_H0-v1":
            raise ValueError("SAC time-factor checkpoint has incompatible output scaling")
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
        extra_neighbors = ((e.neighbor_slots - 3) * e.neighbor_dim
                           if e.action_space == "continuous" and e.neighbor_slots > 3 else 0)
        actor_current_dim = e.current_obs_dim - extra_neighbors
        if e.action_space == "discrete9":
            self.actor = CategoricalActor(
                e.obs_dim, a.hidden, a.ue_hidden, e.capacity
            ).to(device)
        else:
            self.actor = GaussianActor(
                actor_current_dim if e.observation_history == "one_step" else e.obs_dim - extra_neighbors,
                a.hidden,
                a.log_std_min,
                a.log_std_max,
                state_dependent_std=a.mappo_state_dependent_std,
                ue_hidden=a.ue_hidden,
                count_scale=e.capacity,
            ).to(device)
        if e.observation_history == "one_step":
            add_history_inputs(self.actor, actor_current_dim)
        self.value = mlp(e.state_dim, 1, a.hidden).to(device)
        self.normalizer = ValueNormalizer().to(device)
        if a.critic_parameterization == "time_factor":
            self.average_normalizer = ValueNormalizer().to(device)
            self.horizon_factor = DiscountedHorizon(e.horizon, a.discount).to(device)
            with torch.no_grad():
                self.value[-1].weight.div_(self.horizon_factor.table[-1])
                self.value[-1].bias.div_(self.horizon_factor.table[-1])
        if extra_neighbors:
            add_neighbor_inputs(self.actor, actor_current_dim, extra_neighbors)
        if a.actor_sharing == "independent":
            self.actor = IndependentGaussianActors(self.actor, e.num_uavs)
        self.actor_optimizers = [torch.optim.Adam(
            actor.parameters(), lr=a.learning_rate("actor"), eps=1e-5
        ) for actor in policy_actor_modules(self.actor)]
        # Preserve the public single-optimizer attribute for historical callers.
        self.actor_opt = self.actor_optimizers[0] if a.actor_sharing == "shared" else None
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
        if self.cfg.algo.critic_parameterization == "time_factor":
            return self.value_components(state)[0]
        return self.normalizer.denormalize(
            self.value(
                torch.as_tensor(state, device=self.device, dtype=torch.float32)
            ).squeeze(-1)
        )

    def value_components(self, state):
        """Differentiable raw cumulative value, average coverage and factor."""
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        factor = self.horizon_factor(state)
        average = self.average_normalizer.denormalize(self.value(state).squeeze(-1))
        return factor * average, average, factor

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
        if a.critic_parameterization == "time_factor":
            factors = self.horizon_factor(state)
            if (factors <= 0).any():
                raise ValueError("Rollout current states must precede true termination")
            average_targets = returns / factors
            self.average_normalizer.update(average_targets, self.value[-1], self.value_opt)
            self.normalizer.update_statistics(returns)
        else:
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
                independent = a.actor_sharing == "independent"
                per_actor_loss = -surrogate.mean(0) - a.entropy_coef * entropy.mean(0)
                actor_loss = (-surrogate.mean() - a.entropy_coef * entropy.mean())
                for optimizer in self.actor_optimizers:
                    optimizer.zero_grad(set_to_none=True)
                # Each independent block receives its own sample-mean loss,
                # without an extra 1/U factor or global gradient clipping.
                (per_actor_loss.sum() if independent else actor_loss).backward()
                actor_modules = policy_actor_modules(self.actor)
                actor_grads = [nn.utils.clip_grad_norm_(actor.parameters(), a.max_grad_norm)
                               for actor in actor_modules]
                actor_grad = torch.stack(actor_grads).mean()
                encoder_grads = [gradient_norm(actor.encoder) for actor in actor_modules]
                for optimizer in self.actor_optimizers:
                    optimizer.step()
                if a.critic_parameterization == "time_factor":
                    raw_value, average_prediction, _ = self.value_components(state[indices])
                    predicted = self.normalizer.normalize(raw_value)
                else:
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
                            encoder_grad_norm=torch.stack(encoder_grads).mean(),
                            **(dict(history_input_grad_norm=torch.stack([actor.net[0].weight.grad[:,
                                    self.cfg.env.current_obs_dim:self.cfg.env.current_obs_dim + 4].norm()
                                    for actor in actor_modules]).mean(),
                                    history_input_weight_norm=torch.stack([actor.net[0].weight[:,
                                    self.cfg.env.current_obs_dim:self.cfg.env.current_obs_dim + 4].detach().norm()
                                    for actor in actor_modules]).mean())
                               if self.cfg.env.observation_history == "one_step" else {}),
                            **{f"agent_{i}_{key}": value for i in range(self.cfg.env.num_uavs)
                                for key, value in dict(
                                    actor_loss=per_actor_loss[i],
                                    entropy=entropy[:,i].mean(),
                                    ppo_approx_kl=(old_logp[indices,i]-logp[:,i]).mean(),
                                    ppo_clip_fraction=((ratio[:,i]-1).abs()>a.ppo_clip).float().mean(),
                                ).items()},
                            **({f"agent_{i}_{key}": value for i in range(self.cfg.env.num_uavs)
                                for key, value in dict(
                                    actor_grad_norm=actor_grads[i],encoder_grad_norm=encoder_grads[i],
                                    history_input_grad_norm=actor_modules[i].net[0].weight.grad[:,
                                        self.cfg.env.current_obs_dim:self.cfg.env.current_obs_dim+4].norm()
                                        if self.cfg.env.observation_history == "one_step" else torch.zeros((),device=self.device),
                                ).items()} if independent else {}),
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
                            **(
                                dict(
                                    critic_value_clip_fraction=((predicted.detach()-normalized_old[indices]).abs()>a.ppo_clip).float().mean(),
                                    critic_clipped_loss_fraction=(
                                        nn.functional.huber_loss(normalized_old[indices]+(predicted.detach()-normalized_old[indices]).clamp(-a.ppo_clip,a.ppo_clip),normalized_returns[indices],reduction="none",delta=10.)
                                        > nn.functional.huber_loss(predicted.detach(),normalized_returns[indices],reduction="none",delta=10.)
                                    ).float().mean() if a.value_clipping else torch.zeros((),device=self.device),
                                    critic_average_prediction_mean=average_prediction.detach().mean(),
                                    critic_average_prediction_outside=((average_prediction.detach()<0)|(average_prediction.detach()>1)).float().mean(),
                                    critic_average_target_mean=average_targets.mean(),
                                    critic_average_target_outside=((average_targets<0)|(average_targets>1)).float().mean(),
                                    critic_return_normalizer_mean=self.normalizer.mean,
                                    critic_return_normalizer_std=self.normalizer.std,
                                    critic_average_normalizer_mean=self.average_normalizer.mean,
                                    critic_average_normalizer_std=self.average_normalizer.std,
                                ) if a.critic_parameterization == "time_factor" else {}
                            ),
                        )
                    )
                )
        self.updates += 1
        return {k: float(np.mean([row[k] for row in metrics])) for k in metrics[0]}

    def state_dict(self):
        return {
            **{
                name: getattr(self, name).state_dict()
                for name in ("actor", "value", "normalizer", "value_opt")
            },
            **({"actor_opt": self.actor_opt.state_dict()} if self.cfg.algo.actor_sharing == "shared"
               else {"actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers]}),
            "actor_sharing": self.cfg.algo.actor_sharing,
            **({"actor_agent_ids": list(range(self.cfg.env.num_uavs))}
               if self.cfg.algo.actor_sharing == "independent" else {}),
            "critic_parameterization": self.cfg.algo.critic_parameterization,
            "observation_coordinates": self.cfg.env.observation_coordinates,
            "mappo_state_dependent_std": self.cfg.algo.mappo_state_dependent_std,
            "observation_history": history_specification(self.cfg.env),
            **({"average_normalizer": self.average_normalizer.state_dict(),
                "horizon_factor": self.horizon_factor.state_dict()}
               if self.cfg.algo.critic_parameterization == "time_factor" else {}),
            "updates": self.updates,
        }

    def load_state_dict(self, data):
        if data.get("actor_sharing", "shared") != self.cfg.algo.actor_sharing:
            raise ValueError("Cannot load a different actor_sharing mode")
        if self.cfg.algo.actor_sharing == "independent" and (
            data.get("actor_agent_ids") != list(range(self.cfg.env.num_uavs))
            or len(data.get("actor_optimizers", [])) != self.cfg.env.num_uavs
        ):
            raise ValueError("Independent actor count/order/optimizer state is incompatible")
        if data.get("observation_history", {"mode": "none"}) != history_specification(self.cfg.env):
            raise ValueError("MAPPO observation_history changed during loading")
        if data.get("observation_coordinates", "absolute") != self.cfg.env.observation_coordinates:
            raise ValueError("MAPPO observation coordinates changed during loading")
        if data.get("mappo_state_dependent_std", False) != self.cfg.algo.mappo_state_dependent_std:
            raise ValueError("MAPPO standard deviation type changed during loading")
        if data.get("critic_parameterization", "direct") != self.cfg.algo.critic_parameterization:
            raise ValueError("Cannot load a different critic parameterization")
        for name in ("actor", "value", "normalizer", "value_opt"):
            getattr(self, name).load_state_dict(data[name])
        if self.cfg.algo.actor_sharing == "shared":
            self.actor_opt.load_state_dict(data["actor_opt"])
        else:
            for optimizer, state in zip(self.actor_optimizers, data["actor_optimizers"]):
                optimizer.load_state_dict(state)
        if self.cfg.algo.critic_parameterization == "time_factor":
            self.average_normalizer.load_state_dict(data["average_normalizer"])
            self.horizon_factor.load_state_dict(data["horizon_factor"])
        self.updates = data["updates"]


def make_learner(cfg, device):
    if cfg.algo.name == "diffusion_dppo":
        from .diffusion_dppo.learner import DiffusionLearner
        return DiffusionLearner(cfg, device)
    return {"sac": SAC, "mappo": MAPPO, "matd3": MATD3}[cfg.algo.name](cfg, device)
