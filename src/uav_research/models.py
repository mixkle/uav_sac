from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical
from .observations import TensorObservation


def mlp(input_dim, output_dim, hidden):
    layers = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, 0.01 if output_dim > 1 else 1.0)
    return layers


def disk_transform(z):
    """Bijective R² -> open unit disk: a=z/sqrt(1+||z||²).

    Jacobian eigenvalues: (1+r²)^(-1/2), (1+r²)^(-3/2).
    Thus log|det J|=-2*log(1+r²). No boundary probability mass.
    """
    norm2 = z.square().sum(dim=-1, keepdim=True)
    return z * torch.rsqrt(1 + norm2), -2 * torch.log1p(norm2).squeeze(-1)


def disk_inverse(action):
    norm2 = action.square().sum(dim=-1, keepdim=True)
    return action * torch.rsqrt((1 - norm2).clamp_min(torch.finfo(action.dtype).eps))


def project_disk(action):
    return action / action.norm(dim=-1, keepdim=True).clamp_min(1.0)


class UEEncoder(nn.Module):
    def __init__(self, fixed_dim, hidden, count_scale):
        super().__init__()
        self.count_scale = count_scale
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.output_dim = fixed_dim + hidden + 1
        self.hidden = hidden

    def forward(self, obs):
        if not isinstance(obs, TensorObservation):
            raise ValueError("Expected structured TensorObservation")
        if len(obs.users):
            encoded = self.net(obs.users)
            sums = torch.segment_reduce(encoded, "sum", lengths=obs.lengths)
            means = sums / obs.lengths.clamp_min(1)[:, None]
        else:
            means = obs.fixed.new_zeros((obs.lengths.numel(), self.hidden))
        shape = obs.fixed.shape[:-1]
        return torch.cat(
            [
                obs.fixed,
                means.reshape(*shape, self.hidden),
                (obs.lengths / self.count_scale).reshape(*shape, 1),
            ],
            dim=-1,
        )


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden,
        log_std_min=-5.0,
        log_std_max=2.0,
        state_dependent_std=True,
        ue_hidden=32,
        count_scale=20,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.encoder = UEEncoder(obs_dim, ue_hidden, count_scale)
        self.net = mlp(self.encoder.output_dim, 4 if state_dependent_std else 2, hidden)
        if not state_dependent_std:
            self.log_std = nn.Parameter(torch.zeros(2))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def distribution(self, obs):
        output = self.net(self.encoder(obs))
        if self.state_dependent_std:
            mean, log_std = output.chunk(2, dim=-1)
        else:
            mean, log_std = output, self.log_std.expand_as(output)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        # Inputs/actions and learner metrics have explicit finite checks. Rechecking
        # every Normal parameter would synchronize CUDA during each policy call.
        return Normal(mean, log_std.exp(), validate_args=False)

    def sample(self, obs, deterministic=False, *, with_std=False):
        distribution = self.distribution(obs)
        z = distribution.mean if deterministic else distribution.rsample()
        action, jacobian = disk_transform(z)
        result = action, distribution.log_prob(z).sum(-1) - jacobian, z
        if with_std:
            return (*result, distribution.stddev.detach().mean())
        return result

    def evaluate_latent(self, obs, z):
        distribution = self.distribution(obs)
        _, jacobian = disk_transform(z)
        log_prob = distribution.log_prob(z).sum(-1) - jacobian
        _, sampled_jacobian = disk_transform(distribution.rsample())
        entropy = distribution.entropy().sum(-1) + sampled_jacobian
        return log_prob, entropy


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, hidden, ue_hidden=32, count_scale=20):
        super().__init__()
        self.encoder = UEEncoder(obs_dim, ue_hidden, count_scale)
        self.net = mlp(self.encoder.output_dim, 9, hidden)

    def distribution(self, obs):
        return Categorical(logits=self.net(self.encoder(obs)), validate_args=False)

    def sample(self, obs, deterministic=False):
        distribution = self.distribution(obs)
        action = (
            distribution.probs.argmax(-1) if deterministic else distribution.sample()
        )
        return action, distribution.log_prob(action), action

    def evaluate_actions(self, obs, actions):
        distribution = self.distribution(obs)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            distribution.probs.max(-1).values,
        )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, hidden, ue_hidden=32, count_scale=20):
        super().__init__()
        self.encoder = UEEncoder(obs_dim, ue_hidden, count_scale)
        self.net = mlp(self.encoder.output_dim, 2, hidden)

    def forward(self, obs):
        return disk_transform(self.net(self.encoder(obs)))[0]


class TwinQ(nn.Module):
    def __init__(self, state_dim, num_uavs, hidden):
        super().__init__()
        self.q1 = mlp(state_dim + 2 * num_uavs, 1, hidden)
        self.q2 = mlp(state_dim + 2 * num_uavs, 1, hidden)

    def forward(self, state, actions):
        x = torch.cat([state, actions.flatten(start_dim=1)], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNormalizer(nn.Module):
    """Running return statistics with PopArt output rescaling."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("variance", torch.tensor(1.0))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @property
    def std(self):
        return self.variance.clamp_min(1e-4).sqrt()

    def normalize(self, value):
        return (value - self.mean) / self.std

    def denormalize(self, value):
        return value * self.std + self.mean

    @torch.no_grad()
    def update(self, values, last_layer, optimizer):
        values = values.detach().flatten()
        old_mean, old_std = self.mean.clone(), self.std.clone()
        n = values.numel()
        mean, var = values.mean(), values.var(unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        new_mean = self.mean + delta * (n / total)
        m2 = (
            self.variance * self.count
            + var * n
            + delta.square() * self.count * n / total
        )
        self.mean.copy_(new_mean)
        self.variance.copy_((m2 / total).float().clamp_min(1e-4))
        self.count.copy_(total)
        scale = old_std / self.std
        last_layer.weight.mul_(scale)
        last_layer.bias.copy_(
            (old_std * last_layer.bias + old_mean - self.mean) / self.std
        )
        for parameter in last_layer.parameters():
            state = optimizer.state.get(parameter, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(scale)
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(scale.square())
