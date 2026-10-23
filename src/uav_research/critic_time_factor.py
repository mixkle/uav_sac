"""Offline-only paired critics: direct return versus analytic horizon factor.

No production MAPPO policy, normalizer or checkpoint semantics are changed.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from torch import nn

from .critic_calibration import tensor_states
from .runtime import atomic_save, rng_state, restore_rng

ARMS = ('a_direct', 'b_factor_return_loss', 'c_factor_average_loss')
CHECKPOINT_KIND = 'offline_critic_time_factor_v1'


def horizon_factors(remaining, gamma):
    n = np.asarray(remaining)
    if not np.isfinite(n).all() or (n < 0).any() or (n != np.rint(n)).any():
        raise ValueError('Remaining action counts must be nonnegative integers')
    if not 0 <= gamma <= 1:
        raise ValueError('Discount must be in [0,1]')
    if gamma == 1:
        return n.astype(np.float64)
    if gamma == 0:
        return (n > 0).astype(np.float64)
    # Stable near gamma=1; floating-point labels may exceed 1 by roundoff only.
    return -np.expm1(np.log(gamma) * n) / (1 - gamma)


def target_statistics(fit_returns, gamma):
    g = np.asarray(fit_returns, dtype=np.float64)
    if g.ndim != 2 or not np.isfinite(g).all() or not g.shape[1]:
        raise ValueError('Expected finite [scene,time] fit returns')
    factors = horizon_factors(np.arange(g.shape[1], 0, -1), gamma)
    c = g / factors
    if (c < -1e-12).any() or (c > 1 + 1e-12).any():
        raise ValueError('Average-coverage targets must be in [0,1]')
    return dict(return_mean=float(g.mean()), return_std=max(float(g.std(ddof=0)), 1e-6),
                average_mean=float(c.mean()), average_std=max(float(c.std(ddof=0)), 1e-6),
                count=g.size)


class OfflineCritic(nn.Module):
    """Linear normalized output; horizon comes solely from state[:, -1]."""

    def __init__(self, source_value, arm, statistics, horizon, gamma):
        super().__init__()
        if arm not in ARMS or horizon <= 0:
            raise ValueError('Invalid critic arm/horizon')
        self.arm, self.horizon, self.gamma = arm, horizon, gamma
        self.statistics = dict(statistics)
        self.network = copy.deepcopy(source_value)
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()
        for key in ('return_mean', 'return_std', 'average_mean', 'average_std'):
            self.register_buffer(key, torch.tensor(statistics[key], dtype=torch.float64))
        self.register_buffer('factor_table', torch.tensor(
            horizon_factors(np.arange(horizon + 1), gamma), dtype=torch.float64))

    def factors(self, states):
        time = states[..., -1] * self.horizon
        nearest = time.round()
        if (not torch.isfinite(time).all() or (time < -1e-4).any()
                or (time > self.horizon + 1e-4).any()
                or (time-nearest).abs().max() > 1e-4):
            raise ValueError('State time must encode an integer action count')
        return self.factor_table[self.horizon - nearest.long()].to(states.dtype)

    def components(self, states):
        z = self.network(states).squeeze(-1)
        h = self.factors(states)
        if self.arm == 'a_direct':
            value = z * self.return_std.to(z.dtype) + self.return_mean.to(z.dtype)
            # Termination is analytic for every arm; terminal rows are not fitted.
            value = torch.where(h > 0, value, torch.zeros_like(value))
            average = torch.where(h > 0, value / h.clamp_min(1), torch.zeros_like(value))
        else:
            average = z * self.average_std.to(z.dtype) + self.average_mean.to(z.dtype)
            value = h * average
            average = torch.where(h > 0, average, torch.zeros_like(average))
        return value, average, h

    def forward(self, states):
        return self.components(states)[0]

    def loss(self, states, returns):
        value, average, h = self.components(states)
        if (h <= 0).any():
            raise ValueError('Terminal states do not have average-coverage labels')
        if self.arm == 'c_factor_average_loss':
            prediction = (average-self.average_mean.to(value.dtype))/self.average_std.to(value.dtype)
            label = (returns/h-self.average_mean.to(value.dtype))/self.average_std.to(value.dtype)
        else:
            prediction = (value-self.return_mean.to(value.dtype))/self.return_std.to(value.dtype)
            label = (returns-self.return_mean.to(value.dtype))/self.return_std.to(value.dtype)
        return nn.functional.huber_loss(prediction, label, delta=10.0)


def make_optimizer(model, learning_rate=5e-4):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5,
                            betas=(0.9, 0.999), weight_decay=0)


def fit_epoch(model, optimizer, dataset, batch_size, generator, max_grad_norm=.5):
    states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
    targets = dataset['returns'].reshape(-1)
    order = generator.permutation(len(states))
    model.train()
    loss_sum, grad_sum, batches = 0., 0., 0
    for start in range(0, len(states), batch_size):
        idx = order[start:start+batch_size]
        x = tensor_states(states[idx], next(model.parameters()).device)
        y = torch.tensor(targets[idx], dtype=torch.float32, device=x.device)
        loss = model.loss(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite offline critic loss/gradient; no automatic retries')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite critic parameter')
        loss_sum += loss.item() * len(idx)
        grad_sum += norm.item()
        batches += 1
    return dict(loss=loss_sum/len(states), gradient_norm=grad_sum/batches,
                optimizer_steps=batches,
                permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest())


@torch.no_grad()
def predict(model, dataset, batch_size=512):
    states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
    v, c = [], []
    model.eval()
    for start in range(0, len(states), batch_size):
        x = tensor_states(states[start:start+batch_size], next(model.parameters()).device)
        value, average, _ = model.components(x)
        v.append(value.cpu().numpy()); c.append(average.cpu().numpy())
    shape = dataset['returns'].shape
    return dict(value=np.concatenate(v).reshape(shape), average=np.concatenate(c).reshape(shape))


def error_metrics(predicted, labels):
    p, y = np.asarray(predicted, np.float64), np.asarray(labels, np.float64)
    if p.shape != y.shape or p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('Invalid critic prediction array')
    difference = p-y
    pc, yc = p-p.mean(0), y-y.mean(0)
    denominator = np.sqrt((pc**2).sum()*(yc**2).sum())
    varies = np.max(np.abs(pc)) > 1e-12 * max(1., float(np.max(np.abs(p))))
    stages = {}
    for name, sl in [('all', slice(None)), ('first100', slice(0,100)),
                     ('middle100', slice(150,250)), ('last100', slice(-100,None)),
                     ('last20', slice(-20,None)), ('last1', slice(-1,None))]:
        e = difference[:, sl]
        if not e.size:
            continue
        stages[name] = dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.abs(e).mean()),
                            bias=float(e.mean()), prediction_mean=float(p[:, sl].mean()),
                            label_mean=float(y[:, sl].mean()))
    return dict(stages=stages, time_centered_correlation=float((pc*yc).sum()/denominator)
                if varies and denominator > 1e-12 else None)


def score(prediction, returns, gamma):
    h = horizon_factors(np.arange(returns.shape[1], 0, -1), gamma)
    return dict(value=error_metrics(prediction['value'], returns),
                average=error_metrics(prediction['average'], returns/h),
                value_range_violation_fraction=float(((prediction['value'] < 0) |
                                                       (prediction['value'] > h)).mean()),
                average_range_violation_fraction=float(((prediction['average'] < 0) |
                                                         (prediction['average'] > 1)).mean()))


def save_offline(path, model, optimizer, generator, epoch, experiment):
    atomic_save(path, dict(kind=CHECKPOINT_KIND, arm=model.arm, epoch=epoch,
        horizon=model.horizon, discount=model.gamma, statistics=model.statistics,
        output_semantics='cumulative_return' if model.arm == 'a_direct' else 'discounted_average_coverage',
        loss_semantics='average' if model.arm == 'c_factor_average_loss' else 'cumulative_return',
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        permutation_rng=copy.deepcopy(generator.bit_generator.state), rng=rng_state(),
        experiment=experiment))


def restore_offline(path, model, optimizer, generator):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data.get('kind') != CHECKPOINT_KIND or data['arm'] != model.arm
            or data['horizon'] != model.horizon or data['discount'] != model.gamma
            or data['statistics'] != model.statistics):
        raise ValueError('Incompatible offline critic checkpoint')
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    generator.bit_generator.state = data['permutation_rng']
    restore_rng(data['rng'])
    return data


def paired_bootstrap(predictions, targets, replicates=2000, seed=950001):
    """Scene-resampled delta RMSE/MAE, later method minus earlier method."""
    names = list(predictions)
    scenes = len(targets)
    draws = np.random.default_rng(seed).integers(scenes, size=(replicates, scenes))
    result = []
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = predictions[left]-targets, predictions[right]-targets
            for stage, sl in [('all', slice(None)), ('last20', slice(-20,None)), ('last1', slice(-1,None))]:
                for metric in ['rmse', 'mae']:
                    if metric == 'rmse':
                        am, bm = (a[:,sl]**2).mean(1), (b[:,sl]**2).mean(1)
                        samples = np.sqrt(bm[draws].mean(1))-np.sqrt(am[draws].mean(1))
                        delta = np.sqrt(bm.mean())-np.sqrt(am.mean())
                    else:
                        am, bm = np.abs(a[:,sl]).mean(1), np.abs(b[:,sl]).mean(1)
                        samples = bm[draws].mean(1)-am[draws].mean(1)
                        delta = bm.mean()-am.mean()
                    low, high = np.quantile(samples, [.025,.975])
                    result.append(dict(left=left,right=right,stage=stage,metric=metric,
                        delta=float(delta),low=float(low),high=float(high)))
    return result
