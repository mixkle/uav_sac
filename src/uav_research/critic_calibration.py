"""Offline finite-horizon value calibration with an unchanged MAPPO actor."""
from __future__ import annotations
import hashlib
from pathlib import Path
import json
import time
import numpy as np
import torch
from .runtime import atomic_json, isolated_rng, restore_rng
from .vector import VectorEnvironment


def discounted_episode_returns(rewards, discount):
    """[episode,time] full returns, with zero value after genuine termination."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or not np.isfinite(rewards).all():
        raise ValueError('Expected finite [episode,time] rewards')
    if not 0 <= discount <= 1:
        raise ValueError('Discount must be in [0,1]')
    returns = np.empty_like(rewards)
    carry = np.zeros(len(rewards), dtype=np.float64)
    for t in reversed(range(rewards.shape[1])):
        carry = rewards[:, t] + discount * carry
        returns[:, t] = carry
    return returns


def actor_digest(actor):
    digest = hashlib.sha256()
    for name, tensor in actor.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect_complete_episodes(cfg, policy, scene_seeds, action_seed, output):
    """Persist states before actions; collect complete episodes without autoreset."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seeds = list(scene_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Expected distinct scene seeds')
    horizon, dim = cfg.env.horizon, cfg.env.state_dim
    states = np.lib.format.open_memmap(output/'states.npy', mode='w+', dtype=np.float32,
                                     shape=(len(seeds), horizon, dim))
    rewards = np.empty((len(seeds), horizon), np.float64)
    coverage = np.empty_like(rewards)
    actions_digest = hashlib.sha256()
    stamp = time.perf_counter()
    with isolated_rng():
        torch.manual_seed(action_seed)
        for start in range(0, len(seeds), cfg.run.num_envs):
            selected = seeds[start:start+cfg.run.num_envs]
            indices = np.arange(start, start+len(selected))
            with VectorEnvironment(cfg.env, len(selected), 'batch') as vector:
                obs, state, _ = vector.reset(selected)
                for t in range(horizon):
                    np.testing.assert_allclose(state[:, -1], t/horizon, atol=1e-7)
                    states[indices, t] = state
                    actions = policy.act(obs, False)
                    actions_digest.update(actions.tobytes())
                    obs, state, _, done, truncated, infos = vector.step(actions)
                    if truncated.any() or not np.array_equal(done, np.full(len(selected), t == horizon-1)):
                        raise AssertionError('Calibration requires genuine full-episode termination')
                    rewards[indices, t] = [info['reward'] for info in infos]
                    coverage[indices, t] = [info['coverage'] for info in infos]
            print(f'Collected {output.name}: {start+len(selected)}/{len(seeds)} episodes', flush=True)
    states.flush()
    returns = discounted_episode_returns(rewards, cfg.algo.discount)
    np.save(output/'returns.npy', returns)
    np.savez_compressed(output/'episodes.npz', scene_seeds=np.asarray(seeds),
                        rewards=rewards, coverage=coverage)
    atomic_json(output/'metadata.json', dict(scene_seeds=seeds, action_seed=action_seed,
                horizon=horizon, state_dim=dim, discount=cfg.algo.discount,
                joint_transitions=len(seeds)*horizon,
                actor_sha256=actor_digest(policy.actor), action_sequence_sha256=actions_digest.hexdigest(),
                elapsed_seconds=time.perf_counter()-stamp,
                state_bytes=states.nbytes, coverage_mean=float(coverage.mean()),
                episode_return_mean=float(rewards.sum(1).mean())))
    del states
    return load_dataset(output)


def load_dataset(output):
    output = Path(output)
    metadata = json.loads((output/'metadata.json').read_text())
    state = np.load(output/'states.npy', mmap_mode='r')
    returns = np.load(output/'returns.npy')
    if state.shape[:2] != returns.shape or state.shape[-1] != metadata['state_dim']:
        raise ValueError('Dataset shape mismatch')
    return dict(states=state, returns=returns, metadata=metadata)


def tensor_states(states, device):
    # Copies one minibatch only, including read-only memmap slices.
    return torch.from_numpy(np.array(states, dtype=np.float32, copy=True)).to(device)


def rebase_normalizer_from_fit_targets(policy, returns):
    """Freeze fresh fit-only target statistics and preserve raw predictions (PopArt)."""
    probe = policy.value[-1].weight.detach().clone()
    policy.normalizer.count.zero_()
    policy.normalizer.update(torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=policy.device),
                             policy.value[-1], policy.value_opt)
    return dict(mean=policy.normalizer.mean.item(), std=policy.normalizer.std.item(),
                count=policy.normalizer.count.item(), output_layer_changed=not torch.equal(probe, policy.value[-1].weight))


def configure_calibration_optimizer(policy, *, reset_adam=False, learning_rate=None):
    """Change only Adam history or LR, after the common PopArt rebase."""
    if learning_rate is not None and (not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Learning rate must be finite and positive')
    if reset_adam:
        if len(policy.value_opt.param_groups) != 1:
            raise ValueError('Calibration expects a single Adam parameter group')
        options = dict(policy.value_opt.defaults)
        options['lr'] = policy.value_opt.param_groups[0]['lr']
        policy.value_opt = torch.optim.Adam(policy.value.parameters(), **options)
    if learning_rate is not None:
        for group in policy.value_opt.param_groups:
            group['lr'] = float(learning_rate)
    return dict(reset_adam=reset_adam, learning_rate=policy.value_opt.param_groups[0]['lr'],
                initialized_parameter_states=len(policy.value_opt.state))


def restore_calibration_checkpoint(policy, checkpoint, sample_count):
    """Resume value/Adam/statistics and epoch permutations without a fresh rebase."""
    if sample_count <= 0 or int(checkpoint['epoch']) < 0:
        raise ValueError('Invalid calibration dataset size or checkpoint epoch')
    for key in ('value', 'normalizer'):
        getattr(policy, key).load_state_dict(checkpoint[key])
    policy.value_opt.load_state_dict(checkpoint['value_optimizer'])
    restore_rng(checkpoint['rng'])
    generator = np.random.default_rng(checkpoint['experiment']['permutation_seed'])
    if 'permutation_rng' in checkpoint:
        generator.bit_generator.state = checkpoint['permutation_rng']
    else:
        # Earlier offline checkpoints saved the seed, but not this independent RNG.
        # fit_critic_epoch consumes exactly one full permutation per completed epoch.
        for _ in range(int(checkpoint['epoch'])):
            generator.permutation(sample_count)
    return generator


def fit_critic_epoch(policy, dataset, batch_size, permutation_rng):
    states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
    returns = dataset['returns'].reshape(-1)
    permutation = permutation_rng.permutation(len(states))
    loss_sum, grad_sum, batches = 0.0, 0.0, 0
    policy.value.train()
    for start in range(0, len(states), batch_size):
        indices = permutation[start:start+batch_size]
        x = tensor_states(states[indices], policy.device)
        y = torch.as_tensor(returns[indices], dtype=torch.float32, device=policy.device)
        predicted = policy.value(x).squeeze(-1)
        target = policy.normalizer.normalize(y)
        # No old prediction, bootstrap target, or value clipping in this experiment.
        loss = torch.nn.functional.huber_loss(predicted, target, delta=10.0)
        policy.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.value.parameters(), policy.cfg.algo.max_grad_norm)
        policy.value_opt.step()
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite critic calibration update')
        loss_sum += loss.item()*len(indices); grad_sum += grad.item(); batches += 1
    return dict(normalized_huber_loss=loss_sum/len(states), value_grad_norm=grad_sum/batches,
                optimizer_steps=batches)


@torch.no_grad()
def predict_critic(policy, dataset, batch_size=512):
    states = dataset['states'].reshape(-1, dataset['states'].shape[-1])
    chunks = []
    policy.value.eval()
    for start in range(0, len(states), batch_size):
        x = tensor_states(states[start:start+batch_size], policy.device)
        chunks.append(policy.normalizer.denormalize(policy.value(x).squeeze(-1)).cpu().numpy())
    return np.concatenate(chunks).reshape(dataset['returns'].shape)


def calibration_metrics(predicted, returns, reward_upper_bound):
    predicted, returns = np.asarray(predicted, np.float64), np.asarray(returns, np.float64)
    error = predicted-returns
    if predicted.shape != returns.shape or predicted.ndim != 2 or not np.isfinite(predicted).all():
        raise ValueError('Expected finite episode/time prediction arrays')
    variance = returns.var()
    result = dict(raw_rmse=float(np.sqrt(np.mean(error**2))), raw_mae=float(np.mean(np.abs(error))),
                  raw_bias=float(error.mean()), explained_variance=float(1-error.var()/variance) if variance else 0.0,
                  final_step_rmse=float(np.sqrt(np.mean(error[:, -1]**2))),
                  final_step_prediction_mean=float(predicted[:, -1].mean()),
                  final_step_return_mean=float(returns[:, -1].mean()),
                  final_step_upper_bound_violations=float((predicted[:, -1] > reward_upper_bound).mean()),
                  negative_prediction_fraction=float((predicted < 0).mean()))
    h = predicted.shape[1]
    bounds = [('first100', 0, min(100,h)), ('middle', min(100,h), max(min(100,h),h-100)),
              ('last100', max(0,h-100),h), ('last20',max(0,h-20),h)]
    result['time_bins'] = {name: dict(rmse=float(np.sqrt(np.mean(error[:,a:b]**2))),
                          bias=float(error[:,a:b].mean())) for name,a,b in bounds if b>a}
    return result
