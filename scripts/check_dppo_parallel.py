"""Bounded 1/2-GPU saved-chain gradient check and identical-rollout benchmark."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch

from uav_research.config import load_config
from uav_research.runtime import seed_all, rng_state, restore_rng, atomic_json
from uav_research.diffusion_dppo.storage import configure, load
from uav_research.diffusion_dppo.learner import DiffusionLearner, DppoRollout
from uav_research.diffusion_dppo.environment import ServiceVector
from uav_research.diffusion_dppo.parallel import evaluate_chain_parallel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--vector-steps', type=int, default=128)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.diffusion.actor_update_devices = []
    device = configure(cfg)
    if torch.cuda.device_count() < 2:
        raise ValueError('Expose two CUDA devices for this diagnostic')
    saved = load(args.resume, cfg, {'dppo_full'})
    seed_all(cfg.run.seed)
    learner = DiffusionLearner(cfg, device)
    learner.load_state_dict(saved['learner'])
    vector = ServiceVector(cfg.env, cfg.run.num_envs, cfg.run.vector_backend)
    vector.restore(saved['environments'])
    restore_rng(saved['rng'])
    obs, state = saved['observation'].copy(), saved['state'].copy()
    rollout = DppoRollout()
    started = time.perf_counter()
    try:
        for _ in range(args.vector_steps):
            action, chain = learner.actor.sample(obs)
            value = learner.values(state).cpu().numpy()
            no, ns, reward, done, trunc, _ = vector.step(action)
            rollout.append(obs, chain, state=state, reward=reward, value=value,
                next_value=learner.values(ns).cpu().numpy(), terminated=done, truncated=trunc)
            if (done | trunc).any():
                from uav_research.diffusion_dppo.observations import stack
                parts = []
                for i in range(cfg.run.num_envs):
                    if done[i] or trunc[i]:
                        item, ns[i], _ = vector.reset_at(i)
                    else:
                        item = no.take(np.arange(i*cfg.env.num_uavs, (i+1)*cfg.env.num_uavs))
                    parts.append(item)
                no = stack(parts)
            obs, state = no, ns
        collection = time.perf_counter()-started
    finally:
        vector.close()
    # Representative ragged groups include all UAVs from multiple times.
    packed, chains, _ = rollout.packed()
    index = np.arange(min(576, packed.groups))
    o, c = packed.take(index), chains.take(index)
    saved_rng = rng_state()
    grad = []
    predictions = []
    for devices in ([], [torch.device('cuda:0'), torch.device('cuda:1')]):
        learner.actor_opt.zero_grad(set_to_none=True)
        result = evaluate_chain_parallel(learner.actor, o, c, devices)
        weights = torch.linspace(-1, 1, result.numel(), device=device).reshape_as(result)
        (result*weights).mean().backward()
        predictions.append(result.detach().cpu())
        grad.append(torch.cat([x.grad.detach().flatten().cpu() if x.grad is not None else
            torch.zeros(x.numel()) for x in learner.actor.parameters()]))
    prediction_error = float((predictions[0]-predictions[1]).abs().max())
    gradient_relative = float(torch.linalg.vector_norm(grad[0]-grad[1]) /
                              torch.linalg.vector_norm(grad[0]).clamp_min(1e-12))
    if prediction_error > .005 or gradient_relative > .005:
        raise AssertionError((prediction_error, gradient_relative))
    rows = []
    # Equal rollout, initial optimizer/model state and sample permutation.
    for name, devices, micro in [('single_8192', [], 8192),
                                 ('single_16384', [], 16384),
                                 ('dual_16384', ['cuda:0', 'cuda:1'], 16384)]:
        trial = copy.deepcopy(cfg)
        trial.diffusion.actor_update_devices = devices
        trial.diffusion.microbatch_agents = micro
        learner = DiffusionLearner(trial, device)
        learner.load_state_dict(saved['learner'])
        restore_rng(saved_rng)
        for dev in range(2):
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        started = time.perf_counter()
        metrics = learner.update(rollout)
        for dev in range(2):
            torch.cuda.synchronize(dev)
        row = dict(mode=name, update_seconds=time.perf_counter()-started,
            optimizer_steps=learner.optimizer_steps-saved['learner']['optimizer_steps'],
            peak_gpu_mib=[torch.cuda.max_memory_allocated(dev)/2**20 for dev in range(2)],
            metrics=metrics)
        rows.append(row)
        print(row, flush=True)
        del learner
        torch.cuda.empty_cache()
    atomic_json(Path(args.output)/'result.json', dict(passed=True,
        source_steps=saved['env_steps'], joint_transitions=args.vector_steps*cfg.run.num_envs,
        collection_seconds=collection, probability_max_abs_difference=prediction_error,
        gradient_relative_l2=gradient_relative, timings=rows))


if __name__ == '__main__':
    main()
