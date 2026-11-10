"""Isolated production-size SAC timing; never exports trained weights.

Replay capacity is reduced to 1024 for timing. Scene, network, batch size and
update mathematics come from the pure-coverage 1M configuration. A fresh process
per device/thread setting keeps runtime configuration unambiguous.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import uniform_disk
from uav_research.replay import JointReplay
from uav_research.runtime import (
    atomic_json, configure_runtime, memory_stats, resolve_output, seed_all, source_digest,
)
from uav_research.vector import VectorEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    output = resolve_output(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.updates < 1 or args.threads < 1:
        parser.error('updates and threads must be positive')
    c = load_config('configs/sac_coverage_time_factor_relative_central_history_1m.yaml')
    c.run.device, c.run.cpu_threads = args.device, args.threads
    device = configure_runtime(c)
    seed_all(0)
    learner = make_learner(c, device)
    replay = JointReplay(1024, (c.env.num_uavs, c.env.obs_dim), c.env.state_dim, 300000)
    rng = np.random.default_rng(102)
    collection = []
    with VectorEnvironment(c.env, c.run.num_envs, 'batch') as envs:
        obs, state, _ = envs.reset(range(30000, 30000+c.run.num_envs))
        for i in range(64):
            action = uniform_disk(rng, (c.run.num_envs, c.env.num_uavs))
            stamp = time.perf_counter()
            no, ns, reward, term, _, _ = envs.step(action)
            replay.add(obs, state, action, reward, no, ns, term)
            if i >= 8:  # Exclude lazy compilation / warm caches.
                collection.append(time.perf_counter()-stamp)
            obs, state = no, ns

        def update():
            stamp = time.perf_counter()
            batch = replay.sample(c.algo.batch_size, torch.device('cpu'))
            sample = time.perf_counter()-stamp
            stamp = time.perf_counter()
            if device.type == 'cuda':
                batch = {k: v.pin_memory().to(device, non_blocking=True)
                         for k, v in batch.items()}
                torch.cuda.synchronize(device)
            transfer = time.perf_counter()-stamp
            stamp = time.perf_counter()
            metrics = learner.update(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-stamp
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite benchmark metrics')
            return sample, transfer, elapsed

        for _ in range(10):
            update()
        rows = [update() for _ in range(args.updates)]
        stamp = time.perf_counter()
        for _ in range(30):
            learner.act(obs)
        infer = (time.perf_counter()-stamp)/30
        result = dict(
            device=str(device), threads=args.threads, torch=torch.__version__,
            source_hash=source_digest(), updates=args.updates, replay_capacity=1024,
            mean_seconds=dict(zip(['sample', 'transfer', 'update'], np.mean(rows, axis=0).tolist())),
            total_update_mean=float(np.mean(np.sum(rows, axis=1))),
            total_update_p10_p90=np.quantile(np.sum(rows, axis=1), [.1, .9]).tolist(),
            collection_per_transition=float(np.mean(collection)/c.run.num_envs),
            inference_batch16_seconds=infer, formal_weights_discarded=True,
            **memory_stats(device),
        )
        atomic_json(output, result)
        print(result, flush=True)
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                for _ in range(5):
                    update()
            output.with_suffix('.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))


if __name__ == '__main__':
    main()
