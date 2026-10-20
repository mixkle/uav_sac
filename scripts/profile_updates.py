"""Isolate SAC update costs on a fixed physical replay batch (no research run)."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment, uniform_disk
from uav_research.replay import JointReplay
from uav_research.observations import stack_observations
from uav_research.runtime import configure_runtime, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/server_probe.yaml")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--disable-distribution-validation", action="store_true")
    parser.add_argument("--fused-adam", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    device = configure_runtime(cfg)
    seed_all(0)
    if args.disable_distribution_validation:
        torch.distributions.Distribution.set_default_validate_args(False)
    learner = make_learner(cfg, device)
    if args.fused_adam:
        if device.type != "cuda":
            parser.error("--fused-adam requires CUDA")
        for name in ("actor_opt", "critic_opt", "alpha_opt"):
            previous = getattr(learner, name)
            parameters = [p for group in previous.param_groups for p in group["params"]]
            setattr(
                learner,
                name,
                torch.optim.Adam(
                    parameters, lr=previous.param_groups[0]["lr"], fused=True
                ),
            )
    replay = JointReplay(
        cfg.algo.batch_size, (cfg.env.num_uavs, cfg.env.obs_dim), cfg.env.state_dim
    )
    env = UAVEnvironment(cfg.env)
    obs, state, _ = env.reset(123)
    rng = np.random.default_rng(123)
    for _ in range(cfg.algo.batch_size):
        action = uniform_disk(rng, (cfg.env.num_uavs,))
        no, ns, reward, done, _, _ = env.step(action)
        replay.add(
            stack_observations([obs]),
            state[None],
            action[None],
            np.array([reward], np.float32),
            stack_observations([no]),
            ns[None],
            np.array([done]),
        )
        obs, state = no, ns
        if done:
            obs, state, _ = env.reset()
    batch = replay.sample(cfg.algo.batch_size, device)
    for _ in range(10):
        learner.update(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.updates):
        metrics = learner.update(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(10):
            learner.update(batch)
    (args.output / "operators.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    )
    prof.export_chrome_trace(str(args.output / "trace.json"))
    summary = dict(
        update_ms=1000 * elapsed / args.updates,
        updates=args.updates,
        config=cfg.as_dict(),
        validation_disabled=args.disable_distribution_validation,
        fused_adam=args.fused_adam,
        final_metrics=metrics,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"update_ms": summary["update_ms"]}), flush=True)


if __name__ == "__main__":
    main()
