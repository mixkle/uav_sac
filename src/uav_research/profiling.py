"""Isolated CPU stages, reported separately from end-to-end throughput."""

import tempfile
import time
from pathlib import Path
import torch
from .environment import physical_links, UAVEnvironment
from .observations import stack_observations
from .checkpoints import save_full


def stage_probe(cfg, learner, replay, device, repeats=20):
    env = UAVEnvironment(cfg.env)
    obs, _, _ = env.reset(80099)
    raw = stack_observations([obs] * cfg.run.num_envs)
    tensor = raw.to_torch(device)

    def timed(fn):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stamp = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - stamp) / repeats

    with torch.no_grad():
        result = dict(
            physical_links_seconds=timed(
                lambda: physical_links(cfg.env, env.uav_xy, env.user_xy)
            ),
            observation_pack_seconds=timed(
                lambda: stack_observations([obs] * cfg.run.num_envs)
            ),
            encoder_aggregate_seconds=timed(lambda: learner.actor.encoder(tensor)),
            raw_to_tensor_seconds=timed(lambda: raw.to_torch(device)),
        )
    with tempfile.TemporaryDirectory(prefix="uav_checkpoint_probe_") as directory:
        stamp = time.perf_counter()
        save_full(
            Path(directory) / "checkpoint",
            {"learner": learner.state_dict(), "config": cfg.as_dict()},
            replay,
        )
        result["save_seconds"] = time.perf_counter() - stamp
        result["saved_transitions"] = replay.size if replay is not None else 0
        result["saved_replay_bytes"] = (
            replay.effective_bytes if replay is not None else 0
        )
    result["note"] = (
        "Isolated initial-layout microbenchmarks; physical_links is one environment, pack/encoder use num_envs; not additive with end-to-end phases."
    )
    return result
