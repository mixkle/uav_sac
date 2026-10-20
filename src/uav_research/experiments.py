from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
from .algorithms import make_learner
from .environment import UAVEnvironment, uniform_disk
from .evaluate import evaluate, policy_from_checkpoint
from .replay import JointReplay
from .runtime import (
    atomic_json,
    configure_runtime,
    seed_all,
    isolated_rng,
    memory_stats,
    resolve_output,
)
from .vector import VectorEnvironment
from .observations import stack_observations, RawObservation


def assert_observations_equal(a, b):
    for key in ("fixed", "users", "offsets"):
        np.testing.assert_allclose(
            getattr(a, key), getattr(b, key), rtol=1e-6, atol=1e-7
        )


def check_environment(cfg, output, scenes=8, steps=32):
    report = dict(
        obs_shape=[cfg.env.num_uavs, cfg.env.obs_dim],
        state_dim=cfg.env.state_dim,
        per_subband_power_dbm=cfg.env.subband_power_dbm,
        noise_mw=cfg.env.noise_mw,
        bs_positions=cfg.env.bs_positions,
        scenes=[],
    )
    configure_runtime(cfg)
    for seed in range(scenes):
        ref_config = copy.deepcopy(cfg.env)
        ref_config.backend = "numpy"
        fast_config = copy.deepcopy(cfg.env)
        fast_config.backend = "numba"
        reference, fast = UAVEnvironment(ref_config), UAVEnvironment(fast_config)
        ro, rs, ri = reference.reset(seed)
        fo, fs, fi = fast.reset(seed)
        assert_observations_equal(ro, fo)
        np.testing.assert_array_equal(rs, fs)
        rng = np.random.default_rng(seed + 500000)
        for _ in range(min(steps, cfg.env.horizon)):
            action = (
                rng.integers(0, 9, cfg.env.num_uavs)
                if cfg.env.action_space == "discrete9"
                else uniform_disk(rng, (cfg.env.num_uavs,))
            )
            r = reference.step(action)
            f = fast.step(action)
            for a, b in zip(r[:5], f[:5]):
                if isinstance(a, RawObservation):
                    assert_observations_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(reference.C, fast.C)
            np.testing.assert_array_equal(reference.B, fast.B)
            np.testing.assert_allclose(reference.retained, fast.retained, atol=1e-12)
            assert np.all(fast.C.sum(0) <= 1) and np.all(
                fast.C.sum(1) <= cfg.env.capacity
            )
            assert (
                f[5]["robustness"]
                <= (1 - 1 / cfg.env.num_uavs) * f[5]["coverage"] + 1e-12
            )
        report["scenes"].append(
            dict(
                seed=seed,
                uav_initialization=cfg.env.uav_initialization,
                initial=fi,
                final=fast.info,
            )
        )
    report["passed"] = True
    atomic_json(resolve_output(output), report)
    return report


def benchmark(
    cfg,
    output,
    counts=(1, 8, 16, 32),
    backends=("batch", "process"),
    devices=("cpu",),
    steps=512,
):
    rows = []
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            rows.append(dict(device=device, skipped="CUDA unavailable"))
            continue
        for backend in backends:
            for count in counts:
                c = copy.deepcopy(cfg)
                c.run.device = device
                c.run.num_envs = count
                c.run.vector_backend = backend
                c.algo.replay_capacity = max(c.algo.batch_size * 2, count * 2)
                actual_device = configure_runtime(c)
                seed_all(c.run.seed)
                start = time.perf_counter()
                learner = make_learner(c, actual_device)
                replay = (
                    None
                    if c.algo.name == "mappo"
                    else JointReplay(
                        c.algo.replay_capacity,
                        (c.env.num_uavs, c.env.obs_dim),
                        c.env.state_dim,
                        100,
                    )
                )
                rng = np.random.default_rng(80000)
                phases = {
                    key: 0.0
                    for key in (
                        "environment",
                        "inference",
                        "sample",
                        "transfer",
                        "update",
                    )
                }
                with VectorEnvironment(c.env, count, backend) as envs:
                    obs, state, _ = envs.reset(list(range(80000, 80000 + count)))
                    # Compile and warm the environment, inference and learner paths before timing.
                    no, ns, reward, done, truncated, infos = envs.step(
                        rng.integers(0, 9, (count, c.env.num_uavs))
                        if c.env.action_space == "discrete9"
                        else uniform_disk(rng, (count, c.env.num_uavs))
                    )
                    obs, state = no, ns
                    for _ in range(math.ceil(c.algo.batch_size / count)):
                        action = learner.act(obs)
                        no, ns, reward, done, truncated, infos = envs.step(action)
                        if replay is not None:
                            replay.add(obs, state, action, reward, no, ns, done)
                        obs, state = no, ns
                        for index in np.flatnonzero(done | truncated):
                            obs[index], state[index], _ = envs.reset_at(int(index))
                    if c.algo.name != "mappo":
                        learner.update(replay.sample(c.algo.batch_size, actual_device))
                    startup = time.perf_counter() - start
                    measured = time.perf_counter()
                    collected = 0
                    credit = 0
                    updates_before = learner.updates
                    rollout = {
                        key: []
                        for key in (
                            "obs",
                            "state",
                            "actions"
                            if c.env.action_space == "discrete9"
                            else "latent",
                            "logp",
                            "value",
                            "next_value",
                            "reward",
                            "terminated",
                            "truncated",
                        )
                    }
                    while collected < steps:
                        stamp = time.perf_counter()
                        if c.algo.name == "mappo":
                            action, logp, latent, value = learner.collect(obs, state)
                        else:
                            action = learner.act(obs)
                        phases["inference"] += time.perf_counter() - stamp
                        stamp = time.perf_counter()
                        no, ns, reward, done, truncated, infos = envs.step(action)
                        phases["environment"] += time.perf_counter() - stamp
                        if c.algo.name == "mappo":
                            next_value = learner.values(ns).cpu().numpy()
                            for key, val in dict(
                                obs=obs,
                                state=state,
                                **(
                                    {"actions": action}
                                    if c.env.action_space == "discrete9"
                                    else {"latent": latent}
                                ),
                                logp=logp,
                                value=value,
                                next_value=next_value,
                                reward=reward,
                                terminated=done,
                                truncated=truncated,
                            ).items():
                                rollout[key].append(val)
                            if (
                                len(rollout["reward"]) >= c.algo.rollout_steps
                                or collected + count >= steps
                            ):
                                stamp = time.perf_counter()
                                learner.update(
                                    {
                                        k: stack_observations(v)
                                        if k == "obs"
                                        else np.stack(v)
                                        for k, v in rollout.items()
                                    }
                                )
                                phases["update"] += time.perf_counter() - stamp
                                rollout = {key: [] for key in rollout}
                        else:
                            replay.add(obs, state, action, reward, no, ns, done)
                            credit += count
                            while credit >= c.algo.transitions_per_update:
                                stamp = time.perf_counter()
                                batch = replay.sample(
                                    c.algo.batch_size, torch.device("cpu")
                                )
                                phases["sample"] += time.perf_counter() - stamp
                                stamp = time.perf_counter()
                                if actual_device.type == "cuda":
                                    batch = {
                                        k: v.pin_memory().to(
                                            actual_device, non_blocking=True
                                        )
                                        for k, v in batch.items()
                                    }
                                    torch.cuda.synchronize(actual_device)
                                phases["transfer"] += time.perf_counter() - stamp
                                stamp = time.perf_counter()
                                learner.update(batch)
                                phases["update"] += time.perf_counter() - stamp
                                credit -= c.algo.transitions_per_update
                        obs, state = no, ns
                        for index in np.flatnonzero(done | truncated):
                            obs[index], state[index], _ = envs.reset_at(int(index))
                        collected += count
                    wall = time.perf_counter() - measured
                    row = dict(
                        device=device,
                        vector_backend=backend,
                        num_envs=count,
                        algorithm=c.algo.name,
                        startup_seconds=startup,
                        transitions=collected,
                        wall_seconds=wall,
                        transitions_per_second=collected / wall,
                        updates=learner.updates - updates_before,
                        phase_seconds=phases,
                        **memory_stats(actual_device),
                    )
                    if cfg.run.hourly_price:
                        row["cost_per_million"] = (
                            cfg.run.hourly_price
                            * 1e6
                            / (row["transitions_per_second"] * 3600)
                        )
                    from .profiling import stage_probe

                    row["isolated_stages"] = stage_probe(
                        c, learner, replay, actual_device
                    )
                    rows.append(row)
                    print(row, flush=True)
    valid = [row for row in rows if "transitions_per_second" in row]
    report = dict(
        results=rows,
        recommendation=max(valid, key=lambda row: row["transitions_per_second"])
        if valid
        else None,
        note="End-to-end throughput includes learner updates and resets; startup is reported separately.",
    )
    if valid:
        report["estimated_training_hours"] = (
            cfg.run.total_steps
            / report["recommendation"]["transitions_per_second"]
            / 3600
        )
    atomic_json(resolve_output(output), report)
    return report


def diagnose(
    checkpoint, output, seeds=(10000, 10001), candidates=16, lookahead=10, device="cpu"
):
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg, learner = policy_from_checkpoint(checkpoint, torch.device(device))
    evaluate(cfg, learner, seeds, output / "evaluation", trace_count=len(seeds))
    report = dict(checkpoint=str(checkpoint), probes=[])
    if not hasattr(learner, "critic"):
        report["critic_probe"] = (
            "MAPPO has V(s), no joint-action critic to rank actions"
        )
    else:
        with isolated_rng():
            for seed in seeds:
                env = UAVEnvironment(cfg.env)
                obs, state, _ = env.reset(seed)
                original = env.snapshot()
                actions = [
                    learner.act(obs, True),
                    np.zeros((cfg.env.num_uavs, 2), np.float32),
                ]
                rng = np.random.default_rng(seed + 900000)
                actions.extend(uniform_disk(rng, (candidates - 2, cfg.env.num_uavs)))
                actions = np.stack(actions)
                with torch.no_grad():
                    states = torch.as_tensor(
                        np.repeat(state[None], len(actions), axis=0),
                        device=learner.device,
                    )
                    q1, q2 = learner.critic(
                        states, torch.as_tensor(actions, device=learner.device)
                    )
                    estimates = torch.minimum(q1, q2).cpu().numpy()
                returns = []
                for action in actions:
                    env.restore(original)
                    accumulated = 0.0
                    obs, state, reward, done, _, _ = env.step(action)
                    accumulated += reward
                    for t in range(1, min(lookahead, cfg.env.horizon)):
                        if done:
                            break
                        obs, state, reward, done, _, _ = env.step(
                            learner.act(obs, True)
                        )
                        accumulated += cfg.algo.discount**t * reward
                    returns.append(accumulated)
                correlation = (
                    float(np.corrcoef(estimates, returns)[0, 1])
                    if np.std(estimates) > 0 and np.std(returns) > 0
                    else None
                )
                report["probes"].append(
                    dict(
                        scene_seed=seed,
                        q=estimates.tolist(),
                        short_returns=returns,
                        correlation=correlation,
                        policy_action_rank_by_q=int(
                            np.sum(estimates > estimates[0]) + 1
                        ),
                        policy_action_rank_by_return=int(
                            np.sum(np.array(returns) > returns[0]) + 1
                        ),
                    )
                )
    atomic_json(output / "diagnosis.json", report)
    return report
