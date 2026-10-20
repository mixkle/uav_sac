from __future__ import annotations
from .actions import action_specification, validate_action_specification
from collections import defaultdict
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .algorithms import make_learner
from .config import Config, save_config, config_from_dict
from .environment import uniform_disk
from .evaluate import evaluate
from .replay import JointReplay
from .observations import OBSERVATION_VERSION, stack_observations
from .checkpoints import save_full, restore_replay
import shutil
import psutil
from .runtime import (
    configure_runtime,
    seed_all,
    rng_state,
    restore_rng,
    atomic_save,
    atomic_json,
    load_checkpoint,
    metadata,
    memory_stats,
    CSVLogger,
    METRIC_FIELDS,
    resolve_output,
    validate_resume_output,
)
from .vector import VectorEnvironment


def train(cfg: Config, output, resume=None):
    cfg.validate()
    if cfg.run.total_steps % cfg.run.num_envs:
        raise ValueError(
            "total_steps must be divisible by num_envs (counts joint transitions)"
        )
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and resume is None:
        raise ValueError(
            "Run directory already exists; choose a new directory or --resume full checkpoint"
        )
    device = configure_runtime(cfg)
    seed_all(cfg.run.seed)
    start = time.perf_counter()
    elapsed_before = 0.0
    phases = defaultdict(float)
    learner = make_learner(cfg, device)
    exploration = np.random.default_rng(np.random.SeedSequence([cfg.run.seed, 102]))
    replay = (
        None
        if cfg.algo.name == "mappo"
        else JointReplay(
            cfg.algo.replay_capacity,
            (cfg.env.num_uavs, cfg.env.obs_dim),
            cfg.env.state_dim,
            cfg.run.seed + 300000,
        )
    )
    # Replay tensors must never be mapped to CUDA. File mapping avoids a second
    # resident replay during loading; learner/optimizer loaders move their tensors.
    restored = load_checkpoint(resume, "cpu", mmap=True) if resume else None
    if restored:
        if restored.get("kind") != "full":
            raise ValueError("Resume requires a full checkpoint, not best.pt/last.pt")
        validate_resume_output(output, restored)
        validate_action_specification(restored, cfg.env, cfg.algo.name)
        previous = config_from_dict(restored["config"]).as_dict()
        if (
            previous["env"] != cfg.as_dict()["env"]
            or previous["algo"] != cfg.as_dict()["algo"]
        ):
            raise ValueError("Environment/algorithm changed during resume")
        for key in (
            "seed",
            "num_envs",
            "vector_backend",
            "deterministic",
            "evaluation_policy",
            "evaluation_action_seed",
            "validation_seeds",
            "test_seeds",
            "eval_interval",
        ):
            if previous["run"][key] != getattr(cfg.run, key):
                raise ValueError(f"Resume changed run.{key}")
        learner.load_state_dict(restored["learner"])
        if replay is not None:
            restore_replay(resume, replay, restored["replay_metadata"])
        exploration.bit_generator.state = restored["exploration_rng"]
        elapsed_before = restored["elapsed_seconds"]
        phases.update(restored["phase_seconds"])
        best_model = restored.get("best_model")
        best_selection = restored.get("best_selection")
        if math.isfinite(restored["best_score"]):
            if best_model is None or best_selection is None:
                # Legacy runs can migrate when their matching historical best is available.
                origin = Path(resume).parent
                if (origin / "best.pt").exists() and (
                    origin / "best_selection.json"
                ).exists():
                    import json

                    candidate = json.loads((origin / "best_selection.json").read_text())
                    if (
                        candidate["validation_return"] == restored["best_score"]
                        and candidate["env_steps"] <= restored["env_steps"]
                    ):
                        best_model = load_checkpoint(origin / "best.pt")
                        best_selection = candidate
            if best_model is None or best_selection is None:
                raise ValueError(
                    "Legacy checkpoint lacks the historical best model; migrate a complete run "
                    "or create a self-contained checkpoint before resuming"
                )
            if (
                best_model["env_steps"] > restored["env_steps"]
                or best_model["env_steps"] != best_selection["env_steps"]
            ):
                raise ValueError("Best model belongs to a later checkpoint branch")
            atomic_save(output / "best.pt", best_model)
            atomic_json(output / "best_selection.json", best_selection)
    save_config(cfg, output / "config.yaml")
    session_metadata = metadata(cfg, device)
    history_path = output / "sessions.json"
    import json

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({**session_metadata, "resume_from": str(resume) if resume else None})
    atomic_json(history_path, history)
    atomic_json(output / "metadata.json", session_metadata)
    fields = (
        METRIC_FIELDS
        + (
            [f"action_{i}_frequency" for i in range(9)]
            + ["hover_fraction", "categorical_entropy", "max_action_probability"]
            if cfg.env.action_space == "discrete9"
            else []
        )
        + [
            key
            for g in range(cfg.env.num_bs)
            for key in (f"bs_{g}_direct_uavs", f"bs_{g}_only_reachable")
        ]
    )
    logger = CSVLogger(output / "metrics.csv", fields)
    validation_metrics = [
        "coverage", "discovery_fraction", "disconnection_ratio", "reward_primary",
        "throughput_bps", "robustness", "retention_ratio", "return", "worst_fault_coverage",
        "user_displacement_mean", "uav_nearest_neighbor_mean", "uav_close_neighbor_fraction",
    ]
    eval_logger = CSVLogger(
        output / "validation.csv",
        ["env_steps", "elapsed_seconds"] + validation_metrics,
    )
    writer = None
    if cfg.run.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            str(output / "tensorboard"),
            purge_step=restored["env_steps"] if restored else None,
        )
    stop_requested = [False]
    old_handlers = {}

    def request_stop(signum, frame):
        stop_requested[0] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)
        except ValueError:
            pass  # Tests may invoke training from a non-main thread.
    steps = restored["env_steps"] if restored else 0
    credit = restored["update_credit"] if restored else 0
    best = restored["best_score"] if restored else -math.inf
    next_eval = restored["next_eval"] if restored else cfg.run.eval_interval
    next_log = steps + cfg.run.log_interval
    last_full = time.perf_counter()
    latest_metrics = {}
    reason = "completed"
    mappo_checkpoint_safe = True
    manifest = dict(
        status="running",
        algorithm=cfg.algo.name,
        action_specification=action_specification(cfg.env, cfg.algo.name),
        seed=cfg.run.seed,
        env_steps=steps,
        config_hash=cfg.digest(),
        replay_allocated_bytes=replay.allocated_bytes if replay else 0,
        replay_worst_case_bytes=(
            replay.allocated_bytes
            + cfg.algo.replay_capacity * 2 * cfg.env.num_uavs * cfg.env.num_users * 16
        )
        if replay
        else 0,
    )
    atomic_json(output / "manifest.json", manifest)
    print(
        f"Replay worst-case raw storage: {manifest['replay_worst_case_bytes'] / 2**30:.3f} GiB; RSS limit: {cfg.run.memory_limit_gib:g} GiB",
        flush=True,
    )

    def elapsed():
        return elapsed_before + time.perf_counter() - start

    def budget_expired():
        if (
            cfg.run.max_wall_seconds is not None
            and elapsed() >= cfg.run.max_wall_seconds
        ):
            return True
        return (
            cfg.run.budget is not None
            and elapsed() * cfg.run.hourly_price / 3600 >= cfg.run.budget
        )

    def light():
        return dict(
            kind="model",
            action_specification=action_specification(cfg.env, cfg.algo.name),
            observation_version=OBSERVATION_VERSION,
            config=cfg.as_dict(),
            learner=learner.state_dict(),
            env_steps=steps,
        )

    def validate():
        nonlocal best
        stamp = time.perf_counter()
        atomic_save(output / "last.pt", light())
        phases["io"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        rows, summary = evaluate(cfg, learner, cfg.run.validation_seeds)
        phases["evaluation"] += time.perf_counter() - stamp
        means = summary["mean"]
        eval_logger.write(
            dict(
                env_steps=steps,
                elapsed_seconds=elapsed(),
                **{
                    k: means[k]
                    for k in validation_metrics
                },
            )
        )
        atomic_json(output / "validation_latest.json", summary)
        atomic_json(
            output / "validation_history" / f"{steps:012d}.json",
            {"episodes": rows, "summary": summary},
        )
        # A lightweight latest model survives even if the next full save fails.
        stamp = time.perf_counter()
        atomic_save(output / "last.pt", light())
        phases["io"] += time.perf_counter() - stamp
        if writer:
            for key, value in means.items():
                writer.add_scalar(f"validation/{key}", value, steps)
        if means["return"] > best:
            best = means["return"]
            stamp = time.perf_counter()
            atomic_save(output / "best.pt", light())
            phases["io"] += time.perf_counter() - stamp
            atomic_json(
                output / "best_selection.json",
                dict(
                    env_steps=steps,
                    validation_return=best,
                    seeds=cfg.run.validation_seeds,
                ),
            )

    def write_metrics(infos):
        env_means = {
            k: float(np.mean([info[k] for info in infos]))
            for k, v in infos[0].items()
            if isinstance(v, (int, float))
        }
        row = {
            "replay_effective_bytes": replay.effective_bytes if replay else 0,
            **env_means,
            **latest_metrics,
            "component_services": json.dumps(
                [info["component_services"] for info in infos]
            ),
            "env_steps": steps,
            "updates": learner.updates,
            "elapsed_seconds": elapsed(),
            "training_seconds": sum(
                phases[k]
                for k in ("environment", "inference", "sample", "transfer", "update")
            ),
            "transitions_per_second": steps / max(elapsed(), 1e-9),
            **memory_stats(device),
        }
        row.update(
            {
                f"{name}_seconds": phases[name]
                for name in (
                    "environment",
                    "inference",
                    "sample",
                    "transfer",
                    "update",
                    "evaluation",
                    "io",
                )
            }
        )
        if cfg.run.hourly_price:
            row["estimated_cost"] = elapsed() * cfg.run.hourly_price / 3600
        logger.write({k: v for k, v in row.items() if k in fields})
        manifest.update(
            env_steps=steps, updates=learner.updates, elapsed_seconds=elapsed()
        )
        atomic_json(output / "manifest.json", manifest)
        if writer:
            for key, value in row.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(f"train/{key}", value, steps)
        print(
            f"{cfg.algo.name} seed={cfg.run.seed} steps={steps} updates={learner.updates} "
            f"coverage={env_means['coverage']:.3f} robustness={env_means['robustness']:.3f} "
            f"elapsed={elapsed():.1f}s",
            flush=True,
        )

    with VectorEnvironment(cfg.env, cfg.run.num_envs, cfg.run.vector_backend) as envs:
        if restored:
            envs.restore(restored["environments"])
            obs, state = restored["obs"], restored["state"]
            infos = restored["infos"]
            restore_rng(restored["rng"])
        else:
            child_seeds = np.random.SeedSequence([cfg.run.seed, 101]).spawn(
                cfg.run.num_envs
            )
            obs, state, infos = envs.reset(
                [int(seed.generate_state(1)[0]) for seed in child_seeds]
            )

        def full_checkpoint():
            nonlocal last_full
            if cfg.algo.name == "mappo" and not mappo_checkpoint_safe:
                raise RuntimeError(
                    "MAPPO rollout/update failed before a safe checkpoint boundary; previous checkpoint retained"
                )
            stamp = time.perf_counter()
            atomic_save(output / "last.pt", light())
            manifest.update(
                env_steps=steps,
                updates=learner.updates,
                checkpoint_stage="saving",
                elapsed_seconds=elapsed(),
            )
            atomic_json(output / "manifest.json", manifest)
            data = light()
            data.update(
                kind="full",
                environments=envs.snapshot(),
                obs=obs.copy(),
                state=state.copy(),
                infos=infos,
                rng=rng_state(),
                exploration_rng=exploration.bit_generator.state,
                update_credit=credit,
                best_score=best,
                best_model=load_checkpoint(output / "best.pt")
                if math.isfinite(best)
                else None,
                best_selection=json.loads((output / "best_selection.json").read_text())
                if math.isfinite(best)
                else None,
                next_eval=next_eval,
                elapsed_seconds=elapsed(),
                phase_seconds=dict(phases),
            )
            path = output / f"resume_{steps:012d}"
            save_full(path, data, replay, cfg.run.memory_limit_gib * 2**30)
            files = sorted(
                p
                for p in output.glob("resume_*")
                if p.is_dir() and not p.name.endswith(".tmp")
            )
            for old in files[:-2]:
                shutil.rmtree(old)
            phases["io"] += time.perf_counter() - stamp
            last_full = time.perf_counter()
            manifest.update(
                checkpoint_stage="saved",
                resume_checkpoint=str(path),
                elapsed_seconds=elapsed(),
            )
            atomic_json(output / "manifest.json", manifest)
            return path

        try:
            if not restored and not budget_expired():
                validate()
            while steps < cfg.run.total_steps:
                if stop_requested[0]:
                    reason = "signal"
                    break
                if budget_expired():
                    reason = "budget_or_wall_limit"
                    break
                if cfg.algo.name == "mappo":
                    mappo_checkpoint_safe = False
                    rollout = defaultdict(list)
                    count = min(
                        cfg.algo.rollout_steps,
                        (cfg.run.total_steps - steps) // cfg.run.num_envs,
                    )
                    for _ in range(count):
                        stamp = time.perf_counter()
                        action, logp, latent, value = learner.collect(obs, state)
                        phases["inference"] += time.perf_counter() - stamp
                        stamp = time.perf_counter()
                        no, ns, reward, done, truncated, infos = envs.step(action)
                        phases["environment"] += time.perf_counter() - stamp
                        next_value = learner.values(ns).cpu().numpy()
                        for key, value_ in dict(
                            obs=obs,
                            state=state,
                            **(
                                {"actions": action}
                                if cfg.env.action_space == "discrete9"
                                else {"latent": latent}
                            ),
                            logp=logp,
                            value=value,
                            next_value=next_value,
                            reward=reward,
                            terminated=done,
                            truncated=truncated,
                        ).items():
                            rollout[key].append(value_)
                        steps += cfg.run.num_envs
                        obs, state = no, ns
                        for index in np.flatnonzero(done | truncated):
                            obs[index], state[index], _ = envs.reset_at(int(index))
                        if stop_requested[0] or budget_expired():
                            break
                    stamp = time.perf_counter()
                    latest_metrics = learner.update(
                        {
                            k: stack_observations(v) if k == "obs" else np.stack(v)
                            for k, v in rollout.items()
                        }
                    )
                    phases["update"] += time.perf_counter() - stamp
                    mappo_checkpoint_safe = True
                else:
                    stamp = time.perf_counter()
                    action = (
                        uniform_disk(exploration, (cfg.run.num_envs, cfg.env.num_uavs))
                        if steps < cfg.algo.warmup_steps
                        else learner.act(obs)
                    )
                    phases["inference"] += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    no, ns, reward, done, truncated, infos = envs.step(action)
                    phases["environment"] += time.perf_counter() - stamp
                    replay.add(obs, state, action, reward, no, ns, done)
                    previous = steps
                    steps += cfg.run.num_envs
                    credit += max(0, steps - max(previous, cfg.algo.warmup_steps))
                    if replay.size >= cfg.algo.batch_size:
                        while credit >= cfg.algo.transitions_per_update:
                            stamp = time.perf_counter()
                            batch = replay.sample(
                                cfg.algo.batch_size, torch.device("cpu")
                            )
                            phases["sample"] += time.perf_counter() - stamp
                            stamp = time.perf_counter()
                            if device.type == "cuda":
                                batch = {
                                    k: v.pin_memory().to(device, non_blocking=True)
                                    for k, v in batch.items()
                                }
                                torch.cuda.synchronize(device)
                            phases["transfer"] += time.perf_counter() - stamp
                            stamp = time.perf_counter()
                            latest_metrics = learner.update(batch)
                            phases["update"] += time.perf_counter() - stamp
                            credit -= cfg.algo.transitions_per_update
                    obs, state = no, ns
                    for index in np.flatnonzero(done | truncated):
                        obs[index], state[index], _ = envs.reset_at(int(index))
                if (
                    any(not np.isfinite(v) for v in latest_metrics.values())
                    or not np.isfinite(reward).all()
                ):
                    raise FloatingPointError(
                        "Nonfinite reward/learner metric; inspect anomaly checkpoint"
                    )
                if (
                    psutil.Process().memory_info().rss
                    > cfg.run.memory_limit_gib * 2**30
                ):
                    raise MemoryError(
                        "Configured resident-memory limit exceeded; no users or transitions were discarded"
                    )
                if steps >= next_log:
                    write_metrics(infos)
                    next_log = steps + cfg.run.log_interval
                if steps >= next_eval and not budget_expired():
                    validate()
                    next_eval = (
                        (steps // cfg.run.eval_interval) + 1
                    ) * cfg.run.eval_interval
                if time.perf_counter() - last_full >= cfg.run.checkpoint_minutes * 60:
                    full_checkpoint()
            if (
                reason == "completed"
                and steps > 0
                and steps % cfg.run.eval_interval
                and not budget_expired()
            ):
                validate()
            write_metrics(infos)
            resume_path = full_checkpoint()
            manifest.update(
                status="completed" if reason == "completed" else "stopped",
                stop_reason=reason,
                env_steps=steps,
                updates=learner.updates,
                elapsed_seconds=elapsed(),
                phase_seconds=dict(phases),
                resume_checkpoint=str(resume_path),
                best_validation_return=best if math.isfinite(best) else None,
                **memory_stats(device),
            )
            atomic_json(output / "manifest.json", manifest)
        except BaseException as error:
            manifest.update(
                status="failed",
                env_steps=steps,
                error=repr(error),
                elapsed_seconds=elapsed(),
            )
            atomic_json(output / "manifest.json", manifest)
            try:
                full_checkpoint()
                if envs.envs:
                    frame = envs.envs[0].trace_frame()
                    np.savez_compressed(output / "anomaly.npz", **frame)
            except Exception as save_error:
                manifest["checkpoint_error"] = repr(save_error)
                atomic_json(output / "manifest.json", manifest)
            raise
        finally:
            logger.close()
            eval_logger.close()
            if writer:
                writer.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return manifest
