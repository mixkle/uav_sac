"""Manual online training; safe-point checkpoints and isolated validation RNG."""
import copy
from pathlib import Path
import signal
import time
import numpy as np

from ..config import save_config
from ..runtime import (seed_all, rng_state, metadata,
                       atomic_json, atomic_save, CSVLogger, memory_stats)
from ..topk_diagnostic import file_hash
from .environment import ServiceVector
from .observations import stack, chain_diagnostics
from .learner import DiffusionLearner, DppoRollout
from .storage import (artifact, load, ensure_budget, save_online_full, Progress,
                      configure as configure_runtime, restore_online_rng as restore_rng)
from .evaluation import evaluate


def train_online(cfg, output, resume=None, pretrained=None, stop_after_updates=None):
    cfg.validate()
    output = Path(output)
    ensure_budget(cfg, output)
    if not resume and any((output/name).exists() for name in ("manifest.json", "last.pt", "best.pt", "step0.pt")):
        raise FileExistsError("Online run exists; use full checkpoint --resume or a new output")
    output.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(cfg)
    seed_all(cfg.run.seed)
    progress = Progress(cfg.diffusion.heartbeat_seconds)
    progress(f"DPPO initialize device={device} environments={cfg.run.num_envs} algorithm={cfg.algo.name} policy_steps={cfg.diffusion.steps if cfg.algo.name == 'diffusion_dppo' else 1}", True)
    learner = DiffusionLearner(cfg, device)
    progress(f"actor PPO devices={[str(d) for d in learner.actor_update_devices] or [str(device)]}; "
             f"microbatch_agents={cfg.diffusion.microbatch_agents}; critic and sampling={device}", True)
    steps, best, best_step, next_eval, elapsed_before = 0, -float("inf"), 0, cfg.run.eval_interval, 0.
    provenance = {}
    restored = None
    if resume:
        restored = load(resume, cfg, {"dppo_full"})
        if restored["num_envs"] != cfg.run.num_envs:
            raise ValueError("Cannot change environment count on resume")
        for log_name in ("metrics.csv", "validation.csv"):
            path = output/log_name
            if path.exists():
                import csv
                if any(int(row["env_steps"]) > restored["env_steps"] for row in csv.DictReader(path.open())):
                    raise ValueError("Output contains future steps; resume into a new output directory")
        learner.load_state_dict(restored["learner"])
        steps, best, best_step, next_eval = [restored[k] for k in ("env_steps", "best_score", "best_step", "next_eval")]
        elapsed_before = restored["elapsed_seconds"]
        provenance = restored["provenance"]
        if steps >= cfg.run.total_steps:
            raise ValueError("Resume checkpoint already reaches requested total budget")
        if not (output/"best.pt").exists():
            atomic_save(output/"best.pt", restored["best_artifact"])
    elif cfg.algo.name == "diffusion_dppo":
        source = Path(pretrained or cfg.diffusion.pretrained_checkpoint)
        initial = load(source, cfg, {"dppo_pretrain"})
        learner.actor.load_state_dict(initial["actor"])
        provenance = dict(pretrained_path=str(source), pretrained_sha256=file_hash(source),
                          pretraining_epoch=initial["epoch"], pretraining_provenance=initial["provenance"])
    save_config(cfg, output/"config.yaml")
    atomic_json(output/"metadata.json", dict(metadata(cfg, device), provenance=provenance,
        actor_update_devices=[str(d) for d in learner.actor_update_devices] or [str(device)],
        actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
        critic_parameters=sum(p.numel() for p in learner.value.parameters()),
        probability=("coordinate sum; per UAV, per DDPM step ratio; mean over samples/UAV/steps"
            if cfg.algo.name == "diffusion_dppo" else "coordinate sum; one joint Gaussian ratio per UAV"),
        resource_allocation="service slots and association only", reset="baseline_greedy_then_topk"))
    vector = ServiceVector(cfg.env, cfg.run.num_envs, cfg.run.vector_backend)
    worker_seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg.run.seed, 101]).spawn(cfg.run.num_envs)]
    if restored:
        vector.restore(restored["environments"])
        obs = restored["observation"].copy()
        state = restored["state"].copy()
        restore_rng(restored["rng"])
    else:
        obs, state, _ = vector.reset(worker_seeds)
    start = time.perf_counter()
    phase_times = copy.deepcopy(restored.get("phase_times", {})) if restored else {}
    phase_times = dict(evaluation_seconds=phase_times.get("evaluation_seconds", 0.),
                       io_seconds=phase_times.get("io_seconds", 0.))
    last_save = time.monotonic()
    stop = [False]

    def request_stop(signum, _):
        stop[0] = True
        progress(f"signal={signum}: finish current PPO rollout/update, then save a complete checkpoint", True)

    previous_handlers = {}
    if __import__("threading").current_thread() is __import__("threading").main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, request_stop)

    def elapsed():
        return elapsed_before+time.perf_counter()-start

    def model():
        return artifact(cfg, "model", env_steps=steps, learner=learner.state_dict(), provenance=provenance)

    def light():
        stamp = time.perf_counter()
        atomic_save(output/"last.pt", model())
        phase_times["io_seconds"] += time.perf_counter()-stamp

    def full():
        stamp = time.perf_counter()
        snapshot = artifact(cfg, "full", env_steps=steps, learner=learner.state_dict(),
            environments=vector.snapshot(), observation=obs.copy(), state=state.copy(),
            rng=rng_state(), best_score=best, best_step=best_step, next_eval=next_eval,
            num_envs=cfg.run.num_envs, elapsed_seconds=elapsed(), provenance=provenance,
            phase_times=copy.deepcopy(phase_times),
            best_artifact=load(output/"best.pt", kinds={"dppo_model"}))
        path = save_online_full(output, cfg, snapshot)
        phase_times["io_seconds"] += time.perf_counter()-stamp
        return path

    validation = CSVLogger(output/"validation.csv", ["env_steps", "elapsed_seconds", "return", "task_return",
        "coverage", "throughput_bps", "last100_coverage", "final_coverage", "best_score", "best_step"])
    logger = None
    if cfg.run.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output/"tensorboard")
    else:
        writer = None

    def validate():
        nonlocal best, best_step
        light()  # Preserve current weights before expensive validation.
        progress(f"validation begin env_steps={steps}", True)
        stamp = time.perf_counter()
        result = evaluate(cfg, learner.actor, cfg.run.validation_seeds, progress=progress)
        phase_times["evaluation_seconds"] += time.perf_counter()-stamp
        summary = result["summary"]
        score = summary["return"]  # No boundary costs: task return == train return.
        if score > best:
            best, best_step = score, steps
            atomic_save(output/"best.pt", model())
        validation.write(dict(env_steps=steps, elapsed_seconds=elapsed(),
            **{k: summary[k] for k in ("return", "coverage", "throughput_bps", "last100_coverage", "final_coverage")},
            task_return=score, best_score=best, best_step=best_step))
        progress(f"validation env_steps={steps} return={score:.6f} coverage={summary['coverage']:.4%} best_step={best_step}", True)

    try:
        if not restored:
            atomic_save(output/"step0.pt", model())
            validate()
            full()
        while steps < cfg.run.total_steps and not stop[0]:
            remaining_vectors = (cfg.run.total_steps-steps)//cfg.run.num_envs
            horizon = min(cfg.algo.rollout_steps, remaining_vectors)
            rollout = DppoRollout()
            infos_history = []
            inference, environment, encoding, denoising, packing_transfer = 0., 0., 0., 0., 0.
            progress(f"rollout begin env_steps={steps} vector_steps={horizon}", True)
            for t in range(horizon):
                stamp = time.perf_counter()
                action, chain = learner.actor.sample(obs)
                timing = learner.actor.last_sample_timing
                encoding += timing["encoding_seconds"]
                denoising += timing["denoising_seconds"]
                packing_transfer += timing["packing_transfer_seconds"]
                value = learner.values(state).cpu().numpy()
                inference += time.perf_counter()-stamp
                stamp = time.perf_counter()
                new_obs, next_state, reward, done, truncated, infos = vector.step(action)
                for info, diagnostics in zip(infos, chain_diagnostics(chain, cfg.env.num_uavs)):
                    info.update(diagnostics)
                environment += time.perf_counter()-stamp
                next_value = learner.values(next_state).cpu().numpy()
                rollout.append(obs, chain, state=state, reward=reward, value=value,
                    next_value=next_value, terminated=done, truncated=truncated)
                infos_history.extend(infos)
                # next_value was computed from terminal observations before any reset.
                if (done | truncated).any():
                    parts = []
                    u = cfg.env.num_uavs
                    for i in range(cfg.run.num_envs):
                        if done[i] or truncated[i]:
                            item, next_state[i], _ = vector.reset_at(i)
                        else:
                            item = new_obs.take(np.arange(i*u, (i+1)*u))
                        parts.append(item)
                    new_obs = stack(parts)
                obs, state = new_obs, next_state
                steps += cfg.run.num_envs
                progress(f"collect env_steps={steps}/{cfg.run.total_steps} vector_step={t+1}/{horizon}")
            stamp = time.perf_counter()
            metrics = learner.update(rollout, lambda epoch, count:
                progress(f"PPO env_steps={steps} epoch={epoch}/{cfg.algo.ppo_epochs} optimizer_steps={count}"))
            update = time.perf_counter()-stamp
            ensure_budget(cfg, output)
            scalar_keys = sorted(k for k, v in infos_history[0].items() if np.isscalar(v))
            row = {k: float(np.mean([info[k] for info in infos_history])) for k in scalar_keys}
            row.update(metrics, env_steps=steps, updates=learner.updates,
                optimizer_steps=learner.optimizer_steps, elapsed_seconds=elapsed(),
                inference_seconds=inference, environment_seconds=environment, update_seconds=update,
                encoding_seconds=encoding, denoising_seconds=denoising, packing_transfer_seconds=packing_transfer,
                **memory_stats(device))
            if logger is None:
                logger = CSVLogger(output/"metrics.csv", list(row))
            logger.write(row)
            if writer:
                for k, v in row.items():
                    writer.add_scalar(k, v, steps)
                writer.flush()
            progress(f"update complete env_steps={steps} reward={row['reward']:.6f} coverage={row['coverage']:.4%} PPO={update:.2f}s", True)
            light()
            if steps >= next_eval or steps == cfg.run.total_steps:
                while next_eval <= steps:
                    next_eval += cfg.run.eval_interval
                validate()
            if time.monotonic()-last_save >= cfg.run.checkpoint_minutes*60:
                full()
                last_save = time.monotonic()
            if cfg.run.max_wall_seconds and elapsed()-elapsed_before >= cfg.run.max_wall_seconds:
                stop[0] = True
            if stop_after_updates is not None and learner.updates >= stop_after_updates:
                stop[0] = True
            atomic_json(output/"manifest.json", dict(env_steps=steps, updates=learner.updates,
                best_score=best, best_step=best_step, elapsed_seconds=elapsed(), status="running", **phase_times))
        light()
        checkpoint = full()
        result = dict(env_steps=steps, updates=learner.updates, optimizer_steps=learner.optimizer_steps,
            best_score=best, best_step=best_step, elapsed_seconds=elapsed(), full_checkpoint=str(checkpoint),
            status="complete" if steps == cfg.run.total_steps else "interrupted", **phase_times, **memory_stats(device))
        atomic_json(output/"manifest.json", result)
        return result
    except BaseException as exc:
        atomic_json(output/"failure.json", dict(error=repr(exc), env_steps=steps,
            updates=learner.updates, note="Use last complete checkpoint; partial PPO updates are not resumable"))
        try:
            ensure_budget(cfg, output)
            atomic_save(output/"failure_context.pt", artifact(cfg, "failure", env_steps=steps,
                learner=learner.state_dict(), rollout=rollout if "rollout" in locals() else None,
                observation=obs, state=state, rng=rng_state()))
        except (OSError, RuntimeError, MemoryError):
            pass
        raise
    finally:
        vector.close()
        validation.close()
        if logger:
            logger.close()
        if writer:
            writer.close()
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def export_actor(checkpoint, output):
    state = load(checkpoint, kinds={"dppo_model", "dppo_full", "dppo_pretrain"})
    from .config import from_dict
    cfg = from_dict(state["config"])
    actor = state["actor"] if state["kind"] == "dppo_pretrain" else state["learner"]["actor"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(output, artifact(cfg, "actor", actor=actor,
        env_steps=state.get("env_steps", 0), source_sha256=file_hash(checkpoint if Path(checkpoint).is_file() else Path(checkpoint)/"metadata.pt")))
    return dict(output=str(output))


class DeployedServiceActor:
    def __init__(self, path, device="cpu"):
        from .config import from_dict
        from .policy import make_service_actor
        data = load(path, kinds={"dppo_actor"})
        self.cfg = from_dict(data["config"])
        self.actor = make_service_actor(self.cfg).to(device)
        self.actor.load_state_dict(data["actor"])
        self.actor.eval()

    def act(self, obs, deterministic=False):
        return self.actor.sample(obs, deterministic=deterministic)[0]
