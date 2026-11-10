"""Explicit two-stage commands. None deploy or access a server."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..config import load_config
from ..runtime import seed_all, resolve_output, atomic_json, memory_stats
from .storage import configure as configure_runtime
from .config import DppoConfig


def benchmark(cfg, output, vector_steps=8, microbatches=(32, 128), pretrained=None):
    from .environment import ServiceVector
    from .learner import DiffusionLearner, DppoRollout
    from .storage import Progress, ensure_budget
    output = Path(output)
    ensure_budget(cfg, output)
    output.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(cfg)
    seed_all(cfg.run.seed)
    learner = DiffusionLearner(cfg, device)
    if pretrained:
        from .storage import load
        from .config import from_dict
        source=load(pretrained,kinds={"dppo_pretrain"})
        if from_dict(source["config"]).env != cfg.env or cfg.algo.name != "diffusion_dppo":
            raise ValueError("Benchmark pretraining physical/observation config mismatch")
        learner.actor.load_state_dict(source["actor"])
    vector = ServiceVector(cfg.env, cfg.run.num_envs, cfg.run.vector_backend)
    progress = Progress(cfg.diffusion.heartbeat_seconds)
    try:
        obs, state, _ = vector.reset(list(range(990000, 990000+cfg.run.num_envs)))
        # Warm the physical kernels and policy outside measured intervals.
        snap = vector.snapshot()
        action, _ = learner.actor.sample(obs)
        vector.step(action)
        vector.restore(snap)
        initial = copy.deepcopy(learner.state_dict())
        rollout = DppoRollout()
        infer, physical = 0., 0.
        for i in range(vector_steps):
            t = time.perf_counter()
            action, chain = learner.actor.sample(obs)
            value = learner.values(state).cpu().numpy()
            infer += time.perf_counter()-t
            t = time.perf_counter()
            no, ns, r, done, trunc, _ = vector.step(action)
            physical += time.perf_counter()-t
            nv = learner.values(ns).cpu().numpy()
            rollout.append(obs, chain, state=state, reward=r, value=value,
                           next_value=nv, terminated=done, truncated=trunc)
            obs, state = no, ns
            if done.any():
                raise ValueError("Benchmark vector_steps must be shorter than a complete episode")
        results = []
        for micro in microbatches:
            cfg.diffusion.microbatch_agents = micro
            learner.load_state_dict(initial)
            seed_all(cfg.run.seed+10)
            start = time.perf_counter()
            learner.update(rollout, lambda e, s: progress(f"benchmark microbatch={micro} epoch={e} updates={s}"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update = time.perf_counter()-start
            results.append(dict(microbatch_agents=micro, update_seconds=update,
                estimated_online_training_seconds_excluding_validation=(infer+physical)*cfg.run.total_steps/(vector_steps*cfg.run.num_envs)
                    +update*(cfg.algo.rollout_steps/vector_steps)*int(np.ceil(cfg.run.total_steps/(cfg.run.num_envs*cfg.algo.rollout_steps))),
                **memory_stats(device)))
        result = dict(device=str(device), vector_steps=vector_steps, environments=cfg.run.num_envs,
            joint_transitions=vector_steps*cfg.run.num_envs, inference_seconds=infer,
            environment_seconds=physical, results=results,
            note="Rough linear estimate; short minibatches and excluded validation/IO limit accuracy",
            actor_parameters=sum(p.numel() for p in learner.actor.parameters()))
        atomic_json(output/"benchmark.json", result)
        return result
    finally:
        vector.close()


def main(argv):
    parser = argparse.ArgumentParser(description="Set-conditioned DDPM + denoising-transition PPO")
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("collect", "pretrain", "train", "all", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml")
        p.add_argument("--output", required=True)
        p.add_argument("--device")
        if name in ("pretrain", "train"):
            p.add_argument("--resume")
        if name == "pretrain":
            p.add_argument("--demonstrations", required=True)
        if name == "train":
            p.add_argument("--pretrained")
            p.add_argument("--steps", type=int)
        if name == "benchmark":
            p.add_argument("--pretrained")
            p.add_argument("--vector-steps", type=int, default=8)
            p.add_argument("--microbatches", type=int, nargs="+", default=[32, 128])
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--moment-repeats", type=int, default=64)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--moment-repeats", type=int, default=64)
    p = sub.add_parser("export-actor")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.output = str(resolve_output(args.output))
    if args.phase in ("collect", "pretrain", "train", "all", "benchmark"):
        cfg = load_config(args.config)
        if not isinstance(cfg, DppoConfig):
            parser.error("Use a diffusion_dppo config")
        if args.device:
            cfg.run.device = args.device
        if args.phase == "train" and args.steps is not None:
            cfg.run.total_steps = args.steps
        cfg.validate()
    if args.phase in ("collect", "pretrain", "all") and cfg.algo.name != "diffusion_dppo":
        parser.error("Gaussian service MAPPO trains from scratch; use train")
    if args.phase == "collect":
        from .data import collect_demonstrations
        result = collect_demonstrations(cfg, args.output)
    elif args.phase == "pretrain":
        from .data import pretrain
        result = pretrain(cfg, args.demonstrations, args.output, args.resume)
    elif args.phase == "train":
        from .runner import train_online
        result = train_online(cfg, args.output, args.resume, args.pretrained)
    elif args.phase == "all":
        from .data import collect_demonstrations, pretrain
        from .runner import train_online
        root = Path(args.output)
        collect_demonstrations(cfg, root/"demonstrations")
        pretrain(cfg, root/"demonstrations", root/"pretrain")
        result = train_online(cfg, root, pretrained=root/"pretrain"/"best.pt")
    elif args.phase == "evaluate":
        from .report import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, args.output, args.device, moment_repeats=args.moment_repeats)["summary"]
    elif args.phase == "report":
        from .report import report
        from .storage import load
        from .config import from_dict
        report_cfg = None if args.rebuild else from_dict(load(Path(args.run)/"last.pt")["config"])
        result = report(args.run, args.output, args.device, args.rebuild, args.moment_repeats,
            include_baseline=(report_cfg is None or (report_cfg.algo.name == "diffusion_dppo" and not report_cfg.env.motion_scaling["enabled"])))
    elif args.phase == "export-actor":
        from .runner import export_actor
        result = export_actor(args.checkpoint, args.output)
    else:
        result = benchmark(cfg, args.output, args.vector_steps, args.microbatches, args.pretrained)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
