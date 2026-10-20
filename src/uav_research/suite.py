"""Sequential experiment stages with an evidence-based multiseed gate."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
from .config import AlgoConfig
from .evaluate import Baseline, evaluate, policy_from_checkpoint, check_gate
from .runtime import atomic_json, configure_runtime, resolve_output
from .train import train


def environment_hash(cfg):
    data = cfg.as_dict()["env"]
    data.pop("backend")
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def run_suite(cfg, output, stage="pilot", gate=None, include_matd3=False):
    root = resolve_output(output)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frozen = None
    if stage == "multiseed":
        if gate is None:
            raise ValueError("multiseed requires --gate passed gate.json")
        report = json.loads(Path(gate).read_text())
        if not report.get("passed"):
            raise ValueError("Single-seed gate failed; diagnose before multiseed")
        if report.get("environment_hash") != environment_hash(cfg):
            raise ValueError("Environment differs from passed gate")
        frozen = report.get("frozen_algorithms")
        if frozen is None:
            raise ValueError(
                "Gate lacks frozen configurations; regenerate with gate command"
            )
        if cfg.run.total_steps < report["training_steps"]:
            raise ValueError(
                "Multiseed training budget is smaller than the passed pilot"
            )
        seeds = range(5)
    else:
        seeds = [0]
    jobs = []
    for seed in seeds:
        for name in ("sac", "mappo", "sac_no_robustness"):
            c = copy.deepcopy(cfg)
            c.run.seed = seed
            if frozen is not None:
                c.algo = AlgoConfig(**frozen[name])
            else:
                c.algo.name = "sac" if name == "sac_no_robustness" else name
            if name == "sac_no_robustness":
                c.env.robustness_weight = 0.0
            jobs.append((name, seed, c))
        if include_matd3:
            c = copy.deepcopy(cfg)
            c.run.seed = seed
            c.algo.name = "matd3"
            jobs.append(("matd3", seed, c))
    manifest = dict(stage=stage, status="running", jobs=[])
    atomic_json(root / "suite.json", manifest)
    evaluation_paths = {}
    for name, seed, c in jobs:
        if cfg.run.max_wall_seconds is not None:
            remaining_wall = cfg.run.max_wall_seconds - (time.perf_counter() - started)
            if remaining_wall <= 0:
                manifest.update(status="wall_stopped")
                break
            c.run.max_wall_seconds = remaining_wall
        if cfg.run.budget is not None:
            remaining = (
                cfg.run.budget
                - (time.perf_counter() - started) * cfg.run.hourly_price / 3600
            )
            if remaining <= 0:
                manifest.update(status="budget_stopped")
                break
            c.run.budget = remaining
        directory = root / name / f"seed{seed}"
        status = train(c, directory)
        manifest["jobs"].append(
            dict(method=name, seed=seed, run=str(directory), status=status["status"])
        )
        atomic_json(root / "suite.json", manifest)
        if status["status"] != "completed":
            manifest["status"] = "stopped"
            break
        device = configure_runtime(c)
        ecfg, policy = policy_from_checkpoint(directory / "best.pt", device)
        evaluation = directory / ("test" if stage == "multiseed" else "validation")
        evaluate(
            ecfg,
            policy,
            ecfg.run.test_seeds if stage == "multiseed" else ecfg.run.validation_seeds,
            evaluation,
            trace_count=2,
        )
        evaluation_paths[(name, seed)] = evaluation
    else:
        for name in ("static", "random", "heuristic"):
            c = copy.deepcopy(cfg)
            c.run.seed = 0
            evaluate(
                c,
                Baseline(c, name),
                c.run.test_seeds if stage == "multiseed" else c.run.validation_seeds,
                root / "baselines" / name,
                method=name,
                trace_count=1,
            )
        if stage != "multiseed":
            result = check_gate(
                evaluation_paths[("sac", 0)],
                evaluation_paths[("mappo", 0)],
                evaluation_paths[("sac_no_robustness", 0)],
                root / "gate.json",
            )
            manifest["gate_passed"] = result["passed"]
        manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(root / "suite.json", manifest)
    return manifest
