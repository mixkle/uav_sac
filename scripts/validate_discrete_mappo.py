"""Local-only discrete MAPPO acceptance and CPU measurements; never starts formal training."""

from pathlib import Path
import copy
import torch
from uav_research.config import load_config
from uav_research.runtime import atomic_json, configure_runtime
from uav_research.train import train
from uav_research.evaluate import evaluate, policy_from_checkpoint, Baseline
from uav_research.deployment import export_actor
from uav_research.plotting import curves, trace_plot
from uav_research.experiments import benchmark, check_environment


def main():
    root = Path("analysis/discrete_mappo_20261013")
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    check_environment(cfg, root / "environment_check.json", scenes=8, steps=32)
    first = train(cfg, root / "smoke")
    cfg.run.total_steps = 288
    train(cfg, root / "smoke", first["resume_checkpoint"])
    ecfg, policy = policy_from_checkpoint(
        root / "smoke" / "best.pt", torch.device("cpu")
    )
    for name, candidate in [("mappo", policy)] + [
        (n, Baseline(ecfg, n)) for n in ("static", "random", "heuristic")
    ]:
        evaluate(
            ecfg,
            candidate,
            [10000, 10001],
            root / "evaluation" / name,
            method=name,
            trace_count=1,
        )
    export_actor(root / "smoke" / "best.pt", root / "actor.pt")
    curves([root / "smoke"], root / "figures")
    trace_plot(root / "evaluation" / "mappo" / "trace_10000.npz", root / "figures")
    cfg = load_config("configs/discrete_mappo_wsl_50k.yaml")
    cfg.run.total_steps = 4096
    cfg.run.validation_seeds = [10000, 10001]
    cfg.run.eval_interval = 2048
    cfg.run.log_interval = 2048
    cfg.run.tensorboard = False
    atomic_json(root / "local_overrides.json", cfg.as_dict())
    configure_runtime(cfg)
    train(cfg, root / "main_4096")
    for threads in (1, 4, 8):
        probe = copy.deepcopy(cfg)
        probe.run.cpu_threads = threads
        benchmark(
            probe,
            root / f"benchmark_threads_{threads}.json",
            counts=(16,),
            backends=("batch",),
            steps=2048,
        )


if __name__ == "__main__":
    main()
