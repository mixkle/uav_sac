"""CPU tuning on a real saved batch. Only temporary learners are updated."""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.runtime import (
    configure_runtime,
    seed_all,
    load_checkpoint,
    atomic_json,
)
from uav_research.observations import TensorObservation
from uav_research.algorithms import make_learner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("runs/wsl_sac_50k_seed0"))
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.run / "config.yaml")
    saved = load_checkpoint(args.run / "last.pt")["learner"]
    # A mid-training block with nontrivial detected-user lists, not an empty launch batch.
    data = torch.load(
        args.run / "resume_000000050000/replay_025600.pt",
        weights_only=False,
        map_location="cpu",
    )
    batch = {
        k: v
        for k, v in data.items()
        if k in ("state", "next_state", "actions", "reward", "terminated")
    }
    for key, label in [("obs", "current"), ("next_obs", "next")]:
        batch[key] = TensorObservation(
            data[key], data[label + "_users"], torch.diff(data[label + "_offsets"])
        )
    results = []
    for threads in args.threads:
        for foreach in [False, True]:
            cfg.run.device = "cpu"
            cfg.run.cpu_threads = threads
            device = configure_runtime(cfg)
            seed_all(710)
            learner = make_learner(cfg, device)
            learner.load_state_dict(copy.deepcopy(saved))
            for name in ["actor_opt", "critic_opt", "alpha_opt"]:
                for group in getattr(learner, name).param_groups:
                    group["foreach"] = foreach
            for _ in range(5):
                learner.update(batch)
            durations = []
            for _ in range(args.updates):
                stamp = time.perf_counter()
                metrics = learner.update(batch)
                durations.append(time.perf_counter() - stamp)
            assert all(np.isfinite(v) for v in metrics.values())
            row = dict(
                cpu_threads=threads,
                foreach=foreach,
                mean_update_ms=float(np.mean(durations) * 1000),
                median_update_ms=float(np.median(durations) * 1000),
                updates=args.updates,
            )
            results.append(row)
            print(row, flush=True)
            del learner
    report = dict(
        batch_size=len(batch["reward"]),
        ue_rows=len(batch["obs"].users),
        next_ue_rows=len(batch["next_obs"].users),
        source_run=str(args.run),
        results=results,
        recommendation=min(results, key=lambda r: r["mean_update_ms"]),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        new_training_run=False,
    )
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
