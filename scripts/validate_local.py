"""Reproduce local integrity acceptance. Never connects to servers or trains >4096 steps."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uav_research.runtime import source_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("analysis/beacon_v2_20261013"))
    p.add_argument("--skip-memory", action="store_true")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    jobs = []

    def run(name, command):
        stamp = time.perf_counter()
        print(f"Local validation: {name}", flush=True)
        with (logs / (name + ".log")).open("w") as f:
            status = subprocess.run(
                [sys.executable, *command], stdout=f, stderr=subprocess.STDOUT
            )
        job = dict(
            name=name,
            command=command,
            seconds=time.perf_counter() - stamp,
            exit_code=status.returncode,
        )
        jobs.append(job)
        (root / "validation_jobs.json").write_text(json.dumps(jobs, indent=2))
        if status.returncode:
            print((logs / (name + ".log")).read_text()[-5000:])
            raise SystemExit(status.returncode)

    def cli(name, *argv):
        run(name, ["-m", "uav_research", *map(str, argv)])

    run("tests", ["-m", "pytest", "-q", "--disable-warnings"])
    cli(
        "check_env",
        "check-env",
        "--config",
        "configs/main.yaml",
        "--output",
        root / "check_env.json",
    )
    if not args.skip_memory:
        run(
            "memory",
            [
                "scripts/check_full_replay_checkpoint.py",
                "--output",
                str(root / "memory"),
            ],
        )
    cli("calibration", "calibrate", "--output", root / "calibration")
    for algorithm in ["sac", "mappo", "matd3"]:
        smoke = root / "smoke" / algorithm
        main_run = root / "main" / algorithm
        cli(
            "smoke_" + algorithm,
            "train",
            "--config",
            "configs/smoke.yaml",
            "--algo",
            algorithm,
            "--output",
            smoke,
        )
        cli(
            "resume_" + algorithm,
            "train",
            "--config",
            "configs/smoke.yaml",
            "--algo",
            algorithm,
            "--steps",
            288,
            "--resume",
            smoke / "resume_000000000256",
            "--output",
            smoke,
        )
        cli(
            "smoke_eval_" + algorithm,
            "evaluate",
            "--checkpoint",
            smoke / "last.pt",
            "--output",
            smoke / "evaluation",
        )
        cli(
            "export_" + algorithm,
            "export-actor",
            "--checkpoint",
            smoke / "last.pt",
            "--output",
            smoke / "actor.pt",
        )
        cli(
            "smoke_plot_" + algorithm,
            "plot",
            "--runs",
            smoke,
            "--evaluations",
            smoke / "evaluation",
            "--traces",
            smoke / "evaluation/trace_10000.npz",
            "--output",
            smoke / "figures",
        )
        cli(
            "main_" + algorithm,
            "train",
            "--config",
            "configs/local_validation.yaml",
            "--algo",
            algorithm,
            "--steps",
            4096,
            "--output",
            main_run,
        )
        for checkpoint in ["best", "last"]:
            cli(
                f"eval_{algorithm}_{checkpoint}",
                "evaluate",
                "--checkpoint",
                main_run / (checkpoint + ".pt"),
                "--traces",
                1,
                "--output",
                main_run / ("evaluation_" + checkpoint),
            )
    for method in ["static", "random", "heuristic"]:
        cli(
            "baseline_" + method,
            "evaluate",
            "--config",
            "configs/local_validation.yaml",
            "--baseline",
            method,
            "--traces",
            1,
            "--output",
            root / "baselines" / method,
        )
    cli(
        "calibration_plot",
        "plot",
        "--calibration",
        root / "calibration",
        "--output",
        root / "figures/calibration",
    )
    for method in ["static", "random", "scripted_expansion", "expanded_static"]:
        cli(
            "topology_" + method,
            "plot",
            "--traces",
            root / "calibration" / f"trace_{method}_30000.npz",
            "--output",
            root / "figures/calibration" / method,
        )
    main_runs = [root / "main" / a for a in ["sac", "mappo", "matd3"]]
    evaluations = [r / "evaluation_last" for r in main_runs] + [
        root / "baselines" / b for b in ["static", "random", "heuristic"]
    ]
    cli(
        "main_plot",
        "plot",
        "--runs",
        *main_runs,
        "--evaluations",
        *evaluations,
        "--traces",
        main_runs[0] / "evaluation_last/trace_10000.npz",
        "--output",
        root / "figures/short_training",
    )
    cli(
        "benchmark",
        "benchmark",
        "--config",
        "configs/local_validation.yaml",
        "--counts",
        1,
        2,
        4,
        8,
        "--backends",
        "batch",
        "--devices",
        "cpu",
        "--steps",
        256,
        "--output",
        root / "benchmark.json",
    )
    (root / "acceptance.json").write_text(
        json.dumps(
            dict(
                passed=True,
                source_hash=source_digest(),
                tests_and_jobs=len(jobs),
                formal_training=False,
                cuda_verified=False,
                held_out_test_used=False,
            ),
            indent=2,
        )
    )
    print(f"Local acceptance completed: {root}", flush=True)


if __name__ == "__main__":
    main()
