"""Run a short experiment with process-tree CPU/RAM and GPU telemetry."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--metrics", type=Path, help="Training CSV to tag warmup/update phases"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "command.json").write_text(json.dumps(command))
    fields = [
        "elapsed_seconds",
        "epoch_seconds",
        "cpu_percent_one_core",
        "processes",
        "rss_tree_mib",
        "gpu_util_percent",
        "gpu_memory_mib",
        "gpu_power_w",
        "cgroup_memory_mib",
        "training_env_steps",
        "training_updates",
    ]
    started = time.monotonic()
    samples = []
    with (
        (args.output / "stdout.log").open("w") as log,
        (args.output / "resources.csv").open("w", newline="") as resource,
    ):
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        root = psutil.Process(process.pid)
        writer = csv.DictWriter(resource, fieldnames=fields)
        writer.writeheader()
        previous = {}
        previous_time = started
        try:
            while process.poll() is None:
                now = time.monotonic()
                total_cpu, rss, count = 0.0, 0, 0
                try:
                    children = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    children = []
                current = {}
                for child in children:
                    try:
                        times = child.cpu_times()
                        value = times.user + times.system
                        current[child.pid] = value
                        total_cpu += max(0.0, value - previous.get(child.pid, value))
                        rss += child.memory_info().rss
                        count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                row = dict(
                    elapsed_seconds=now - started,
                    epoch_seconds=time.time(),
                    cpu_percent_one_core=100
                    * total_cpu
                    / max(now - previous_time, 1e-9),
                    processes=count,
                    rss_tree_mib=rss / 2**20,
                )
                previous, previous_time = current, now
                try:
                    gpu = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used,power.draw",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=5,
                        )
                        .splitlines()[0]
                        .split(",")
                    )
                    row.update(
                        zip(
                            ("gpu_util_percent", "gpu_memory_mib", "gpu_power_w"),
                            map(float, gpu),
                        )
                    )
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                memory = Path("/sys/fs/cgroup/memory.current")
                if memory.exists():
                    row["cgroup_memory_mib"] = int(memory.read_text()) / 2**20
                if args.metrics is not None and args.metrics.exists():
                    with args.metrics.open() as file:
                        records = list(csv.DictReader(file))
                    if records:
                        for field in ("env_steps", "updates"):
                            row[f"training_{field}"] = int(records[-1][field])
                writer.writerow(row)
                resource.flush()
                samples.append(row)
                time.sleep(args.interval)
        except BaseException:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
            raise
    summary = dict(
        exit_code=process.returncode,
        wall_seconds=time.monotonic() - started,
        samples=len(samples),
    )
    for key in fields[2:]:
        values = [r[key] for r in samples if key in r]
        if values:
            summary[key] = dict(mean=sum(values) / len(values), maximum=max(values))
    learning = [r for r in samples if r.get("training_updates", 0) > 0]
    summary["learning_samples"] = len(learning)
    summary["learning_resources"] = {}
    for key in ("cpu_percent_one_core", "gpu_util_percent", "gpu_memory_mib"):
        values = [r[key] for r in learning if key in r]
        if values:
            summary["learning_resources"][key] = dict(
                mean=sum(values) / len(values), maximum=max(values)
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
