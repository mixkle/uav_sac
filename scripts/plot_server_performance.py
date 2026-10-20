"""Rebuild server throughput/resource figures from saved measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-quota", type=float, default=10.0)
    args = parser.parse_args()
    output = args.data / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    runs = [
        json.loads((args.data / "benchmarks" / name).read_text())["results"]
        for name in ("new_host_before.json", "new_host_after.json")
    ]
    labels = [f"{r['device'].upper()} / {r['num_envs']} envs" for r in runs[0]]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(len(labels))
    for offset, rows, label, color in zip(
        (-0.19, 0.19), runs, ("Before", "Optimized"), ("#8797A7", "#247BA0")
    ):
        values = [r["transitions_per_second"] for r in rows]
        bars = ax.bar(x + offset, values, 0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Joint transitions / second")
    ax.set_ylim(0, max(r["transitions_per_second"] for r in runs[1]) * 1.22)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    save(fig, "throughput")
    with (args.data / "profiles/optimized_seed0/resources.csv").open() as file:
        rows = list(csv.DictReader(file))
    t = np.array([float(r["elapsed_seconds"]) for r in rows])
    first = next(
        (
            float(r["elapsed_seconds"])
            for r in rows
            if r.get("training_updates") and int(r["training_updates"]) > 0
        ),
        None,
    )
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    axes[0].plot(
        t,
        [float(r["cpu_percent_one_core"]) / args.cpu_quota for r in rows],
        color="#247BA0",
        label=f"CPU / {args.cpu_quota:g}-core quota",
    )
    axes[0].set_ylabel("CPU quota (%)")
    axes[1].plot(
        t,
        [float(r["gpu_util_percent"]) for r in rows],
        color="#E07A5F",
        label="GPU utilization",
    )
    axes[1].set_ylabel("GPU use (%)")
    axes[2].plot(
        t,
        [float(r["rss_tree_mib"]) / 1024 for r in rows],
        color="#247BA0",
        label="Process RSS",
    )
    axes[2].plot(
        t,
        [float(r["gpu_memory_mib"]) / 1024 for r in rows],
        color="#E07A5F",
        label="GPU memory",
    )
    axes[2].set_ylabel("Memory (GiB)")
    axes[2].set_xlabel("Elapsed time (seconds)")
    for ax in axes:
        ax.grid(alpha=0.18)
        if first is not None:
            ax.axvline(
                first, color="#666666", ls="--", lw=1, label="First logged update"
            )
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "resources")


if __name__ == "__main__":
    main()
