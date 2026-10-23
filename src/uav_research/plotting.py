from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import numpy as np

# Headless, deterministic export on WSL2 and servers.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse
from .config import load_config
from .runtime import atomic_json, resolve_output

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
STYLES = ["-", "--", "-.", ":"]
METRICS = [
    "coverage",
    "throughput_bps",
    "robustness",
    "retention_ratio",
    "worst_fault_coverage",
    "inference_ms_per_uav",
]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure, output, name):
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for ext in ("pdf", "svg", "png"):
        figure.savefig(output / f"{name}.{ext}", bbox_inches="tight")
    plt.close(figure)


def read_csv(path):
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def label(cfg, method=None):
    method = method or cfg.algo.name
    return f"{method} (λ={cfg.env.robustness_weight:g})"


def comparable_environment(data):
    result = dict(data)
    result.pop("backend", None)
    result.pop("robustness_weight", None)
    return result


def bootstrap_mean(values, rng, repeats=2000):
    values = np.asarray(values)
    if len(values) < 2:
        return None, None
    sample = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return tuple(np.quantile(sample, [0.025, 0.975]))


def curves(run_dirs, output, smooth=1):
    style()
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    environment = None
    group_seeds = {}
    frozen_algorithms = {}
    for directory in run_dirs:
        directory = Path(directory)
        cfg = load_config(directory / "config.yaml")
        setting = comparable_environment(cfg.as_dict()["env"])
        if environment is not None and setting != environment:
            raise ValueError("Curve environments differ; plot separate scenarios")
        environment = setting
        name = label(cfg)
        seeds = group_seeds.setdefault(name, set())
        if cfg.run.seed in seeds:
            raise ValueError("Duplicate training seed in curve group")
        seeds.add(cfg.run.seed)
        algorithm = cfg.as_dict()["algo"]
        if name in frozen_algorithms and frozen_algorithms[name] != algorithm:
            raise ValueError("Cannot aggregate different algorithm hyperparameters")
        frozen_algorithms[name] = algorithm
        rows = read_csv(directory / "validation.csv")
        if rows:
            groups.setdefault(label(cfg), []).append(rows)
    if not groups:
        raise ValueError("No validation curves found")
    for axis_key, filename, xlabel in (
        ("env_steps", "sample_efficiency", "Joint environment transitions"),
        ("elapsed_seconds", "wall_efficiency", "Wall-clock hours"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
        for gi, (name, runs) in enumerate(groups.items()):
            max_common = min(float(run[-1][axis_key]) for run in runs)
            min_common = max(float(run[0][axis_key]) for run in runs)
            grid = (
                np.linspace(min_common, max_common, 100)
                if max_common > min_common
                else np.array([max_common])
            )
            for ax, metric in zip(axes, ["coverage", "throughput_bps", "robustness"]):
                values = []
                for rows in runs:
                    x = np.array([float(row[axis_key]) for row in rows])
                    y = np.array([float(row[metric]) for row in rows])
                    order = np.argsort(x, kind="stable")
                    values.append(np.interp(grid, x[order], y[order]))
                values = np.stack(values)
                mean = values.mean(0)
                displayed = mean.copy()
                if smooth > 1:
                    displayed = np.array(
                        [
                            mean[max(0, i - smooth + 1) : i + 1].mean()
                            for i in range(len(mean))
                        ]
                    )
                scale = 1e6 if metric == "throughput_bps" else 1.0
                x = grid / 3600 if axis_key == "elapsed_seconds" else grid
                color = COLORS[gi % len(COLORS)]
                ax.plot(
                    x,
                    displayed / scale,
                    label=name,
                    color=color,
                    linestyle=STYLES[gi % len(STYLES)],
                )
                if len(runs) > 1:
                    rng = np.random.default_rng(0)
                    sampled = values[
                        rng.integers(0, len(runs), (2000, len(runs)))
                    ].mean(1)
                    lower, upper = np.quantile(sampled, [0.025, 0.975], axis=0)
                    ax.fill_between(
                        x, lower / scale, upper / scale, color=color, alpha=0.14
                    )
                ax.set(
                    xlabel=xlabel,
                    ylabel="Throughput (Mbps)"
                    if metric == "throughput_bps"
                    else metric.replace("_", " ").title(),
                )
                with (output / f"{filename}_{gi}_{metric}_raw.csv").open(
                    "w", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow([axis_key, "mean"])
                    writer.writerows(zip(grid, mean))
        axes[0].legend(fontsize=7)
        save_figure(fig, output, filename)

    for index, directory in enumerate(run_dirs):
        cfg = load_config(Path(directory) / "config.yaml")
        if cfg.env.action_space != "discrete9":
            continue
        rows = read_csv(Path(directory) / "metrics.csv")
        x = [float(row["env_steps"]) for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        from .actions import ACTION_NAMES

        for i, name in enumerate(ACTION_NAMES):
            axes[0].plot(
                x, [float(row[f"action_{i}_frequency"]) for row in rows], label=name
            )
        for metric in (
            "categorical_entropy",
            "max_action_probability",
            "hover_fraction",
        ):
            axes[1].plot(
                x,
                [float(row[metric]) if row.get(metric) else np.nan for row in rows],
                label=metric,
            )
        for ax in axes:
            ax.set_xlabel("Joint environment transitions")
            ax.legend(fontsize=6, ncol=2)
        axes[0].set_ylabel("Action frequency (logged vector step)")
        axes[1].set_ylabel("Categorical diagnostics")
        save_figure(fig, output, f"discrete_actions_{index}")


def hierarchical_results(evaluation_dirs, output, repeats=2000):
    """Training-seed resampling, then paired scene resampling within each seed."""
    style()
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    environment = None
    for directory in evaluation_dirs:
        directory = Path(directory)
        rows = read_csv(directory / "episodes.csv")
        from .config import Config, EnvConfig, AlgoConfig, RunConfig

        data = json.loads((directory / "evaluation_config.json").read_text())
        setting = comparable_environment(data["env"])
        if environment is not None and setting != environment:
            raise ValueError(
                "Evaluation environments differ; use --parameter for sensitivity plots"
            )
        environment = setting
        cfg = Config(
            EnvConfig(**data["env"]),
            AlgoConfig(**data["algo"]),
            RunConfig(**data["run"]),
        )
        name = label(cfg, rows[0]["method"])
        train_seed = int(rows[0]["train_seed"])
        if train_seed in groups.setdefault(name, {}):
            raise ValueError(f"Duplicate training seed for {name}")
        groups[name][train_seed] = {int(row["scene_seed"]): row for row in rows}
    if not groups:
        raise ValueError("No evaluation tables")
    summaries = []
    for name, seeds in groups.items():
        means = {
            metric: float(
                np.mean(
                    [
                        np.mean([float(r[metric]) for r in scenes.values()])
                        for scenes in seeds.values()
                    ]
                )
            )
            for metric in METRICS
        }
        rng = np.random.default_rng(0)
        draws = {metric: [] for metric in METRICS}
        if len(seeds) >= 2:
            seed_keys = list(seeds)
            for _ in range(repeats):
                selected = rng.choice(seed_keys, len(seed_keys), replace=True)
                sums = {metric: [] for metric in METRICS}
                for seed in selected:
                    scenes = list(seeds[seed].values())
                    chosen = rng.integers(0, len(scenes), len(scenes))
                    for metric in METRICS:
                        sums[metric].append(
                            np.mean([float(scenes[i][metric]) for i in chosen])
                        )
                for metric in METRICS:
                    draws[metric].append(float(np.mean(sums[metric])))
        for metric in METRICS:
            low, high = (
                tuple(np.quantile(draws[metric], [0.025, 0.975]))
                if draws[metric]
                else (None, None)
            )
            summaries.append(
                dict(
                    method=name,
                    metric=metric,
                    mean=means[metric],
                    ci_low=low,
                    ci_high=high,
                    training_seeds=len(seeds),
                )
            )
    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paired = []
    names = list(groups)
    for j, name in enumerate(names):
        for other in names[j + 1 :]:
            common = sorted(set(groups[name]) & set(groups[other]))
            if len(common) < 2:
                continue
            differences = {metric: [] for metric in METRICS}
            rng = np.random.default_rng(0)
            for _ in range(repeats):
                selected = rng.choice(common, len(common), replace=True)
                per_seed = {metric: [] for metric in METRICS}
                for seed in selected:
                    a, b = groups[name][seed], groups[other][seed]
                    scenes = sorted(set(a) & set(b))
                    if not scenes:
                        raise ValueError("No paired evaluation scenes")
                    chosen = rng.choice(scenes, len(scenes), replace=True)
                    for metric in METRICS:
                        per_seed[metric].append(
                            np.mean(
                                [
                                    float(a[s][metric]) - float(b[s][metric])
                                    for s in chosen
                                ]
                            )
                        )
                for metric in METRICS:
                    differences[metric].append(np.mean(per_seed[metric]))
            for metric, draws in differences.items():
                low, high = np.quantile(draws, [0.025, 0.975])
                paired.append(
                    dict(
                        first=name,
                        second=other,
                        metric=metric,
                        ci_low=float(low),
                        ci_high=float(high),
                    )
                )
    atomic_json(
        output / "statistics.json",
        dict(
            results=summaries,
            paired_differences=paired,
            bootstrap_repeats=repeats,
            note="No training-seed confidence interval is computed for a single seed.",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    for ax, metric in zip(axes, ["coverage", "throughput_bps", "robustness"]):
        rows = [r for r in summaries if r["metric"] == metric]
        scale = 1e6 if metric == "throughput_bps" else 1
        for i, row in enumerate(rows):
            error = (
                None
                if row["ci_low"] is None
                else np.array(
                    [[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
                )
                / scale
            )
            ax.errorbar(
                i,
                row["mean"] / scale,
                yerr=error,
                fmt="o",
                color=COLORS[i % len(COLORS)],
                capsize=3,
            )
        ax.set_xticks(
            range(len(rows)),
            [r["method"] for r in rows],
            rotation=30,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel(
            "Throughput (Mbps)" if metric == "throughput_bps" else metric.title()
        )
    save_figure(fig, output, "method_comparison")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i, name in enumerate(groups):
        by_metric = {r["metric"]: r["mean"] for r in summaries if r["method"] == name}
        ax.scatter(
            by_metric["coverage"],
            by_metric["retention_ratio"],
            color=COLORS[i % len(COLORS)],
            label=name,
        )
    ax.set(xlabel="Coverage", ylabel="Fault retention ratio")
    ax.legend(fontsize=7)
    save_figure(fig, output, "coverage_robustness_tradeoff")
    return summaries


def trace_plot(path, output):
    style()
    data = np.load(path)
    positions = data["uav_xy"]
    users = data["user_xy"]
    bs = data["bs_positions"]
    indices = [0, (len(positions) - 1) // 2, len(positions) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, t in zip(axes, indices):
        served = data["C"][t].sum(0) > 0
        sensed = data['sensed'][t].any(axis=0) if 'sensed' in data else served
        only_sensed = sensed & ~served
        ax.scatter(
            users[t, ~sensed, 0], users[t, ~sensed, 1], s=6, c="#999999", alpha=0.55
        )
        ax.scatter(users[t, only_sensed, 0], users[t, only_sensed, 1], s=7, c=COLORS[3], alpha=0.7)
        ax.scatter(
            users[t, served, 0], users[t, served, 1], s=8, c="#009E73", alpha=0.7
        )
        for i in range(positions.shape[1]):
            ax.plot(
                positions[: t + 1, i, 0],
                positions[: t + 1, i, 1],
                lw=0.8,
                color=COLORS[i % len(COLORS)],
                alpha=0.6,
            )
            for j in range(i):
                if data["A"][t, i, j]:
                    ax.plot(
                        positions[t, [i, j], 0],
                        positions[t, [i, j], 1],
                        c="#0072B2",
                        lw=0.7,
                        alpha=0.5,
                    )
            ax.annotate(str(i + 1), positions[t, i], fontsize=7)
            for g in range(len(bs)):
                if data["B"][t, i, g] == 1:
                    ax.plot(
                        [positions[t, i, 0], bs[g, 0]],
                        [positions[t, i, 1], bs[g, 1]],
                        c="#D55E00",
                        ls="--",
                        lw=0.8,
                    )
        ax.scatter(
            positions[t, :, 0], positions[t, :, 1], marker="^", s=35, c="#0072B2"
        )
        ax.scatter(bs[:, 0], bs[:, 1], marker="*", s=130, c="#D55E00", clip_on=False)
        ax.set(
            xlim=(0, float(data["width"])),
            ylim=(0, float(data["height"])),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"Step {t}",
        )
    legend = [
        Line2D(
            [],
            [],
            marker="*",
            color="#D55E00",
            linestyle="none",
            markersize=9,
            label="BS gateway",
        ),
        Line2D([], [], marker="^", color="#0072B2", linestyle="none", label="UAV"),
        Line2D(
            [], [], marker="o", color="#009E73", linestyle="none", label="Served UE"
        ),
        Line2D(
            [], [], marker="o", color="#999999", linestyle="none", label="Not sensed UE"
        ),
        Line2D([], [], marker='o', color=COLORS[3], linestyle='none', label='Sensed, unserved UE'),
        Line2D([], [], color="#0072B2", label="UAV link"),
        Line2D([], [], color="#D55E00", linestyle="--", label="Direct BS link"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=6,
        fontsize=7,
    )
    save_figure(fig, output, f"topology_{data['scene_seed'].item()}")
    if "discovery_fraction" in data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(
            data["discovery_fraction"], label="Currently discovered", color=COLORS[0]
        )
        ax.plot(data["coverage"], label="Currently served", color=COLORS[1])
        ax.plot(
            data["cumulative_discovery_fraction"],
            label="Ever discovered (history diagnostic)",
            color=COLORS[2],
            ls="--",
        )
        ax.set(xlabel="Task step", ylabel="UE fraction", ylim=(-0.02, 1.02))
        ax.legend(fontsize=7)
        save_figure(fig, output, f"discovery_service_{data['scene_seed'].item()}")
    if "sensed" in data:
        lengths = data["sensed"].sum(-1).ravel()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(
            lengths, bins=np.arange(lengths.max() + 2) - 0.5, color=COLORS[0], alpha=0.8
        )
        ax.set(xlabel="Currently discovered UE per UAV", ylabel="UAV-time group count")
        save_figure(fig, output, f"set_lengths_{data['scene_seed'].item()}")
    coverage = data["reward_terms"][1:, 0]
    faults = data["fault_coverage"][1:]
    losses = coverage[:, None] - faults
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].boxplot(
        losses,
        showfliers=False,
        tick_labels=[str(i + 1) for i in range(losses.shape[1])],
    )
    axes[0].set(xlabel="Failed UAV", ylabel="Immediate coverage loss")
    axes[1].plot(coverage, label="Nominal coverage", color=COLORS[0])
    axes[1].plot(faults.mean(1), label="Mean post-failure coverage", color=COLORS[1])
    axes[1].plot(
        faults.min(1), label="Worst post-failure coverage", color=COLORS[2], ls="--"
    )
    axes[1].set(xlabel="Task step", ylabel="Coverage")
    axes[1].legend(fontsize=7)
    save_figure(fig, output, f"faults_{data['scene_seed'].item()}")


def sensitivity(evaluation_dirs, parameter, output):
    style()
    output = resolve_output(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for directory in evaluation_dirs:
        directory = Path(directory)
        cfg = json.loads((directory / "evaluation_config.json").read_text())
        value = (
            len(cfg["env"]["bs_positions"])
            if parameter == "num_bs"
            else cfg["env"][parameter]
        )
        rows = read_csv(directory / "episodes.csv")
        key = rows[0]["method"]
        if parameter != "robustness_weight":
            key += f" (λ={cfg['env']['robustness_weight']:g})"
        group = groups.setdefault(key, {})
        group.setdefault(value, []).append(rows)
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    rng = np.random.default_rng(0)
    for gi, (name, settings) in enumerate(groups.items()):
        for ax, metric in zip(axes, ["coverage", "throughput_bps", "retention_ratio"]):
            points = []
            for setting, runs in sorted(settings.items()):
                seeds = [int(run[0]["train_seed"]) for run in runs]
                if len(seeds) != len(set(seeds)):
                    raise ValueError("Duplicate seed in sensitivity setting")
                values = [np.mean([float(row[metric]) for row in run]) for run in runs]
                mean = float(np.mean(values))
                low, high = bootstrap_mean(values, rng)
                results.append(
                    dict(
                        method=name,
                        parameter=parameter,
                        value=setting,
                        metric=metric,
                        mean=mean,
                        ci_low=low,
                        ci_high=high,
                        training_seeds=len(runs),
                    )
                )
                points.append((setting, mean, low, high))
            scale = 1e6 if metric == "throughput_bps" else 1
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points]) / scale
            ax.plot(
                x,
                y,
                marker="o",
                label=name,
                color=COLORS[gi % len(COLORS)],
                linestyle=STYLES[gi % len(STYLES)],
            )
            if all(p[2] is not None for p in points):
                ax.fill_between(
                    x,
                    np.array([p[2] for p in points]) / scale,
                    np.array([p[3] for p in points]) / scale,
                    color=COLORS[gi % len(COLORS)],
                    alpha=0.14,
                )
            ax.set(
                xlabel=parameter.replace("_", " "),
                ylabel="Throughput (Mbps)"
                if metric == "throughput_bps"
                else metric.replace("_", " ").title(),
            )
    axes[0].legend(fontsize=7)
    save_figure(fig, output, f"sensitivity_{parameter}")
    atomic_json(output / f"sensitivity_{parameter}.json", results)
    return results


def calibration_plots(directory, output):
    """Rebuild physical calibration figures directly from exported CSV files."""
    style()
    directory = Path(directory)
    rows = read_csv(directory / "curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for i, mode in enumerate(dict.fromkeys(r["mode"] for r in rows)):
        selected = [r for r in rows if r["mode"] == mode]
        x = [int(r["step"]) for r in selected]
        for ax, key, title in zip(
            axes,
            ["discovery_fraction", "coverage", "robustness"],
            [
                "Currently discovered UE fraction",
                "Served UE fraction",
                "Post-failure retained coverage",
            ],
        ):
            ax.plot(
                x,
                [float(r[key]) for r in selected],
                color=COLORS[i],
                ls=STYLES[i],
                label=mode.replace("_", " "),
            )
            ax.set(xlabel="Task step", ylabel=title, ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=6)
    save_figure(fig, output, "calibration_expansion")
    summaries = read_csv(directory / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, mode in zip(axes, ["static", "expanded_static"]):
        grid = np.zeros((3, 3))
        for i, s in enumerate([0, 5, 10]):
            for j, d in enumerate([-6, -3, 0]):
                subset = [
                    r
                    for r in summaries
                    if r["mode"] == mode and r["variant"] == f"sensing{s}_downlink{d}"
                ]
                grid[i, j] = np.mean([float(r["coverage"]) for r in subset])
        img = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set(
            xticks=range(3),
            xticklabels=[-6, -3, 0],
            yticks=range(3),
            yticklabels=[0, 5, 10],
            xlabel="Downlink SINR threshold (dB)",
            ylabel="Beacon SNR threshold (dB)",
            title=mode.replace("_", " "),
        )
        for (i, j), val in np.ndenumerate(grid):
            ax.text(
                j,
                i,
                f"{val:.1%}",
                ha="center",
                va="center",
                color="white" if val < 0.5 else "black",
            )
        fig.colorbar(img, ax=ax, label="Mean coverage", shrink=0.8)
    save_figure(fig, output, "calibration_thresholds")
    cfg = load_config(directory / "config.yaml")
    traces = sorted(directory.glob("trace_static_*.npz"))
    if traces:
        initialization_plot(traces[0], cfg, output)


def initialization_plot(trace_path, cfg, output):
    """Initial UE/launch sample; all points come from the saved scenario."""
    style()
    data = np.load(trace_path)
    xy, uav = data["user_xy"][0], data["uav_xy"][0]
    e = cfg.env
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(
        xy[:, 0], xy[:, 1], s=10, c=COLORS[2], alpha=0.65, label=f"UE (M={len(xy)})"
    )
    ax.scatter(uav[:, 0], uav[:, 1], s=25, marker="^", c=COLORS[0], label="Initial UAV")
    bs = np.asarray(e.bs_positions)
    ax.scatter(
        bs[:, 0], bs[:, 1], s=110, marker="*", c=COLORS[1], clip_on=False, label="BS"
    )
    if e.user_initialization in ("gaussian_mixture", "uniform_interior"):
        lower = np.array([e.width, e.height]) * e.user_initial_margin_fraction
        ax.add_patch(
            Rectangle(
                lower,
                e.width - 2 * lower[0],
                e.height - 2 * lower[1],
                fill=False,
                color="#666666",
                ls="--",
                label="Initial UE bounds",
            )
        )
    if e.user_initialization == "gaussian_mixture":
        centers = np.asarray(e.user_cluster_centers) * [e.width, e.height]
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=40,
            marker="+",
            color="#000000",
            label="Mixture centers",
        )
        for center in centers:
            ax.add_patch(
                Ellipse(
                    center,
                    2 * e.width * e.user_cluster_std_fraction,
                    2 * e.height * e.user_cluster_std_fraction,
                    fill=False,
                    color="#666666",
                    alpha=0.6,
                    lw=0.8,
                )
            )
    ax.set(
        xlim=(0, e.width),
        ylim=(0, e.height),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            "Uniform UE initialization"
            if e.user_initialization in ("uniform", "uniform_interior")
            else "Truncated Gaussian-mixture UE initialization"
        ),
    )
    ax.legend(fontsize=7, loc="upper right")
    save_figure(fig, output, "ue_initialization")
