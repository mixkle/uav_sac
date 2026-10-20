"""Non-training physical calibration; scripted layouts are diagnostics, not baselines."""

from __future__ import annotations
import copy
import csv
from pathlib import Path
import numpy as np
from .environment import UAVEnvironment, uniform_disk
from .actions import nearest_actions
from .config import save_config
from .runtime import atomic_json, configure_runtime, metadata

FIELDS = [
    "coverage",
    "throughput_bps",
    "robustness",
    "discovery_fraction",
    "cumulative_discovery_fraction",
    "sensing_rejected_users",
    "downlink_rejected_users",
    "backhaul_rejected_users",
    "capacity_rejected_users",
    "expansion_distance_mean",
    "observed_count_mean",
    "observed_count_max",
    "disconnected_uavs",
    "disconnection_ratio",
    "disconnection_penalty",
]


def expanded_layout(e):
    if e.num_uavs == 18 and e.num_bs == 3:
        # Three six-node grids near the reference corner gateways. This is a
        # fixed physical diagnostic, never a reset rule or a policy baseline.
        grid = np.array([(x, y) for y in [350, 950] for x in [350, 950, 1550]], float)
        return np.concatenate([
            grid,
            grid * [1, -1] + [0, 3500],
            grid * [-1, 1] + [3500, 0],
        ]) * [e.width / 3500, e.height / 3500]
    if e.num_uavs != 16:
        raise ValueError(
            "Fixed-layout calibration requires the 16-UAV or 18-UAV/3-BS scene"
        )
    return np.array(
        [(x, y) for y in [120, 460, 1140, 1480] for x in [180, 520, 860, 1200]], float
    ) * [e.width / 1600, e.height / 1600]


def calibrate(cfg, output, seeds=range(30000, 30008)):
    cfg.validate()
    configure_runtime(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "config.yaml")
    atomic_json(
        output / "metadata.json", metadata(cfg, __import__("torch").device("cpu"))
    )
    seeds = list(seeds)
    if set(seeds) & (set(cfg.run.validation_seeds) | set(cfg.run.test_seeds)):
        raise ValueError("Calibration must not use validation/test seeds")
    variants = [("default", {})]
    variants += [
        (f"sensing{s}_downlink{d}", dict(sensing_threshold_db=s, sinr_threshold_db=d))
        for s in [cfg.env.sensing_threshold_db + delta for delta in [-5, 0, 5]]
        for d in [cfg.env.sinr_threshold_db + delta for delta in [-3, 0, 3]]
    ]
    variants += [
        (f"{field}_{factor:g}", {field: getattr(cfg.env, field) * factor})
        for field in ["uu_range", "ug_range"]
        for factor in [0.8, 1.2]
    ]
    summaries, curves = [], []
    for name, changes in variants:
        e = copy.deepcopy(cfg.env)
        for key, value in changes.items():
            setattr(e, key, value)
        target = expanded_layout(e)
        for mode in ["static", "random", "scripted_expansion", "expanded_static"]:
            mode_curves = []
            for seed in seeds:
                env = UAVEnvironment(e)
                env.reset(seed)
                if mode == "expanded_static":
                    env.set_positions(target, env.user_xy)
                rng = np.random.default_rng([seed, 901])
                history = [env.info.copy()]
                frames = (
                    [env.trace_frame()]
                    if name == "default" and seed == seeds[0]
                    else []
                )
                for t in range(e.horizon):
                    if mode == "random":
                        action = uniform_disk(rng, (e.num_uavs,))
                    elif mode == "scripted_expansion":
                        delta = (target - env.uav_xy) / (e.max_speed * e.dt)
                        # Reach the fixed targets progressively over the first 160 steps.
                        action = delta / max(1, 160 - t)
                        action /= np.maximum(
                            1, np.linalg.norm(action, axis=-1, keepdims=True)
                        )
                    else:
                        action = np.zeros((e.num_uavs, 2), np.float32)
                    if e.action_space == "discrete9":
                        if mode == "random":
                            action = rng.integers(0, 9, e.num_uavs)
                        else:
                            if mode == "scripted_expansion":
                                delta = (target - env.uav_xy) / (e.max_speed * e.dt)
                                action = delta / np.maximum(
                                    1, np.linalg.norm(delta, axis=-1, keepdims=True)
                                )
                            action = nearest_actions(action)
                    _, _, _, _, _, info = env.step(action)
                    history.append(info)
                    if frames:
                        frames.append(env.trace_frame())
                row = dict(
                    variant=name,
                    mode=mode,
                    scene_seed=seed,
                    sensing_threshold_db=e.sensing_threshold_db,
                    sinr_threshold_db=e.sinr_threshold_db,
                    uu_range=e.uu_range,
                    ug_range=e.ug_range,
                )
                for key in FIELDS:
                    row[key] = float(np.mean([h[key] for h in history[1:]]))
                    row["initial_" + key] = history[0][key]
                    row["final_" + key] = history[-1][key]
                    row["last100_" + key] = float(
                        np.mean([h[key] for h in history[-100:]])
                    )
                row["first_service_step"] = history[-1]["first_service_step"]
                row["zero_reward_steps"] = sum(h["reward"] == 0 for h in history[1:])
                row["zero_reward_initial_steps"] = next(
                    (i for i, h in enumerate(history[1:]) if h["reward"] > 0), e.horizon
                )
                summaries.append(row)
                if name == "default":
                    mode_curves.append([[h[k] for k in FIELDS] for h in history])
                if frames:
                    np.savez_compressed(
                        output / f"trace_{mode}_{seed}.npz",
                        **{k: np.stack([f[k] for f in frames]) for k in frames[0]},
                        bs_positions=np.array(e.bs_positions),
                        width=e.width,
                        height=e.height,
                        scene_seed=seed,
                        reward_weights=[float(e.reward_primary == 'coverage'), e.throughput_weight, e.robustness_weight, -e.disconnection_weight],
                        sensing_reward_weight=float(e.reward_primary == 'sensing'),
                        reward_primary_name=e.reward_primary,
                        reward_contribution_names=e.reward_component_names,
                    )
            if mode_curves:
                values = np.mean(mode_curves, axis=0)
                curves += [
                    dict(mode=mode, step=t, **dict(zip(FIELDS, vals)))
                    for t, vals in enumerate(values)
                ]
        print(
            f"Calibrated {name}: {len(seeds)} independent scenes, four layouts",
            flush=True,
        )
    for name, rows in [("summary.csv", summaries), ("curves.csv", curves)]:
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(
        seeds=seeds,
        variants=len(variants),
        episodes=len(summaries),
        horizon=cfg.env.horizon,
        note="Paired UE movement; scripted target layouts use no UE positions, but are physical diagnostics, not decentralized policy baselines. Defaults are never changed by this scan.",
    )
    atomic_json(output / "report.json", report)
    return report
