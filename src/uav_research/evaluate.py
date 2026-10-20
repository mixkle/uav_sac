from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import Config
from .actions import nearest_actions, validate_action_specification
from .environment import uniform_disk
from .algorithms import make_learner
from .runtime import atomic_json, load_checkpoint, isolated_rng, resolve_output
from .vector import VectorEnvironment


class Baseline:
    def __init__(self, cfg, name, seed=0):
        self.cfg, self.name, self.rng = cfg.env, name, np.random.default_rng(seed)

    def act(self, obs, deterministic=False):
        e = self.cfg
        flat = obs.fixed.reshape(-1, e.obs_dim)
        if self.name == "static":
            return (
                np.full(obs.fixed.shape[:-1], 8, np.int64)
                if e.action_space == "discrete9"
                else np.zeros((*obs.fixed.shape[:-1], 2), np.float32)
            )
        if self.name == "random":
            return (
                self.rng.integers(0, 9, obs.fixed.shape[:-1])
                if e.action_space == "discrete9"
                else uniform_disk(self.rng, obs.fixed.shape[:-1])
            )
        if self.name != "heuristic":
            raise ValueError(self.name)
        output = np.zeros((len(flat), 2), np.float32)
        scale = np.array([e.width, e.height])
        for i, row in enumerate(flat):
            own = row[:2] * scale
            start = e.self_dim + 2 * e.num_bs
            neighbors = row[start : start + 3 * e.neighbor_dim].reshape(
                3, e.neighbor_dim
            )
            users = obs.users[obs.offsets[i] : obs.offsets[i + 1]]
            neighbor_positions = neighbors[neighbors[:, 0] > 0, 1:3] * scale
            user_positions = users[:, :2] * scale + own
            vector = np.zeros(2)
            if len(user_positions):
                vector += (user_positions.mean(0) - own) / 100
            if len(neighbor_positions):
                delta = neighbor_positions - own
                distances = np.linalg.norm(delta, axis=1)
                # 0.25 cohesion when connected; 1.0 when self hop equals sentinel.
                cohesion = 1.0 if row[3] >= 1 else 0.25
                vector += cohesion * delta.mean(0) / 100
                vector -= np.sum(
                    delta
                    / np.maximum(distances[:, None], 1)
                    * (np.maximum(0, 100 - distances) / 100)[:, None],
                    axis=0,
                )
            norm = np.linalg.norm(vector)
            output[i] = (
                vector / max(norm, 1) if norm > 1e-6 else uniform_disk(self.rng, ())
            )
        output = output.reshape(*obs.fixed.shape[:-1], 2)
        return nearest_actions(output) if e.action_space == "discrete9" else output


EPISODE_FIELDS = [
    "reward_primary",
    "final_reward_primary",
    "last100_reward_primary",
    "last100_discovery_fraction",
    "user_speed_mean",
    "user_displacement_mean",
    "final_user_displacement_mean",
    "uav_nearest_neighbor_mean",
    "uav_close_neighbor_fraction",
    "final_uav_nearest_neighbor_mean",
    "final_uav_close_neighbor_fraction",
    "last100_uav_nearest_neighbor_mean",
    "last100_uav_close_neighbor_fraction",
    "scene_seed",
    "train_seed",
    "method",
    "steps",
    "return",
    "coverage",
    "throughput_bps",
    "tau",
    "robustness",
    "retention_ratio",
    "worst_fault_coverage",
    "zero_coverage_episode",
    "zero_coverage_steps",
    "final_coverage",
    "final_throughput_bps",
    "final_robustness",
    "final_retention_ratio",
    "inference_ms_per_uav",
    "component_count",
    "boundary_clip_fraction",
    "boundary_reflection_fraction",
    "gateway_switch_coverage",
    "disconnected_uavs",
    "disconnection_ratio",
    "disconnection_penalty",
    "final_disconnection_ratio",
    "final_disconnection_penalty",
    "last100_disconnection_ratio",
    "last100_disconnection_penalty",
    "reachable_all_gateways",
    "discovered_users",
    "discovery_fraction",
    "cumulative_discovery_fraction",
    "discovered_unserved_users",
    "sensing_rejected_users",
    "downlink_rejected_users",
    "backhaul_rejected_users",
    "capacity_rejected_users",
    "expansion_distance_mean",
    "observed_count_mean",
    "observed_count_max",
    "actual_speed_mean",
    "first_service_step",
    "zero_reward_steps",
    "zero_reward_initial_steps",
    "negative_reward_steps",
    "first_positive_reward_step",
    "final_discovery_fraction",
    "final_cumulative_discovery_fraction",
    "last100_coverage",
    "last100_throughput_bps",
    "last100_robustness",
    "last100_retention_ratio",
]


def evaluation_fields(cfg):
    return (
        EPISODE_FIELDS
        + (
            [f"action_{i}_frequency" for i in range(9)] + ["hover_fraction"]
            if cfg.env.action_space == "discrete9"
            else []
        )
        + [
            key
            for g in range(cfg.env.num_bs)
            for key in (f"bs_{g}_direct_uavs", f"bs_{g}_only_reachable")
        ]
    )


def _evaluation_episodes(cfg, policy, seeds, trace_count):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("Evaluation requires at least one scene seed")
    for start_index in range(0, len(seeds), cfg.run.num_envs):
        selected = seeds[start_index : start_index + cfg.run.num_envs]
        count = len(selected)
        with VectorEnvironment(cfg.env, count, "batch") as vector:
            obs, _, _ = vector.reset(selected)
            baselines = (
                [Baseline(cfg, policy.name, int(seed) + 700000) for seed in selected]
                if isinstance(policy, Baseline)
                else None
            )
            histories = [[] for _ in selected]
            frames = [
                [env.trace_frame()] if start_index + i < trace_count else []
                for i, env in enumerate(vector.envs)
            ]
            inference_seconds = 0.0
            for _ in range(cfg.env.horizon):
                stamp = time.perf_counter()
                actions = (
                    np.stack(
                        [baseline.act(o, True) for baseline, o in zip(baselines, obs)]
                    )
                    if baselines
                    else policy.act(obs, cfg.run.evaluation_policy == "deterministic")
                )
                inference_seconds += time.perf_counter() - stamp
                obs, _, _, done, truncated, infos = vector.step(actions)
                for i, info in enumerate(infos):
                    histories[i].append(info)
                    if frames[i]:
                        frames[i].append(vector.envs[i].trace_frame())
                if (done | truncated).all():
                    break
            for i, seed in enumerate(selected):
                yield (
                    start_index + i,
                    seed,
                    histories[i],
                    frames[i],
                    inference_seconds / count,
                )


def evaluate(cfg: Config, policy, seeds, output=None, method=None, trace_count=0, reward_curves=False):
    rows = []
    curve_rows = []
    diagnostic_names = ['coverage', 'discovery_fraction', 'disconnection_ratio',
                        'user_displacement_mean', 'uav_nearest_neighbor_mean',
                        'uav_close_neighbor_fraction']
    output = resolve_output(output) if output is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    method = method or (policy.name if isinstance(policy, Baseline) else cfg.algo.name)
    # Diagnostic and validation sampling cannot perturb training RNG streams.
    with isolated_rng():
        if cfg.run.evaluation_policy == "sample":
            torch.manual_seed(cfg.run.evaluation_action_seed)
        for index, seed, infos, frames, inference_seconds in _evaluation_episodes(
            cfg, policy, seeds, trace_count
        ):
            scalar_keys = [
                k for k, v in infos[0].items() if isinstance(v, (float, int))
            ]
            means = {
                key: float(np.mean([info[key] for info in infos]))
                for key in scalar_keys
            }
            coverage_sum = sum(info["coverage"] for info in infos)
            row = {key: means[key] for key in evaluation_fields(cfg) if key in means}
            row.update(
                scene_seed=int(seed),
                train_seed=cfg.run.seed,
                method=method,
                steps=len(infos),
                return_=sum(info["reward"] for info in infos),
                retention_ratio=sum(info["robustness"] for info in infos) / coverage_sum
                if coverage_sum
                else 0.0,
                zero_coverage_episode=int(coverage_sum == 0),
                zero_coverage_steps=sum(info["coverage"] == 0 for info in infos),
                inference_ms_per_uav=1000
                * inference_seconds
                / (len(infos) * cfg.env.num_uavs),
            )
            row["return"] = row.pop("return_")
            for key in ("coverage", "throughput_bps", "robustness", "retention_ratio", "disconnection_ratio", "disconnection_penalty", "reward_primary", "user_displacement_mean", "uav_nearest_neighbor_mean", "uav_close_neighbor_fraction"):
                row[f"final_{key}"] = infos[-1][key]
            row["first_service_step"] = infos[-1]["first_service_step"]
            row["zero_reward_steps"] = sum(info["reward"] == 0 for info in infos)
            row["zero_reward_initial_steps"] = next(
                (i for i, info in enumerate(infos) if info["reward"] != 0), len(infos)
            )
            row["negative_reward_steps"] = sum(info["reward"] < 0 for info in infos)
            row["first_positive_reward_step"] = next(
                (i for i, info in enumerate(infos) if info["reward"] > 0), -1
            )
            for key in ("discovery_fraction", "cumulative_discovery_fraction"):
                row["final_" + key] = infos[-1][key]
            tail = infos[-100:]
            for key in ("coverage", "throughput_bps", "robustness", "disconnection_ratio", "disconnection_penalty", "discovery_fraction", "reward_primary", "uav_nearest_neighbor_mean", "uav_close_neighbor_fraction"):
                row["last100_" + key] = float(np.mean([info[key] for info in tail]))
            tail_coverage = sum(info["coverage"] for info in tail)
            row["last100_retention_ratio"] = (
                sum(info["robustness"] for info in tail) / tail_coverage
                if tail_coverage
                else 0
            )
            rows.append(row)
            if reward_curves:
                raw=np.array([[info[k] for k in ('reward_primary','tau','robustness','disconnection_ratio')]
                              for info in infos],dtype=np.float64)
                actual=np.array([info['reward'] for info in infos],dtype=np.float64)
                weights=np.array([1,cfg.env.throughput_weight,cfg.env.robustness_weight,-cfg.env.disconnection_weight])
                np.testing.assert_allclose((raw*weights).sum(-1),actual,atol=1e-12,rtol=1e-12)
                diagnostics = np.array([[info[k] for k in diagnostic_names] for info in infos])
                curve_rows.append((int(seed),raw,actual,diagnostics))
            if output and frames:
                arrays = {
                    key: np.stack([frame[key] for frame in frames]) for key in frames[0]
                }
                np.savez_compressed(
                    output / f"trace_{seed}.npz",
                    **arrays,
                    bs_positions=np.array(cfg.env.bs_positions),
                    width=cfg.env.width,
                    height=cfg.env.height,
                    scene_seed=seed,
                    reward_primary_name=cfg.env.reward_primary,
                    reward_term_names=np.array(['coverage','tau','robustness','disconnection_ratio']),
                    reward_contribution_names=np.array(cfg.env.reward_component_names),
                    sensing_reward_weight=float(cfg.env.reward_primary == 'sensing'),
                    reward_weights=np.array(
                        [float(cfg.env.reward_primary == 'coverage'), cfg.env.throughput_weight, cfg.env.robustness_weight, -cfg.env.disconnection_weight]
                    ),
                )
    if output:
        if reward_curves:
            raw=np.stack([item[1] for item in curve_rows])
            np.savez_compressed(output/'reward_curves.npz',
                scene_seeds=np.array([item[0] for item in curve_rows]),
                steps=np.arange(1,raw.shape[1]+1), raw=raw, contributions=raw*weights,
                reward=np.stack([item[2] for item in curve_rows]),reward_weights=weights,
                names=np.array(cfg.env.reward_component_names),
                raw_names=np.array([cfg.env.primary_reward_metric,'tau','robustness','disconnection_ratio']),
                reward_primary_name=cfg.env.reward_primary,
                diagnostic_names=np.array(diagnostic_names),
                diagnostics=np.stack([item[3] for item in curve_rows]))
        with (output / "episodes.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=evaluation_fields(cfg))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows)
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "evaluation_config.json", cfg.as_dict())
        provenance = getattr(
            policy, "provenance", {"kind": "baseline", "method": method}
        )
        atomic_json(output / "evaluation_source.json", provenance)
    return rows, summarize(rows)


def summarize(rows):
    keys = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (int, float)) and k not in ("scene_seed", "train_seed")
    ]
    return {
        "episodes": len(rows),
        "method": rows[0]["method"],
        "train_seed": rows[0]["train_seed"],
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "std_across_scenes": {
            key: float(np.std([row[key] for row in rows])) for key in keys
        },
    }


def policy_from_checkpoint(path, device):
    data = load_checkpoint(path, device)
    config = data["config"]
    from .config import config_from_dict

    cfg = config_from_dict(config)
    validate_action_specification(data, cfg.env, cfg.algo.name)
    learner = make_learner(cfg, device)
    learner.load_state_dict(data["learner"])
    manifest_path = Path(path).parent / "manifest.json"
    metrics_path = Path(path).parent / "metrics.csv"
    stable = metrics_path.exists()
    if metrics_path.exists():
        with metrics_path.open() as file:
            for row in csv.DictReader(file):
                for key, value in row.items():
                    if value and key not in ("method",):
                        try:
                            stable = stable and bool(np.isfinite(float(value)))
                        except ValueError:
                            continue
    learner.provenance = dict(
        kind="checkpoint",
        path=str(path),
        checkpoint_env_steps=data["env_steps"],
        training_config=config,
        finite_training_metrics=stable,
        training_manifest=json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None,
    )
    return cfg, learner


def check_gate(sac_dir, mappo_dir, ablation_dir, output):
    def read(directory):
        with (Path(directory) / "episodes.csv").open() as file:
            return list(csv.DictReader(file))

    sac, mappo, ablation = map(read, (sac_dir, mappo_dir, ablation_dir))
    seed_sets = [
        {int(row["scene_seed"]) for row in data} for data in (sac, mappo, ablation)
    ]
    if any(
        len(data) != len(seeds)
        for data, seeds in zip((sac, mappo, ablation), seed_sets)
    ):
        raise ValueError("Gate cannot use duplicate scene seeds")
    if len(set(map(frozenset, seed_sets))) != 1:
        raise ValueError("Gate requires paired validation scenes")
    configs = [
        json.loads((Path(p) / "evaluation_config.json").read_text())
        for p in (sac_dir, mappo_dir, ablation_dir)
    ]
    sources = [
        json.loads((Path(p) / "evaluation_source.json").read_text())
        for p in (sac_dir, mappo_dir, ablation_dir)
    ]
    if any("training_config" not in source for source in sources):
        raise ValueError(
            "Gate requires checkpoint training_config provenance; re-evaluate old results"
        )
    training = [source["training_config"] for source in sources]
    # Evaluation may override reward weights, but must not hide training differences.
    for evaluated, trained in zip(configs, training):
        ev, tr = dict(evaluated["env"]), dict(trained["env"])
        for item in (ev, tr):
            item.pop("robustness_weight")
            item.pop("backend")
        if ev != tr or evaluated["algo"] != trained["algo"]:
            raise ValueError(
                "Gate evaluation/training environments differ or algorithm changed"
            )
    configs = training
    stable_complete = all(
        s.get("finite_training_metrics")
        and s.get("training_manifest")
        and s["training_manifest"].get("status") == "completed"
        for s in sources
    )
    step_counts = [
        s.get("training_manifest", {}).get("env_steps")
        if s.get("training_manifest")
        else None
        for s in sources
    ]
    if len(set(step_counts)) != 1 or None in step_counts:
        raise ValueError("Gate requires equal, recorded training budgets")
    if any(c["run"]["seed"] != configs[0]["run"]["seed"] for c in configs):
        raise ValueError("Gate training seeds must match")
    if any(seed in configs[0]["run"]["test_seeds"] for seed in seed_sets[0]):
        raise ValueError("Test scenes cannot be used for the development gate")
    if seed_sets[0] != set(configs[0]["run"]["validation_seeds"]):
        raise ValueError("Gate requires the complete configured validation set")
    envs = [dict(c["env"]) for c in configs]
    for env in envs:
        env.pop("robustness_weight")
        env.pop("backend")
    if envs[0] != envs[1] or envs[0] != envs[2]:
        raise ValueError("Gate environments differ")
    if configs[0]["env"]["robustness_weight"] != configs[1]["env"]["robustness_weight"]:
        raise ValueError("SAC and MAPPO must use the same training reward weights")
    if (
        configs[0]["algo"]["name"] != "sac"
        or configs[1]["algo"]["name"] != "mappo"
        or configs[2]["algo"]["name"] != "sac"
    ):
        raise ValueError("Gate expects SAC, MAPPO and SAC ablation")
    if configs[2]["env"]["robustness_weight"] != 0:
        raise ValueError("Ablation must have lambda=0")
    if configs[0]["env"]["robustness_weight"] <= 0:
        raise ValueError("Main SAC must have positive lambda")
    if configs[0]["algo"] != configs[2]["algo"]:
        raise ValueError(
            "Single-factor ablation must keep all SAC algorithm hyperparameters identical"
        )
    for key in (
        "seed",
        "num_envs",
        "total_steps",
        "eval_interval",
        "validation_seeds",
        "test_seeds",
        "deterministic",
    ):
        if configs[0]["run"][key] != configs[2]["run"][key]:
            raise ValueError(f"Single-factor ablation changed run.{key}")
    metrics = ["coverage", "throughput_bps", "robustness", "retention_ratio"]
    means = [
        {k: float(np.mean([float(row[k]) for row in data])) for k in metrics}
        for data in (sac, mappo, ablation)
    ]
    comparison = {
        k: {
            "sac": means[0][k],
            "mappo": means[1][k],
            "ablation": means[2][k],
            "beats_mappo": means[0][k] > means[1][k],
        }
        for k in metrics
    }
    robust_effect = means[0]["retention_ratio"] > means[2]["retention_ratio"]
    passed = (
        all(row["beats_mappo"] for row in comparison.values())
        and robust_effect
        and stable_complete
    )
    from .suite import environment_hash
    from .config import EnvConfig, AlgoConfig, RunConfig

    gate_cfg = Config(
        EnvConfig(**configs[0]["env"]),
        AlgoConfig(**configs[0]["algo"]),
        RunConfig(**configs[0]["run"]),
    )
    report = dict(
        passed=passed,
        comparison=comparison,
        robustness_ablation_positive=robust_effect,
        stable_complete_training=bool(stable_complete),
        training_steps=step_counts[0],
        environment_hash=environment_hash(gate_cfg),
        frozen_algorithms={
            "sac": configs[0]["algo"],
            "mappo": configs[1]["algo"],
            "sac_no_robustness": configs[2]["algo"],
        },
        scope="single-training-seed development evidence, not statistical significance",
        next_step="eligible_for_multiseed" if passed else "diagnose_before_multiseed",
    )
    atomic_json(resolve_output(output), report)
    return report
