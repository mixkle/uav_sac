from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math
import yaml


@dataclass
class EnvConfig:
    width: float = 1600.0
    height: float = 1600.0
    num_uavs: int = 16
    num_users: int = 180
    capacity: int = 20
    altitude: float = 100.0
    bs_positions: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 1600.0, 0.0]]
    )
    horizon: int = 600
    dt: float = 1.0
    max_speed: float = 10.0
    user_speed_std: float = 1.0
    user_speed_clip: float = 3.0
    user_initialization: str = "gaussian_mixture"
    user_initial_margin_fraction: float = 0.10
    user_cluster_std_fraction: float = 0.08
    user_cluster_centers: list[list[float]] = field(
        default_factory=lambda: [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]
    )
    uu_range: float = 400.0
    ug_range: float = 350.0
    frequency_hz: float = 2e9
    bandwidth_hz: float = 20e6
    total_power_dbm: float = 30.0
    noise_density_dbm_hz: float = -174.0
    noise_figure_db: float = 9.0
    los_a: float = 9.61
    los_b: float = 0.16
    los_extra_db: float = 1.0
    nlos_extra_db: float = 20.0
    sinr_threshold_db: float = -3.0
    rate_reference_bps: float = 5e6
    reward_primary: str = "coverage"
    throughput_weight: float = 0.1
    robustness_weight: float = 0.5
    # Missing in historical configs: retain their original reward semantics.
    disconnection_weight: float = 0.0
    neighbor_slots: int = 3
    sinr_scale_db: float = 30.0
    action_space: str = "continuous"
    uav_boundary: str = "clip"
    uav_initialization: str = "uniform"
    launch_radius: float = 100.0
    beacon_power_dbm: float = 0.0
    beacon_bandwidth_hz: float = 180e3
    beacon_noise_figure_db: float = 9.0
    sensing_threshold_db: float = 5.0
    backend: str = "numba"

    @property
    def primary_reward_metric(self):
        return "discovery_fraction" if self.reward_primary == "sensing" else "coverage"

    @property
    def reward_component_names(self):
        return [self.reward_primary, "throughput", "robustness", "disconnection"]

    @property
    def num_bs(self):
        return len(self.bs_positions)

    @property
    def self_dim(self):
        return 5 + self.num_uavs

    @property
    def neighbor_dim(self):
        return 6 + self.num_uavs

    @property
    def fixed_obs_dim(self):
        return self.self_dim + 2 * self.num_bs + self.neighbor_slots * self.neighbor_dim

    @property
    def obs_dim(self):
        # Fixed raw part only; UE lists are separate, never padded to this dimension.
        return self.fixed_obs_dim

    @property
    def beacon_noise_dbm(self):
        return (
            self.noise_density_dbm_hz
            + 10 * math.log10(self.beacon_bandwidth_hz)
            + self.beacon_noise_figure_db
        )

    @property
    def state_dim(self):
        u, m, g = self.num_uavs, self.num_users, self.num_bs
        return 3 * (u + m + g) + u * u + u * g + 2 * u * m + 1

    @property
    def subband_power_dbm(self):
        return self.total_power_dbm - 10 * math.log10(self.capacity)

    @property
    def noise_mw(self):
        dbm = (
            self.noise_density_dbm_hz
            + 10 * math.log10(self.bandwidth_hz / self.capacity)
            + self.noise_figure_db
        )
        return 10 ** (dbm / 10)


@dataclass
class AlgoConfig:
    name: str = "sac"
    hidden: int = 256
    ue_hidden: int = 32
    actor_lr: float | None = None
    critic_lr: float | None = None
    temperature_lr: float = 1e-4
    initial_alpha: float = 0.01
    target_entropy: float | None = None
    discount: float = 0.995
    polyak: float = 0.005
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    batch_size: int = 256
    replay_capacity: int = 50000
    warmup_steps: int = 10000
    transitions_per_update: int = 4
    rollout_steps: int = 128
    ppo_epochs: int = 10
    ppo_minibatches: int = 4
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_clipping: bool = True
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    td3_policy_delay: int = 2
    td3_target_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_exploration_noise: float = 0.1

    def learning_rate(self, role):
        value = getattr(self, f"{role}_lr")
        return (
            value
            if value is not None
            else (
                5e-4
                if self.name == "mappo"
                else (1e-4 if self.name == "sac" and role == "actor" else 3e-4)
            )
        )


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"
    num_envs: int = 16
    vector_backend: str = "batch"
    cpu_threads: int = 1
    numba_threads: int = 4
    total_steps: int = 2000000
    eval_interval: int = 50000
    validation_seeds: list[int] = field(
        default_factory=lambda: list(range(10000, 10032))
    )
    test_seeds: list[int] = field(default_factory=lambda: list(range(20000, 20100)))
    log_interval: int = 1000
    checkpoint_minutes: float = 30.0
    max_wall_seconds: float | None = None
    hourly_price: float | None = None
    budget: float | None = None
    tensorboard: bool = True
    memory_limit_gib: float = 12.0
    deterministic: bool = True
    evaluation_policy: str = "deterministic"
    evaluation_action_seed: int = 880001


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self):
        e, a, r = self.env, self.algo, self.run
        positives = (
            e.width,
            e.height,
            e.altitude,
            e.num_uavs,
            e.num_users,
            e.capacity,
            e.horizon,
            e.dt,
            e.max_speed,
            e.uu_range,
            e.ug_range,
            e.frequency_hz,
            e.bandwidth_hz,
            e.rate_reference_bps,
            e.sinr_scale_db,
            e.beacon_bandwidth_hz,
            a.hidden,
            a.ue_hidden,
            a.batch_size,
            a.replay_capacity,
            a.transitions_per_update,
            a.rollout_steps,
            a.ppo_epochs,
            a.ppo_minibatches,
            r.num_envs,
            r.cpu_threads,
            r.numba_threads,
            r.total_steps,
            r.eval_interval,
            r.log_interval,
            r.checkpoint_minutes,
        )
        if any(x <= 0 for x in positives):
            raise ValueError(
                "Counts, scales, intervals and physical dimensions must be positive"
            )
        counts = (
            e.num_uavs,
            e.num_users,
            e.capacity,
            e.horizon,
            a.hidden,
            a.ue_hidden,
            a.batch_size,
            a.replay_capacity,
            a.warmup_steps,
            a.transitions_per_update,
            a.rollout_steps,
            a.ppo_epochs,
            a.ppo_minibatches,
            a.td3_policy_delay,
            r.num_envs,
            r.cpu_threads,
            r.numba_threads,
            r.total_steps,
            r.eval_interval,
            r.log_interval,
            r.seed,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in counts
        ):
            raise ValueError("Counts and seed must be integers")
        if (
            e.user_speed_std < 0
            or e.user_speed_clip < 0
            or e.throughput_weight < 0
            or e.robustness_weight < 0
            or e.disconnection_weight < 0
        ):
            raise ValueError("Speed scales and reward weights must be nonnegative")
        if not math.isfinite(e.disconnection_weight):
            raise ValueError("Disconnection reward weight must be finite")
        if e.reward_primary not in ("coverage", "sensing"):
            raise ValueError("reward_primary must be coverage or sensing")
        if (
            a.learning_rate("actor") <= 0
            or a.learning_rate("critic") <= 0
            or a.temperature_lr <= 0
            or a.max_grad_norm <= 0
        ):
            raise ValueError("Learning rates and gradient clipping must be positive")
        if not e.bs_positions or any(len(p) != 3 for p in e.bs_positions):
            raise ValueError("bs_positions must contain [x,y,z] rows")
        if any(
            not (0 <= p[0] <= e.width and 0 <= p[1] <= e.height and p[2] == 0)
            for p in e.bs_positions
        ):
            raise ValueError("BS positions must be on the ground inside the region")
        if e.ug_range <= e.altitude:
            raise ValueError(
                "ug_range must exceed altitude for nonzero direct-access area"
            )
        if len({tuple(p) for p in e.bs_positions}) != e.num_bs:
            raise ValueError("BS positions must be distinct")
        if e.uav_boundary not in ("clip", "reflect"):
            raise ValueError("Unknown UAV boundary mode")
        if r.evaluation_policy not in ("deterministic", "sample"):
            raise ValueError("Unknown evaluation policy")
        if not isinstance(r.evaluation_action_seed, int) or r.evaluation_action_seed < 0:
            raise ValueError("Evaluation action seed must be a nonnegative integer")
        if e.action_space not in ("continuous", "discrete9"):
            raise ValueError("Unknown action space")
        if e.action_space == "discrete9" and a.name != "mappo":
            raise ValueError("discrete9 requires MAPPO")
        if e.uav_initialization not in ("uniform", "bs_launch"):
            raise ValueError("Unknown UAV initialization; use uniform or bs_launch")
        if e.uav_initialization == "bs_launch":
            if e.bs_positions != [[0.0, 0.0, 0.0], [0.0, e.height, 0.0]]:
                raise ValueError("BS launch requires the two left-corner BS")
            if e.num_uavs % 2:
                raise ValueError("BS launch requires an even UAV count")
            if (
                not 0
                < e.launch_radius
                <= min(e.width, e.height, math.sqrt(e.ug_range**2 - e.altitude**2))
            ):
                raise ValueError(
                    "Launch radius must fit the region and direct gateway range"
                )
        if r.memory_limit_gib <= 0:
            raise ValueError("memory_limit_gib must be positive")
        if e.user_initialization not in ("uniform", "uniform_interior", "gaussian_mixture"):
            raise ValueError("Unknown UE initialization; use uniform, uniform_interior or gaussian_mixture")
        if e.user_initialization in ("uniform_interior", "gaussian_mixture"):
            margin = e.user_initial_margin_fraction
            if not 0 <= margin < 0.5:
                raise ValueError("UE initialization margin must be in [0,0.5)")
        if e.user_initialization == "gaussian_mixture":
            if e.user_cluster_std_fraction <= 0:
                raise ValueError("UE mixture dispersion must be positive")
            if not e.user_cluster_centers or any(
                len(center) != 2 or any(not margin <= v <= 1 - margin for v in center)
                for center in e.user_cluster_centers
            ):
                raise ValueError(
                    "UE mixture centers must be normalized xy inside the initialization margin"
                )
        if e.backend not in ("numpy", "numba") or r.vector_backend not in (
            "batch",
            "process",
        ):
            raise ValueError("Unknown environment/vector backend")
        if e.neighbor_slots != 3:
            raise ValueError("The paper observation fixes exactly three neighbor slots")
        if a.name not in ("sac", "mappo", "matd3"):
            raise ValueError("Unknown learner")
        if not isinstance(a.value_clipping, bool):
            raise ValueError("value_clipping must be a boolean")
        if (
            not 0 < a.discount <= 1
            or not 0 < a.polyak <= 1
            or not 0 <= a.gae_lambda <= 1
        ):
            raise ValueError("Invalid discount, Polyak coefficient or GAE lambda")
        if (
            a.replay_capacity < a.batch_size
            or a.warmup_steps < 0
            or a.log_std_min >= a.log_std_max
        ):
            raise ValueError("Invalid replay/warmup/log-std configuration")
        if a.initial_alpha <= 0 or a.td3_policy_delay < 1:
            raise ValueError("Invalid temperature/policy delay")
        if set(r.validation_seeds) & set(r.test_seeds):
            raise ValueError("Validation and test seeds must be disjoint")
        if not r.validation_seeds or not r.test_seeds:
            raise ValueError("Evaluation seed sets cannot be empty")
        if r.budget is not None and (
            r.hourly_price is None or r.hourly_price <= 0 or r.budget <= 0
        ):
            raise ValueError("A budget requires a positive hourly_price and budget")
        if r.max_wall_seconds is not None and r.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        # Reject NaN/inf even in fields that otherwise pass comparison checks.
        json.dumps(asdict(self), allow_nan=False)
        return self

    def as_dict(self):
        data = asdict(self)
        for role in ("actor", "critic"):
            data["algo"][f"{role}_lr"] = self.algo.learning_rate(role)
        return data

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()


def load_config(path: str | Path | None = None) -> Config:
    data = {} if path is None else yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(data)


def config_from_dict(data: dict) -> Config:
    """Explicit modes for new runs; preserve historical BS-launch configurations."""
    unknown = set(data) - {"env", "algo", "run"}
    if unknown:
        raise ValueError(f"Unknown config sections: {unknown}")
    env = dict(data.get("env", {}))
    if "uav_initialization" not in env and "launch_radius" in env:
        env["uav_initialization"] = "bs_launch"
    return Config(
        EnvConfig(**env),
        AlgoConfig(**data.get("algo", {})),
        RunConfig(**data.get("run", {})),
    ).validate()


def save_config(cfg: Config, path: Path):
    path.write_text(yaml.safe_dump(cfg.as_dict(), sort_keys=False))
