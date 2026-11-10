"""Configuration for the two-stage experiment; historical Config stays intact."""
from dataclasses import dataclass, field, asdict, replace
import math

from ..config import Config, EnvConfig, AlgoConfig, RunConfig


@dataclass
class DiffusionSettings:
    version: str = "set-dppo-topk-v1"
    association: str = "topk_score"
    steps: int = 8
    cosine_offset: float = 0.008
    max_beta: float = 0.999
    std_floor: float = 0.05
    ue_width: int = 64
    context_width: int = 128
    time_width: int = 32
    pair_width: int = 64
    value_clip: float = 0.2
    microbatch_agents: int = 32
    # Runtime-only data parallelism for actor PPO updates. Sampling and the
    # single team critic stay on run.device; [] preserves the historical path.
    actor_update_devices: list[str] = field(default_factory=list)
    teacher_checkpoint: str = "runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt"
    teacher_step: int = 671744
    teacher_action_seed: int = 980001
    teacher_score_scale_db: float = 10.0
    fit_seeds: list[int] = field(default_factory=lambda: list(range(300000, 300128)))
    selection_seeds: list[int] = field(default_factory=lambda: list(range(301000, 301032)))
    pretrain_epochs: int = 50
    pretrain_batch: int = 1024
    pretrain_lr: float = 3e-4
    pretrain_seed: int = 0
    selection_noise_seed: int = 981001
    pretrain_mov_weight: float = 1.0
    pretrain_score_weight: float = 1.0
    pretrained_checkpoint: str = "runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain/best.pt"
    minimum_free_gib: float = 15.0
    gpu_limit_gib: float = 20.0
    heartbeat_seconds: float = 30.0


@dataclass
class DppoConfig(Config):
    diffusion: DiffusionSettings = field(default_factory=DiffusionSettings)

    def validate(self):
        # Reuse all existing physical/config checks without enabling this
        # algorithm in the legacy Gaussian learner or observation interface.
        Config(self.env, replace(self.algo, name="mappo"), self.run).validate()
        d, e, a = self.diffusion, self.env, self.algo
        if a.name not in ("diffusion_dppo", "gaussian_service_mappo") or d.version != "set-dppo-topk-v1":
            raise ValueError("Expected diffusion_dppo / set-dppo-topk-v1")
        if a.name == "gaussian_service_mappo" and d.actor_update_devices:
            raise ValueError("Gaussian service MAPPO uses one GPU")
        if d.association != "topk_score" or e.action_space != "continuous":
            raise ValueError("Diffusion DPPO requires continuous topk_score")
        if (a.actor_sharing != "shared" or a.critic_parameterization != "time_factor"
                or a.entropy_coef != 0 or a.mappo_state_dependent_std):
            raise ValueError("DPPO requires shared diffusion actor, time_factor and entropy_coef=0")
        if e.reward_primary != "coverage" or any((e.robustness_weight,
                e.disconnection_weight, e.boundary_penalty_weight)):
            raise ValueError("This experiment supports coverage + throughput only")
        if e.observation_history != "one_step" or e.observation_coordinates != "relative":
            raise ValueError("DPPO requires relative observations and one_step history")
        for name in ("steps", "ue_width", "context_width", "time_width", "pair_width",
                     "microbatch_agents", "pretrain_epochs", "pretrain_batch"):
            x = getattr(d, name)
            if not isinstance(x, int) or isinstance(x, bool) or x <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d.time_width % 2 or d.steps < 2:
            raise ValueError("time_width must be even, steps >= 2")
        devices = d.actor_update_devices
        if (not isinstance(devices, list) or any(not isinstance(x, str) or
                not x.startswith("cuda:") or not x[5:].isdigit() for x in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError("actor_update_devices must contain unique explicit cuda:N devices")
        if devices and devices[0] != self.run.device:
            raise ValueError("First actor update device must equal run.device")
        if not (0 < d.max_beta < 1 and d.cosine_offset >= 0 and 0 < d.std_floor < 1):
            raise ValueError("Invalid diffusion schedule")
        for name in ("value_clip", "teacher_score_scale_db", "pretrain_lr",
                     "pretrain_mov_weight", "pretrain_score_weight", "gpu_limit_gib",
                     "heartbeat_seconds"):
            if not math.isfinite(getattr(d, name)) or getattr(d, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(d.minimum_free_gib) or d.minimum_free_gib < 0:
            raise ValueError("minimum_free_gib must be nonnegative")
        sets = [d.fit_seeds, d.selection_seeds, self.run.validation_seeds, self.run.test_seeds]
        if any(not s or len(s) != len(set(s)) or any(type(x) is not int or x < 0 for x in s) for s in sets):
            raise ValueError("Scene sets must be nonempty, unique nonnegative integers")
        if any(set(sets[i]) & set(sets[j]) for i in range(4) for j in range(i)):
            raise ValueError("Demonstration, selection, validation and test scenes overlap")
        if self.run.total_steps % self.run.num_envs:
            raise ValueError("Joint transition budget must be divisible by num_envs")
        if self.run.evaluation_policy != "sample":
            raise ValueError("DPPO validation uses the sampled policy; deterministic execution is auxiliary only")
        for key in ("teacher_step", "teacher_action_seed", "pretrain_seed", "selection_noise_seed"):
            if type(getattr(d, key)) is not int or getattr(d, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 0 < a.ppo_clip < 1:
            raise ValueError("PPO clipping must lie in (0,1)")
        return self


def from_dict(data):
    if set(data) - {"env", "algo", "run", "diffusion"}:
        raise ValueError("Unknown DPPO config section")
    return DppoConfig(EnvConfig(**data.get("env", {})),
                      AlgoConfig(**data.get("algo", {})), RunConfig(**data.get("run", {})),
                      DiffusionSettings(**data.get("diffusion", {}))).validate()


def baseline_differences(cfg, baseline):
    """All physical parameters must match the selected capacity-10 baseline."""
    return {k: [v, asdict(cfg.env)[k]] for k, v in asdict(baseline.env).items()
            if v != asdict(cfg.env)[k]}


def contract(cfg):
    """Strict online resume/deployment semantics, excluding runtime/budget paths."""
    d = asdict(cfg.diffusion)
    for key in ("microbatch_agents", "actor_update_devices", "teacher_checkpoint", "pretrained_checkpoint",
                "minimum_free_gib", "gpu_limit_gib", "heartbeat_seconds"):
        d.pop(key)
    run_semantics = {k: getattr(cfg.run, k) for k in ("seed", "num_envs", "eval_interval",
        "validation_seeds", "test_seeds", "evaluation_policy", "evaluation_action_seed",
        "best_metric", "deterministic")}
    env = asdict(cfg.env)
    # Omit newly added legacy defaults to preserve hashes/contracts of old runs.
    if env["motion_scaling"] == dict(enabled=False, min_ratio=.25, hop_growth=.4):
        env.pop("motion_scaling")
    if env["history_motion_source"] == "command":
        env.pop("history_motion_source")
    return dict(env=env, algo=cfg.as_dict()["algo"], diffusion=d, run=run_semantics)
