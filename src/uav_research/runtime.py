from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import importlib.metadata
import numpy as np
import torch


def configure_runtime(cfg):
    torch.set_num_threads(cfg.run.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import numba

        numba.set_num_threads(
            min(cfg.run.numba_threads, numba.config.NUMBA_NUM_THREADS)
        )
    except ImportError:
        pass
    torch.use_deterministic_algorithms(cfg.run.deterministic)
    if cfg.run.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not cfg.run.deterministic
    device = cfg.run.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return torch.device(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(data):
    random.setstate(data["python"])
    np.random.set_state(data["numpy"])
    torch.set_rng_state(data["torch"].cpu())
    if data["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda"]])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False))
    os.replace(tmp, path)


def atomic_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        # Preserve the previous valid file if serialization/rename failed.
        tmp.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu", *, mmap=False):
    # Checkpoints contain NumPy/RNG data. Load only locally generated trusted files.
    from .observations import OBSERVATION_VERSION

    target = Path(path)
    if target.is_dir():
        target = target / "metadata.pt"
    data = torch.load(target, map_location=device, weights_only=False, mmap=mmap)
    if (
        data.get("kind") in ("model", "full", "actor")
        and data.get("observation_version") != OBSERVATION_VERSION
    ):
        raise ValueError(
            "Old observation checkpoint is incompatible; beacon-set-v2 required"
        )
    return data


def resolve_output(path):
    """Relative artifacts use the data disk on AutoDL; absolute paths stay explicit."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    root = os.environ.get("UAV_DATA_ROOT")
    if root is None:
        try:
            if Path("/root/autodl-tmp").is_dir():
                root = "/root/autodl-tmp/uav_sac"
        except PermissionError:
            pass
    return Path(root).expanduser() / path if root else path


def validate_resume_output(output, restored):
    """Never append a rollback branch to a directory containing its future."""
    step = restored["env_steps"]
    manifest = output / "manifest.json"
    later = (
        manifest.exists()
        and json.loads(manifest.read_text()).get("env_steps", 0) > step
    )
    for name in ("metrics.csv", "validation.csv"):
        path = output / name
        if path.exists():
            with path.open() as file:
                later = later or any(
                    int(row["env_steps"]) > step for row in csv.DictReader(file)
                )
    for name in ("best.pt", "last.pt"):
        path = output / name
        if path.exists():
            later = later or load_checkpoint(path)["env_steps"] > step
    if later:
        raise ValueError(
            "Output contains steps after this checkpoint; resume into a new output directory"
        )


def source_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def container_limits():
    """Host CPU/RAM totals can exceed the resources allocated to a container."""
    limits = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            value = path.read_text().strip()
            if name == "cpu.max":
                quota, period = value.split()
                limits["cpu_quota_cores"] = (
                    None if quota == "max" else int(quota) / int(period)
                )
            else:
                limits["memory_limit_bytes"] = None if value == "max" else int(value)
    if hasattr(os, "sched_getaffinity"):
        limits["affinity_cpus"] = sorted(os.sched_getaffinity(0))
    return limits


def metadata(cfg, device):
    versions = {}
    for package in (
        "numpy",
        "numba",
        "torch",
        "PyYAML",
        "matplotlib",
        "tensorboard",
        "pytest",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    worker_seeds = [
        int(seed.generate_state(1)[0])
        for seed in np.random.SeedSequence([cfg.run.seed, 101]).spawn(cfg.run.num_envs)
    ]
    return dict(
        config_hash=cfg.digest(),
        source_hash=source_digest(),
        python=sys.version,
        platform=platform.platform(),
        cpu=platform.processor(),
        logical_cpus=os.cpu_count(),
        container_limits=container_limits(),
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        dependencies=versions,
        command=sys.argv,
        seed=cfg.run.seed,
        random_streams=dict(
            worker_initialization_seeds=worker_seeds,
            worker_movement_seed_sequences=[[seed, 201] for seed in worker_seeds],
            replay_seed=cfg.run.seed + 300000,
            exploration_seed_sequence=[cfg.run.seed, 102],
            validation_seeds=cfg.run.validation_seeds,
            test_seeds=cfg.run.test_seeds,
        ),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def memory_stats(device):
    # VmHWM is peak RSS in KiB on Linux, including WSL2.
    stats = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                stats["peak_rss_mib"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    child_peaks = 0.0
    try:
        children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
        for child in children:
            for line in Path(f"/proc/{child}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    child_peaks += int(line.split()[1]) / 1024
    except OSError:
        pass
    stats["sum_process_peak_rss_mib"] = stats.get("peak_rss_mib", 0) + child_peaks
    if device.type == "cuda":
        stats["peak_gpu_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return stats


METRIC_FIELDS = [
    "reward_primary",
    "user_speed_mean",
    "user_displacement_mean",
    "uav_nearest_neighbor_mean",
    "uav_close_neighbor_fraction",
    "discovered_users",
    "discovery_fraction",
    "cumulative_discovery_fraction",
    "discovered_unserved_users",
    "sensing_rejected_users",
    "downlink_rejected_users",
    "backhaul_rejected_users",
    "capacity_rejected_users",
    "first_service_step",
    "zero_coverage_steps",
    "expansion_distance_mean",
    "actual_speed_mean",
    "observed_count_mean",
    "observed_count_max",
    "encoder_grad_norm",
    "replay_effective_bytes",
    "env_steps",
    "updates",
    "elapsed_seconds",
    "training_seconds",
    "transitions_per_second",
    "coverage",
    "throughput_bps",
    "tau",
    "robustness",
    "retention_ratio",
    "reward",
    "worst_fault_coverage",
    "component_services",
    "disconnected_uavs",
    "disconnection_ratio",
    "disconnection_penalty",
    "mean_hops",
    "capacity_utilization",
    "component_count",
    "gateway_switch_coverage",
    "reachable_all_gateways",
    "reachable_none",
    "candidate_count_mean",
    "candidate_count_max",
    "sinr_db_p10",
    "sinr_db_p50",
    "sinr_db_p90",
    "action_norm_mean",
    "boundary_clip_fraction",
    "boundary_reflection_fraction",
    "q_loss",
    "actor_loss",
    "value_loss",
    "q_grad_norm",
    "actor_grad_norm",
    "alpha",
    "joint_entropy",
    "q_mean",
    "target_mean",
    "td_abs",
    "policy_std",
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "value_return_mean",
    "sample_seconds",
    "transfer_seconds",
    "environment_seconds",
    "inference_seconds",
    "update_seconds",
    "evaluation_seconds",
    "io_seconds",
    "peak_rss_mib",
    "sum_process_peak_rss_mib",
    "peak_gpu_mib",
    "estimated_cost",
]


class CSVLogger:
    def __init__(self, path, fields):
        path = Path(path)
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open() as file:
                if next(csv.reader(file)) != fields:
                    raise ValueError(f"CSV schema changed: {path}")
        self.file = path.open("a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields, extrasaction="raise")
        if not exists:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
