"""Atomic artifacts, explicit experiment contracts, bounded-resource guards."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import time
import numpy as np
import psutil
import torch

from ..runtime import atomic_json
from ..topk_diagnostic import file_hash
from .config import from_dict, contract
from .observations import OBSERVATION_VERSION
from . import VERSION


def configure(cfg):
    from ..runtime import configure_runtime
    device = configure_runtime(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def restore_online_rng(data):
    """Unused replica GPUs do not constrain subsequent inference/CPU recovery.

    Sampling and randperm always use logical GPU0 (run.device); saved secondary
    GPU states are retained when visible. Replica forward evaluation draws no
    randomness. This permits a dual-card run to resume on the single-card path.
    """
    from ..runtime import restore_rng
    copied = dict(data)
    if copied.get('cuda') is not None and torch.cuda.is_available():
        copied['cuda'] = copied['cuda'][:torch.cuda.device_count()]
    restore_rng(copied)


def ensure_budget(cfg, output, estimated_bytes=0):
    output = Path(output)
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    needed = estimated_bytes + cfg.diffusion.minimum_free_gib*2**30
    if free < needed:
        raise OSError(f"Insufficient disk: free={free/2**30:.2f}GiB, required={needed/2**30:.2f}GiB")
    rss = psutil.Process().memory_info().rss
    if rss > cfg.run.memory_limit_gib*2**30:
        raise MemoryError(f"RSS {rss/2**30:.2f}GiB exceeds configured limit")
    if (str(cfg.run.device).startswith("cuda") or cfg.run.device == "auto") and torch.cuda.is_available():
        selected = cfg.run.device if str(cfg.run.device).startswith("cuda") else None
        devices = cfg.diffusion.actor_update_devices or [selected]
        for dev in devices:
            if torch.cuda.memory_allocated(dev) > cfg.diffusion.gpu_limit_gib*2**30:
                raise MemoryError(f"Configured CUDA memory limit exceeded on {dev}")


@contextmanager
def atomic_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.mkdir(exist_ok=False)
    try:
        yield temporary
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def observation_version(cfg):
    return ("service-set5-executed-history-v2" if cfg.env.history_motion_source == "post_reflection_displacement"
            else OBSERVATION_VERSION)


def artifact(cfg, phase, **data):
    return dict(kind="dppo_"+phase, version=VERSION, observation_version=observation_version(cfg),
                action_specification=dict(movement="unit-disk-v1", scores="sigmoid-candidate-logits-v1",
                    association="pre-move-topk-post-move-gate-v1", distribution=("gaussian-service-latents-v1" if cfg.algo.name == "gaussian_service_mappo" else "ddpm-gaussian-transitions-v1")),
                config=cfg.as_dict(), contract=contract(cfg), **data)


def load(path, cfg=None, kinds=None):
    path = Path(path)
    if path.is_dir():
        import json
        index = json.loads((path/"index.json").read_text())
        if file_hash(path/"metadata.pt") != index["metadata_sha256"]:
            raise ValueError("Complete checkpoint checksum mismatch")
        path = path/"metadata.pt"
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if data.get("version") != VERSION or data.get("observation_version") != observation_version(from_dict(data["config"])):
        raise ValueError("Not a compatible diffusion DPPO artifact")
    if data.get("contract") != contract(from_dict(data["config"])):
        raise ValueError("Artifact configuration and saved experiment contract disagree")
    saved_cfg = from_dict(data["config"])
    expected_distribution = ("gaussian-service-latents-v1" if saved_cfg.algo.name == "gaussian_service_mappo" else "ddpm-gaussian-transitions-v1")
    if data.get("action_specification", {}).get("distribution") != expected_distribution:
        raise ValueError("Artifact policy distribution does not match config")
    if kinds and data.get("kind") not in kinds:
        raise ValueError(f"Wrong DPPO artifact phase: {data.get('kind')}")
    if cfg is not None and data.get("contract") != contract(cfg):
        raise ValueError("Cannot resume a different DPPO experiment")
    return data


def save_online_full(output, cfg, data):
    path = Path(output)/"checkpoints"/f"full_{data['env_steps']:09d}"
    ensure_budget(cfg, output)
    if not path.exists():
        with atomic_directory(path) as temporary:
            torch.save(data, temporary/"metadata.pt")
            atomic_json(temporary/"index.json", {"metadata_sha256": file_hash(temporary/"metadata.pt")})
    saved = sorted(path.parent.glob("full_[0-9]*"))
    for old in saved[:-2]:
        if old.is_dir() and (old/"metadata.pt").exists():
            shutil.rmtree(old)
    return path


def save_arrays(path, arrays, meta):
    with atomic_directory(path) as temporary:
        for name, value in arrays.items():
            np.save(temporary/(name+".npy"), value, allow_pickle=False)
        atomic_json(temporary/"metadata.json", dict(meta, files={
            x.name: file_hash(x) for x in sorted(temporary.glob("*.npy"))}))


class Progress:
    def __init__(self, seconds=30):
        self.seconds, self.last = seconds, 0.

    def __call__(self, message, force=False):
        now = time.monotonic()
        if force or now-self.last >= self.seconds:
            print(time.strftime("%H:%M:%S"), message, flush=True)
            self.last = now
