"""Full worst-visible replay I/O acceptance, bounded chunks, no training budget."""

import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import threading

import numpy as np
import psutil
import torch
from uav_research.config import load_config
from uav_research.replay import JointReplay
from uav_research.checkpoints import save_full, restore_replay
from uav_research.runtime import load_checkpoint, memory_stats
from uav_research.observations import OBSERVATION_VERSION, TensorObservation


def digests(replay):
    result = {
        k: hashlib.sha256(memoryview(v).cast("B")).hexdigest()
        for k, v in replay.arrays.items()
    }
    for name in ("users", "next_users"):
        h = hashlib.sha256()
        for array in getattr(replay, name):
            h.update(memoryview(array).cast("B"))
        result[name] = h.hexdigest()
    for name in ("offsets", "next_offsets"):
        result[name] = hashlib.sha256(
            memoryview(getattr(replay, name)).cast("B")
        ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=50000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config("configs/main.yaml")
    e = cfg.env
    proc = psutil.Process()
    monitoring = threading.Event()
    samples = []
    phase = ["fill"]

    def monitor():
        while not monitoring.wait(0.1):
            samples.append((time.time(), phase[0], proc.memory_info().rss))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    replay = JointReplay(args.capacity, (e.num_uavs, e.obs_dim), e.state_dim, 19)
    rng = np.random.default_rng(73)
    for array in replay.arrays.values():
        for start in range(0, len(array), 256):
            block = array[start : start + 256]
            if array.dtype == np.bool_:
                block[:] = rng.integers(0, 2, block.shape, dtype=np.int8)
            else:
                block[:] = rng.standard_normal(block.shape, dtype=np.float32)
    for i in range(replay.capacity):
        for users, offsets in [
            (replay.users, replay.offsets),
            (replay.next_users, replay.next_offsets),
        ]:
            users[i] = rng.standard_normal(
                (e.num_uavs * e.num_users, 4), dtype=np.float32
            )
            offsets[i] = np.arange(e.num_uavs + 1) * e.num_users
    replay.size, replay.cursor = replay.capacity, 12345 % replay.capacity
    expected = digests(replay)
    raw_bytes = replay.effective_bytes
    path = args.output / "full"
    phase[0] = "save"
    before = time.perf_counter()
    save_full(
        path, {"kind": "full", "observation_version": OBSERVATION_VERSION}, replay
    )
    save_seconds = time.perf_counter() - before
    expected_sample = replay.sample(256, torch.device("cpu"))
    size_bytes = sum(p.stat().st_size for p in path.iterdir())
    del replay, array, block, users, offsets
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    phase[0] = "restore"
    replay = JointReplay(args.capacity, (e.num_uavs, e.obs_dim), e.state_dim)
    before = time.perf_counter()
    loaded = load_checkpoint(path)
    restore_replay(path, replay, loaded["replay_metadata"])
    load_seconds = time.perf_counter() - before
    assert replay.cursor == 12345 % replay.capacity and replay.size == replay.capacity
    phase[0] = "verify"
    assert digests(replay) == expected
    actual = replay.sample(256, torch.device("cpu"))
    for key in actual:
        if isinstance(actual[key], TensorObservation):
            for field in ("fixed", "users", "lengths"):
                torch.testing.assert_close(
                    getattr(actual[key], field),
                    getattr(expected_sample[key], field),
                    rtol=0,
                    atol=0,
                )
        else:
            torch.testing.assert_close(
                actual[key], expected_sample[key], rtol=0, atol=0
            )
    monitoring.set()
    thread.join()
    peaks = {
        p: max(r for _, q, r in samples if p == q) / 2**30
        for p in set(q for _, q, _ in samples)
    }
    peak = memory_stats(torch.device("cpu"))["peak_rss_mib"] / 1024
    result = dict(
        capacity=replay.capacity,
        all_users_visible=True,
        replay_bytes=raw_bytes,
        checkpoint_bytes=size_bytes,
        save_seconds=save_seconds,
        load_seconds=load_seconds,
        all_arrays_sha256_match=True,
        next_sample_exact=True,
        phase_peak_rss_gib=peaks,
        peak_rss_gib=peak,
        rss_limit_gib=12,
        passed=peak <= 12,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    (args.output / "rss_samples.json").write_text(json.dumps(samples))
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], "RSS exceeded 12 GiB; optimize before formal training"
    shutil.rmtree(path)  # Only this script's generated, verified synthetic checkpoint.


if __name__ == "__main__":
    main()
