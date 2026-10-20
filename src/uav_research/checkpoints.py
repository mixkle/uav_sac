"""Atomic directory publication; bounded-memory replay save/load."""

import json
import os
from pathlib import Path
import shutil
import hashlib
import torch
from .observations import OBSERVATION_VERSION


def save_full(path, data, replay, max_rss_bytes=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        return path
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        chunks = []
        if replay is not None:
            for start, stop, chunk in replay.chunks():
                name = f"replay_{start:06d}.pt"
                file = temporary / name
                if max_rss_bytes is not None:
                    import psutil

                    if psutil.Process().memory_info().rss > max_rss_bytes:
                        raise MemoryError(
                            "RSS budget exceeded while saving replay; previous checkpoint preserved"
                        )
                torch.save(chunk, file)
                digest = hashlib.sha256()
                with file.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                chunks.append(
                    dict(name=name, start=start, stop=stop, sha256=digest.hexdigest())
                )
        torch.save(
            {
                **data,
                "observation_version": OBSERVATION_VERSION,
                "replay_metadata": replay.metadata() if replay is not None else None,
            },
            temporary / "metadata.pt",
        )
        (temporary / "index.json").write_text(json.dumps(chunks))
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path


def restore_replay(path, replay, metadata):
    if replay is None:
        return
    replay.restore_metadata(metadata)
    expected_start = 0
    for chunk in json.loads((Path(path) / "index.json").read_text()):
        if (
            chunk["start"] != expected_start
            or not chunk["start"] < chunk["stop"] <= replay.size
        ):
            raise ValueError("Invalid replay chunk index")
        file = Path(path) / chunk["name"]
        digest = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != chunk["sha256"]:
            raise ValueError("Replay checkpoint checksum mismatch")
        data = torch.load(file, map_location="cpu", weights_only=False)
        replay.restore_chunk(chunk["start"], chunk["stop"], data)
        del data
        expected_start = chunk["stop"]
    if expected_start != replay.size:
        raise ValueError("Incomplete replay checkpoint")
