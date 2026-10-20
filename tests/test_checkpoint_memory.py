import json
import numpy as np
import pytest
import torch
from uav_research.replay import JointReplay
from uav_research.runtime import atomic_save, load_checkpoint
from uav_research.checkpoints import save_full, restore_replay
from uav_research.observations import RawObservation


@pytest.mark.parametrize("size", [7, 20])
def test_chunked_checkpoint_shares_dense_storage_and_restores(tmp_path, size):
    replay = JointReplay(20, (2, 3), 5, 31)
    replay.size, replay.cursor = size, size % 20
    for array in replay.arrays.values():
        array[:] = np.arange(array.size).reshape(array.shape) % 3
    for i in range(size):
        replay.users[i] = np.ones((i % 4, 4), np.float32) * i
        replay.next_users[i] = np.zeros((0, 4), np.float32)
        replay.offsets[i] = [0, i % 4, i % 4]
    first = next(replay.chunks(4))[2]
    assert first["state"].data_ptr() == replay.arrays["state"].ctypes.data
    path = tmp_path / "full"
    save_full(path, {}, replay)
    other = JointReplay(20, (2, 3), 5)
    restore_replay(path, other, load_checkpoint(path)["replay_metadata"])
    for key in replay.arrays:
        np.testing.assert_array_equal(
            replay.arrays[key][:size], other.arrays[key][:size]
        )
    a, b = replay.sample(5, torch.device("cpu")), other.sample(5, torch.device("cpu"))
    for key in a:
        if key in ("obs", "next_obs"):
            for field in ("fixed", "users", "lengths"):
                torch.testing.assert_close(
                    getattr(a[key], field), getattr(b[key], field), rtol=0, atol=0
                )
        else:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    index = json.loads((path / "index.json").read_text())
    file = path / index[0]["name"]
    file.write_bytes(file.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        restore_replay(path, other, load_checkpoint(path)["replay_metadata"])


def test_failed_atomic_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    atomic_save(path, {"step": 12})

    def fail(data, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated"):
        atomic_save(path, {"step": 24})
    assert load_checkpoint(path)["step"] == 12
    assert not path.with_suffix(".pt.tmp").exists()


def test_failed_chunk_save_retains_previous_directory(tmp_path, monkeypatch):
    old = tmp_path / "old"
    save_full(old, {"step": 12}, None)

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_full(tmp_path / "new", {}, None)
    assert load_checkpoint(old)["step"] == 12
    assert not (tmp_path / "new").exists() and not (tmp_path / "new.tmp").exists()


def test_raw_replay_copies_and_aligns_users_after_wrap(tmp_path):
    replay = JointReplay(3, (2, 1), 1, 7)
    for step in range(5):
        obs = RawObservation(
            np.ones((1, 2, 1), np.float32) * step,
            np.ones((step, 4), np.float32) * step,
            np.array([0, step, step]),
        )
        replay.add(
            obs,
            np.array([[step]], np.float32),
            np.zeros((1, 2, 2), np.float32),
            np.array([step]),
            obs,
            np.array([[step]], np.float32),
            np.zeros(1, bool),
        )
        obs.users.fill(-99)
    for i in range(3):
        value = int(replay.arrays["reward"][i])
        assert len(replay.users[i]) == value
        assert (replay.users[i] == value).all()
    obs = replay.observations([0, 0, 1])
    assert obs.fixed.shape == (3, 2, 1)
    assert len(obs.offsets) == 7


def test_old_model_checkpoint_explicitly_rejected(tmp_path):
    path = tmp_path / "old.pt"
    atomic_save(path, {"kind": "model"})
    with pytest.raises(ValueError, match="Old observation"):
        load_checkpoint(path)
