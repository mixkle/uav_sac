from __future__ import annotations
import numpy as np
import torch
from .observations import RawObservation


class JointReplay:
    """One ring slot per aligned joint transition; variable UE arrays per slot."""

    def __init__(self, capacity, obs_shape, state_dim, seed=0):
        self.capacity, self.size, self.cursor = int(capacity), 0, 0
        self.rng = np.random.default_rng(seed)
        self.obs_shape = obs_shape
        shapes = dict(
            obs=obs_shape,
            next_obs=obs_shape,
            state=(state_dim,),
            next_state=(state_dim,),
            actions=(obs_shape[0], 2),
            reward=(),
            terminated=(),
        )
        self.arrays = {
            k: np.empty(
                (capacity, *shape), np.bool_ if k == "terminated" else np.float32
            )
            for k, shape in shapes.items()
        }
        self.users = [None] * capacity
        self.next_users = [None] * capacity
        self.offsets = np.zeros((capacity, obs_shape[0] + 1), np.int64)
        self.next_offsets = np.zeros_like(self.offsets)

    def add(self, obs, state, actions, reward, next_obs, next_state, terminated):
        n = len(reward)
        if n > self.capacity:
            raise ValueError("Collection batch exceeds replay capacity")
        indices = (self.cursor + np.arange(n)) % self.capacity
        for k, v in dict(
            obs=obs.fixed,
            state=state,
            actions=actions,
            reward=reward,
            next_obs=next_obs.fixed,
            next_state=next_state,
            terminated=terminated,
        ).items():
            self.arrays[k][indices] = v
        u = self.obs_shape[0]
        for j, index in enumerate(indices):
            for value, users, offsets in [
                (obs, self.users, self.offsets),
                (next_obs, self.next_users, self.next_offsets),
            ]:
                start, end = value.offsets[j * u : (j + 1) * u + 1][[0, -1]]
                users[index] = value.users[start:end].copy()
                offsets[index] = value.offsets[j * u : (j + 1) * u + 1] - start
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def observations(self, indices, next_=False):
        users, offsets, key = (
            (self.next_users, self.next_offsets, "next_obs")
            if next_
            else (self.users, self.offsets, "obs")
        )
        # Stored groups are already validated. Pack once instead of constructing
        # and validating one RawObservation per sampled transition.
        indices = np.asarray(indices, np.int64)
        lengths = np.diff(offsets[indices], axis=1).ravel()
        selected = [users[i] for i in indices]
        return RawObservation(
            self.arrays[key][indices],
            np.concatenate(selected) if selected else np.empty((0, 4), np.float32),
            np.r_[0, np.cumsum(lengths)],
        )

    def sample(self, batch_size, device):
        if self.size < batch_size:
            raise ValueError("Insufficient replay")
        indices = self.rng.integers(0, self.size, batch_size)
        out = {
            k: torch.from_numpy(a[indices]).to(device)
            for k, a in self.arrays.items()
            if k not in ("obs", "next_obs")
        }
        out["obs"] = self.observations(indices).to_torch(device)
        out["next_obs"] = self.observations(indices, True).to_torch(device)
        return out

    @property
    def allocated_bytes(self):
        return (
            sum(a.nbytes for a in self.arrays.values())
            + self.offsets.nbytes
            + self.next_offsets.nbytes
        )

    @property
    def effective_bytes(self):
        return self.allocated_bytes + sum(
            a.nbytes
            for items in (self.users, self.next_users)
            for a in items
            if a is not None
        )

    def metadata(self):
        return dict(
            capacity=self.capacity,
            size=self.size,
            cursor=self.cursor,
            rng=self.rng.bit_generator.state,
        )

    def chunks(self, count=256):
        for start in range(0, self.size, count):
            stop = min(start + count, self.size)
            ids = np.arange(start, stop)
            data = {k: torch.from_numpy(a[start:stop]) for k, a in self.arrays.items()}
            for label, next_ in (("current", False), ("next", True)):
                obs = self.observations(ids, next_)
                data[label + "_users"] = torch.from_numpy(obs.users)
                data[label + "_offsets"] = torch.from_numpy(obs.offsets)
            yield start, stop, data

    def restore_chunk(self, start, stop, data):
        for k, a in self.arrays.items():
            a[start:stop] = data[k].numpy()
        u = self.obs_shape[0]
        for label, users, offsets in (
            ("current", self.users, self.offsets),
            ("next", self.next_users, self.next_offsets),
        ):
            flat, off = data[label + "_users"].numpy(), data[label + "_offsets"].numpy()
            for j, i in enumerate(range(start, stop)):
                begin, end = off[j * u : (j + 1) * u + 1][[0, -1]]
                users[i] = flat[begin:end].copy()
                offsets[i] = off[j * u : (j + 1) * u + 1] - begin

    def restore_metadata(self, data):
        if data["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed during resume")
        self.size, self.cursor = data["size"], data["cursor"]
        self.rng.bit_generator.state = data["rng"]
