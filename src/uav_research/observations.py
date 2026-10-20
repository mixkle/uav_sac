"""Raw observations: fixed features and unpadded, UAV-grouped UE rows."""

from dataclasses import dataclass
import numpy as np
import torch

OBSERVATION_VERSION = "beacon-set-v2"


@dataclass
class TensorObservation:
    fixed: torch.Tensor
    users: torch.Tensor
    lengths: torch.Tensor

    def to(self, device, non_blocking=False):
        return TensorObservation(
            *(
                x.to(device, non_blocking=non_blocking)
                for x in (self.fixed, self.users, self.lengths)
            )
        )

    def pin_memory(self):
        return TensorObservation(
            *(x.pin_memory() for x in (self.fixed, self.users, self.lengths))
        )


@dataclass
class RawObservation:
    fixed: np.ndarray
    users: np.ndarray
    offsets: np.ndarray

    def __post_init__(self):
        groups = int(np.prod(self.fixed.shape[:-1]))
        if self.users.ndim != 2 or self.users.shape[1] != 4:
            raise ValueError("UE rows must have shape [L,4]")
        if (
            self.offsets.shape != (groups + 1,)
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.users)
        ):
            raise ValueError("Invalid observation group offsets")
        if (np.diff(self.offsets) < 0).any():
            raise ValueError("Offsets must be nondecreasing")

    def copy(self):
        return RawObservation(self.fixed.copy(), self.users.copy(), self.offsets.copy())

    def flatten_batch(self):
        return RawObservation(
            self.fixed.reshape(-1, *self.fixed.shape[-2:]), self.users, self.offsets
        )

    def __len__(self):
        return len(self.fixed)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if self.fixed.ndim < 3:
            raise ValueError(
                "Select environments/transitions, not individual UAV groups"
            )
        flat = self.flatten_batch()
        indices = np.arange(len(flat.fixed))[index]
        single = np.ndim(indices) == 0
        indices = np.atleast_1d(indices)
        u = flat.fixed.shape[1]
        arrays, lengths = [], []
        for i in indices:
            begin, end = flat.offsets[int(i) * u : (int(i) + 1) * u + 1][[0, -1]]
            arrays.append(flat.users[begin:end])
            lengths.extend(np.diff(flat.offsets[int(i) * u : (int(i) + 1) * u + 1]))
        fixed = flat.fixed[indices]
        if single:
            fixed = fixed[0]
        return RawObservation(
            fixed,
            np.concatenate(arrays) if arrays else np.empty((0, 4), np.float32),
            np.r_[0, np.cumsum(lengths)].astype(np.int64),
        )

    def __setitem__(self, index, value):
        items = list(self)
        items[index] = value
        rebuilt = stack_observations(items)
        self.fixed, self.users, self.offsets = (
            rebuilt.fixed,
            rebuilt.users,
            rebuilt.offsets,
        )

    def to_torch(self, device):
        return TensorObservation(
            torch.as_tensor(self.fixed, device=device),
            torch.as_tensor(self.users, device=device),
            torch.as_tensor(np.diff(self.offsets), device=device),
        )

    @property
    def nbytes(self):
        return self.fixed.nbytes + self.users.nbytes + self.offsets.nbytes


def stack_observations(items):
    items = list(items)
    if not items:
        raise ValueError("Cannot stack no observations")
    return RawObservation(
        np.stack([x.fixed for x in items]),
        np.concatenate([x.users for x in items]),
        np.r_[0, np.cumsum(np.concatenate([np.diff(x.offsets) for x in items]))].astype(
            np.int64
        ),
    )


def as_tensor_observation(obs, device):
    if isinstance(obs, RawObservation):
        return obs.to_torch(device)
    if isinstance(obs, TensorObservation):
        return obs.to(device)
    raise ValueError("Actor requires beacon-set-v2 structured raw observations")
