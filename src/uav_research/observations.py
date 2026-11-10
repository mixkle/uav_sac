"""Raw observations: fixed features and unpadded, UAV-grouped UE rows."""

from dataclasses import dataclass
import numpy as np
import torch

OBSERVATION_VERSION = "beacon-set-v2"


def history_specification(cfg):
    """Extra fixed features; absence in historical artifacts means no history."""
    if cfg.observation_history == "none":
        return {"mode": "none"}
    if cfg.history_motion_source == "post_reflection_displacement":
        return dict(mode="one_step", version=2,
            fields=["previous_actual_dx", "previous_actual_dy", "previous_hop", "previous_sensed_count"],
            action="post_reflection_displacement", motion_scale="max_speed*dt",
            hop_scale=cfg.num_uavs+1, count_scale=cfg.capacity)
    return dict(mode="one_step", version=1,
                fields=["previous_command_x", "previous_command_y", "previous_hop", "previous_sensed_count"],
                action="normalized_command_before_boundary", hop_scale=cfg.num_uavs + 1,
                count_scale=cfg.capacity)


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

    def split_agents(self):
        """Preserve batch axes and one UAV axis; UE rows remain UAV-local.

        The row ownership map is built once for the whole actor bank, not once
        per user or per actor. This also keeps zero-length groups explicit.
        """
        if self.fixed.ndim < 2:
            raise ValueError("Expected [...,U,F] fixed observation")
        u = self.fixed.shape[-2]
        lengths = self.lengths.reshape(-1, u)
        owner = torch.repeat_interleave(
            torch.arange(u, device=self.lengths.device).repeat(len(lengths)),
            self.lengths,
            output_size=len(self.users),
        )
        return [
            TensorObservation(self.fixed[..., i:i+1, :], self.users[owner == i],
                              lengths[:, i].contiguous())
            for i in range(u)
        ]


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
