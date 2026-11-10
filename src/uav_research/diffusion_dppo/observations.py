"""Ragged local sets with explicit routing IDs, never supplied as network inputs."""
from dataclasses import dataclass
import numpy as np
import torch

from ..observations import RawObservation

OBSERVATION_VERSION = "service-set5-history-v1"


def offsets(lengths):
    return np.r_[0, np.cumsum(lengths)].astype(np.int64)


@dataclass
class ServiceObservation:
    fixed: np.ndarray
    users: np.ndarray
    offsets: np.ndarray
    ue_ids: np.ndarray
    service_mask: np.ndarray

    def __post_init__(self):
        n = int(np.prod(self.fixed.shape[:-1]))
        if (self.users.ndim != 2 or self.users.shape[1] != 5 or
                self.offsets.shape != (n + 1,) or self.offsets[0] != 0 or
                self.offsets[-1] != len(self.users) or np.any(self.offsets[1:] < self.offsets[:-1])
                or self.ue_ids.shape != (len(self.users),)
                or self.service_mask.shape != (len(self.users),)):
            raise ValueError("Invalid five-feature service observation/offsets")
        if (self.offsets.dtype.kind not in "iu" or self.ue_ids.dtype.kind not in "iu"
                or self.service_mask.dtype != np.bool_):
            raise ValueError("Offsets/UE IDs must be integers; service mask must be boolean")

    @property
    def groups(self):
        return len(self.offsets) - 1

    @property
    def candidate_lengths(self):
        prefix = np.r_[0, np.cumsum(self.service_mask, dtype=np.int64)]
        return np.diff(prefix[self.offsets])

    @property
    def candidate_ids(self):
        return self.ue_ids[self.service_mask]

    def copy(self):
        return ServiceObservation(*(x.copy() for x in (self.fixed, self.users,
                    self.offsets, self.ue_ids, self.service_mask)))

    def take(self, indices):
        """Select flattened UAV samples, preserving duplicates and empty groups."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        rows = [np.arange(self.offsets[i], self.offsets[i+1]) for i in indices]
        rows = np.concatenate(rows) if rows else np.empty(0, np.int64)
        return ServiceObservation(self.fixed.reshape(self.groups, -1)[indices].copy(),
            self.users[rows].copy(), offsets(np.diff(self.offsets)[indices]),
            self.ue_ids[rows].copy(), self.service_mask[rows].copy())

    def legacy(self):
        return RawObservation(self.fixed.copy(), self.users[:, [0, 1, 2, 4]].copy(),
                              self.offsets.copy())

    def tensor(self, device):
        device = torch.device(device)
        def tensor(array):
            x = torch.as_tensor(array).contiguous()
            if device.type == "cuda":
                return x.pin_memory().to(device, non_blocking=True)
            return x.to(device)
        return TensorServiceObservation(
            tensor(self.fixed.reshape(self.groups, -1)), tensor(self.users),
            tensor(np.diff(self.offsets)), tensor(np.flatnonzero(self.service_mask)),
            tensor(self.candidate_lengths))


@dataclass
class TensorServiceObservation:
    fixed: torch.Tensor
    users: torch.Tensor
    lengths: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_lengths: torch.Tensor


def stack(items):
    items = list(items)
    if not items:
        raise ValueError("Cannot stack empty service observations")
    return ServiceObservation(np.stack([x.fixed for x in items]),
        np.concatenate([x.users for x in items]),
        offsets(np.concatenate([np.diff(x.offsets) for x in items])),
        np.concatenate([x.ue_ids for x in items]),
        np.concatenate([x.service_mask for x in items]))


@dataclass
class ServiceAction:
    movement: np.ndarray
    scores: np.ndarray
    ue_ids: np.ndarray
    offsets: np.ndarray

    def split(self, u):
        movement = self.movement.reshape(-1, u, 2)
        return [ServiceAction(movement[e].copy(),
                self.scores[self.offsets[e*u]:self.offsets[(e+1)*u]].copy(),
                self.ue_ids[self.offsets[e*u]:self.offsets[(e+1)*u]].copy(),
                self.offsets[e*u:(e+1)*u+1].copy() - self.offsets[e*u])
                for e in range(len(movement))]


@dataclass
class DiffusionChain:
    movement: np.ndarray  # [N,L+1,2], order z_L,...,z_0
    scores: np.ndarray    # [total candidates,L+1]
    candidate_offsets: np.ndarray
    logp: np.ndarray      # [N,L], each is a coordinate SUM
    kind: str = "diffusion"

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        rows = [np.arange(self.candidate_offsets[i], self.candidate_offsets[i+1])
                for i in indices]
        rows = np.concatenate(rows) if rows else np.empty(0, np.int64)
        return DiffusionChain(self.movement[indices], self.scores[rows],
            offsets(np.diff(self.candidate_offsets)[indices]), self.logp[indices], self.kind)


def stack_chains(chains):
    return DiffusionChain(np.concatenate([x.movement for x in chains]),
        np.concatenate([x.scores for x in chains]),
        offsets(np.concatenate([np.diff(x.candidate_offsets) for x in chains])),
        np.concatenate([x.logp for x in chains]), chains[0].kind)


def chain_diagnostics(chain, u):
    """Per-environment distribution diagnostics, not policy entropy estimates."""
    rows = []
    for begin in range(0, len(chain.movement), u):
        end = begin+u
        scores = chain.scores[chain.candidate_offsets[begin]:chain.candidate_offsets[end], -1]
        x = chain.movement[begin:end, -1]
        rows.append(dict(movement_latent_norm_mean=float(np.linalg.norm(x, axis=-1).mean()),
            movement_latent_std=float(x.std()),
            score_latent_mean=float(scores.mean()) if len(scores) else 0.,
            score_latent_std=float(scores.std()) if len(scores) else 0.,
            score_saturation_fraction=float((np.abs(scores) > 10).mean()) if len(scores) else 0.,
            **{(f"denoise_{chain.logp.shape[1]-k}_logp" if chain.kind == "diffusion" else "gaussian_joint_logp"): float(chain.logp[begin:end, k].mean())
               for k in range(chain.logp.shape[1])}))
    return rows
