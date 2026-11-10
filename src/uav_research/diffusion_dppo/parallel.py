"""Optional single-process actor data parallelism without changing artifacts.

Only saved-chain density evaluation is replicated. PyTorch's differentiable
broadcast/copy reduces replica gradients to the original actor before its one
global gradient clip and Adam step. There is no extra rollout, critic update,
denoising randomness, or per-device optimizer.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import replicate, parallel_apply


class _ChainEvaluation(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation, chain):
        return self.actor.evaluate_chain(observation, chain)


def evaluate_chain_parallel(actor, observation, chain, devices):
    if len(devices) < 2 or observation.groups < 2:
        return actor.evaluate_chain(observation, chain)
    devices = devices[:min(len(devices), observation.groups)]
    # Split UAV samples, preserving each variable-size candidate set and chain.
    indices = np.array_split(np.arange(observation.groups), len(devices))
    inputs = [(observation.take(index), chain.take(index)) for index in indices]
    replicas = replicate(_ChainEvaluation(actor), devices)
    results = parallel_apply(replicas, inputs, devices=devices)
    # .to remains differentiable. The caller applies the original full-batch
    # sample mean, so unequal shard sizes do not change gradient weighting.
    primary = next(actor.parameters()).device
    return torch.cat([result.to(primary) for result in results], dim=0)
