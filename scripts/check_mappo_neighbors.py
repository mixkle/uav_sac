"""Initialization and physical-control check; never trains or creates a run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.observations import as_tensor_observation
from uav_research.runtime import configure_runtime, seed_all
from uav_research.vector import VectorEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neighbors', type=int, choices=[4, 5, 6, 7, 8], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(f'configs/mappo_capacity10_neighbors{args.neighbors}_1m_server.yaml')
    base = copy.deepcopy(cfg)
    base.env.neighbor_slots = 3
    device = configure_runtime(cfg)
    seed_all(cfg.run.seed)
    reference = make_learner(base, device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
    seed_all(cfg.run.seed)
    expanded = make_learner(cfg, device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    for key, value in reference.value.state_dict().items():
        assert torch.equal(value, expanded.value.state_dict()[key])
    extra = (args.neighbors - 3) * cfg.env.neighbor_dim
    point = base.env.current_obs_dim
    assert not expanded.actor.net[0].weight[:, point:point + extra].any()
    old = VectorEnvironment(base.env, cfg.run.num_envs, cfg.run.vector_backend)
    new = VectorEnvironment(cfg.env, cfg.run.num_envs, cfg.run.vector_backend)
    try:
        seeds = list(range(30000, 30000 + cfg.run.num_envs))
        old_obs, old_state, _ = old.reset(seeds)
        obs, state, _ = new.reset(seeds)
        np.testing.assert_array_equal(state, old_state)
        np.testing.assert_array_equal(obs.fixed[..., :point], old_obs.fixed[..., :point])
        np.testing.assert_array_equal(obs.fixed[..., -4:], old_obs.fixed[..., -4:])
        np.testing.assert_array_equal(obs.users, old_obs.users)
        with torch.no_grad():
            a = reference.actor.distribution(as_tensor_observation(old_obs, device))
            b = expanded.actor.distribution(as_tensor_observation(obs, device))
            torch.testing.assert_close(a.loc, b.loc, rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(a.scale, b.scale, rtol=1e-5, atol=2e-6)
        seed_all(991001)
        expected = reference.act(old_obs)
        seed_all(991001)
        actual = expanded.act(obs)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        old_step = old.step(actual)
        new_step = new.step(actual)
        for index in [1, 2, 3, 4]:
            np.testing.assert_array_equal(old_step[index], new_step[index])
        assert reference.updates == expanded.updates == 0
        assert not reference.actor_opt.state and not expanded.actor_opt.state
        result = dict(
            check='passed', device=str(device), gpu=torch.cuda.get_device_name(0),
            neighbors=args.neighbors, num_envs=cfg.run.num_envs,
            fixed_observation=cfg.env.fixed_obs_dim, actor_input=expanded.actor.net[0].in_features,
            critic_input=cfg.env.state_dim,
            baseline_actor_parameters=sum(p.numel() for p in reference.actor.parameters()),
            actor_parameters=sum(p.numel() for p in expanded.actor.parameters()),
            critic_parameters=sum(p.numel() for p in expanded.value.parameters()),
            initial_action_max_abs_difference=float(np.max(np.abs(actual - expected))),
            initial_policy_equal_within_tolerance=True,
            initial_critic_and_rng_exact=True, fixed_action_physics_exact=True,
            optimization_updates=expanded.updates, formal_training_started=False,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        old.close()
        new.close()


if __name__ == '__main__':
    main()
