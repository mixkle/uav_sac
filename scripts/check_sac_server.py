"""Isolated CUDA acceptance; never starts the formal 1M experiment."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from uav_research.config import load_config
from uav_research.deployment import export_actor, DeployedActor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.runtime import atomic_json, load_checkpoint, resolve_output
from uav_research.train import train


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = resolve_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    c = load_config('configs/smoke.yaml')
    c.algo.name = 'sac'
    c.algo.critic_parameterization = 'time_factor'
    c.env.observation_history = 'one_step'
    c.env.observation_coordinates = 'relative'
    c.env.reward_primary = 'coverage'
    c.env.throughput_weight = c.env.robustness_weight = c.env.disconnection_weight = 0.
    c.run.best_metric = 'task_return'
    c.run.device = 'cuda'
    c.run.cpu_threads = 8
    c.run.numba_threads = 4
    c.run.total_steps = 64
    c.run.eval_interval = 32
    c.run.validation_seeds = [30020]
    c.run.evaluation_policy = 'sample'
    c.algo.warmup_steps = 16
    c.algo.batch_size = 8
    c.algo.replay_capacity = 128
    c.run.tensorboard = False
    full = train(c, output / 'continuous')
    part = copy.deepcopy(c)
    part.run.total_steps = 32
    split = train(part, output / 'split')
    split = train(c, output / 'split', split['resume_checkpoint'])
    full_data = load_checkpoint(full['resume_checkpoint'])
    split_data = load_checkpoint(split['resume_checkpoint'])
    for key in ('learner', 'environments', 'rng', 'replay_metadata'):
        assert_exact(full_data[key], split_data[key])
    export_actor(output / 'continuous' / 'last.pt', output / 'actor.pt')
    ec, policy = policy_from_checkpoint(output / 'continuous' / 'last.pt', 'cuda')
    deployed = DeployedActor(output / 'actor.pt', device='cuda')
    obs, _, _ = UAVEnvironment(ec.env).reset(30021)
    np.testing.assert_array_equal(policy.act(obs, True), deployed.act(obs))
    _, summary = evaluate(ec, policy, [30021], output / 'small_validation', trace_count=1)
    assert np.isfinite(summary['mean']['return'])
    # Production scene and batch size; explicit short-test overrides only.
    main_cfg = load_config('configs/sac_coverage_time_factor_relative_central_history_server_1m.yaml')
    main_cfg.run.total_steps = 2048
    main_cfg.algo.warmup_steps = 256
    main_cfg.algo.replay_capacity = 4096
    main_cfg.run.validation_seeds = [30020]
    main_cfg.run.eval_interval = 2048
    main_cfg.run.log_interval = 256
    main_cfg.run.tensorboard = False
    result = train(main_cfg, output / 'main_short')
    assert result['status'] == 'completed' and result['updates'] == 448
    restored = load_checkpoint(result['resume_checkpoint'])
    assert restored['env_steps'] == 2048
    atomic_json(output / 'acceptance.json', dict(
        cuda_exact_resume=True, cuda_export_actions_exact=True,
        small_validation=summary, main_short=result,
        wall_seconds=time.perf_counter()-started,
        formal_training_started=False,
        main_overrides=dict(total_steps=2048, warmup_steps=256, replay_capacity=4096,
                            validation_seeds=[30020], eval_interval=2048,
                            log_interval=256, tensorboard=False),
    ))
    print('CUDA recovery/export and 2048-transition main-scene acceptance passed.', flush=True)


if __name__ == '__main__':
    main()
