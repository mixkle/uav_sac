"""Local acceptance only: six small pipelines and at most 2048 main-scene transitions per group."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from report_mappo_training import build_report
from report_ue_reward_ablations import GROUPS, compare
from uav_research.config import load_config
from uav_research.deployment import DeployedActor, export_actor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.plotting import curves, trace_plot
from uav_research.runtime import atomic_json, load_checkpoint
from uav_research.train import train


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('analysis/ue_reward_ablation_acceptance_six_groups_20261019'))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reports, results, runs = {}, {}, []
    for name, (primary, speed, mu) in GROUPS.items():
        print(f'Checking {name}', flush=True)
        cfg = load_config('configs/discrete_mappo_smoke.yaml')
        cfg.env.reward_primary = primary
        cfg.env.user_speed_std = speed
        cfg.env.throughput_weight = cfg.env.robustness_weight = 0.0
        cfg.env.disconnection_weight = mu
        cfg.run.cpu_threads = 1
        cfg.run.validation_seeds = [30030, 30031]
        cfg.run.total_steps = 64
        cfg.run.eval_interval = 32
        small = root / 'small' / name
        uninterrupted = train(cfg, small / 'continuous')
        half = deepcopy(cfg)
        half.run.total_steps = 32
        first = train(half, small / 'resumed')
        continued = train(cfg, small / 'resumed', first['resume_checkpoint'])
        exact(load_checkpoint(uninterrupted['resume_checkpoint'])['learner'],
              load_checkpoint(continued['resume_checkpoint'])['learner'])
        ecfg, policy = policy_from_checkpoint(small / 'continuous' / 'last.pt', torch.device('cpu'))
        evaluate(ecfg, policy, ecfg.run.validation_seeds, small / 'evaluation', trace_count=1, reward_curves=True)
        export_actor(small / 'continuous' / 'last.pt', small / 'actor.pt')
        obs, _, _ = UAVEnvironment(ecfg.env).reset(30030)
        np.testing.assert_array_equal(DeployedActor(small / 'actor.pt').act(obs), policy.act(obs, True))
        curves([small / 'continuous'], small / 'figures')
        trace_plot(small / 'evaluation' / 'trace_30030.npz', small / 'figures')

        cfg = load_config(f'configs/discrete_mappo_ablation_{name}.yaml')
        cfg.run.total_steps = 2048
        cfg.run.validation_seeds = [30030, 30031]
        cfg.run.eval_interval = 2048
        cfg.run.log_interval = 2048
        cfg.run.tensorboard = False
        # Keep all 16 environments, physical parameters and PPO update settings.
        run = root / 'main' / name
        manifest = train(cfg, run)
        assert manifest['env_steps'] == 2048 and manifest['updates'] == 1
        reports[name] = build_report(run, root / 'comparison' / name)
        runs.append(run)
        results[name] = dict(
            exact_cpu_resume=True, actor_export=True, small_steps=64,
            main_steps=manifest['env_steps'], main_ppo_updates=manifest['updates'],
            main_elapsed_seconds=manifest['elapsed_seconds'], peak_rss_mib=manifest['peak_rss_mib'],
            main_validation_scenes=cfg.run.validation_seeds,
            main_mean=reports[name]['evaluations']['last']['summary']['mean'],
        )
    compare(runs, root / 'comparison', reports)
    atomic_json(root / 'acceptance.json', dict(
        groups=results, elapsed_seconds=time.perf_counter() - start,
        purpose='Implementation acceptance only, not a formal 250k experiment or evidence of policy superiority.',
        overrides=dict(total_steps=2048, validation_seeds=[30030, 30031], eval_interval=2048,
                       log_interval=2048, tensorboard=False),
    ))
    print(f'Acceptance artifacts: {root}', flush=True)


if __name__ == '__main__':
    main()
