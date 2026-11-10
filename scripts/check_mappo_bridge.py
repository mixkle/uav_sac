"""Bridge acceptance: initial SAC pairing and at most 2048 transitions per MAPPO.

Formal 100k configurations are read-only; outputs cannot overwrite existing runs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate
from uav_research.runtime import atomic_json, configure_runtime, load_checkpoint, seed_all
from uav_research.train import train


CONFIGS = {
    'M0': 'configs/discrete_mappo_full_reward_time_factor_relative_100k.yaml',
    'M1': 'configs/continuous_mappo_full_reward_time_factor_relative_100k.yaml',
}


def actor_digest(actor):
    h = hashlib.sha256()
    for key, value in actor.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('analysis/mappo_bridge_implementation_checks'))
    p.add_argument('--sac-run', type=Path, default=Path('runs/sac_full_reward_time_factor_relative_100k_seed0'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    configs = {label: load_config(path) for label, path in CONFIGS.items()}
    source = load_checkpoint(args.sac_run / 'best.pt')
    if source['env_steps'] != 0:
        raise ValueError('Initial actor reference must be SAC step 0')
    initial = {}
    for label, cfg in configs.items():
        configure_runtime(cfg); seed_all(cfg.run.seed)
        model = make_learner(cfg, torch.device('cpu'))
        if label == 'M1':
            for key, value in model.actor.state_dict().items():
                torch.testing.assert_close(value, source['learner']['actor'][key], rtol=0, atol=0)
        _, summary = evaluate(cfg, model, cfg.run.validation_seeds, args.output/f'initial_{label}')
        if label == 'M1':
            expected = json.loads((args.sac_run/'validation_history/000000000000.json').read_text())['summary']['mean']
            for key in ['coverage', 'return', 'discovery_fraction', 'disconnection_ratio']:
                np.testing.assert_allclose(summary['mean'][key], expected[key], rtol=1e-12, atol=1e-12)
        initial[label] = dict(actor_sha256=actor_digest(model.actor), summary=summary)
    # Action-independent scene and UE streams are identical between M0 and M1.
    environments = [UAVEnvironment(configs[k].env) for k in ['M0', 'M1']]
    for env in environments: env.reset(30020)
    np.testing.assert_array_equal(environments[0].uav_xy, environments[1].uav_xy)
    for _ in range(4):
        environments[0].step(np.full(configs['M0'].env.num_uavs, 8, np.int64))
        environments[1].step(np.zeros((configs['M1'].env.num_uavs, 2), np.float32))
        np.testing.assert_array_equal(environments[0].state, environments[1].state)
    records = {}
    for label, cfg in configs.items():
        cfg = copy.deepcopy(cfg)
        cfg.run.total_steps = 2048; cfg.run.validation_seeds = [30020]
        cfg.run.eval_interval = 2048; cfg.run.log_interval = 2048
        cfg.run.tensorboard = False
        result = train(cfg, args.output/f'{label}_short')
        assert result['status'] == 'completed' and result['updates'] == 1
        model = load_checkpoint(args.output/f'{label}_short/last.pt')
        if 'replay_metadata' in model:
            raise ValueError('MAPPO unexpectedly allocated replay')
        records[label] = result
    atomic_json(args.output/'acceptance.json', dict(
        initial=initial, main_short=records, formal_training_started=False,
        short_overrides=dict(total_steps=2048,validation_seeds=[30020],eval_interval=2048,
                             log_interval=2048,tensorboard=False),
        source_sac_initial_checkpoint_sha256=hashlib.sha256((args.sac_run/'best.pt').read_bytes()).hexdigest()))
    print(json.dumps({k:dict(seconds=v['elapsed_seconds'], peak_rss_mib=v['peak_rss_mib'])
                      for k,v in records.items()}, indent=2))


if __name__ == '__main__':
    main()
