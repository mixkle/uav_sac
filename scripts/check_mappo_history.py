"""M1-C-H acceptance only: initial pairing, <=2048 main transitions, small reports."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from report_mappo_history import build, validate_configs, LABELS
from report_mappo_bridge import digest
from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.deployment import export_actor, DeployedActor
from uav_research.runtime import configure_runtime, seed_all, atomic_json, load_checkpoint
from uav_research.train import train

BASE = 'configs/continuous_mappo_full_reward_time_factor_relative_central'
REFERENCE = Path('runs/continuous_mappo_full_reward_time_factor_relative_central_100k_seed0')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('analysis/mappo_history_implementation_checks'))
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    configs = [load_config(f'{BASE}{suffix}_100k.yaml') for suffix in ['', '_history']]
    validate_configs(configs)
    files = [Path(f'{BASE}{suffix}_100k.yaml') for suffix in ['', '_history']]
    files += [REFERENCE/'best.pt', REFERENCE/'last.pt', REFERENCE/'validation.csv']
    hashes = {str(f): digest(f) for f in files}
    models, initial, states = [], {}, []
    for label, cfg in zip(LABELS, configs):
        device = configure_runtime(cfg); seed_all(cfg.run.seed)
        model = make_learner(cfg, device); models.append(model)
        states.append(torch.get_rng_state().clone())
        _, summary = evaluate(cfg, model, cfg.run.validation_seeds, out/f'initial_{label}')
        initial[label] = summary['mean']
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    for k, value in models[0].value.state_dict().items():
        torch.testing.assert_close(value, models[1].value.state_dict()[k], rtol=0, atol=0)
    # Encoder, all later layers, and existing first-layer columns are identical.
    a, b = models[0].actor.state_dict(), models[1].actor.state_dict()
    split = configs[0].env.current_obs_dim
    for k in a:
        if k == 'net.0.weight':
            torch.testing.assert_close(a[k][:, :split], b[k][:, :split], rtol=0, atol=0)
            torch.testing.assert_close(a[k][:, split:], b[k][:, split+4:], rtol=0, atol=0)
            assert b[k][:, split:split+4].count_nonzero() == 0
        else:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    expected = json.loads((REFERENCE/'validation_history/000000000000.json').read_text())['summary']['mean']
    for key in ['coverage', 'return', 'task_return', 'discovery_fraction', 'disconnection_ratio']:
        ref = expected.get(key, expected['return'])
        for label in LABELS:
            np.testing.assert_allclose(initial[label][key], ref, rtol=1e-8, atol=1e-8)
    cfg = copy.deepcopy(configs[1]); cfg.run.total_steps = 2048
    cfg.run.validation_seeds = [30020]; cfg.run.eval_interval = cfg.run.log_interval = 2048
    cfg.run.tensorboard = False
    result = train(cfg, out/'main_short')
    assert result['status'] == 'completed' and result['env_steps'] == 2048 and result['updates'] == 1
    export_actor(out/'main_short/last.pt', out/'main_short/actor.pt')
    exported = DeployedActor(out/'main_short/actor.pt')
    _, trained = policy_from_checkpoint(out/'main_short/last.pt', 'cpu')
    obs, _, _ = UAVEnvironment(cfg.env).reset(30020)
    np.testing.assert_array_equal(exported.act(obs), trained.act(obs, True))
    artifact = load_checkpoint(result['resume_checkpoint'])
    assert artifact['observation_history']['mode'] == 'one_step'
    assert all(e['history'].shape == (18, 4) for e in artifact['environments'])
    small_runs = []
    for label, mode in zip(LABELS, ['none', 'one_step']):
        small = load_config('configs/discrete_mappo_smoke.yaml')
        small.env.action_space = 'continuous'; small.env.observation_coordinates = 'relative'
        small.env.observation_history = mode; small.env.uav_boundary = 'reflect'
        small.env.user_initialization = 'uniform_interior'; small.env.user_initial_margin_fraction = 1/7
        small.env.throughput_weight = .1; small.env.robustness_weight = .5; small.env.disconnection_weight = .1
        small.algo.mappo_state_dependent_std = True; small.algo.critic_parameterization = 'time_factor'
        small.run.cpu_threads = 8; small.run.numba_threads = 4
        small.run.total_steps = 64; small.run.eval_interval = 32; small.run.log_interval = 16
        small.run.validation_seeds = [30020, 30021]; small.run.tensorboard = False
        small.run.evaluation_policy = 'sample'; small.run.evaluation_action_seed = 880001
        small.run.best_metric = 'task_return'
        small.validate()
        path = out/f'{label}_small'; train(small, path); small_runs.append(path)
    build(small_runs, out/'small_report', samples=8, diagnostic_seeds=[30030, 30031],
          action_seeds=[880001, 880002], branch_source=None)
    assert hashes == {f: digest(f) for f in hashes}
    record = dict(initial=initial, main_short=result, formal_training_started=False,
                  short_overrides=dict(total_steps=2048, validation_seeds=[30020], eval_interval=2048,
                                       log_interval=2048, tensorboard=False),
                  small_report='small_report/report.md', source_hashes=hashes,
                  initial_critic_and_rng_identical=True, new_history_weights_initially_zero=True)
    atomic_json(out/'acceptance.json', record)
    lines = ['# M1-C-H 实现验收', '', '以下仅为正确性与短流程验收，正式100k未启动。', '',
             '| 模型 | 未训练策略覆盖 | 未训练策略任务回报 |', '|---|---:|---:|']
    for label in LABELS:
        lines.append(f'|{label}|{initial[label]["coverage"]:.6%}|{initial[label]["task_return"]:.6f}|')
    lines += ['', f'主场景：2048条联合转移、1轮PPO，耗时{result["elapsed_seconds"]:.2f}s，进程峰值RSS {result["peak_rss_mib"]:.2f} MiB。',
              '初始actor已有参数、critic参数和构建后随机状态核对一致，新增4列权重为零。',
              '[小场景训练、MC验证与可重建图表](small_report/report.md)。短流程结果不用于判断历史输入有效性。',
              '原运行与两份正式配置文件哈希前后保持不变；完整检查点包含一步历史。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
