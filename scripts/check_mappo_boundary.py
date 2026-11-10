"""Initial-policy pairing and <=2048 main-scene transitions per NEW M1 experiment."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate
from uav_research.runtime import atomic_json, configure_runtime, seed_all
from uav_research.train import train
from check_mappo_bridge import actor_digest
from report_mappo_boundary import PREFIX, LABELS, DEFAULT_RUNS, build, digest, validate_configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_boundary_implementation_checks'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(f'configs/{PREFIX}{s}_100k.yaml') for s in ['', '_central','_boundary','_central_boundary']]
    configs={k:load_config(p) for k,p in zip(LABELS,paths)};validate_configs(configs)
    reference=Path(DEFAULT_RUNS[0]);sources=[*paths,reference/'validation_history/000000000000.json']
    source_hashes={str(p):digest(p) for p in sources}
    expected=json.loads((reference/'validation_history/000000000000.json').read_text())['summary']['mean']
    initial={};actor_hash=None
    for label,cfg in configs.items():
        configure_runtime(cfg);seed_all(cfg.run.seed);model=make_learner(cfg,torch.device('cpu'))
        current=actor_digest(model.actor)
        if actor_hash is not None and current!=actor_hash:raise ValueError('Initial actor changed')
        actor_hash=current
        _,summary=evaluate(cfg,model,cfg.run.validation_seeds,args.output/f'initial_{label}')
        if label in ['M1','M1-B']:
            for key in ['coverage','discovery_fraction','disconnection_ratio']:
                np.testing.assert_allclose(summary['mean'][key],expected[key],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(summary['mean']['task_return'],expected['return'],atol=1e-12,rtol=1e-12)
        if label=='M1-CB':
            for key in ['coverage','task_return','discovery_fraction','disconnection_ratio']:
                np.testing.assert_allclose(summary['mean'][key],initial['M1-C']['summary']['mean'][key],atol=1e-12,rtol=1e-12)
        initial[label]=dict(actor_sha256=current,summary=summary)
    # Check scene initialization and action-independent UE motion using calibration seeds.
    envs=[UAVEnvironment(c.env) for c in configs.values()]
    for e in envs:e.reset(30020)
    for e in envs[1:]:np.testing.assert_array_equal(e.uav_xy,envs[0].uav_xy)
    action=np.tile([.1,.2],(configs['M1'].env.num_uavs,1))
    for _ in range(5):
        for e in envs:e.step(action)
        for i,j in [(0,2),(1,3)]:np.testing.assert_array_equal(envs[i].state,envs[j].state)
        for e in envs[1:]:assert e.movement_rng.bit_generator.state==envs[0].movement_rng.bit_generator.state
    records={}
    for label in LABELS[1:]:
        c=copy.deepcopy(configs[label]);c.run.total_steps=2048;c.run.validation_seeds=[30020]
        c.run.eval_interval=c.run.log_interval=2048;c.run.tensorboard=False
        result=train(c,args.output/f'{label}_short')
        assert result['status']=='completed' and result['env_steps']==2048 and result['updates']==1
        records[label]=result
    # Persist a separate small four-way report, never present it as a formal result.
    tiny={}
    for label in LABELS:
        c=load_config('configs/discrete_mappo_smoke.yaml')
        c.env.action_space='continuous';c.env.observation_coordinates='relative'
        c.env.throughput_weight=.1;c.env.robustness_weight=.5;c.env.disconnection_weight=.1
        c.env.boundary_penalty_margin_m=100.;c.env.boundary_penalty_weight=configs[label].env.boundary_penalty_weight
        c.env.user_initialization=configs[label].env.user_initialization
        c.env.user_initial_margin_fraction=configs[label].env.user_initial_margin_fraction
        c.algo.mappo_state_dependent_std=True;c.algo.critic_parameterization='time_factor'
        c.run.cpu_threads=8;c.run.total_steps=64;c.run.eval_interval=32;c.run.log_interval=16
        c.run.validation_seeds=[30020,30021];c.run.best_metric=configs[label].run.best_metric
        c.run.tensorboard=False;tiny[label]=c.validate()
    validate_configs(tiny);runs=[]
    for label,c in tiny.items():
        path=args.output/f'{label}_small';train(c,path);runs.append(path)
    build(runs,args.output/'small_report',samples=8)
    if not all(digest(p)==v for p,v in source_hashes.items()):raise ValueError('Formal config/source modified')
    atomic_json(args.output/'acceptance.json',dict(initial=initial,main_short=records,formal_training_started=False,
                source_sha256=source_hashes,main_short_overrides=dict(total_steps=2048,validation_seeds=[30020],eval_interval=2048,log_interval=2048,tensorboard=False),
                small_report='small_report/report.md',small_config_note='4 UAV,24 UE,400 m,32-step episodes,64 transitions; calibration only'))
    lines=['# 边界实验实现验收','', '正式训练未启动；以下不是正式性能对照。','',
           '|组别|初始覆盖|初始任务回报|','|---|---:|---:|']
    for label,x in initial.items():
        m=x['summary']['mean'];lines.append(f"|{label}|{m['coverage']:.6%}|{m['task_return']:.6f}|")
    lines+=['','|主场景短测|联合转移|PPO轮数|耗时(s)|峰值RSS(MiB)|','|---|---:|---:|---:|---:|']
    for label,x in records.items():lines.append(f"|{label}|2048|1|{x['elapsed_seconds']:.2f}|{x['peak_rss_mib']:.2f}|")
    lines+=['','峰值RSS为同进程累计观测，不能解释为三组独立内存差值。',
            '[小场景四组报告与可重建图表](small_report/report.md)。小场景与主场景覆盖项见acceptance.json。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(seconds=v['elapsed_seconds'],peak_rss_mib=v['peak_rss_mib']) for k,v in records.items()},indent=2))


if __name__=='__main__':main()
