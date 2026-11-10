"""Acceptance only: paired initial validation and <=2048 main transitions/group."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from report_mappo_actor_sharing import BASE, MODES, build, publish, validate_configs
from report_mappo_bridge import digest
from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.deployment import export_actor, DeployedActor
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.runtime import configure_runtime, seed_all, atomic_json, load_checkpoint
from uav_research.train import train
from uav_research.vector import VectorEnvironment


def benchmark(cfg,model):
    with VectorEnvironment(cfg.env,cfg.run.num_envs) as vector:
        raw,_,_=vector.reset([30020+i for i in range(cfg.run.num_envs)])
        tensor=raw.to_torch('cpu')
        stages={'raw_to_tensor':lambda:raw.to_torch('cpu'),
                'split_agent_rows':lambda:tensor.split_agents(),
                'distribution_including_encoder':lambda:model.actor.distribution(tensor)}
        result={}
        with torch.inference_mode():
            for name,call in stages.items():
                for _ in range(3):call()
                values=[]
                for _ in range(32):
                    start=time.perf_counter();call();values.append((time.perf_counter()-start)*1000)
                result[name]=dict(median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path('analysis/mappo_actor_sharing_implementation_checks'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    configs=[load_config(f'configs/{BASE.format(mode)}.yaml') for mode in MODES];validate_configs(configs)
    sources=[Path(f'configs/{BASE.format(mode)}.yaml') for mode in MODES]
    sources += [Path('runs/continuous_mappo_full_reward_time_factor_relative_central_history_100k_seed0')/name
                for name in ['best.pt','last.pt','config.yaml','validation.csv']]
    sources += list(Path('src/uav_research').glob('*.py'))
    sources += [Path(__file__),Path('scripts/report_mappo_actor_sharing.py')]
    hashes={str(p):digest(p) for p in sources}
    models=[];rngs=[];initial={};bench={};short={};started=time.perf_counter()
    for mode,cfg in zip(MODES,configs):
        device=configure_runtime(cfg);seed_all(cfg.run.seed);model=make_learner(cfg,device)
        models.append(model);rngs.append(torch.get_rng_state().clone())
        _,summary=evaluate(cfg,model,cfg.run.validation_seeds,out/f'initial_{mode}',trace_count=2,reward_curves=True)
        initial[mode]=summary['mean'];bench[mode]=benchmark(cfg,model)
        np.testing.assert_allclose(initial[mode]['coverage'],.27228541666666667,atol=1e-10,rtol=0)
        np.testing.assert_allclose(initial[mode]['return'],108.91416666666667,atol=1e-8,rtol=0)
        print(f'{mode} initial: {summary["mean"]["coverage"]:.10%}, return {summary["mean"]["return"]:.10f}',flush=True)
    assert torch.equal(rngs[0],rngs[1])
    for k,v in models[0].value.state_dict().items():assert torch.equal(v,models[1].value.state_dict()[k])
    for actor in models[1].actor.actors:
        for k,v in models[0].actor.state_dict().items():assert torch.equal(v,actor.state_dict()[k])
    with np.load(out/'initial_shared/reward_curves.npz') as a,np.load(out/'initial_independent/reward_curves.npz') as b:
        np.testing.assert_array_equal(a['reward'],b['reward'])
    # Matrix batch sizes differ across implementations; permit only sub-mm
    # coordinate rounding while requiring exactly equal connectivity/service.
    for seed in configs[0].run.validation_seeds[:2]:
        name=f'trace_{seed}.npz'
        with np.load(out/'initial_shared'/name) as a,np.load(out/'initial_independent'/name) as b:
            np.testing.assert_allclose(a['uav_xy'],b['uav_xy'],atol=1e-3,rtol=0)
            for key in ['user_xy','A','B','C']:np.testing.assert_array_equal(a[key],b[key])
    for mode,cfg in zip(MODES,configs):
        trial=copy.deepcopy(cfg);trial.run.total_steps=2048
        trial.run.validation_seeds=[30020];trial.run.eval_interval=trial.run.log_interval=2048
        trial.run.tensorboard=False
        folder=out/f'{mode}_main_short';result=train(trial,folder);short[mode]=result
        assert result['env_steps']==2048 and result['updates']==1 and result['status']=='completed'
        full=load_checkpoint(result['resume_checkpoint'])
        assert full['learner']['actor_sharing']==mode
        for states in [full['learner']['value_opt'],*(full['learner']['actor_optimizers'] if mode=='independent' else [full['learner']['actor_opt']])]:
            assert all(int(s['step'])==40 for s in states['state'].values())
        _,restored=policy_from_checkpoint(folder/'last.pt','cpu')
        export_actor(folder/'last.pt',folder/'actor.pt');deployed=DeployedActor(folder/'actor.pt')
        obs,*_=UAVEnvironment(trial.env).reset(30020)
        np.testing.assert_array_equal(restored.act(obs,True),deployed.act(obs))
        with VectorEnvironment(trial.env,trial.run.num_envs) as vector:
            vector.restore(full['environments'])
            for env,snap in zip(vector.envs,full['environments']):
                np.testing.assert_array_equal(env.history,snap['history'])
        print(f'{mode} main short: {result["elapsed_seconds"]:.2f}s, peak {result["peak_rss_mib"]:.1f} MiB',flush=True)
    small_runs=[]
    for mode in MODES:
        cfg=load_config('configs/discrete_mappo_smoke.yaml')
        cfg.env.action_space='continuous';cfg.env.observation_coordinates='relative';cfg.env.observation_history='one_step'
        cfg.env.uav_boundary='reflect';cfg.env.user_initialization='uniform_interior';cfg.env.user_initial_margin_fraction=1/7
        for key in ['throughput_weight','robustness_weight','disconnection_weight','boundary_penalty_weight']:setattr(cfg.env,key,0.)
        cfg.algo.mappo_state_dependent_std=True;cfg.algo.critic_parameterization='time_factor';cfg.algo.actor_sharing=mode
        cfg.run.cpu_threads=8;cfg.run.numba_threads=4;cfg.run.total_steps=64;cfg.run.eval_interval=32;cfg.run.log_interval=16
        cfg.run.validation_seeds=[30020,30021];cfg.run.tensorboard=False;cfg.run.best_metric='task_return'
        cfg.run.evaluation_policy='sample';cfg.run.evaluation_action_seed=880001;cfg.validate()
        path=out/f'{mode}_small';train(cfg,path);small_runs.append(path)
    comparison=build(small_runs,out/'small_report',samples=8,action_seeds=[880001,880002])
    publish(out/'small_report',comparison)
    assert hashes=={str(p):digest(p) for p in sources}
    record=dict(initial=initial,benchmark=bench,main_short=short,source_sha256=hashes,
        paired_initial_rng_and_critic=True,main_budget_per_group=2048,formal_training_started=False,
        source_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    atomic_json(out/'checks.json',record)
    lines=['# 独立Actor实现验收','',
           '仅初始验证、短训和小场景完整报告；未启动两组正式100k训练。','',
           '|组别|初始覆盖|初始纯覆盖回报|主短训秒|峰值RSS MiB|Actor参数量|','|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        result=short[mode];lines.append(f'|{mode}|{initial[mode]["coverage"]:.8%}|{initial[mode]["return"]:.8f}|{result["elapsed_seconds"]:.2f}|{result["peak_rss_mib"]:.1f}|{result["actor_parameters"]}|')
    lines+=['','主场景每组只采集2048条联合转移，一轮PPO/每actor40次优化，critic40次优化。完整恢复状态和actor导出已核对；小场景完成训练、验证、MC、逐UAV诊断、拓扑、对照报告和无重新采样的重建。','',
        '初始参数、构建后随机流、critic一致，32场景覆盖和纯覆盖回报复现。独立网络与共享网络矩阵批次不同，浮点坐标只允许亚毫米级差异，拓扑与服务计数必须一致。','',
        '微基准见checks.json：每次16环境，32次重复，报告raw转Tensor、UE行按UAV分组、含编码器的完整分布前向中位数/p95。分组耗时包含在前向中，不额外相加为端到端耗时。',
        '参数/梯度隔离、逐actor均值损失、按actor梯度裁剪、单次critic更新及CPU连续/恢复一致性另由专项测试覆盖。','']
    (out/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
