"""One-action counterfactual value audit from saved good deployments; no training.

Only the first joint action is intervened on. All later actions follow the
same frozen actor. Persistent hovering is a different policy, not Q^pi(s,a).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_generalization import actor_digest, interval
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from uav_research.boundary import reward_contributions
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import history_specification, stack_observations
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


SOURCE = Path('analysis/mappo_central_best_deployment_branches_100k_seed0')
HISTORY = Path('analysis/mappo_central_history_100k_seed0')
DEFAULT_OUT = 'analysis/mappo_central_action_values_100k_seed0'
BASE = 'continuous_mappo_full_reward_time_factor_relative_central'
RUNS = {label:Path(f'runs/{BASE}{suffix}_100k_seed0') for label,suffix in
        [('M1-C',''),('M1-C-H','_history')]}
MODES = ['policy','half','hover']
FIELDS = ['reward','coverage','discovery_fraction','disconnection_ratio','robustness','tau',
          'actual_speed_mean','uav_nearest_neighbor_mean','boundary_within_150_fraction',
          'boundary_reflection_fraction']


def intervene(action, mode, step):
    """First transition only; return a copy and never modify the actor sample."""
    result = np.array(action, copy=True)
    if mode not in MODES:
        raise ValueError(mode)
    if step == 0:
        if mode == 'half': result *= .5
        elif mode == 'hover': result.fill(0)
    return result


def evidence(rewards, values, discount, lam=.95, span=128):
    """Arrays [..., time], values include the true terminal V=0.

    MC and all advantages use full environment reward, never actor entropy.
    128 is a diagnostic fragment length, not reconstructed training alignment.
    """
    r, v = np.asarray(rewards,np.float64), np.asarray(values,np.float64)
    if v.shape != (*r.shape[:-1],r.shape[-1]+1):
        raise ValueError('Need one value per state including terminal')
    np.testing.assert_array_equal(v[...,-1],0)
    g = np.zeros_like(r)
    carry = np.zeros(r.shape[:-1])
    for t in range(r.shape[-1]-1,-1,-1):
        carry = r[...,t] + discount*carry
        g[...,t] = carry
    delta = r + discount*v[...,1:] - v[...,:-1]
    length = min(span,r.shape[-1])
    result = dict(mc_q=g[...,0],td1=delta[...,0],
        gae128=(delta[...,:length]*(discount*lam)**np.arange(length)).sum(-1),
        gae_full=(delta*(discount*lam)**np.arange(r.shape[-1])).sum(-1),
        reward1=r[...,0],next_value=v[...,1],initial_value=v[...,0],
        next_mc=(g[...,0]-r[...,0])/discount,
        future_mc_error=v[...,1]-(g[...,0]-r[...,0])/discount)
    np.testing.assert_allclose((delta*discount**np.arange(r.shape[-1])).sum(-1),
                               result['mc_q']-v[...,0],rtol=1e-11,atol=1e-10)
    return result


def paired_interval(values, seed=950001):
    """Repeat x scene; scene CI plus nested scene/future-repeat CI."""
    values = np.asarray(values)
    result = interval(values.mean(0),seed)
    rng = np.random.default_rng(seed)
    r,s = values.shape
    scenes = rng.integers(s,size=(2000,s))
    repeats = rng.integers(r,size=(2000,s,r))
    bootstrap = values[repeats,scenes[...,None]].mean((1,2))
    result.update(nested_low=float(np.quantile(bootstrap,.025)),
                  nested_high=float(np.quantile(bootstrap,.975)))
    return result


def run_candidate(cfg, policy, case, mode, repeat):
    """Paired current action noise and UE/actor futures across candidate modes."""
    snapshots = copy.deepcopy(case['snapshots'])
    if repeat:
        for snap,record in zip(snapshots,case['records']):
            snap['movement_rng'] = np.random.default_rng(np.random.SeedSequence(
                [982001,record['seed'],case['time'],repeat])).bit_generator.state
    user_hash = hashlib.sha256()
    with isolated_rng(), torch.inference_mode():
        if repeat == 0: torch.set_rng_state(case['torch_rng'].clone())
        else: torch.manual_seed(int(np.random.SeedSequence(
            [981001,case['time'],case['start'],repeat]).generate_state(1)[0]))
        with VectorEnvironment(cfg.env,len(snapshots),'batch') as vector:
            vector.restore(snapshots)
            obs = stack_observations([e.obs for e in vector.envs])
            state = np.stack([e.state for e in vector.envs])
            values = [policy.values(state).cpu().numpy()]
            rows, components = [], []
            for k in range(cfg.env.horizon-case['time']):
                original = policy.act(obs,deterministic=False)
                action = intervene(original,mode,k)
                if k == 0:
                    first_action = action.copy()
                    raw_action = original.copy()
                    initial_xy = np.stack([e.uav_xy for e in vector.envs])
                obs,state,_,done,truncated,infos = vector.step(action)
                assert np.all(done == (case['time']+k == cfg.env.horizon-1)) and not truncated.any()
                if k == 0 and mode == 'hover':
                    np.testing.assert_array_equal(initial_xy,np.stack([e.uav_xy for e in vector.envs]))
                if cfg.env.observation_history == 'one_step':
                    for env,command in zip(vector.envs,action):
                        np.testing.assert_array_equal(env.history[:,:2],command)
                values.append(policy.values(state).cpu().numpy())
                rows.append([[info[f] for f in FIELDS] for info in infos])
                contributions = np.stack([reward_contributions(info,cfg.env) for info in infos])
                np.testing.assert_allclose(contributions.sum(-1),[i['reward'] for i in infos],atol=1e-12,rtol=1e-12)
                components.append(contributions)
                user_hash.update(np.stack([e.user_xy for e in vector.envs]).tobytes())
            rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    data = dict(diagnostics=np.swapaxes(np.array(rows),0,1),
                values=np.stack(values,-1).astype(np.float64),
                contributions=np.swapaxes(np.array(components),0,1),
                first_actions=first_action,raw_first_actions=raw_action)
    assert all(np.isfinite(v).all() for v in data.values())
    return data,dict(ue_path_sha256=user_hash.hexdigest(),final_actor_rng_sha256=rng_hash)


def module_hashes(policy):
    return {key:actor_digest(getattr(policy,key)) for key in
            ['actor','value','normalizer','average_normalizer'] if hasattr(policy,key)}


def build(output,repeats=8):
    if repeats < 2:
        raise ValueError('At least two future repetitions are needed')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(RUNS['M1-C']/'config.yaml')
    cfg.run.cpu_threads=8; cfg.run.numba_threads=4
    configure_runtime(cfg)
    cases=torch.load(SOURCE/'branch_states.pt',map_location='cpu',weights_only=False)
    histories=torch.load(HISTORY/'branch_histories.pt',map_location='cpu',weights_only=False)
    sources=[SOURCE/'branch_states.pt',SOURCE/'experiment.json',HISTORY/'branch_histories.pt',
             HISTORY/'branches.json',Path(__file__)]
    sources += [run/name for run in RUNS.values() for name in ['last.pt','config.yaml']]
    sources += [HISTORY/'branches/last'/f't{at}/curves.npz' for at in sorted({c['time'] for c in cases})]
    sources += list(Path('src/uav_research').glob('*.py'))
    hashes={str(p):digest(p) for p in sources}
    meta=dict(status='running',repeats=repeats,intervention_steps=1,modes=MODES,
        cpu_threads=8,numba_threads=4,future_actor_seed_base=981001,future_ue_seed_base=982001,
        source_sha256=hashes,groups={},checks=[],records={str(t):[r for c in cases if c['time']==t for r in c['records']]
                                                       for t in sorted({c['time'] for c in cases})})
    atomic_json(out/'experiment.json',meta)
    started=time.perf_counter()
    ue_references={}
    for label,run in RUNS.items():
        cfg,policy=policy_from_checkpoint(run/'last.pt','cpu')
        configure_runtime(cfg)
        policy.actor.eval().requires_grad_(False)
        policy.value.eval().requires_grad_(False)
        assert policy.provenance['checkpoint_env_steps']==100000
        assert cfg.env.boundary_penalty_weight==0 and cfg.env.max_speed==30
        assert not set(r['seed'] for c in cases for r in c['records']).intersection(cfg.run.test_seeds)
        before=module_hashes(policy)
        meta['groups'][label]=dict(run=str(run),checkpoint_steps=100000,module_hashes=before,
                                  config=cfg.as_dict(),completed_cases=[])
        for time_index in sorted({c['time'] for c in cases}):
            parts=[]
            for original_case in [c for c in cases if c['time']==time_index]:
                case=copy.deepcopy(original_case)
                if cfg.env.observation_history=='one_step':
                    for i,snap in enumerate(case['snapshots']):
                        snap.update(observation_history='one_step',history_specification=history_specification(cfg.env),
                                    history=histories[(case['start'],case['time'])][i].copy())
                        np.testing.assert_array_equal(snap['history'][:,:2],snap['action'])
                results={mode:[] for mode in MODES}
                for repeat in range(repeats):
                    paired=[]
                    for mode in MODES:
                        data,audit=run_candidate(cfg,policy,case,mode,repeat)
                        results[mode].append(data); paired.append(audit)
                        if repeat==0 and mode=='policy':
                            reference=HISTORY/'branches/last'/f't{time_index}/curves.npz'
                            with np.load(reference) as saved:
                                fields=list(saved['fields']); left=saved[label][0,case['start']:case['start']+len(case['records'])]
                                common=[f for f in FIELDS if f in fields]
                                np.testing.assert_array_equal(data['diagnostics'][...,[FIELDS.index(f) for f in common]],
                                                              left[...,[fields.index(f) for f in common]])
                        if mode!='policy':
                            np.testing.assert_array_equal(data['raw_first_actions'],results['policy'][-1]['raw_first_actions'])
                            np.testing.assert_array_equal(data['values'][:,0],results['policy'][-1]['values'][:,0])
                    assert len({v['ue_path_sha256'] for v in paired})==1
                    assert len({v['final_actor_rng_sha256'] for v in paired})==1
                    key=(time_index,case['start'],repeat)
                    if key in ue_references: assert ue_references[key]==paired[0]['ue_path_sha256']
                    else: ue_references[key]=paired[0]['ue_path_sha256']
                    meta['checks'].append(dict(model=label,time=time_index,start=case['start'],repeat=repeat,
                                              **paired[0],paired=True))
                    print(f'{label}, t{time_index}, scenes {case["start"]}-{case["start"]+len(case["records"])-1}, future {repeat+1}/{repeats}: 3 first-action branches complete ({time.perf_counter()-started:.1f}s)',flush=True)
                packed={key:np.stack([np.stack([d[key] for d in results[m]]) for m in MODES])
                        for key in results['policy'][0]}
                target=out/label/f't{time_index}'
                target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/f'chunk_{case["start"]}.npz',**packed)
                parts.append(packed)
                meta['groups'][label]['completed_cases'].append(dict(time=time_index,start=case['start']))
                atomic_json(out/'experiment.json',meta)
            merged={k:np.concatenate([part[k] for part in parts],axis=2) for k in parts[0]}
            np.savez_compressed(out/label/f't{time_index}/curves.npz',**merged,fields=np.array(FIELDS),
                contribution_names=np.array(cfg.env.reward_component_names),modes=np.array(MODES),
                scene_seeds=np.array([r['seed'] for r in meta['records'][str(time_index)]]),
                episode_steps=np.arange(time_index+1,cfg.env.horizon+1),discount=cfg.algo.discount,
                gae_lambda=cfg.algo.gae_lambda)
        assert module_hashes(policy)==before
        assert all(p.grad is None for module in [policy.actor,policy.value] for p in module.parameters())
        meta['groups'][label]['module_hashes_unchanged']=True
    assert hashes=={str(p):digest(p) for p in sources}
    meta.update(status='completed',elapsed_seconds=time.perf_counter()-started,source_hashes_unchanged=True,
                original_continuations_exactly_reproduced=True)
    atomic_json(out/'experiment.json',meta)
    publish(out)


def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else None


def publish(output):
    out=Path(output); meta=json.loads((out/'experiment.json').read_text())
    assert meta['status']=='completed'
    rows,scene_rows,ranking,curves={},{},[],{}
    summary_rows=[]; raw_predictions={}
    for label in meta['groups']:
        for at,records in meta['records'].items():
            t=int(at)
            with np.load(out/label/f't{t}/curves.npz') as data:
                diagnostic=data['diagnostics']; rewards=diagnostic[...,FIELDS.index('reward')]
                values=data['values']; discount=float(data['discount']); lam=float(data['gae_lambda'])
                scores=evidence(rewards,values,discount,lam)
                component_q=np.stack([evidence(data['contributions'][...,j],np.zeros_like(values),discount,lam)['mc_q']
                                      for j in range(data['contributions'].shape[-1])],-1)
                np.testing.assert_allclose(component_q.sum(-1),scores['mc_q'],atol=1e-10,rtol=1e-11)
                raw_predictions[f'{label}_t{t}']=scores
                cov=diagnostic[...,FIELDS.index('coverage')]
                curves[(label,t)]=(rewards,cov,discount)
            masks=dict(all=np.ones(len(records),bool),good=np.array([r['good'] for r in records]))
            for mi,mode in enumerate(MODES[1:],1):
                differences={key:value[mi]-value[0] for key,value in scores.items()}
                np.testing.assert_allclose(differences['td1']-differences['mc_q'],
                    discount*differences['future_mc_error'],atol=1e-10,rtol=1e-11)
                repeat_count,scene_count=differences['mc_q'].shape
                draws=np.random.default_rng(950001).integers(repeat_count,size=(2000,repeat_count))
                mc_draws=differences['mc_q'][draws].mean(1)
                lows,highs=np.quantile(mc_draws,[.025,.975],axis=0)
                for subset,mask in masks.items():
                    if not mask.any(): continue
                    for metric,delta in differences.items():
                        if metric in ['initial_value','next_mc','future_mc_error']: continue
                        summary_rows.append(dict(model=label,time=t,subset=subset,candidate=mode,
                            metric=metric,scenes=int(mask.sum()),**paired_interval(delta[:,mask])))
                    mc=differences['mc_q'].mean(0)
                    for score in ['td1','gae128','gae_full']:
                        pred=differences[score].mean(0)
                        confirmed=mask & ((lows>0)|(highs<0))
                        selected=mask & (np.abs(mc)>1e-9)
                        ranking.append(dict(model=label,time=t,subset=subset,candidate=mode,score=score,
                            scenes=int(mask.sum()),positive_mc_scenes=int((mc[mask]>0).sum()),
                            positive_predicted_scenes=int((pred[mask]>0).sum()),
                            sign_agreement=float((mc[selected]*pred[selected]>0).mean()) if selected.any() else None,
                            repeat_interval_excludes_zero_scenes=int(confirmed.sum()),
                            sign_agreement_interval_excludes_zero=float((mc[confirmed]*pred[confirmed]>0).mean()) if confirmed.any() else None,
                            correlation=correlation(mc[mask],pred[mask]),
                            relative_advantage_rmse=float(np.sqrt(np.mean((pred[mask]-mc[mask])**2)))))
                for i,record in enumerate(records):
                    scene_rows[(label,t,mode,i)]=dict(model=label,time=t,candidate=mode,**record,
                        **{k:float(v[:,i].mean()) for k,v in differences.items()},
                        mc_repeat_ci_low=float(lows[i]),mc_repeat_ci_high=float(highs[i]))
            rows[(label,t)]=scores
    write_csv(out/'paired_scores.csv',summary_rows)
    write_csv(out/'scene_scores.csv',list(scene_rows.values()))
    write_csv(out/'ranking.csv',ranking)
    # Arrays are retained at the repeat/scene level; never call finite MC an exact Q label.
    np.savez_compressed(out/'predictions.npz',**{f'{key}_{metric}':value
        for key,scores in raw_predictions.items() for metric,value in scores.items()})
    atomic_json(out/'summary.json',dict(paired_scores=summary_rows,ranking=ranking))
    style(); figures=out/'figures'
    for label in meta['groups']:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ri,at in enumerate(meta['records']):
            t=int(at); mask=np.array([r['good'] for r in meta['records'][at]])
            scores=rows[(label,t)]
            for mi,mode in enumerate(MODES[1:],1):
                mc=(scores['mc_q'][mi]-scores['mc_q'][0]).mean(0)[mask]
                for ci,score in enumerate(['td1','gae128']):
                    pred=(scores[score][mi]-scores[score][0]).mean(0)[mask]
                    axes[ri,ci].scatter(mc,pred,s=22,alpha=.7,label=mode,color=COLORS[mi-1])
            for ci,score in enumerate(['td1','gae128']):
                ax=axes[ri,ci]; ax.axhline(0,color='grey',lw=.8); ax.axvline(0,color='grey',lw=.8)
                ax.set(xlabel='Paired MC return difference vs policy',ylabel=f'Predicted {score} difference',
                       title=f'{label}, t{t}, {mask.sum()} preselected good states')
                ax.legend()
        save_figure(fig,figures,f'{label}_action_ranking')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ri,label in enumerate(meta['groups']):
        for ci,at in enumerate(meta['records']):
            t=int(at); mask=np.array([r['good'] for r in meta['records'][at]])
            rewards,_,discount=curves[(label,t)]
            for mi,mode in enumerate(MODES[1:],1):
                delta=(rewards[mi]-rewards[0]).mean(0)[mask].mean(0)
                cumulative=np.cumsum(delta*discount**np.arange(len(delta)))
                axes[ri,ci].plot(np.arange(1,len(delta)+1),cumulative,label=mode,color=COLORS[mi-1])
            axes[ri,ci].axhline(0,color='grey',lw=.8)
            axes[ri,ci].set(xlabel='Steps after the one-action intervention',ylabel='Cumulative discounted reward difference',
                            title=f'{label}, t{t}; then resume original actor')
            axes[ri,ci].legend()
    save_figure(fig,figures,'effect_arrival')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ri,label in enumerate(meta['groups']):
        for ci,at in enumerate(meta['records']):
            t=int(at); ax=axes[ri,ci]
            for mi,mode in enumerate(MODES[1:]):
                selected=[next(r for r in summary_rows if (r['model'],r['time'],r['subset'],r['candidate'],r['metric'])
                              ==(label,t,'good',mode,key)) for key in ['mc_q','td1','gae128','reward1']]
                y=np.array([r['mean'] for r in selected]);lo=np.array([r['nested_low'] for r in selected]);hi=np.array([r['nested_high'] for r in selected])
                ax.errorbar(np.arange(4)+(mi-.5)*.15,y,yerr=np.stack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),
                            marker='o',capsize=3,ls='none',label=mode,color=COLORS[mi])
            ax.axhline(0,color='grey',lw=.8)
            ax.set(xticks=np.arange(4),xticklabels=['MC','TD1','GAE128','Immediate'],
                   ylabel='Difference vs policy',title=f'{label}, t{t}; good states')
            ax.legend()
    save_figure(fig,figures,'mean_action_values')
    lines=['# 好部署状态下的一步动作价值诊断','',
        '两个冻结100k模型，沿用原策略t150/t200的全部32个固定状态及预定义good子集（24/28）。每状态8次配对延续；只改第一个联合动作为原采样、半幅或悬停，之后全部恢复各自原策略。没有训练。',
        '第0次保留旧随机流，原策略整段结果精确复现旧分支报告；其余重复同时重采未来actor噪声和未来UE运动噪声，各候选保持配对，不改变当前物理状态。历史组沿用准确重建的原前缀历史，干预后的历史记录实际指令。','',
        '## 定义','',
        '- MC为完整环境四项奖励的终止折扣回报，不含actor熵；有限重复近似该一步动作方案随后跟随原策略的价值，不能称为精确Q。当前原动作也随重复重新采样，因此这里比较的是三种一步动作方案的期望，而非一个固定动作向量。',
        '- TD1 = r + gamma V(s_next) - V(s)；报告候选减原策略。共同V(s)偏差在该差值中抵消。',
        '- GAE128使用后续最多128步TD残差、gamma=.995、lambda=.95，仅为诊断截段，未重建真实训练rollout边界或优势归一化。另存完整剩余回合GAE及lambda=1的MC恒等式核验。',
        '- 只改变当前一步不等于持续悬停。持续悬停收益更好，不要求这一步悬停随后恢复旧策略也更好；整段行为的MC不能直接作一步V bootstrap标签。','',
        '## 较好部署子集：候选减原策略','',
        '|模型|起点|候选|MC差值[场景+未来重复区间]|TD1差值|GAE128差值|',
        '|---|---:|---|---:|---:|---:|']
    for label in meta['groups']:
        for at in meta['records']:
            for mode in MODES[1:]:
                chosen={r['metric']:r for r in summary_rows if (r['model'],r['time'],r['subset'],r['candidate'])==(label,int(at),'good',mode)}
                q=chosen['mc_q']
                lines.append(f'|{label}|{at}|{mode}|{q["mean"]:+.4f} [{q["nested_low"]:+.4f},{q["nested_high"]:+.4f}]|{chosen["td1"]["mean"]:+.4f}|{chosen["gae128"]["mean"]:+.4f}|')
    lines+=['','![均值与区间](figures/mean_action_values.png)','',
        '![原组动作排序](figures/M1-C_action_ranking.png)','',
        '![历史组动作排序](figures/M1-C-H_action_ranking.png)','',
        '![收益何时出现](figures/effect_arrival.png)','',
        '## 解释限制','',
        '场景区间先平均8次重复；nested区间同时重采场景和每场景未来重复，2000次、950001。t150和t200分开，不当成64个独立场景；都是研究诊断集，不是最终保留测试集。8次延续仍可能无法确定某些状态的MC动作排序。',
        'ranking.csv报告全部状态的符号一致率，以及每状态重复区间不跨零的子集；后者是有选择的描述，不能当作无偏总体准确率。不要把噪声很小的TD预测与有限MC均值当作无误差的真值比较。',
        '共享团队V不显式读联合动作，只通过下一状态判断。准确预测平均回报不保证20 m左右状态变化对应的价值差准确。逐状态核验 DeltaTD-DeltaMC = gamma * Delta(next-value prediction error)，以区分共同偏差和动作相对价值误差。',
        '本轮是联合动作干预，不足以单独证明每个agent的信用分配正确或错误。标准MAPPO归一化优势、逐agent概率比和熵正则都会影响实际更新；这里的原始优势符号不等于某个训练minibatch必然增大或减小动作概率。',
        '若MC区间跨零，就不能宣称已判定critic排序错误或actor违背正确价值信号；若单步收益不清晰但已有持续悬停收益明确，应继续检查多步保持的协调与信用传播，而非直接要求一步动作有同样收益。','',
        '全部逐步值、奖励分量、有效指令、逐状态评分和来源保留；源模型、归一化器及源码哈希不变，完整MC分量求和、终止零值和lambda=1恒等式通过。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([r for r in summary_rows if r['subset']=='good' and r['metric'] in ['mc_q','td1','gae128']],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=DEFAULT_OUT)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if args.rebuild: publish(args.output)
    else: build(args.output,args.repeats)
