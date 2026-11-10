"""Paired 5/20-step holds, then the frozen actor; matched K-step bootstrap.

Reuses exact policy continuations and one-step interventions from the previous
audit. No model/normalizer/optimizer update or production-code change.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_mappo_action_values import (
    FIELDS, MODES, SOURCE, HISTORY, RUNS, module_hashes, paired_interval,
)
from check_mappo_generalization import interval
from report_mappo_bridge import digest
from compare_online_time_factor import write_csv
from uav_research.boundary import reward_contributions
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import history_specification, stack_observations
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


PREVIOUS = Path('analysis/mappo_central_action_values_100k_seed0')
DEFAULT_OUT = 'analysis/mappo_central_action_blocks_100k_seed0'
BLOCKS = [5,20]


def block_action(original, mode, step, length):
    if mode not in MODES or length < 1:
        raise ValueError('Expected a valid mode and positive block length')
    action=np.array(original,copy=True)
    if step < length:
        if mode=='half':action*=.5
        elif mode=='hover':action.fill(0)
    return action


def block_scores(rewards, values, gamma, length):
    r=np.asarray(rewards,np.float64);v=np.asarray(values,np.float64)
    if v.shape!=(*r.shape[:-1],r.shape[-1]+1) or length < 1:
        raise ValueError('Invalid reward/value alignment or block length')
    np.testing.assert_array_equal(v[...,-1],0)
    n=r.shape[-1]; k=min(length,n)
    tail=np.zeros(r.shape[:-1]);returns=np.empty_like(r)
    for j in range(n-1,-1,-1):
        tail=r[...,j]+gamma*tail;returns[...,j]=tail
    prefix=(r[...,:k]*gamma**np.arange(k)).sum(-1)
    future=returns[...,k] if k<n else np.zeros_like(tail)
    boot=prefix+gamma**k*v[...,k]
    result=dict(mc=returns[...,0],bootstrap=boot-v[...,0],
        prefix=prefix,tail_mc=gamma**k*future,tail_prediction=gamma**k*v[...,k],
        boundary_error=v[...,k]-future,initial_value=v[...,0])
    np.testing.assert_allclose(result['mc'],result['prefix']+result['tail_mc'],atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(boot-result['mc'],gamma**k*result['boundary_error'],atol=1e-10,rtol=1e-11)
    return result


def run_block(cfg, policy, case, mode, repeat, length):
    snaps=copy.deepcopy(case['snapshots'])
    if repeat:
        for snap,record in zip(snaps,case['records']):
            snap['movement_rng']=np.random.default_rng(np.random.SeedSequence(
                [982001,record['seed'],case['time'],repeat])).bit_generator.state
    path_hash=hashlib.sha256()
    with isolated_rng(),torch.inference_mode():
        if repeat==0:torch.set_rng_state(case['torch_rng'].clone())
        else:torch.manual_seed(int(np.random.SeedSequence(
            [981001,case['time'],case['start'],repeat]).generate_state(1)[0]))
        with VectorEnvironment(cfg.env,len(snaps),'batch') as vector:
            vector.restore(snaps)
            obs=stack_observations([e.obs for e in vector.envs])
            state=np.stack([e.state for e in vector.envs])
            origin=np.stack([e.uav_xy for e in vector.envs])
            values=[policy.values(state).cpu().numpy()]
            diagnostics,contributions,raw,effective=[],[],[],[]
            for j in range(cfg.env.horizon-case['time']):
                original=policy.act(obs,deterministic=False)
                action=block_action(original,mode,j,length)
                if j<=length:raw.append(original.copy());effective.append(action.copy())
                obs,state,_,done,truncated,infos=vector.step(action)
                assert np.all(done==(case['time']+j==cfg.env.horizon-1)) and not truncated.any()
                if mode=='hover' and j<length:
                    np.testing.assert_array_equal(origin,np.stack([e.uav_xy for e in vector.envs]))
                if cfg.env.observation_history=='one_step':
                    for env,command in zip(vector.envs,action):
                        np.testing.assert_array_equal(env.history[:,:2],command)
                values.append(policy.values(state).cpu().numpy())
                diagnostics.append([[info[f] for f in FIELDS] for info in infos])
                terms=np.stack([reward_contributions(info,cfg.env) for info in infos])
                np.testing.assert_allclose(terms.sum(-1),[i['reward'] for i in infos],atol=1e-12,rtol=1e-12)
                contributions.append(terms)
                path_hash.update(np.stack([e.user_xy for e in vector.envs]).tobytes())
            rng_hash=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    data=dict(diagnostics=np.swapaxes(np.array(diagnostics),0,1),
              values=np.stack(values,-1).astype(np.float64),
              contributions=np.swapaxes(np.array(contributions),0,1),
              prefix_raw=np.swapaxes(np.stack(raw),0,1),
              prefix_effective=np.swapaxes(np.stack(effective),0,1))
    assert all(np.isfinite(v).all() for v in data.values())
    return data,dict(ue_path_sha256=path_hash.hexdigest(),final_actor_rng_sha256=rng_hash)


def adapt_case(case,cfg,histories):
    result=copy.deepcopy(case)
    if cfg.env.observation_history=='one_step':
        for i,snap in enumerate(result['snapshots']):
            snap.update(observation_history='one_step',history_specification=history_specification(cfg.env),
                        history=histories[(case['start'],case['time'])][i].copy())
            np.testing.assert_array_equal(snap['history'][:,:2],snap['action'])
    return result


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(RUNS['M1-C']/'config.yaml');configure_runtime(cfg)
    previous=json.loads((PREVIOUS/'experiment.json').read_text())
    assert previous['status']=='completed' and previous['repeats']==8
    cases=torch.load(SOURCE/'branch_states.pt',map_location='cpu',weights_only=False)
    histories=torch.load(HISTORY/'branch_histories.pt',map_location='cpu',weights_only=False)
    inputs=[PREVIOUS/'experiment.json',SOURCE/'branch_states.pt',HISTORY/'branch_histories.pt',
            Path(__file__),Path('scripts/check_mappo_action_values.py')]
    inputs += [run/name for run in RUNS.values() for name in ['last.pt','config.yaml']]
    inputs += [PREVIOUS/label/f't{t}/curves.npz' for label in RUNS for t in [150,200]]
    inputs += list(Path('src/uav_research').glob('*.py'))
    hashes={str(p):digest(p) for p in inputs}
    expected={(r['model'],r['time'],r['start'],r['repeat']):r for r in previous['checks']}
    meta=dict(status='running',records=previous['records'],blocks=BLOCKS,repeats=8,
        cpu_threads=8,numba_threads=4,baseline_source=str(PREVIOUS),source_sha256=hashes,
        groups={},checks=[],preflight=[],actor_updates=0,critic_updates=0)
    atomic_json(out/'experiment.json',meta);started=time.perf_counter()
    for label,run in RUNS.items():
        cfg,policy=policy_from_checkpoint(run/'last.pt','cpu');configure_runtime(cfg)
        assert cfg.run.cpu_threads==8 and cfg.run.numba_threads==4
        assert policy.provenance['checkpoint_env_steps']==100000 and cfg.env.max_speed==30
        assert previous['groups'][label]['config']==cfg.as_dict()
        policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
        before=module_hashes(policy)
        assert before==previous['groups'][label]['module_hashes']
        meta['groups'][label]=dict(config=cfg.as_dict(),run=str(run),module_hashes=before,completed=[])
        for at in [150,200]:
            with np.load(PREVIOUS/label/f't{at}/curves.npz') as saved:
                assert list(saved['fields'])==FIELDS and list(saved['modes'])==MODES
                reference={key:saved[key].copy() for key in ['diagnostics','values','contributions','raw_first_actions']}
                names=saved['contribution_names'].copy();seeds=saved['scene_seeds'].copy()
            parts={k:[] for k in BLOCKS}
            for original_case in [c for c in cases if c['time']==at]:
                case=adapt_case(original_case,cfg,histories)
                begin=case['start'];end=begin+len(case['records'])
                results={k:{mode:[] for mode in MODES[1:]} for k in BLOCKS}
                if at==150 and begin==0:
                    # K=1 must reproduce the existing runner before any longer holds.
                    for mi,mode in enumerate(MODES[1:],1):
                        data,_=run_block(cfg,policy,case,mode,0,1)
                        for key in ['diagnostics','values','contributions']:
                            np.testing.assert_array_equal(data[key],reference[key][mi,0,begin:end])
                        meta['preflight'].append(dict(model=label,mode=mode,K1_exact=True))
                for repeat in range(8):
                    for length in BLOCKS:
                        for mi,mode in enumerate(MODES[1:],1):
                            data,audit=run_block(cfg,policy,case,mode,repeat,length)
                            old=expected[(label,at,begin,repeat)]
                            assert all(audit[k]==old[k] for k in ['ue_path_sha256','final_actor_rng_sha256'])
                            np.testing.assert_array_equal(data['prefix_raw'][:,0],reference['raw_first_actions'][0,repeat,begin:end])
                            np.testing.assert_array_equal(data['values'][:,0],reference['values'][0,repeat,begin:end,0])
                            if length==20:
                                shorter=results[5][mode][-1]
                                np.testing.assert_array_equal(data['diagnostics'][:,:5],shorter['diagnostics'][:,:5])
                                np.testing.assert_array_equal(data['values'][:,:6],shorter['values'][:,:6])
                                np.testing.assert_array_equal(data['prefix_effective'][:,:5],shorter['prefix_effective'][:,:5])
                            results[length][mode].append(data)
                            meta['checks'].append(dict(model=label,time=at,start=begin,repeat=repeat,
                                                      length=length,mode=mode,**audit))
                    print(f'{label}, t{at}, scenes {begin}-{end-1}, future {repeat+1}/8: K5/K20 half/hover complete ({time.perf_counter()-started:.1f}s)',flush=True)
                for length in BLOCKS:
                    packed={key:np.stack([np.stack([d[key] for d in results[length][mode]])
                                           for mode in MODES[1:]]) for key in results[length]['half'][0]}
                    folder=out/label/f'K{length}/t{at}';folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'chunk_{begin}.npz',**packed)
                    parts[length].append(packed)
                meta['groups'][label]['completed'].append(dict(time=at,start=begin))
                atomic_json(out/'experiment.json',meta)
            for length in BLOCKS:
                merged={key:np.concatenate([p[key] for p in parts[length]],axis=2) for key in parts[length][0]}
                for key in ['diagnostics','values','contributions']:
                    merged[key]=np.concatenate([reference[key][:1],merged[key]],axis=0)
                np.savez_compressed(out/label/f'K{length}/t{at}/curves.npz',**merged,
                    fields=np.array(FIELDS),modes=np.array(MODES),command_modes=np.array(MODES[1:]),
                    contribution_names=names,scene_seeds=seeds,discount=cfg.algo.discount,
                    episode_steps=np.arange(at+1,cfg.env.horizon+1),block_steps=length)
        assert module_hashes(policy)==before
        assert all(p.grad is None for net in [policy.actor,policy.value] for p in net.parameters())
        meta['groups'][label]['module_hashes_unchanged']=True
    assert hashes=={str(p):digest(p) for p in inputs}
    meta.update(status='completed',elapsed_seconds=time.perf_counter()-started,source_hashes_unchanged=True,
                K1_preflight_exact=True,first_five_steps_identical=True)
    atomic_json(out/'experiment.json',meta);publish(out)


def publish(output):
    out=Path(output);meta=json.loads((out/'experiment.json').read_text());assert meta['status']=='completed'
    summary,ranking,scene_rows,duration_rows,components=[],[],[],[],[]
    all_scores,all_curves={},{}
    for label in meta['groups']:
        for at,records in meta['records'].items():
            t=int(at);masks=dict(all=np.ones(len(records),bool),good=np.array([r['good'] for r in records]))
            for length in [1,5,20]:
                path=Path(meta['baseline_source'])/label/f't{t}/curves.npz' if length==1 else out/label/f'K{length}/t{t}/curves.npz'
                with np.load(path) as d:
                    diagnostics=d['diagnostics'];rewards=diagnostics[...,FIELDS.index('reward')]
                    gamma=float(d['discount']);scores=block_scores(rewards,d['values'],gamma,length)
                    contribution=d['contributions'];component_names=list(d['contribution_names'])
                np.testing.assert_allclose((contribution*gamma**np.arange(rewards.shape[-1])[:,None]).sum(-2).sum(-1),
                                           scores['mc'],atol=1e-10,rtol=1e-11)
                k=min(length,rewards.shape[-1]);cov=diagnostics[...,FIELDS.index('coverage')]
                scores.update(release_coverage=cov[...,k-1],last100_coverage=cov[...,-100:].mean(-1),
                    post_release20_coverage=cov[...,k:k+20].mean(-1),final_coverage=cov[...,-1])
                all_scores[(label,t,length)]=scores;all_curves[(label,t,length)]=(rewards,cov,gamma)
                for mi,mode in enumerate(MODES[1:],1):
                    delta={key:value[mi]-value[0] for key,value in scores.items()}
                    np.testing.assert_allclose(delta['bootstrap']-delta['mc'],gamma**k*delta['boundary_error'],atol=1e-10,rtol=1e-11)
                    draws=np.random.default_rng(950001).integers(8,size=(2000,8))
                    intervals={key:np.quantile(delta[key][draws].mean(1),[.025,.975],axis=0) for key in ['mc','bootstrap']}
                    means={key:delta[key].mean(0) for key in ['mc','bootstrap']}
                    for subset,mask in masks.items():
                        for key,value in delta.items():
                            if key=='initial_value':continue
                            summary.append(dict(model=label,time=t,length=length,mode=mode,subset=subset,
                                metric=key,scenes=int(mask.sum()),**paired_interval(value[:,mask])))
                        mc,pred=means['mc'][mask],means['bootstrap'][mask]
                        confirmed=mask & ((intervals['mc'][0]>0)|(intervals['mc'][1]<0))
                        opposite=mask & (((intervals['mc'][0]>0)&(intervals['bootstrap'][1]<0))|
                                        ((intervals['mc'][1]<0)&(intervals['bootstrap'][0]>0)))
                        ranking.append(dict(model=label,time=t,length=length,mode=mode,subset=subset,
                            scenes=int(mask.sum()),sign_agreement=float((mc*pred>0).mean()),
                            correlation=float(np.corrcoef(mc,pred)[0,1]) if mc.std()>1e-12 and pred.std()>1e-12 else None,
                            relative_rmse=float(np.sqrt(np.mean((mc-pred)**2))),
                            mc_repeat_interval_excludes_zero=int(confirmed.sum()),
                            both_nonzero_opposite_intervals=int(opposite.sum())))
                        for j,name in enumerate(component_names):
                            c=((contribution[mi,...,j]-contribution[0,...,j])*gamma**np.arange(rewards.shape[-1])).sum(-1)
                            components.append(dict(model=label,time=t,length=length,mode=mode,subset=subset,
                                                    component=str(name),**paired_interval(c[:,mask])))
                    for i,record in enumerate(records):
                        scene_rows.append(dict(model=label,time=t,length=length,mode=mode,**record,
                            **{key:float(value[:,i].mean()) for key,value in delta.items()},
                            mc_low=float(intervals['mc'][0,i]),mc_high=float(intervals['mc'][1,i]),
                            bootstrap_low=float(intervals['bootstrap'][0,i]),bootstrap_high=float(intervals['bootstrap'][1,i])))
            for left,right in [(1,5),(1,20),(5,20)]:
                for mi,mode in enumerate(MODES[1:],1):
                    for subset,mask in masks.items():
                        for metric in ['mc','last100_coverage','final_coverage']:
                            a=all_scores[(label,t,left)][metric];b=all_scores[(label,t,right)][metric]
                            delta=(b[mi]-b[0])-(a[mi]-a[0])
                            duration_rows.append(dict(model=label,time=t,mode=mode,subset=subset,
                                comparison=f'K{right}-K{left}',metric=metric,**paired_interval(delta[:,mask])))
    for name,rows in [('paired_scores.csv',summary),('ranking.csv',ranking),('scene_scores.csv',scene_rows),
                      ('duration_differences.csv',duration_rows),('reward_components.csv',components)]:write_csv(out/name,rows)
    atomic_json(out/'summary.json',dict(paired_scores=summary,ranking=ranking,duration_differences=duration_rows))
    style();figures=out/'figures'
    for label in meta['groups']:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,at in enumerate(meta['records']):
            t=int(at)
            for ci,mode in enumerate(MODES[1:]):
                ax=axes[ri,ci]
                for j,key in enumerate(['mc','bootstrap']):
                    chosen=[next(r for r in summary if (r['model'],r['time'],r['length'],r['mode'],r['subset'],r['metric'])==
                                 (label,t,k,mode,'good',key)) for k in [1,5,20]]
                    y=np.array([r['mean'] for r in chosen]);lo=np.array([r['nested_low'] for r in chosen]);hi=np.array([r['nested_high'] for r in chosen])
                    ax.errorbar(np.arange(3)+(j-.5)*.12,y,yerr=np.stack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),
                                marker='o',ls='-',capsize=3,label=key,color=COLORS[j])
                ax.axhline(0,color='grey',lw=.8);ax.set(xticks=np.arange(3),xticklabels=['1','5','20'],
                    xlabel='Intervention length K (steps)',ylabel='Difference vs original policy',title=f'{label}, t{t}, {mode}')
                ax.legend()
        save_figure(fig,figures,f'{label}_bootstrap_vs_mc')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ri,label in enumerate(meta['groups']):
        for ci,at in enumerate(meta['records']):
            t=int(at);mask=np.array([r['good'] for r in meta['records'][at]])
            for color,length in zip(COLORS,[1,5,20]):
                rewards,_,gamma=all_curves[(label,t,length)]
                delta=(rewards[2]-rewards[0]).mean(0)[mask].mean(0)
                axes[ri,ci].plot(np.arange(1,len(delta)+1),np.cumsum(delta*gamma**np.arange(len(delta))),
                                label=f'hover K={length}',color=color)
                axes[ri,ci].axvline(length,color=color,ls=':',alpha=.5)
            axes[ri,ci].axhline(0,color='grey',lw=.8);axes[ri,ci].legend()
            axes[ri,ci].set(xlabel='Steps after intervention begins',ylabel='Cumulative discounted reward difference',title=f'{label}, t{t}; return to actor after K')
    save_figure(fig,figures,'hover_effect_arrival')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ri,label in enumerate(meta['groups']):
        for ci,at in enumerate(meta['records']):
            t=int(at);mask=np.array([r['good'] for r in meta['records'][at]])
            for color,length in zip(COLORS,[1,5,20]):
                _,cov,_=all_curves[(label,t,length)]
                axes[ri,ci].plot(np.arange(1,cov.shape[-1]+1),cov[2].mean(0)[mask].mean(0)*100,label=f'hover K={length}',color=color)
            _,base,_=all_curves[(label,t,1)]
            axes[ri,ci].plot(np.arange(1,base.shape[-1]+1),base[0].mean(0)[mask].mean(0)*100,label='original policy',color='black',ls='--')
            axes[ri,ci].legend();axes[ri,ci].set(xlabel='Steps after intervention begins',ylabel='Coverage (%)',title=f'{label}, t{t}; good states')
    save_figure(fig,figures,'hover_coverage_after_release')
    lines=['# 5步／20步保持与对应bootstrap诊断','',
        '冻结M1-C、M1-C-H的100k模型，沿用t150/t200各32个固定快照，good子集24/28；每场景8套配对未来actor和UE噪声。原策略及K1完整复用旧数据，新增K5/K20半幅和悬停，之后恢复原策略，不训练。','',
        '## 定义与核对','',
        'K步bootstrap为 sum(j=0..K-1) gamma^j r_(t+j) + gamma^K V(s_(t+K)) − V(s_t)。只在强制保持结束后使用原策略V，不能用一步TD去评价整段保持。MC使用原回合真实终止、完整环境奖励，不含actor熵。',
        '逐状态核对：Delta bootstrap − Delta MC = gamma^K * Delta[V(s_(t+K))−G_(t+K)]；并分解MC差值为保持期间实际折扣收益与恢复原策略后的折扣收益。',
        'K1预检逐元素复现旧干预；K5/K20前5步轨迹及价值相同；所有分支UE路径与动作随机流消耗与旧实验一致。悬停期间UAV位置不变，放开后恢复当前actor完整采样。历史记录实际指令；actor/critic/归一化器和源文件哈希不变。','',
        '## 悬停：较好部署子集','',
        '|模型|起点|K|MC差值[nested区间]|对应K步bootstrap差值[nested区间]|保持期间收益差|恢复后收益差|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in meta['groups']:
        for at in meta['records']:
            for length in [1,5,20]:
                chosen={r['metric']:r for r in summary if (r['model'],r['time'],r['length'],r['mode'],r['subset'])==(label,int(at),length,'hover','good')}
                q,b=chosen['mc'],chosen['bootstrap']
                lines.append(f'|{label}|{at}|{length}|{q["mean"]:+.3f} [{q["nested_low"]:+.3f},{q["nested_high"]:+.3f}]|{b["mean"]:+.3f} [{b["nested_low"]:+.3f},{b["nested_high"]:+.3f}]|{chosen["prefix"]["mean"]:+.3f}|{chosen["tail_mc"]["mean"]:+.3f}|')
    lines+=['','![原组](figures/M1-C_bootstrap_vs_mc.png)','',
            '![历史组](figures/M1-C-H_bootstrap_vs_mc.png)','',
            '![收益出现时间](figures/hover_effect_arrival.png)','',
            '![解除保持后的覆盖](figures/hover_coverage_after_release.png)','',
            '## 统计和解释边界','',
            '每场景先平均8次；同时重采场景和每场景未来重复的nested区间2000次、种子950001，另存仅重采场景区间。不是训练种子区间。t150、t200单独比较，不视为64个独立场景；保留测试集未用。MC有限、逐状态排序仍有不确定性，区间筛选只作探索性描述。',
            '半幅干预在每个状态重新采样当前actor再缩小，不是回放旧动作；悬停固定当前UAV位置，UE继续移动。历史组来自原策略前缀，是相同布局的干预诊断，不是自身on-policy泛化测试。',
            'K越大既改变了干预行为，也改变bootstrap时刻；不能把所有变化只归因于估计器长短。K步量评价的是一段联合动作，不是标准逐agent PPO的单步优势，不能据此直接替换GAE或认定哪个agent应该负责。',
            '完整奖励及逐分量MC均保存；覆盖改善与任务回报改善分别报告。若保持期收益为正但恢复后被抵消，提示原策略继续破坏布局；若MC收益清晰但匹配K步bootstrap反向，则继续检查保持结束状态的相对价值误差。','',
            '逐场景评分、半幅结果、持续时间配对差值和奖励分量分别见scene_scores.csv、paired_scores.csv、duration_differences.csv、reward_components.csv。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([r for r in summary if r['subset']=='good' and r['mode']=='hover' and r['metric'] in ['mc','bootstrap']],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=DEFAULT_OUT)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if args.rebuild:publish(args.output)
    else:build(args.output)
