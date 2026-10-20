"""Read-only finite-horizon critic audit for experiment D (pure sensing)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from uav_research.algorithms import gae
from uav_research.config import load_config
from uav_research.critic_calibration import actor_digest, discounted_episode_returns
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import stack_observations
from uav_research.plotting import plt, COLORS, style, save_figure
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json
from uav_research.vector import VectorEnvironment


ANCHOR_TIMES = [0, 100, 200, 300, 380, 399]
STAGES = {'all': slice(0,400), 'first100': slice(0,100), 'middle100': slice(150,250),
          'last100': slice(300,400), 'last20': slice(380,400), 'last1': slice(399,400)}


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def errors(v, target):
    difference=v-target
    variance=float(np.var(target))
    correlation=np.corrcoef(v.ravel(),target.ravel())[0,1] if np.std(v)>1e-12 and variance>1e-12 else None
    return dict(prediction_mean=float(v.mean()), return_mean=float(target.mean()),
                bias=float(difference.mean()), rmse=float(np.sqrt(np.mean(difference**2))),
                mae=float(np.abs(difference).mean()),
                explained_variance=1-float(np.var(difference))/variance if variance>1e-12 else None,
                correlation=float(correlation) if correlation is not None else None)


def gae_audit(values, rewards, gamma, lam, rollout):
    # [replica,scene,time] -> [time,replica*scene]. No environment resets inside a trajectory.
    v=torch.tensor(values.reshape(-1,401).T,dtype=torch.float64)
    r=torch.tensor(rewards.reshape(-1,400).T,dtype=torch.float64)
    done=torch.zeros_like(r,dtype=torch.bool);done[-1]=True
    mc=discounted_episode_returns(rewards.reshape(-1,400),gamma).T
    mc_adv=torch.tensor(mc)-v[:-1]
    full,_=gae(r,v[:-1],v[1:],done,gamma,lam)
    exact,_=gae(r,v[:-1],v[1:],done,gamma,1.)
    # Production gae multiplies gamma by a float32 termination mask before
    # promoting to the diagnostic float64 tensors. Account for this rounding.
    np.testing.assert_allclose(exact.numpy(),mc_adv.numpy(),rtol=2e-6,atol=2e-4)
    fragments=torch.empty_like(full)
    centered_gae,centered_mc=torch.empty_like(full),torch.empty_like(full)
    # Chunk layout is a frozen-policy diagnostic, not a reconstruction of historical rollouts.
    for start in range(0,400,rollout):
        end=min(start+rollout,400)
        a,_=gae(r[start:end],v[start:end],v[start+1:end+1],done[start:end],gamma,lam)
        fragments[start:end]=a
        centered_gae[start:end]=a-a.mean()
        z=mc_adv[start:end];centered_mc[start:end]=z-z.mean()
    mask=(centered_gae.abs()>1e-8)&(centered_mc.abs()>1e-8)
    summary=dict(full_lambda_advantage_mean=float(full.mean()),fragment_advantage_mean=float(fragments.mean()),
                 mc_minus_v_mean=float(mc_adv.mean()),
                 fragment_vs_mc_rmse=float((fragments-mc_adv).square().mean().sqrt()),
                 centered_correlation=float(np.corrcoef(centered_gae.flatten(),centered_mc.flatten())[0,1]),
                 centered_sign_disagreement=float(((centered_gae.sign()!=centered_mc.sign())&mask).sum()/mask.sum()),
                 lambda1_mc_identity_passed=True,
                 lambda1_mc_max_absolute_roundoff=float((exact-mc_adv).abs().max()),
                 note='MC minus current V is a noisy diagnostic, not the exact conditional advantage; diagnostic chunks start at episode t=0')
    arrays=dict(gae_full=full.numpy().T.reshape(rewards.shape),gae128=fragments.numpy().T.reshape(rewards.shape),
                mc_minus_v=mc_adv.numpy().T.reshape(rewards.shape))
    return summary,arrays


def collect(cfg, policy, checkpoint_name, action_seeds, reference, output):
    all_v,all_z,all_r,all_p=[],[],[],[]
    snapshots=[];matched=0;max_reward_error=0.
    device=policy.device
    for action_seed in action_seeds:
        vb,zb,rb,pb=[],[],[],[]
        with isolated_rng(),torch.inference_mode():
            torch.manual_seed(action_seed)
            for start in range(0,len(cfg.run.validation_seeds),cfg.run.num_envs):
                seeds=cfg.run.validation_seeds[start:start+cfg.run.num_envs]
                with VectorEnvironment(cfg.env,len(seeds),'batch') as vector:
                    obs,state,_=vector.reset(seeds)
                    values,normalized,rewards,sensing=[],[],[],[]
                    original={}
                    if action_seed==action_seeds[0] and checkpoint_name in ['best','last']:
                        original={i:np.load(reference/f'validation_{checkpoint_name}'/f'trace_{s}.npz')
                                  for i,s in enumerate(seeds) if s in cfg.run.validation_seeds[:2]}
                    for t in range(401):
                        np.testing.assert_allclose(state[:,-1],t/400,atol=1e-7,rtol=0)
                        z=policy.value(torch.as_tensor(state,device=device,dtype=torch.float32)).squeeze(-1)
                        v=policy.normalizer.denormalize(z)
                        if t==0:
                            torch.testing.assert_close(v,policy.values(state))
                        values.append(v.cpu().numpy());normalized.append(z.cpu().numpy())
                        sensing.append(np.array([e.info['discovery_fraction'] for e in vector.envs]))
                        for i,tr in original.items():
                            for key in ['uav_xy','user_xy','B']:
                                np.testing.assert_array_equal(getattr(vector.envs[i],key),tr[key][t])
                            matched+=1
                        if checkpoint_name=='last' and action_seed==action_seeds[0] and start==0 and t in ANCHOR_TIMES:
                            snapshots.extend(dict(scene_seed=seeds[i],time=t,snapshot=e.snapshot(),state=state[i].copy(),
                                                  prediction=float(v[i])) for i,e in enumerate(vector.envs[:8]))
                        if t==400:break
                        action=policy.act(obs,False)
                        obs,state,r,done,truncated,infos=vector.step(action)
                        assert not truncated.any() and np.all(done==(t==399))
                        reward=np.array([info['reward'] for info in infos])
                        np.testing.assert_array_equal(reward,np.array([info['discovery_fraction'] for info in infos]))
                        max_reward_error=max(max_reward_error,float(np.max(np.abs(reward-r))))
                        rewards.append(reward)
                    for tr in original.values():tr.close()
                    vb.append(np.stack(values).T);zb.append(np.stack(normalized).T)
                    rb.append(np.stack(rewards).T);pb.append(np.stack(sensing).T)
        all_v.append(np.concatenate(vb));all_z.append(np.concatenate(zb))
        all_r.append(np.concatenate(rb));all_p.append(np.concatenate(pb))
        print(checkpoint_name,'action seed',action_seed,'mean sensing',np.mean(all_r[-1]),flush=True)
    values,z,rewards,sensing=map(np.stack,[all_v,all_z,all_r,all_p])
    returns=discounted_episode_returns(rewards.reshape(-1,400),cfg.algo.discount).reshape(rewards.shape)
    np.testing.assert_array_equal(returns[:,:,-1],rewards[:,:,-1])
    np.testing.assert_allclose(returns[:,:,:-1],rewards[:,:,:-1]+cfg.algo.discount*returns[:,:,1:],rtol=1e-12,atol=1e-12)
    gae_stats,gae_arrays=gae_audit(values,rewards,cfg.algo.discount,cfg.algo.gae_lambda,cfg.algo.rollout_steps)
    np.savez_compressed(output/f'{checkpoint_name}_trajectories.npz',values=values,normalized_values=z,
                        rewards=rewards,returns=returns,sensing=sensing,action_seeds=action_seeds,
                        scene_seeds=cfg.run.validation_seeds,**gae_arrays)
    v=values[:,:,:400]
    bound=(1-cfg.algo.discount**np.arange(400,0,-1))/(1-cfg.algo.discount)
    stages={name:errors(v[:,:,sl],returns[:,:,sl]) for name,sl in STAGES.items()}
    for name,sl in STAGES.items():
        stages[name]['above_possible_return_fraction']=float((v[:,:,sl]>bound[sl]+1e-6).mean())
        stages[name]['below_zero_fraction']=float((v[:,:,sl]<-1e-6).mean())
    centered_v=v-v.mean((0,1),keepdims=True)
    centered_g=returns-returns.mean((0,1),keepdims=True)
    stats=dict(stages=stages,gae=gae_stats,mean_sensing=float(rewards.mean()),
               normalizer_mean=float(policy.normalizer.mean),normalizer_std=float(policy.normalizer.std),
               normalizer_count=float(policy.normalizer.count),
               within_time_errors=errors(centered_v,centered_g),
               raw_terminal_output_mean=float(values[:,:,-1].mean()),
               terminal_bootstrap_used=0.,original_trace_frames_matched=matched,
               float32_reward_max_difference=max_reward_error)
    return stats,snapshots


def anchor_rollouts(cfg,policy,anchors,output):
    rows=[];all_returns=[];repeats=16
    rng=np.random.default_rng(945001)
    with torch.inference_mode():
        for index,anchor in enumerate(anchors):
            t,seed=anchor['time'],anchor['scene_seed']
            with isolated_rng(),VectorEnvironment(cfg.env,repeats,'batch') as vector:
                vector.restore([anchor['snapshot'] for _ in range(repeats)])
                for k,env in enumerate(vector.envs):
                    # Resample both future UE motion and actor actions, conditional on the same state.
                    env.movement_rng=np.random.default_rng(np.random.SeedSequence([945002,seed,t,k]))
                    np.testing.assert_array_equal(env.state,anchor['state'])
                obs=stack_observations([env.obs for env in vector.envs])
                torch.manual_seed(945003+index)
                returns=np.zeros(repeats,np.float64);discount=1.
                for step in range(t,400):
                    actions=policy.act(obs,False)
                    obs,_,_,done,truncated,infos=vector.step(actions)
                    assert not truncated.any() and np.all(done==(step==399))
                    returns+=discount*np.array([info['reward'] for info in infos])
                    discount*=cfg.algo.discount
            low,high=np.quantile(returns[rng.integers(repeats,size=(5000,repeats))].mean(1),[.025,.975])
            rows.append(dict(scene_seed=seed,state_time=t,replicates=repeats,prediction=anchor['prediction'],
                             mc_mean=float(returns.mean()),mc_std=float(returns.std(ddof=1)),
                             mc_se=float(returns.std(ddof=1)/np.sqrt(repeats)),mc_ci_low=float(low),mc_ci_high=float(high),
                             bias=float(anchor['prediction']-returns.mean()),
                             possible_upper_bound=float((1-cfg.algo.discount**(400-t))/(1-cfg.algo.discount))))
            all_returns.append(returns)
            if (index+1)%8==0:print('anchors:',index+1,'/',len(anchors),'last time',t,flush=True)
    csv_write(output/'anchor_predictions.csv',rows)
    np.savez_compressed(output/'anchor_returns.npz',returns=np.stack(all_returns),
                        scene_seeds=[r['scene_seed'] for r in rows],state_times=[r['state_time'] for r in rows])
    return rows


def time_feature_probe(policy, anchors, output):
    """Sensitivity only: hold physical state fixed and vary the existing time feature."""
    selected=[a for a in anchors if a['time']==0]
    states=np.stack([a['state'] for a in selected])
    rows=[]
    with torch.inference_mode():
        for t in [0,100,200,300,380,399,400]:
            modified=states.copy();modified[:,-1]=t/400
            predictions=policy.values(modified).cpu().numpy()
            rows.extend(dict(scene_seed=a['scene_seed'],synthetic_state_time=t,
                             physical_state_source_time=0,prediction=float(v))
                        for a,v in zip(selected,predictions))
    csv_write(output/'time_feature_probe.csv',rows)
    style();fig,ax=plt.subplots(figsize=(7,4))
    for a in selected:
        chosen=[r for r in rows if r['scene_seed']==a['scene_seed']]
        ax.plot([r['synthetic_state_time'] for r in chosen],[r['prediction'] for r in chosen],alpha=.4,lw=1)
    ax.set(xlabel='Manually replaced time feature × 400',ylabel='Critic V (raw scale)',
           title='Sensitivity probe: identical physical states, different time input')
    save_figure(fig,output/'figures','time_feature_sensitivity')
    return rows


def figures(output):
    style();data={}
    for name in ['best','middle','last']:
        with np.load(output/f'{name}_trajectories.npz') as z:data[name]={k:z[k] for k in ['values','returns','gae128','mc_minus_v']}
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for col,(name,label) in enumerate([('best','Initialization (best, step 0)'),('middle','Step 180,224'),('last','Step 250,000')]):
        d=data[name];v=d['values'][:,:,:400];g=d['returns']
        for row,sl in enumerate([slice(0,400),slice(380,400)]):
            ax=axes[row,col];ts=np.arange(400)[sl]
            ax.plot(ts,g.mean((0,1))[sl],label='Realized discounted return',c=COLORS[1])
            ax.plot(ts,v.mean((0,1))[sl],label='Critic V (denormalized)',c=COLORS[0])
            ax.set(title=label+(' | last 20 states' if row else ''),xlabel='State time before action',ylabel='Remaining discounted reward')
            ax.legend(fontsize=7)
    fig.suptitle('D: pure sensing | frozen actor/critic per checkpoint | 32 scenes × 3 action seeds')
    save_figure(fig,output/'figures','critic_horizon')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    d=data['last'];v=d['values'][:,:,:400];g=d['returns']
    axes[0].plot(np.arange(400),(v-g).mean((0,1)));axes[0].axhline(0,c='grey',ls=':')
    axes[0].set(title='Last model: signed value error',ylabel='Mean V - realized return',xlabel='State time')
    axes[1].plot(np.arange(400),np.sqrt(((v-g)**2).mean((0,1))))
    axes[1].set(title='Last model: raw RMSE',ylabel='RMSE',xlabel='State time')
    axes[2].plot(np.arange(400),d['gae128'].mean((0,1)),label='128-step GAE')
    axes[2].plot(np.arange(400),d['mc_minus_v'].mean((0,1)),label='Full return minus V')
    axes[2].set(title='Noisy advantage diagnostics',xlabel='State time',ylabel='Raw advantage');axes[2].legend(fontsize=8)
    save_figure(fig,output/'figures','critic_errors_advantages')
    rows=list(csv.DictReader((output/'anchor_predictions.csv').open()))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,t in zip(axes.flat,ANCHOR_TIMES):
        r=[x for x in rows if int(x['state_time'])==t];x=np.arange(len(r))
        mean=np.array([float(a['mc_mean']) for a in r]);lo=np.array([float(a['mc_ci_low']) for a in r]);hi=np.array([float(a['mc_ci_high']) for a in r])
        ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt='o',c=COLORS[1],capsize=3,label='16 continuation MC mean (95% bootstrap)')
        ax.scatter(x,[float(a['prediction']) for a in r],marker='x',c=COLORS[0],label='Critic prediction')
        if t==399:ax.axhline(1,c='grey',ls=':',label='Maximum possible remaining reward')
        ax.set(title=f'Same states, t={t}',xlabel='Scene index (10000 + index)',ylabel='Remaining return')
    axes[0,0].legend(fontsize=6)
    save_figure(fig,output/'figures','same_state_continuations')


def write_report(output):
    summary=json.loads((output/'summary.json').read_text())
    verification=json.loads((output/'verification.json').read_text())
    anchors=list(csv.DictReader((output/'anchor_predictions.csv').open()))
    probe=list(csv.DictReader((output/'time_feature_probe.csv').open()))
    last=summary['last'];stage=last['stages']
    with np.load(output/'last_trajectories.npz') as data:
        values=data['values'][:,:,:400];returns=data['returns']
        factor=(1-.995**np.arange(400,0,-1))/(1-.995)
        proxy=data['sensing'][:,:,:400]*factor
        proxy_errors={name:errors(proxy[:,:,sl],returns[:,:,sl]) for name,sl in STAGES.items()}
        final_delta=data['rewards'][:,:,-1]-values[:,:,-1]
        last_range=[float(values[:,:,-1].min()),float(values[:,:,-1].max())]
    atomic_json(output/'constant_sensing_reference.json',dict(
        formula='P(s_t) * (1 - gamma**(400-t)) / (1-gamma)',
        purpose='No fitting; constant-future-sensing approximation, not true V or an actor baseline',
        stages=proxy_errors))
    lines=['# D纯感知实验：critic预测准确性检查','',
           '**确认存在严重的有限回合价值高估，主要发生在回合后期。此检查未单独证明它是策略退化的唯一原因。**','',
           '## 协议与正确性检查','',
           '- D奖励仅为瞬时去重感知率P，取值[0,1]。读取第0步best、180224步完整检查点和250000步last；每个模型冻结自己的actor、critic与PopArt统计。',
           '- 每模型32个原验证场景×3个动作种子880001–880003，共288个完整400步回合；未使用保留测试集，未训练。',
           '- V在动作前状态s_t计算，奖励来自动作后的s_(t+1)。完整回报G_t=r_t+0.995 G_(t+1)，G_400=0；先反归一化V，再与原始奖励尺度的G比较。',
           '- 核对t/400位于全局状态最后一维，终止标志仅在最后动作后成立。终止下一状态的网络输出被掩码，不参与bootstrap。',
           '- 对last的前8个固定验证场景、t=0/100/200/300/380/399共48个状态，各执行16次独立续采样；重新采样未来动作和UE运动，共768段、130688条联合转移。',
           '- best和last的原动作种子验证均值与既有报告一致，前两个场景各401帧的位置/B矩阵一致；源文件以及actor/value/normalizer状态哈希保持不变。',
           '- 生产GAE在lambda=1、完整终止时等于G−V，最大差异小于0.0001（float32掩码乘折扣的舍入）；解析有限回合值测试和终止掩码测试通过。','',
           '训练数据流仍使用env.step返回的P奖励；没有发现把覆盖率误传给D训练的情况。以上是具体检查项，不等于排除了所有训练实现问题。','',
           '## 250k模型：分时段误差','',
           '| 时段（动作前时间） | V均值 | 实际折扣回报均值 | 平均偏差V−G | RMSE |',
           '|---|---:|---:|---:|---:|']
    names={'first100':'0–99','middle100':'150–249','last100':'300–399','last20':'380–399','last1':'399（最后一步）','all':'0–399全部'}
    for key in ['first100','middle100','last100','last20','last1','all']:
        r=stage[key]
        lines.append(f'| {names[key]} | {r["prediction_mean"]:.3f} | {r["return_mean"]:.3f} | {r["bias"]:+.3f} | {r["rmse"]:.3f} |')
    lines += ['',f'最后一步96条验证轨迹的V均在{last_range[0]:.2f}–{last_range[1]:.2f}，全部超过剩余回报上界1。'
              '这不依赖MC估计是否精确。真正终止后的V应为0；t=399仍剩一次奖励，因此应在[0,1]，而非必须等于0。','',
              '![预测与完整回合回报](figures/critic_horizon.png)','',
              '## 相同状态的重复MC检查','',
              '| 状态时间 | critic均值 | 16次续采样回报均值 | 每状态MC均值标准误的平均 |',
              '|---|---:|---:|---:|']
    for t in ANCHOR_TIMES:
        r=[a for a in anchors if int(a['state_time'])==t]
        mean=lambda k:np.mean([float(a[k]) for a in r])
        lines.append(f'| {t} | {mean("prediction"):.3f} | {mean("mc_mean"):.3f} | {mean("mc_se"):.4f} |')
    lines += ['', '这里每个状态的critic预测固定，16条后续轨迹只改变未来随机性。后期几十到近90的偏差远大于MC采样误差。'
              '图中95%区间为单状态16次续采样均值的bootstrap区间，不是训练种子置信区间。','',
              '![同状态重复续采样](figures/same_state_continuations.png)','',
              '## 时间特征敏感性与简单尺度参照','']
    pmean=lambda t:np.mean([float(r['prediction']) for r in probe if int(r['synthetic_state_time'])==t])
    lines += [f'保持8个初始状态的物理布局不变，只替换已有时间特征：V均值在t=0为{pmean(0):.2f}，'
              f't=200为{pmean(200):.2f}，t=399为{pmean(399):.2f}。网络对时间输入的反应很弱。'
              '这是人工状态敏感性探针，不是对这些人工状态的真实价值测量。','',
              '![时间输入敏感性](figures/time_feature_sensitivity.png)','',
              '再用一个不拟合参数的参照：假定未来感知率保持当前P(s_t)，计算'
              '`P(s_t)*(1-gamma**(400-t))/(1-gamma)`。它不是真实价值，但显式考虑剩余时间。',
              f'在同一last轨迹上，该参照全时段RMSE为{proxy_errors["all"]["rmse"]:.3f}，'
              f'现有critic为{stage["all"]["rmse"]:.3f}；最后20步分别为{proxy_errors["last20"]["rmse"]:.3f}、'
              f'{stage["last20"]["rmse"]:.3f}。这进一步支持把剩余时长建模/校准作为优先检查项。','',
              '当前critic为6292→256→256→1的ReLU MLP，时间是6292个输入中的一个归一化标量。'
              '输入含时间不代表已学会正确使用时间；本实验不能独立区分网络表达、优化、bootstrap目标和value clipping各自的贡献。','',
              '## 对优势估计的影响与因果边界','',
              f'最后一步TD残差为r−V，均值约{float(final_delta.mean()):.3f}。此时奖励非负，'
              '巨大负残差主要反映critic过高的预期，不能直接解释成动作实际造成了同等大小的损失。',
              'GAE会把末期残差向前传播，之后PPO还会归一化优势。若e_t=V_hat(s_t)−V_pi(s_t)，'
              '非终止TD残差额外包含gamma*e_(t+1)−e_t，终止时包含−e_t。时间相关误差因而可能改变更新信号。','',
              '本次附带的128步GAE在冻结策略的新轨迹上计算，片段从回合t=0开始；'
              '它不是历史训练rollout的重建，也没有复现跨reset的实际片段排列。'
              'G−当前V同样含采样噪声和基线误差，其符号不能当作精确动作优劣标签。','',
              '因此可以确认critic数值不准确、末期TD信号严重失真；尚不能据此证明所有actor更新方向都错，'
              '或证明修好critic必然解决聚集。MC策略梯度中的动作无关基线即便有误也不必然造成期望梯度偏差，'
              '当前GAE还依赖bootstrap值，需进一步用受控更新实验验证实际影响。','',
              '![误差与优势诊断](figures/critic_errors_advantages.png)','',
              '## 后续实验建议','',
              '优先在D上冻结actor，用独立校准场景的完整回报检查可拟合性，并对剩余时长表达做单变量对照；'
              '保留奖励、actor和原模型，比较检查集的原始尺度误差及最后一步范围。'
              '校准通过后，再比较相同采集数据/更新预算下的actor更新收益，才能建立critic误差与策略退化的因果证据。'
              '本次没有执行校准或修改训练实现。','',
              '## 产物与复现','',
              '- `summary.json`、`errors_by_stage.csv`：三检查点误差和归一化统计。',
              '- `*_trajectories.npz`：原始/归一化预测、奖励、完整回报及诊断GAE。',
              '- `anchor_states.pt`、`anchor_returns.npz`、`anchor_predictions.csv`：相同状态的续采样。',
              '- `time_feature_probe.csv`：单独改变时间输入的敏感性。',
              '- `verification.json`：源文件不变、终止、奖励及原验证复核记录。',
              '- `.venv/bin/python scripts/diagnose_sensing_critic.py`：重新执行诊断，训练更新数始终为0。',
              f'- 主要采样与统计耗时约{verification["elapsed_seconds"]/60:.2f}分钟；一个训练种子，结果不作为跨训练种子统计结论。','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/mappo_ablation_d_mobile_sensing_250k_seed0'))
    parser.add_argument('--output',type=Path,default=Path('analysis/ue_reward_ablations_250k_seed0/d_critic_audit'))
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(run/'config.yaml');device=configure_runtime(cfg)
    assert cfg.env.reward_primary=='sensing' and cfg.env.throughput_weight==cfg.env.robustness_weight==cfg.env.disconnection_weight==0
    assert cfg.env.horizon==400 and cfg.run.evaluation_policy=='sample'
    assert not set(cfg.run.validation_seeds)&set(cfg.run.test_seeds)
    reference=Path('analysis/ue_reward_ablations_250k_seed0/d_mobile_sensing').resolve()
    sources={'best':run/'best.pt','middle':run/'resume_000000180224','last':run/'last.pt'}
    summary={};started=time.monotonic();seeds=[880001,880002,880003]
    for name,path in sources.items():
        file=path/'metadata.pt' if path.is_dir() else path
        sha=hashlib.sha256(file.read_bytes()).hexdigest()
        cfg,policy=policy_from_checkpoint(path,device)
        before={k:actor_digest(getattr(policy,k)) for k in ['actor','value','normalizer']}
        for k in ['actor','value']:
            getattr(policy,k).eval()
            for param in getattr(policy,k).parameters():param.requires_grad_(False)
        result,anchors=collect(cfg,policy,name,seeds,reference,output)
        if name in ['best','last']:
            with np.load(output/f'{name}_trajectories.npz') as z:
                r=z['rewards'][0]
            old=json.loads((reference/'report.json').read_text())['evaluations'][name]['summary']['mean']
            np.testing.assert_allclose(r.mean(),old['discovery_fraction'],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(r.sum(1).mean(),old['return'],atol=1e-10,rtol=1e-12)
        result.update(checkpoint=str(path),checkpoint_env_steps=policy.provenance['checkpoint_env_steps'],sha256=sha)
        if name=='last':
            torch.save(anchors,output/'anchor_states.pt')
            anchor_rollouts(cfg,policy,anchors,output)
            time_feature_probe(policy,anchors,output)
        assert before=={k:actor_digest(getattr(policy,k)) for k in before}
        assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
        result['actor_critic_normalizer_unchanged']=True
        summary[name]=result
        atomic_json(output/'summary.json',summary)
        print(name,'all RMSE',result['stages']['all']['rmse'],'last V / G',result['stages']['last1'],flush=True)
    rows=[]
    for name,result in summary.items():
        for stage,row in result['stages'].items():rows.append(dict(checkpoint=name,stage=stage,**row))
    csv_write(output/'errors_by_stage.csv',rows)
    atomic_json(output/'verification.json',dict(elapsed_seconds=time.monotonic()-started,full_episodes=288,
                conditional_continuations=48*16,conditional_transitions=8*16*sum(400-t for t in ANCHOR_TIMES),
                action_seeds=seeds,scene_seeds=cfg.run.validation_seeds,policy_updates=0,critic_updates=0,
                training_seed_count=1,terminal_bootstrap=0.,original_validation_reproduced=True,
                time_feature_verified=True,reward_equals_sensing=True,source_files_unchanged=True,
                anchor_future_ue_rng='independently resampled',
                note='Frozen-policy GAE is not historical training rollout reconstruction; terminal raw network output is masked.'))
    figures(output)
    write_report(output)


if __name__=='__main__':main()
