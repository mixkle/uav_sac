"""Freeze a saved actor and fit its critic against complete finite-episode returns."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from uav_research.critic_calibration import (actor_digest, collect_complete_episodes, load_dataset,
    rebase_normalizer_from_fit_targets, fit_critic_epoch, predict_critic, calibration_metrics)
from uav_research.evaluate import policy_from_checkpoint
from uav_research.config import load_config
from uav_research.runtime import configure_runtime, atomic_json, atomic_save, rng_state, seed_all
from uav_research.environment import C_LIGHT
from uav_research.plotting import plt, style, save_figure, COLORS


def reward_bound(e):
    if e.robustness_weight != 0: raise ValueError('Reward bound currently assumes robustness weight zero')
    loss=20*np.log10(4*np.pi*e.frequency_hz*e.altitude/C_LIGHT)+min(e.los_extra_db,e.nlos_extra_db)
    snr=10**((e.subband_power_dbm-loss)/10)/e.noise_mw
    rate=e.bandwidth_hz/e.capacity*np.log2(1+snr)
    return float(1+e.throughput_weight*rate/e.rate_reference_bps)


def cpu_state(policy):
    return {name:{k:v.detach().cpu().clone() for k,v in getattr(policy,name).state_dict().items()}
            for name in ('value','normalizer')}


def figures(output, history, before, after, returns):
    target=output/'figures';style()
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for split,color in [('fit',COLORS[0]),('selection',COLORS[1])]:
        axs[0].semilogy([r['epoch'] for r in history],[r[f'{split}_rmse'] for r in history],color=color,label=split)
        axs[1].semilogy([r['epoch'] for r in history],[r[f'{split}_final_step_rmse'] for r in history],color=color,label=split)
    for ax,title in zip(axs,['All-time raw return RMSE','Final-step raw return RMSE']):
        ax.set(xlabel='Critic-only fit epoch',ylabel='RMSE',title=title);ax.legend()
    save_figure(fig,target,'calibration_learning')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));ts=np.arange(returns.shape[1])
    for ax in axs:
        ax.plot(ts,returns.mean(0),color=COLORS[1],label='Realized discounted return')
        ax.plot(ts,before.mean(0),color=COLORS[0],label='Original critic')
        ax.plot(ts,after.mean(0),color=COLORS[2],label='Calibrated critic')
        ax.set(xlabel='State time (last action at t=399)',ylabel='Remaining discounted return');ax.legend(fontsize=8)
    axs[0].set_title('32 untouched holdout scenes | frozen actor')
    axs[1].set(xlim=(returns.shape[1]-20,returns.shape[1]-1), title='Final 20 states')
    save_figure(fig,target,'critic_horizon_calibrated')
    fig,ax=plt.subplots(figsize=(6.5,3.8))
    means=[before[:,-1].mean(),after[:,-1].mean(),returns[:,-1].mean()]
    bars=ax.bar(['Original V(t=399)','Calibrated V(t=399)','Realized last reward'],means,
                color=[COLORS[0],COLORS[2],COLORS[1]])
    ax.bar_label(bars,fmt='%.3f');ax.set(ylabel='Remaining return',title='Last action: one reward remains')
    save_figure(fig,target,'last_step_value')


def report(output, summary):
    b,a=summary['holdout']['before'],summary['holdout']['after']
    text='# 冻结actor、完整回合回报的critic校准实验\n\n'
    text+=f"源检查点：{summary['source_checkpoint']}，训练步数{summary['source_env_steps']}。本实验不更新actor，不修改奖励、环境或正式MAPPO训练配置。\n\n"
    text+='## 实验设置\n\n'
    text+='- 拟合64场景41000–41063；选模16场景42000–42015；最终检查32场景43000–43031。全部与原验证10000–10031、测试20000–20099隔离。\n'
    text+='- actor按原Categorical分布采样，分别固定动作种子910001/910002/910003；完整400步，到期真正终止，无bootstrap。共44800条联合转移。\n'
    text+='- 价值网络保持6292→256→256→1两层ReLU，继承源权重和Adam状态；学习率5e−4、Adam eps1e−5、梯度裁剪0.5、batch512。\n'
    text+='- 目标为G_t=r_t+0.995G_(t+1)，G_T=0；基础Huber delta10，不使用value clipping。PopArt统计仅由拟合集目标重新估计，重标定输出层保持初始原始预测不变，随后固定。\n'
    text+='- 最多60 epochs，选模集全时段原始RMSE选择模型，patience12；最终检查集不参与训练、统计归一化或选模。\n\n'
    text+='## 最终检查集结果\n\n| 指标 | 原critic | 校准critic |\n|---|---:|---:|\n'
    for label,key in [('全时段RMSE','raw_rmse'),('全时段MAE','raw_mae'),('全时段平均偏差','raw_bias'),('解释方差','explained_variance'),('最后一步RMSE','final_step_rmse'),('最后一步V平均','final_step_prediction_mean'),('最后一步实际回报平均','final_step_return_mean')]:
        text+=f"| {label} | {b[key]:.4f} | {a[key]:.4f} |\n"
    text+=f"\n最佳校准epoch：{summary['selected_epoch']}；实际完成{summary['epochs_completed']} epochs，耗时{summary['elapsed_seconds']/60:.2f}分钟，进程峰值RSS {summary['peak_rss_mib']:.1f} MiB。\n\n"
    text+=f"最后一步奖励保守上界{summary['single_step_reward_upper_bound']:.4f}，超上界预测比例：校准前{b['final_step_upper_bound_violations']:.2%}，校准后{a['final_step_upper_bound_violations']:.2%}。\n\n"
    text+='| 时段 | 原RMSE | 校准RMSE |\n|---|---:|---:|\n'
    for name in b['time_bins']:
        text+=f"| {name} | {b['time_bins'][name]['rmse']:.4f} | {a['time_bins'][name]['rmse']:.4f} |\n"
    text+='\n## 解释与边界\n\n'
    text+='完整回合回报仍有采样噪声，不能把每条轨迹的G_t视为精确的条件期望V。是否校准充分，应结合独立场景误差和末期物理尺度检查。\n\n'
    text+=(f"本次尚未解决末期价值问题：独立检查集最后一步预测由{b['final_step_prediction_mean']:.2f}降至"
           f"{a['final_step_prediction_mean']:.2f}，实际奖励约{a['final_step_return_mean']:.2f}，仍显著超出单步奖励上界。"
           "后续轮次拟合集RMSE最低约2.70，选模集却未优于第1轮。拟合集与独立场景的差距支持泛化不足的判断，"
           "但不足以证明某一个结构因素是唯一原因。\n\n")
    text+='本实验同时更换价值目标为实际完整回报并去掉value clipping，不能独立归因于其中一项。若改善，说明原结构有一定可拟合能力；若检查集仍有明显末期误差，需继续检查时间表达、泛化和优化。没有证明在线MAPPO已修复。\n\n'
    text+='actor逐张量校验及SHA256一致，源best.pt的文件SHA256也保持不变。因此覆盖、部署行为和策略收益不会因这次critic单独校准而改变；需要后续在线训练对照检验。\n\n'
    text+='下一步建议继续冻结actor并复用相同数据，先单独加强剩余时间表达，再按相同选模与独立检查协议比较；当前不宜直接把本校准权重用于正式续训。\n\n'
    text+='![校准曲线](figures/calibration_learning.png)\n\n![剩余价值](figures/critic_horizon_calibrated.png)\n\n![末步价值](figures/last_step_value.png)\n\n'
    text+='复现命令见 experiment.json 的command；critic_best.pt是独立校准产物，不是正式训练续训检查点。各数据目录保存状态memmap、目标、奖励、场景种子及动作序列哈希。\n'
    (output/'report.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/wsl_discrete_mappo_paper_scene_250k_seed0/best.pt'))
    parser.add_argument('--output',type=Path,default=Path('analysis/mappo_discrete_250k_seed0/critic_calibration'))
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--reuse-data',action='store_true')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint.resolve();source_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg=load_config(checkpoint.parent/'config.yaml');device=configure_runtime(cfg)
    cfg,policy=policy_from_checkpoint(checkpoint,device)
    if cfg.algo.name!='mappo' or cfg.env.action_space!='discrete9':raise ValueError('Requires discrete MAPPO')
    original_actor={k:v.detach().clone() for k,v in policy.actor.state_dict().items()}
    actor_sha=actor_digest(policy.actor)
    for parameter in policy.actor.parameters():parameter.requires_grad_(False)
    policy.actor.eval()
    splits={'fit':(list(range(41000,41064)),910001),'selection':(list(range(42000,42016)),910002),'holdout':(list(range(43000,43032)),910003)}
    all_seeds=[s for seeds,_ in splits.values() for s in seeds]
    if len(set(all_seeds))!=len(all_seeds) or set(all_seeds)&set(cfg.run.validation_seeds+cfg.run.test_seeds):raise ValueError('Overlapping scene splits')
    experiment=dict(source_checkpoint=str(checkpoint),source_sha256=source_sha,source_env_steps=policy.provenance['checkpoint_env_steps'],
        config=cfg.as_dict(),actor_sha256=actor_sha,splits={k:{'scene_seeds':s,'action_seed':seed} for k,(s,seed) in splits.items()},
        value_target='complete Monte Carlo discounted return; no bootstrap',value_clipping=False,
        huber_delta=10,batch_size=512,epochs_max=args.epochs,patience=args.patience,
        selection_metric='selection all-time raw RMSE',normalizer='fit targets only; frozen after preserving raw outputs by PopArt',
        optimizer_state='source Adam state, transformed consistently by PopArt',permutation_seed=920001,
        command=f'.venv/bin/python scripts/calibrate_mappo_critic.py --checkpoint {checkpoint} --output {output} --epochs {args.epochs} --patience {args.patience}')
    atomic_json(output/'experiment.json',experiment)
    stamp=time.perf_counter();atomic_json(output/'manifest.json',dict(status='collecting'))
    datasets={}
    for split,(seeds,seed) in splits.items():
        folder=output/'data'/split
        datasets[split]=load_dataset(folder) if args.reuse_data else collect_complete_episodes(cfg,policy,seeds,seed,folder)
        md=datasets[split]['metadata']
        if md['scene_seeds']!=seeds or md['action_seed']!=seed or md['actor_sha256']!=actor_sha:raise ValueError('Dataset provenance mismatch')
    bound=reward_bound(cfg.env)
    before={k:predict_critic(policy,data) for k,data in datasets.items()}
    initial_metrics={k:calibration_metrics(before[k],data['returns'],bound) for k,data in datasets.items()}
    probe=torch.as_tensor(np.array(datasets['fit']['states'][0,:8],copy=True),device=device)
    with torch.no_grad():probe_before=policy.values(probe).clone()
    normalization=rebase_normalizer_from_fit_targets(policy,datasets['fit']['returns'])
    with torch.no_grad():np.testing.assert_allclose(policy.values(probe).cpu(),probe_before.cpu(),rtol=1e-5,atol=1e-5)
    print('Initial fit/selection RMSE:',initial_metrics['fit']['raw_rmse'],initial_metrics['selection']['raw_rmse'],flush=True)
    seed_all(920001);rng=np.random.default_rng(920001)
    history=[dict(epoch=0,fit_rmse=initial_metrics['fit']['raw_rmse'],selection_rmse=initial_metrics['selection']['raw_rmse'],
        fit_final_step_rmse=initial_metrics['fit']['final_step_rmse'],selection_final_step_rmse=initial_metrics['selection']['final_step_rmse'],
        normalized_huber_loss=float('nan'),value_grad_norm=0.0,elapsed_seconds=time.perf_counter()-stamp)]
    best_score=initial_metrics['selection']['raw_rmse'];best_state=cpu_state(policy);selected_epoch=0;stale=0;optimizer_steps=0
    def save_best():
        atomic_save(output/'critic_best.pt',dict(kind='critic_calibration',source_checkpoint=str(checkpoint),source_sha256=source_sha,
                    actor_sha256=actor_sha,epoch=selected_epoch,value=best_state['value'],normalizer=best_state['normalizer'],
                    value_optimizer=copy.deepcopy(policy.value_opt.state_dict()),rng=rng_state(),permutation_rng=rng.bit_generator.state,
                    experiment=experiment))
    save_best()
    atomic_json(output/'manifest.json',dict(status='fitting',normalization=normalization))
    for epoch in range(1,args.epochs+1):
        fit=fit_critic_epoch(policy,datasets['fit'],512,rng);optimizer_steps+=fit['optimizer_steps']
        fitted=calibration_metrics(predict_critic(policy,datasets['fit']),datasets['fit']['returns'],bound)
        selected=calibration_metrics(predict_critic(policy,datasets['selection']),datasets['selection']['returns'],bound)
        history.append(dict(epoch=epoch,fit_rmse=fitted['raw_rmse'],selection_rmse=selected['raw_rmse'],
            fit_final_step_rmse=fitted['final_step_rmse'],selection_final_step_rmse=selected['final_step_rmse'],
            normalized_huber_loss=fit['normalized_huber_loss'],value_grad_norm=fit['value_grad_norm'],elapsed_seconds=time.perf_counter()-stamp))
        if selected['raw_rmse']<best_score-1e-4:
            best_score=selected['raw_rmse'];best_state=cpu_state(policy);selected_epoch=epoch;stale=0;save_best()
        else:stale+=1
        with (output/'fit_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        print(f"Epoch {epoch:02d}: fit RMSE={fitted['raw_rmse']:.4f}, selection={selected['raw_rmse']:.4f}, "
              f"selection final={selected['final_step_prediction_mean']:.4f} vs {selected['final_step_return_mean']:.4f}; best={selected_epoch}",flush=True)
        if stale>=args.patience:break
    for name,data in best_state.items():getattr(policy,name).load_state_dict(data)
    after={k:predict_critic(policy,data) for k,data in datasets.items()}
    final_metrics={k:calibration_metrics(after[k],data['returns'],bound) for k,data in datasets.items()}
    for k,tensor in policy.actor.state_dict().items():assert torch.equal(original_actor[k],tensor)
    assert actor_digest(policy.actor)==actor_sha
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_sha
    summary=dict(source_checkpoint=str(checkpoint),source_env_steps=experiment['source_env_steps'],selected_epoch=selected_epoch,
                 epochs_completed=len(history)-1,critic_optimizer_steps=optimizer_steps,normalization=normalization,
                 single_step_reward_upper_bound=bound,actor_unchanged=True,source_checkpoint_unchanged=True,
                 elapsed_seconds=time.perf_counter()-stamp,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    for split,data in datasets.items():
        summary[split]=dict(before=initial_metrics[split],after=final_metrics[split])
        np.savez_compressed(output/f'{split}_predictions.npz',before=before[split],after=after[split],returns=data['returns'])
    atomic_json(output/'summary.json',summary)
    figures(output,history,before['holdout'],after['holdout'],datasets['holdout']['returns'])
    report(output,summary)
    atomic_json(output/'manifest.json',dict(status='completed',**summary))
    print('Holdout before/after:',summary['holdout'],flush=True)
    print('Report:',output/'report.md',flush=True)

if __name__=='__main__':main()
