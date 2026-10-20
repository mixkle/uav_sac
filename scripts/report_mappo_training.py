"""Rebuild MAPPO loss/reward figures and validate saved best/last policies.

Only reads the training run. No training, baselines, or reserved test scenes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from uav_research.config import load_config
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.plotting import plt, read_csv, save_figure, style, COLORS
from uav_research.runtime import configure_runtime, atomic_json


def numeric(rows, field):
    values = np.array([float(row[field]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {field} in log")
    return values


def smooth(values, window=9):
    return np.array([
        values[max(0, i - window + 1):i + 1].mean()
        for i in range(len(values))
    ])


def training_figures(run, output, cfg):
    rows = read_csv(run / "metrics.csv")
    # A final save may append another timing row at the same completed update.
    rows = list({int(row['env_steps']):row for row in rows}.values())
    validation = read_csv(run / "validation.csv")
    if not rows or not validation:
        raise ValueError("Training and validation logs must be nonempty")
    x = numeric(rows, "env_steps") / 1000
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for ax, field, title, color in zip(
        axes, ["actor_loss", "value_loss"],
        ["Actor loss (includes entropy term)", "Value loss (PopArt normalized)"],
        [COLORS[0], COLORS[1]],
    ):
        y = numeric(rows, field)
        ax.plot(x, y, color=color, alpha=.25, lw=.9, label="Raw: mean over PPO minibatches")
        ax.plot(x, smooth(y), color=color, lw=1.8, label="Trailing mean: 9 logged points")
        ax.set(xlabel="Joint environment transitions (k)", ylabel="Loss", title=title)
        ax.legend(fontsize=7)
    title = f"Discrete MAPPO | seed {cfg.run.seed} | {cfg.run.total_steps:,} transitions"
    fig.suptitle(title, fontsize=11)
    save_figure(fig, output, "loss_curves")

    vx = numeric(validation, "env_steps") / 1000
    vr = numeric(validation, "return")
    best = int(vr.argmax())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    reward = numeric(rows, "reward")
    axes[0].plot(x, reward, color=COLORS[0], alpha=.3, lw=.9, label="Raw logged step")
    axes[0].plot(x, smooth(reward), color=COLORS[0], lw=1.8, label="Trailing mean: 9 log points")
    axes[0].set(
        xlabel="Joint environment transitions (k)", ylabel="Team reward per step",
        title=f"Training: last collected step, mean over {cfg.run.num_envs} envs",
    )
    axes[0].legend(fontsize=7)
    axes[1].plot(vx, vr, "-o", color=COLORS[2], lw=1.5, ms=3,
                 label=f"{len(cfg.run.validation_seeds)} fixed validation scenes")
    axes[1].scatter(vx[best], vr[best], marker="*", s=140, color=COLORS[1], zorder=4,
                    label=f"Best: {vx[best]:.3f}k, return {vr[best]:.2f}")
    axes[1].set(
        xlabel="Joint environment transitions (k)", ylabel="Undiscounted episode return",
        title=f"Validation: {cfg.run.evaluation_policy} policy, {cfg.env.horizon} steps",
    )
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Reward = {cfg.env.reward_primary} + {cfg.env.throughput_weight:g} throughput + "
                 f"{cfg.env.robustness_weight:g} robustness - {cfg.env.disconnection_weight:g} disconnection",
                 fontsize=10)
    save_figure(fig, output, "reward_curves")

    with (output / "training_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_steps", "actor_loss", "actor_loss_trailing9",
                         "value_loss", "value_loss_trailing9", "logged_step_reward",
                         "logged_step_reward_trailing9"])
        actor, value = numeric(rows, "actor_loss"), numeric(rows, "value_loss")
        writer.writerows(zip(numeric(rows, "env_steps").astype(int), actor, smooth(actor),
                             value, smooth(value), reward, smooth(reward)))
    with (output / "validation_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation[0]))
        writer.writeheader()
        writer.writerows(validation)
    return rows, validation


def component_plots(run, output, cfg, metrics, validation):
    """Reconstruct signed reward contributions using the saved run weights."""
    w,lam,mu=cfg.env.throughput_weight,cfg.env.robustness_weight,cfg.env.disconnection_weight
    primary = 'reward_primary' if 'reward_primary' in metrics[0] else cfg.env.primary_reward_metric
    training=np.column_stack([numeric(metrics,primary),w*numeric(metrics,'tau'),
                              lam*numeric(metrics,'robustness'),-numeric(metrics,'disconnection_penalty')])
    np.testing.assert_allclose(training.sum(-1),numeric(metrics,'reward'),rtol=1e-10,atol=1e-12)
    validation_components=[];ratios=[]
    for row in validation:
        step=int(row['env_steps'])
        path=run/'validation_history'/f'{step:012d}.json'
        saved=json.loads(path.read_text());m=saved['summary']['mean']
        steps=m['steps']
        terms=np.array([m.get('reward_primary',m[cfg.env.primary_reward_metric]),w*m['tau'],lam*m['robustness'],-m['disconnection_penalty']])*steps
        np.testing.assert_allclose(terms.sum(),float(row['return']),atol=1e-9,rtol=1e-9)
        validation_components.append(terms);ratios.append(m['disconnection_ratio'])
    validation_components=np.array(validation_components)
    fields=[cfg.env.reward_primary,'weighted_throughput','weighted_robustness','signed_disconnection','total_reward','disconnection_ratio']
    for stage,rows,terms,total,d in [
        ('training',metrics,training,numeric(metrics,'reward'),numeric(metrics,'disconnection_ratio')),
        ('validation',validation,validation_components,numeric(validation,'return'),np.array(ratios))]:
        x=numeric(rows,'env_steps')/1000
        values=np.column_stack([terms,total,d])
        labels=[f'{cfg.env.reward_primary.capitalize()} contribution','Throughput contribution','Robustness contribution',
                'Disconnection contribution (negative)','Total reward','Mean disconnected UAV fraction']
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for i,(ax,label) in enumerate(zip(axes.flat,labels)):
            ax.plot(x,values[:,i],color=COLORS[i],marker='o' if stage=='validation' else None,
                    markersize=3,linewidth=1.3)
            ax.axhline(0,color='gray',lw=.5)
            ax.set(xlabel='Joint environment transitions (k)',ylabel='Fraction' if i==5 else
                   ('Mean episode contribution' if stage=='validation' else 'Mean logged-step contribution'),title=label)
        fig.suptitle(('Training: last logged step over environments' if stage=='training' else
                      f'Validation: mean episode contributions over {len(cfg.run.validation_seeds)} fixed scenes')+
                     f' | w={w:g}, lambda={lam:g}, mu={mu:g}',fontsize=10)
        save_figure(fig,output,f'reward_components_{stage}')
        with (output/f'reward_components_{stage}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['env_steps']+fields)
            writer.writerows(zip(numeric(rows,'env_steps').astype(int),*values.T))


def task_metric_plots(run, output, metrics, validation):
    """Compare physical metrics independently of which one earns reward."""
    history = [json.loads((run / 'validation_history' / f"{int(row['env_steps']):012d}.json").read_text())['summary']['mean']
               for row in validation]
    panels = [
        ('coverage', 'Actual service coverage', 'Fraction'),
        ('discovery_fraction', 'Instantaneous sensing fraction', 'Fraction'),
        ('disconnection_ratio', 'Disconnected UAV fraction', 'Fraction'),
        ('user_displacement_mean', 'UE displacement from reset', 'm'),
        ('uav_nearest_neighbor_mean', 'Mean nearest UAV distance', 'm'),
        ('uav_close_neighbor_fraction', 'UAV with a neighbor closer than 100 m', 'Fraction'),
    ]
    for stage, rows, values in [('training', metrics, metrics), ('validation', validation, history)]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        available = []
        for i, (ax, (key, title, unit)) in enumerate(zip(axes.flat, panels)):
            ax.set(xlabel='Joint environment transitions (k)', ylabel=unit, title=title)
            if key not in values[0]:
                ax.text(.5, .5, 'Not recorded in this historical run', ha='center', transform=ax.transAxes, fontsize=7)
                continue
            ax.plot(numeric(rows, 'env_steps') / 1000, numeric(values, key), color=COLORS[i], lw=1.3)
            available.append(key)
        fig.suptitle(f'{stage.capitalize()}: sensing is instantaneous, not cumulative', fontsize=10)
        save_figure(fig, output, f'task_metrics_{stage}')
        with (output / f'task_metrics_{stage}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['env_steps'] + available)
            writer.writerows(zip(numeric(rows, 'env_steps').astype(int), *(numeric(values, k) for k in available)))
    diagnostics = [('categorical_entropy', 'Categorical entropy'), ('ppo_approx_kl', 'Approximate KL'),
                   ('ppo_clip_fraction', 'PPO clipping fraction'), ('max_action_probability', 'Maximum action probability'),
                   ('actor_grad_norm', 'Actor gradient norm'), ('encoder_grad_norm', 'UE encoder gradient norm')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, diagnostics)):
        ax.set(xlabel='Joint environment transitions (k)', title=title)
        if key in metrics[0] and all(row[key] != '' for row in metrics):
            ax.plot(numeric(metrics, 'env_steps') / 1000, numeric(metrics, key), color=COLORS[i])
    save_figure(fig, output, 'ppo_diagnostics')


def episode_component_plots(evaluations, output):
    colors=[COLORS[0],COLORS[2],COLORS[3],COLORS[1],COLORS[4]]
    for name,item in evaluations.items():
        with np.load(Path(item['reward_curve'])) as data:
            terms=data['contributions'].mean(0);total=data['reward'].mean(0);t=data['steps']
            values=np.column_stack([terms,total])
            np.testing.assert_allclose(terms.sum(-1),total,atol=1e-12,rtol=1e-12)
            fig,axes=plt.subplots(2,3,figsize=(13,7))
            primary = str(data['names'][0]) if 'names' in data else 'coverage'
            labels=[f'{primary.capitalize()} contribution','Throughput contribution','Robustness contribution',
                    'Disconnection contribution (negative)','Total reward']
            for i,(ax,label) in enumerate(zip(axes.flat,labels)):
                ax.plot(t,values[:,i],color=colors[i]);ax.axhline(0,color='gray',lw=.5)
                ax.set(xlabel='Episode step',ylabel='Mean reward contribution',title=label)
            ax=axes[1,2]
            for i,label in enumerate(labels[:4]):ax.plot(t,terms[:,i],color=colors[i],label=label)
            ax.plot(t,total,color=colors[4],label='Total reward',linewidth=2)
            ax.set(xlabel='Episode step',ylabel='Mean contribution',title='Signed components and sum');ax.legend(fontsize=7)
            fig.suptitle(f"{name.capitalize()} at {item['checkpoint_env_steps']:,} transitions | "
                         f"all {len(data['scene_seeds'])} fixed validation scenes",fontsize=11)
            save_figure(fig,output,f'validation_reward_components_{name}')
            with (output/f'validation_reward_components_{name}.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['episode_step',primary,'weighted_throughput',
                    'weighted_robustness','signed_disconnection','total_reward'])
                writer.writerows(zip(t,*values.T))
            if 'diagnostics' in data:
                diagnostics = data['diagnostics'].mean(0)
                names = list(data['diagnostic_names'])
                fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
                for i, (key, title) in enumerate(zip(['coverage', 'discovery_fraction', 'disconnection_ratio'],
                                                    ['Actual service coverage', 'Instantaneous sensing fraction', 'Disconnected UAV fraction'])):
                    axes[i].plot(t, diagnostics[:, names.index(key)], color=COLORS[i])
                    axes[i].set(xlabel='Episode step', ylabel='Fraction', title=title)
                save_figure(fig, output, f'validation_task_metrics_{name}')


def write_report(report, output, cfg, metrics, history):
    manifest=report['manifest'];actual=manifest['env_steps'];e=cfg.env
    text=f"# 离散MAPPO训练验证：实际{actual:,}步\n\n"
    text+=(f"运行目录：{report['run']}。保存配置、完成状态和检查点显示实际预算"
           f"{actual:,}条联合转移、{manifest['updates']}轮PPO，正常完成，耗时{manifest['elapsed_seconds']/60:.2f}分钟。"
           '本次按保存配置验证，没有追加训练。\n\n')
    primary_symbol = 'P' if e.reward_primary == 'sensing' else 'rho'
    text+=(f"奖励：r={primary_symbol}+{e.throughput_weight:g} tau+{e.robustness_weight:g} J−{e.disconnection_weight:g} D。"
           f"场景{e.width:g}×{e.height:g} m、{e.num_uavs} UAV、{e.num_users} UE、{e.num_bs} BS；"
           f"UU {e.uu_range:g} m、UG三维{e.ug_range:g} m、边界{e.uav_boundary}。"
           f"验证使用全部{len(cfg.run.validation_seeds)}个固定场景，{cfg.run.evaluation_policy}动作，"
           f"动作种子{cfg.run.evaluation_action_seed}、批量{cfg.run.num_envs}、完整{e.horizon}步。\n\n")
    text+='| 模型 | 训练步数 | 平均回报 | 平均覆盖 | 平均感知 | 吞吐Mbps | 平均断联比例 | 终态覆盖 | 最后100步覆盖 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for name,item in report['evaluations'].items():
        m=item['summary']['mean']
        text+=(f"| {name} | {item['checkpoint_env_steps']} | {m['return']:.4f} | {m['coverage']:.2%} | "
               f"{m['discovery_fraction']:.2%} | {m['throughput_bps']/1e6:.2f} | {m['disconnection_ratio']:.2%} | {m['final_coverage']:.2%} | {m['last100_coverage']:.2%} |\n")
    text+=f"\n| 模型 | {'感知' if e.reward_primary == 'sensing' else '覆盖'}回报贡献 | 吞吐回报贡献 | 鲁棒回报贡献 | 断联扣分贡献 | 总回报 |\n|---|---:|---:|---:|---:|---:|\n"
    for name,item in report['evaluations'].items():
        m=item['summary']['mean'];h=m['steps']
        terms=[h*m['reward_primary'],h*e.throughput_weight*m['tau'],h*e.robustness_weight*m['robustness'],-h*m['disconnection_penalty']]
        text+=f"| {name} | "+' | '.join(f'{v:.4f}' for v in terms)+f" | {m['return']:.4f} |\n"
    initial=float(history[0]['return']);text+=f"\n训练前step0验证回报{initial:.4f}。"
    for name,item in report['evaluations'].items():text+=f"{name}较初始回报变化{item['summary']['mean']['return']-initial:+.4f}。"
    text+=('使用回报绝对增量描述变化。复核返回、覆盖、吞吐和奖励分量均与训练期记录一致。\n\n'
           'actor loss包含熵正则，value loss使用PopArt归一化；展示原始值和9个日志点的后向均值，'
           'Loss下降不能替代策略性能或critic校准证据。训练奖励/分项为日志时刻最后采集步的环境均值，'
           f'验证训练进度图为完整回合贡献；两者时间尺度不同。回合展开曲线为全部{len(cfg.run.validation_seeds)}场景每步平均，'
           '代表拓扑另固定展示前两个验证场景，没有按表现挑选。单训练种子不绘制训练种子置信区间。\n\n')
    for label,file in [('Loss','loss_curves'),('总奖励','reward_curves'),('训练奖励各项','reward_components_training'),
                       ('验证回报各项','reward_components_validation'),('最佳模型回合奖励各项','validation_reward_components_best'),
                       ('最后模型回合奖励各项','validation_reward_components_last'),
                       ('最佳模型拓扑',f'topology_best_{cfg.run.validation_seeds[0]}'),
                       ('最后模型拓扑',f'topology_last_{cfg.run.validation_seeds[0]}')]:
        text+=f'![{label}](figures/{file}.png)\n\n'
    for label, file in [('训练任务指标', 'task_metrics_training'), ('验证任务指标', 'task_metrics_validation'), ('PPO诊断', 'ppo_diagnostics')]:
        text += f'![{label}](figures/{file}.png)\n\n'
    text+=('拓扑包含初始、中期、终态。蓝色UAV可达任一BS，橙色断联，绿色UE当前被服务，紫色仅感知，灰色未感知；'
           '蓝实线为UU链路、橙虚线为直接BS接入、灰线为轨迹，没有BS–BS转发边。'
           '图输出PNG/PDF/SVG，曲线另有CSV；validation_best/last保存逐场景结果、完整奖励曲线和代表轨迹。\n\n')
    tail=actual% (cfg.run.num_envs*cfg.algo.rollout_steps)
    text+=(f"最后一次PPO采集{tail or cfg.run.num_envs*cfg.algo.rollout_steps}条联合转移；"
           f"最后日志近似KL {float(metrics[-1]['ppo_approx_kl']):.4f}，clip比例{float(metrics[-1]['ppo_clip_fraction']):.2%}。"
           '没有运行baseline、保留测试集或任何新训练。\n')
    (output/'report.md').write_text(text)


def topology(trace, output, checkpoint_name, checkpoint_step, cfg):
    style()
    with np.load(trace) as data:
        uav, ue, bs = data["uav_xy"], data["user_xy"], data["bs_positions"]
        frames = [0, (len(uav) - 1) // 2, len(uav) - 1]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        for ax, t in zip(axes, frames):
            served = data["C"][t].any(axis=0)
            sensed = data['sensed'][t].any(axis=0) if 'sensed' in data else served
            only_sensed = sensed & ~served
            connected = (data["B"][t] <= len(uav[t])).any(axis=1)
            ax.scatter(ue[t, ~sensed, 0], ue[t, ~sensed, 1], s=10, color="#a3a3a3",
                       alpha=.55, zorder=1)
            ax.scatter(ue[t, only_sensed, 0], ue[t, only_sensed, 1], s=12, color=COLORS[3], alpha=.75, zorder=2)
            ax.scatter(ue[t, served, 0], ue[t, served, 1], s=13, color=COLORS[2],
                       alpha=.85, zorder=2)
            for i in range(len(uav[t])):
                if t:
                    ax.plot(uav[:t+1, i, 0], uav[:t+1, i, 1], color="#777777",
                            lw=.55, alpha=.25, zorder=1)
                for j in range(i):
                    if data["A"][t, i, j]:
                        ax.plot(uav[t, [i, j], 0], uav[t, [i, j], 1],
                                color=COLORS[0], alpha=.55, lw=.9, zorder=2)
                for g, gateway in enumerate(bs):
                    if data["B"][t, i, g] == 1:
                        ax.plot([uav[t, i, 0], gateway[0]], [uav[t, i, 1], gateway[1]],
                                color=COLORS[1], ls="--", lw=.9, alpha=.75, zorder=2)
                ax.annotate(str(i+1), uav[t, i], xytext=(3, 3), textcoords="offset points",
                            fontsize=6.5, zorder=5)
            ax.scatter(uav[t, connected, 0], uav[t, connected, 1], marker="^", s=48,
                       color=COLORS[0], edgecolors="white", lw=.5, zorder=4)
            ax.scatter(uav[t, ~connected, 0], uav[t, ~connected, 1], marker="^", s=48,
                       color=COLORS[4], edgecolors="white", lw=.5, zorder=4)
            ax.scatter(bs[:, 0], bs[:, 1], marker="*", s=160, color=COLORS[1],
                       clip_on=False, zorder=5)
            ax.set(xlim=(0, float(data["width"])), ylim=(0, float(data["height"])),
                   aspect="equal", xlabel="x (m)", ylabel="y (m)",
                   title=f"Step {t} | coverage {served.mean():.1%}\n"
                         f"Sensed {sensed.mean():.1%} | reachable UAVs {connected.sum()}/{len(connected)}")
        handles = [
            Line2D([], [], marker="*", color=COLORS[1], ls="none", label="BS"),
            Line2D([], [], marker="^", color=COLORS[0], ls="none", label="Reachable UAV"),
            Line2D([], [], marker="^", color=COLORS[4], ls="none", label="Disconnected UAV"),
            Line2D([], [], marker="o", color=COLORS[2], ls="none", label="Served UE"),
            Line2D([], [], marker="o", color=COLORS[3], ls="none", label="Sensed, unserved UE"),
            Line2D([], [], marker="o", color="#a3a3a3", ls="none", label="Not sensed UE"),
            Line2D([], [], color=COLORS[0], label="UAV link"),
            Line2D([], [], color=COLORS[1], ls="--", label="Direct BS link"),
            Line2D([], [], color="#777777", alpha=.5, label="UAV trajectory"),
        ]
        scene = int(data["scene_seed"])
        fig.suptitle(f"{checkpoint_name.capitalize()} checkpoint ({checkpoint_step:,} transitions)"
                     f" | validation scene {scene} | {cfg.run.evaluation_policy} actions", fontsize=11)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.14),
                   ncol=4, fontsize=8, frameon=False)
        save_figure(fig, output, f"topology_{checkpoint_name}_{scene}")


def validation_figures(evaluations, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for idx, (name, result) in enumerate(evaluations.items()):
        traces = result["traces"]
        with_data = [np.load(p) for p in traces]
        reward = np.stack([data["reward_terms"][1:] for data in with_data])
        if all("reward" in data for data in with_data):
            team_reward = np.stack([data["reward"][1:] for data in with_data])
        else:
            # Legacy traces have no scalar reward or disconnection term.
            weights = result["reward_weights"]
            team_reward = reward[:, :, 0] + weights[0] * reward[:, :, 1] + weights[1] * reward[:, :, 2]
        for data in with_data:
            data.close()
        t = np.arange(1, reward.shape[1] + 1)
        # These are fixed trace scenes only, NOT the full 32-scene summary.
        axes[0].plot(t, reward[:, :, 0].mean(0), color=COLORS[idx], label=name)
        axes[1].plot(t, team_reward.mean(0),
                     color=COLORS[idx], label=name)
    for ax, ylabel in zip(axes, ["Instantaneous coverage", "Team reward per step"]):
        ax.set(xlabel="Episode step", ylabel=ylabel)
        ax.legend()
    fig.suptitle("Fixed validation traces only: scenes 10000 and 10001", fontsize=10)
    save_figure(fig, output, "validation_trace_curves")


def build_report(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    cfg = load_config(run / "config.yaml")
    source_hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest()
                   for name in ('config.yaml','manifest.json','metrics.csv','validation.csv','best.pt','last.pt')}
    if cfg.algo.name != "mappo" or cfg.env.action_space != "discrete9":
        raise ValueError("This report requires discrete MAPPO")
    metrics, history = training_figures(run, figures, cfg)
    evaluation = {}
    device = configure_runtime(cfg)
    for name in ["best", "last"]:
        path = run / f"{name}.pt"
        pcfg, policy = policy_from_checkpoint(path, device)
        if pcfg.env != cfg.env or pcfg.algo != cfg.algo:
            raise ValueError('Checkpoint and saved run configuration differ')
        selected = pcfg.run.validation_seeds
        if set(selected) & set(pcfg.run.test_seeds):
            raise ValueError("Reserved test scenes must not be evaluated here")
        target = output / f"validation_{name}"
        rows, summary = evaluate(pcfg, policy, selected, target, trace_count=2,reward_curves=True)
        step = policy.provenance["checkpoint_env_steps"]
        matching = [row for row in history if int(row["env_steps"]) == step]
        if len(matching) != 1:
            raise ValueError(f"No unique training validation row for checkpoint at {step}")
        for key in ["return", "coverage", "throughput_bps"]:
            np.testing.assert_allclose(summary["mean"][key], float(matching[0][key]),
                                       rtol=1e-9, atol=1e-9)
        original=json.loads((run/'validation_history'/f'{step:012d}.json').read_text())['summary']['mean']
        for key in ['tau','robustness','disconnection_ratio','disconnection_penalty']:
            np.testing.assert_allclose(summary['mean'][key],original[key],rtol=1e-9,atol=1e-9)
        traces = [target / f"trace_{seed}.npz" for seed in selected[:2]]
        for trace in traces:
            topology(trace, figures, name, step, pcfg)
        source = {
            "checkpoint": str(path), "checkpoint_env_steps": step,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": summary, "traces": [str(p) for p in traces],
            "reward_weights": [pcfg.env.throughput_weight, pcfg.env.robustness_weight, pcfg.env.disconnection_weight],
            "reward_curve": str(target/'reward_curves.npz'),
            "matches_training_validation": True,
        }
        evaluation[name] = source
        print(f"{name}: step={step}, validation episodes={len(rows)}, "
              f"return={summary['mean']['return']:.6f}, "
              f"coverage={summary['mean']['coverage']:.4%}", flush=True)
    validation_figures(evaluation, figures)
    component_plots(run,figures,cfg,metrics,history)
    task_metric_plots(run,figures,metrics,history)
    episode_component_plots(evaluation,figures)
    report = {
        "run": str(run), "manifest": json.loads((run / "manifest.json").read_text()),
        "metric_log_points": len(metrics), "validation_log_points": len(history),
        "source_metric_log_rows": len(read_csv(run/'metrics.csv')),
        "reward_weights": [1,cfg.env.throughput_weight,cfg.env.robustness_weight,-cfg.env.disconnection_weight],
        "reward_primary": cfg.env.reward_primary,
        "smoothing": "trailing arithmetic mean over 9 logged points; raw curves shown",
        "training_reward": "snapshot of last collected step, averaged over environments; not rollout or episode mean",
        "loss": "mean over PPO minibatches in each round; actor includes entropy, value uses PopArt",
        "validation": f"{cfg.run.evaluation_policy} best/last policies; all fixed validation scenes; no baselines, no reserved test scenes",
        "evaluation_action_seed": cfg.run.evaluation_action_seed,
        "evaluation_num_envs": cfg.run.num_envs,
        "scenario": {"uav_boundary": cfg.env.uav_boundary, "ug_range": cfg.env.ug_range,
                     "uu_range": cfg.env.uu_range, "horizon": cfg.env.horizon},
        "topology": "first two predesignated validation seeds; initial, midpoint, terminal",
        "evaluations": evaluation,
        "source_file_sha256": source_hashes,
    }
    assert all(hashlib.sha256((run/name).read_bytes()).hexdigest()==sha for name,sha in source_hashes.items())
    atomic_json(output / "report.json", report)
    write_report(report,output,cfg,metrics,history)
    print(f"Artifacts: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run, args.output)


if __name__ == "__main__":
    main()
