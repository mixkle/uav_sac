"""Saved-data reports: task metrics, protocol, deployment and critic MC audit."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch

from ..config import config_from_dict
from ..algorithms import make_learner
from ..runtime import load_checkpoint, atomic_json
from ..topk_diagnostic import file_hash
from ..plotting import style, save_figure, COLORS, read_csv
from .config import from_dict
from .storage import load, ensure_budget, configure as configure_runtime
from .policy import DiffusionActor
from .learner import DiffusionLearner
from .data import teacher_action, teacher_policy
from .evaluation import evaluate


class GaussianExecution(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def sample(self, obs):
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            action = self.actor.sample(obs.to_torch(device), False)[0]
        return action.cpu().numpy(), None


class TeacherExecution(torch.nn.Module):
    def __init__(self, actor, cfg):
        super().__init__()
        self.actor, self.cfg = actor, cfg

    def sample(self, obs):
        action, _, _ = teacher_action(self.actor, obs, self.cfg)
        return action, None


def evaluate_checkpoint(path, output, device="cpu", seeds=None, moment_repeats=64):
    data = load(path, kinds={"dppo_model", "dppo_full", "dppo_pretrain"})
    cfg = from_dict(data["config"])
    cfg.run.device = device
    cfg.diffusion.actor_update_devices = []  # Frozen evaluation never replicates PPO gradients.
    selected_device = configure_runtime(cfg)
    if data["kind"] == "dppo_pretrain":
        actor = DiffusionActor(cfg).to(selected_device)
        actor.load_state_dict(data["actor"])
        critic = None
    else:
        critic = DiffusionLearner(cfg, selected_device)
        critic.load_state_dict(data["learner"])
        actor = critic.actor
    actor.eval()
    result = evaluate(cfg, actor, seeds or cfg.run.validation_seeds, output,
                      critic=critic, moment_repeats=moment_repeats)
    atomic_json(Path(output)/"provenance.json", dict(evaluation_data_version=2, checkpoint=str(path),
        sha256=file_hash(Path(path)/"metadata.pt" if Path(path).is_dir() else path),
        env_steps=data.get("env_steps", 0), pretraining_epoch=data.get("epoch"),
        moment_repeats=moment_repeats, scene_seeds=seeds or cfg.run.validation_seeds,
        action_seed=cfg.run.evaluation_action_seed, device=device))
    return result


def error_metrics(predicted, labels):
    error = predicted-labels
    result = {}
    windows = {"all": slice(None), "first100": slice(0, 100), "middle100": slice(150, 250),
               "last100": slice(-100, None), "last20": slice(-20, None), "last1": slice(-1, None)}
    for name, sl in windows.items():
        if error[:, sl].size:
            x = error[:, sl]
            result[name] = dict(rmse=float(np.sqrt(np.mean(x*x))), mae=float(np.mean(np.abs(x))), bias=float(x.mean()))
    p, g = predicted-predicted.mean(0), labels-labels.mean(0)
    result["time_centered_scene_correlation"] = float(np.corrcoef(p.ravel(), g.ravel())[0, 1]) if p.std() and g.std() else None
    result["normalized_rmse"] = result["all"]["rmse"]/float(labels.std()) if labels.std() else None
    return result


def rebuild(output, run=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output = Path(output)
    manifest = json.loads((output/"report_sources.json").read_text())
    if (output/"training_data").exists():
        run = output/"training_data"
    labels = manifest["labels"]
    style()
    series = {}
    episodes = {}
    for name in labels:
        p = output/name
        with np.load(p/"step_metrics.npz") as saved:
            series[name] = (saved["metrics"].copy(), saved["names"].tolist())
        episodes[name] = read_csv(p/"episodes.csv")
    figdir = output/"figures"
    plots = [(["coverage", "throughput_bps", "reward", "discovery_fraction"], "task_curves"),
        (["boundary_within_150_fraction", "boundary_reflection_fraction", "uav_close_neighbor_fraction", "actual_speed_mean"], "deployment"),
        (["proposal_acceptance", "over_capacity_fraction", "multi_candidate_fraction", "invalidated_proposals"], "protocol")]
    for keys, filename in plots:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, key in zip(axes.flat, keys):
            for j, name in enumerate(labels):
                a, names = series[name]
                if key in names:
                    ax.plot(np.arange(1, a.shape[1]+1), a[:, :, names.index(key)].mean(0), label=name, color=COLORS[j % len(COLORS)])
            ax.set(xlabel="Environment step", ylabel=key)
        axes.flat[0].legend(fontsize=7)
        save_figure(fig, figdir, filename)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for j, name in enumerate(labels):
        a, names = series[name]
        for ax, key, weight in ((axes[0], "coverage", 1.), (axes[1], "tau", .1), (axes[2], "reward", 1.)):
            ax.plot(np.arange(1, a.shape[1]+1), weight*a[:, :, names.index(key)].mean(0),
                    label=name, color=COLORS[j % len(COLORS)])
    for ax, title in zip(axes, ("Coverage contribution", "0.1 × normalized throughput", "Total training reward")):
        ax.set(title=title, xlabel="Environment step", ylabel="Reward")
    axes[0].legend(fontsize=6)
    save_figure(fig, figdir, "reward_components")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, name in enumerate(labels):
        p = output/name/"action_moments.npz"
        if p.exists():
            with np.load(p) as z:
                # [scene, probe-time, UAV, mean/variance, xy]
                axes[0].plot(z["times"], np.linalg.norm(z["moments"][:, :, :, 0], axis=-1).mean((0, 2)),
                    label=name, color=COLORS[j % len(COLORS)])
                axes[1].plot(z["times"], z["moments"][:, :, :, 1].sum(-1).mean((0, 2)),
                    label=name, color=COLORS[j % len(COLORS)])
    if axes[0].lines:
        axes[0].set(xlabel="Environment step", ylabel="Conditional mean command displacement (m)")
        axes[1].set(xlabel="Environment step", ylabel="Trace of conditional displacement covariance (m²)")
        axes[0].legend(fontsize=7)
        save_figure(fig, figdir, "action_randomness")
    else:
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key in zip(axes.flat, ("score_latent_mean", "score_latent_std", "score_saturation_fraction", "near_hover_fraction")):
        for j, name in enumerate(labels):
            a, names = series[name]
            if key in names:
                ax.plot(np.arange(1, a.shape[1]+1), a[:, :, names.index(key)].mean(0), label=name, color=COLORS[j % len(COLORS)])
        ax.set(xlabel="Environment step", ylabel=key)
    axes.flat[0].legend(fontsize=7)
    save_figure(fig, figdir, "score_and_motion_distribution")
    for name in labels:
        for trace in sorted((output/name).glob("trace_*.npz")):
            with np.load(trace) as z:
                fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
                times = [0, min(150, len(z["uav_xy"])-1), min(200, len(z["uav_xy"])-1), len(z["uav_xy"])-1]
                bs = z["bs_positions"]
                for ax, t in zip(axes, times):
                    xy, users, c, graph, hop = [z[k][t] for k in ("uav_xy", "user_xy", "C", "A", "B")]
                    served, seen = c.any(0), z["sensed"][t].any(0)
                    for mask, color, label in ((~seen, "#aaaaaa", "unseen"), (seen & ~served, COLORS[3], "sensed"), (served, COLORS[2], "served")):
                        ax.scatter(users[mask, 0], users[mask, 1], c=color, s=7, label=label)
                    for i in range(len(xy)):
                        for j in range(i):
                            if graph[i, j]:
                                ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=COLORS[0], alpha=.35, lw=.6)
                        for g in range(len(bs)):
                            if hop[i, g] == 1:
                                ax.plot([xy[i, 0], bs[g, 0]], [xy[i, 1], bs[g, 1]], "--", color=COLORS[1], lw=.7)
                        ids = np.flatnonzero(c[i])
                        for m in ids:
                            ax.plot([xy[i, 0], users[m, 0]], [xy[i, 1], users[m, 1]], color=COLORS[2], alpha=.2, lw=.4)
                    linked = hop.min(1) <= len(xy)
                    ax.scatter(xy[:, 0], xy[:, 1], marker="^", s=35, c=np.where(linked, COLORS[0], COLORS[1]))
                    ax.scatter(bs[:, 0], bs[:, 1], marker="*", s=100, c=COLORS[1], clip_on=False)
                    ax.set(xlim=(0, float(z["width"])), ylim=(0, float(z["height"])), aspect="equal", title=f"t={t}, coverage={served.mean():.1%}", xlabel="x (m)", ylabel="y (m)")
                fig.legend(handles=[Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
                    for marker, color, label in (("o", COLORS[2], "served UE"), ("o", COLORS[3], "sensed UE"),
                    ("o", "#aaaaaa", "unseen UE"), ("^", COLORS[0], "connected UAV"),
                    ("^", COLORS[1], "disconnected UAV"), ("*", COLORS[1], "BS"))],
                    loc="lower center", bbox_to_anchor=(.5, -.03), ncol=6, fontsize=7, frameon=False)
                save_figure(fig, figdir, f"{name}_{trace.stem}")
    critic_results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, name in enumerate(labels):
        p = output/name/"critic_predictions.npz"
        if p.exists():
            with np.load(p) as z:
                critic_results[name] = {"cumulative": error_metrics(z["predicted"], z["mc_return"]),
                    "average": error_metrics(z["predicted_average"], z["mc_average"])}
                for ax, pred, truth in ((axes[0], "predicted", "mc_return"), (axes[1], "predicted_average", "mc_average")):
                    ax.plot(z[pred].mean(0), color=COLORS[j % len(COLORS)], label=f"{name}: prediction")
                    ax.plot(z[truth].mean(0), "--", color=COLORS[j % len(COLORS)], label=f"{name}: MC")
    axes[0].set(xlabel="Environment step", ylabel="Cumulative value")
    axes[1].set(xlabel="Environment step", ylabel="Discounted average task reward")
    if critic_results:
        axes[0].legend(fontsize=6)
        save_figure(fig, figdir, "critic_mc")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for j, name in enumerate(labels):
            p = output/name/"critic_predictions.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                times = np.arange(z["predicted"].shape[1])
                for ax, pred, truth in ((axes[0], "predicted", "mc_return"), (axes[1], "predicted_average", "mc_average")):
                    ax.plot(times[-20:], z[pred].mean(0)[-20:], color=COLORS[j % len(COLORS)], label=name+" prediction")
                    ax.plot(times[-20:], z[truth].mean(0)[-20:], "--", color=COLORS[j % len(COLORS)])
        axes[0].set(xlabel="State time before action", ylabel="Last-20 cumulative value")
        axes[1].set(xlabel="State time before action", ylabel="Last-20 average task reward")
        axes[0].legend(fontsize=6)
        save_figure(fig, figdir, "critic_terminal_zoom")
    else:
        plt.close(fig)
    paired = []
    rng = np.random.default_rng(950001)
    for a, b in (("last", "step0"), ("best", "step0"), ("last", "baseline_last"), ("teacher_topk", "baseline_best")):
        if a not in episodes or b not in episodes:
            continue
        left, right = episodes[a], episodes[b]
        if [r["scene_seed"] for r in left] != [r["scene_seed"] for r in right]:
            raise ValueError("Cannot pair different validation scenes")
        draws = rng.integers(len(left), size=(2000, len(left)))
        for metric in ("return", "coverage", "throughput_bps", "last100_coverage", "final_coverage", "maintain_coverage"):
            diff = np.asarray([float(x[metric])-float(y[metric]) for x, y in zip(left, right)])
            ci = np.quantile(diff[draws].mean(1), [.025, .975])
            paired.append(dict(comparison=f"{a} - {b}", metric=metric, difference=float(diff.mean()), low=float(ci[0]), high=float(ci[1])))
    if paired:
        with (output/"paired_scene_differences.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    if run:
        run = Path(run)
        for filename, keys in (("metrics.csv", ["actor_loss", "value_loss", "ppo_approx_kl", "ppo_clip_fraction"]),
                               ("validation.csv", ["return", "coverage", "last100_coverage", "final_coverage"])):
            if (run/filename).exists():
                rows = read_csv(run/filename)
                fig, axes = plt.subplots(2, 2, figsize=(10, 6))
                for ax, key in zip(axes.flat, keys):
                    ax.plot([int(r["env_steps"]) for r in rows], [float(r[key]) for r in rows])
                    ax.set(xlabel="Joint transitions", ylabel=key)
                save_figure(fig, figdir, filename.replace(".csv", ""))
                if filename == "metrics.csv":
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    for ax, suffix in zip(axes, ("kl", "clip")):
                        for step in range(1, 9):
                            key = f"denoise_{step}_{suffix}"
                            if rows and key in rows[0]:
                                ax.plot([int(r["env_steps"]) for r in rows], [float(r[key]) for r in rows], label=f"step {step}")
                        ax.set(xlabel="Joint transitions", ylabel=suffix)
                    axes[0].legend(fontsize=6)
                    save_figure(fig, figdir, "per_denoising_step")
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    for key in ("encoding_seconds", "denoising_seconds", "packing_transfer_seconds", "environment_seconds", "update_seconds"):
                        if rows and key in rows[0]:
                            axes[0].plot([int(r["env_steps"]) for r in rows], [float(r[key]) for r in rows], label=key)
                    axes[0].set(xlabel="Joint transitions", ylabel="Seconds per rollout/update")
                    axes[0].legend(fontsize=6)
                    axes[1].plot([float(r["elapsed_seconds"]) for r in rows], [float(r["coverage"]) for r in rows])
                    axes[1].set(xlabel="Wall time (s), including evaluation", ylabel="Training rollout coverage")
                    save_figure(fig, figdir, "runtime")
        pretrain = run/"pretrain"/"history.csv"
        if pretrain.exists():
            rows = read_csv(pretrain)
            fig, ax = plt.subplots(figsize=(7, 4))
            for k in ("fit_movement_mse", "fit_score_mse", "selection_movement_mse", "selection_score_mse"):
                ax.plot([int(r["epoch"]) for r in rows], [float(r[k]) for r in rows], label=k)
            ax.set(xlabel="Pretraining epoch", ylabel="Noise prediction MSE")
            ax.legend(fontsize=7)
            save_figure(fig, figdir, "pretraining")
    summary = {name: json.loads((output/name/"summary.json").read_text()) for name in labels}
    costs = {}
    if run:
        for phase, path in (("online", Path(run)/"manifest.json"), ("pretraining", Path(run)/"pretrain/summary.json"),
                            ("demonstrations", Path(run)/"demonstrations/dataset.json")):
            if path.exists():
                saved = json.loads(path.read_text())
                costs[phase] = {k: saved[k] for k in ("elapsed_seconds", "peak_rss_mib", "env_steps", "epoch", "batches", "expected_joint_transitions") if k in saved}
    atomic_json(output/"comparison.json", dict(models=summary, critic=critic_results, paired=paired, costs=costs,
        confidence="paired validation-scene bootstrap, not training-seed confidence"))
    actual_steps = manifest.get("sources", {}).get("last", {}).get("env_steps", "见provenance")
    lines = ["# Diffusion DPPO 容量10对照报告", "", f"本次last实际在线步数：**{actual_steps}条联合转移**。示范和预训练成本另计；短流程不能作为正式1M性能结论。", "",
        "奖励为覆盖率＋0.1×归一化吞吐；全部模型采用采样评估。", "",
        "| 模型 | 全程覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐 Mbps | 回合回报 |", "|---|---:|---:|---:|---:|---:|"]
    for name in labels:
        s = summary[name]
        lines.append(f"| {name} | {s['coverage']:.3%} | {s['last100_coverage']:.3%} | {s['final_coverage']:.3%} | {s['throughput_bps']/1e6:.3f} | {s['return']:.4f} |")
    lines += ["", "主要比较同预算last及相对预训练后step0的增益。教师在旧集中式协议训练，新协议含Top-K竞争与一步评分时序；新旧差异不能只归因于扩散结构。", "",
        "Critic在各自on-policy轨迹上与真实终止的两项奖励MC比较，不包含熵或去噪概率。", "",
        "场景区间见 paired_scene_differences.csv；不是训练种子置信区间。示范采集和预训练成本另计，保留测试集未参与本报告。", "",
        "图表：", "", "- [覆盖、吞吐和奖励](figures/task_curves.png)", "- [部署维持和边界行为](figures/deployment.png)",
        "- [提议与竞争](figures/protocol.png)", "- [奖励分量](figures/reward_components.png)",
        "- [条件动作随机性](figures/action_randomness.png)", "- [评分与近悬停](figures/score_and_motion_distribution.png)",
        "- [Critic与MC](figures/critic_mc.png)"]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return dict(output=str(output), models=labels)


def report(run, output, device="cpu", rebuild_only=False, moment_repeats=64, include_baseline=True):
    run, output = Path(run), Path(output)
    if rebuild_only:
        return rebuild(output, run)
    output.mkdir(parents=True, exist_ok=True)
    paths = {k: run/(k+".pt") for k in ("step0", "best", "last")}
    reference = load(paths["last"], kinds={"dppo_model"})
    cfg = from_dict(reference["config"])
    cfg.run.device = device
    cfg.diffusion.actor_update_devices = []
    selected_device = configure_runtime(cfg)
    ensure_budget(cfg, output)
    labels, sources = [], {}
    for name, path in paths.items():
        digest = file_hash(path)
        previous = output/name/"provenance.json"
        cached = json.loads(previous.read_text()) if previous.exists() else {}
        if (cached.get("sha256") != digest or cached.get("moment_repeats") != moment_repeats
                or cached.get("device") != device or cached.get("scene_seeds") != cfg.run.validation_seeds):
            evaluate_checkpoint(path, output/name, device, moment_repeats=moment_repeats)
        labels.append(name)
        source_meta = json.loads((output/name/"provenance.json").read_text())
        sources[name] = dict(path=str(path), sha256=digest, env_steps=source_meta["env_steps"])
    if include_baseline:
        teacher, teacher_source = teacher_policy(cfg, selected_device)
        evaluate(cfg, TeacherExecution(teacher, cfg), cfg.run.validation_seeds, output/"teacher_topk")
        labels.append("teacher_topk")
        sources["teacher_topk"] = teacher_source
        for name, path in (("baseline_best", Path(cfg.diffusion.teacher_checkpoint)),
                           ("baseline_last", Path(cfg.diffusion.teacher_checkpoint).with_name("last.pt"))):
            if not path.exists():
                raise FileNotFoundError(f"Baseline reference missing: {path}")
            data = load_checkpoint(path)
            old_cfg = config_from_dict(data["config"])
            if old_cfg.env != cfg.env:
                raise ValueError("Baseline physical scenario differs")
            learner = make_learner(old_cfg, selected_device)
            learner.load_state_dict(data["learner"])
            evaluate(cfg, GaussianExecution(learner.actor), cfg.run.validation_seeds,
                     output/name, critic=learner, greedy=True)
            labels.append(name)
            sources[name] = dict(path=str(path), sha256=file_hash(path), env_steps=data["env_steps"])
    atomic_json(output/"report_sources.json", dict(labels=labels, sources=sources,
        validation_seeds=cfg.run.validation_seeds, action_seed=cfg.run.evaluation_action_seed,
        moment_repeats=moment_repeats, run=str(run)))
    # Rebuilding uses only saved report inputs; no checkpoint/run directory needed.
    target = output/"training_data"
    target.mkdir(exist_ok=True)
    for filename in ("metrics.csv", "validation.csv", "manifest.json", "config.yaml"):
        if (run/filename).exists():
            shutil.copyfile(run/filename, target/filename)
    pretrained = Path(reference.get("provenance", {}).get("pretrained_path", run/"pretrain/best.pt"))
    demo_source = Path(reference.get("provenance", {}).get("pretraining_provenance", {}).get(
        "dataset_path", pretrained.parent.parent/"demonstrations"))
    for phase, source in (("pretrain", pretrained.parent), ("demonstrations", demo_source)):
        for filename in ("history.csv", "summary.json", "dataset.json"):
            if (source/filename).exists():
                (target/phase).mkdir(exist_ok=True)
                shutil.copyfile(source/filename, target/phase/filename)
    return rebuild(output, run)
