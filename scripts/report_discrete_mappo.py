"""Rebuild acceptance report and main-scene figures from saved artifacts."""

import json
from pathlib import Path
import torch
from uav_research.runtime import configure_runtime
from uav_research.evaluate import policy_from_checkpoint, evaluate
from uav_research.deployment import export_actor
from uav_research.plotting import curves, trace_plot, style, save_figure
import matplotlib.pyplot as plt


def main():
    root = Path("analysis/discrete_mappo_20261013")
    manifest = json.loads((root / "main_4096" / "manifest.json").read_text())
    cfg, policy = policy_from_checkpoint(
        root / "main_4096" / "last.pt", torch.device("cpu")
    )
    configure_runtime(cfg)
    e = cfg.env
    evaluate(
        cfg, policy, cfg.run.validation_seeds, root / "main_evaluation", trace_count=1
    )
    export_actor(root / "main_4096" / "last.pt", root / "main_actor.pt")
    curves([root / "main_4096"], root / "main_figures")
    trace_plot(root / "main_evaluation" / "trace_10000.npz", root / "main_figures")
    rows = {
        n: json.loads((root / f"benchmark_threads_{n}.json").read_text())["results"][0]
        for n in (1, 4, 8)
    }
    style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [str(n) for n in rows],
        [r["transitions_per_second"] for r in rows.values()],
        color="#0072B2",
    )
    ax.set(
        xlabel="PyTorch CPU threads",
        ylabel="Joint transitions / second",
        title="Discrete MAPPO: 2048 transitions, 10 epochs",
    )
    save_figure(fig, root / "main_figures", "cpu_threads")
    text = f"""# 九动作离散 MAPPO 本地验收（2026-10-13）

已新增离散MAPPO版本，连续版本继续可用。完整建模见 [离散文档](../ENVIRONMENT%20MODELING%20DISCRETE%20MAPPO.md)，主配置 `configs/discrete_mappo.yaml`，WSL配置 `configs/discrete_mappo_wsl_50k.yaml`，零鲁棒性兼容配置 `configs/discrete_mappo_no_robustness.yaml`。

## 实现与兼容

世界坐标前=+y、右=+x；0–7为八方向、8为悬停，主场景移动动作等速{e.max_speed * e.dt:g} m/步。输入整数[U]/[E,U]，共用只读动作映射。斜向归一化、边界裁剪、无掩码。图、无线与UE集合共用参数化实现；主场景{e.width:g}×{e.height:g} m、{e.num_users} UE、{e.num_bs} BS，UE初始化{e.user_initialization}、鲁棒性权重{e.robustness_weight:g}。物理参数与连续SAC不同。

分类actor使用原UE集合编码器，输出9个logits；PPO按每架UAV的概率比更新，使用精确分类熵、团队GAE和集中V(s)。每轮重新编码原始集合。无SAC随机warmup、回放池、温度或目标网络。原高斯MAPPO仍可使用连续动作配置。

检查点/actor导出包含action_space、action_mapping_version、policy_distribution；历史缺省动作模式按continuous解释。跨模式恢复与映射版本错误会拒绝加载。完整保存仅在MAPPO更新安全点进行；更新中途失败会保留旧完整检查点，不伪装为可恢复的新状态。真实终止不bootstrap，预算尾端的短rollout正常更新。

静止、随机、局部启发式已适配整数动作。轨迹包含action_ids、action和actual_displacement；训练动作频率是日志对应向量步的统计，评估动作频率是整回合均值。图表输出PDF/SVG/PNG，训练曲线禁止混合不同动作空间的实验。

## 正确性与完整流程

本次测试状态与数量以tests.txt原始日志为准；测试覆盖精确中断恢复。测试原始日志：`analysis/discrete_mappo_20261013/tests.txt`。包含连续版本回归、离散映射/非法动作/边界/悬停、标量与批量/进程一致、分类概率与熵、encoder梯度、终止/片段GAE、三个baseline、精确中断恢复、actor导出评估绘图、短预算及保存失败保护。

小配置完成256条→恢复288条、两个验证场景、三种baseline、actor导出和图表。主场景使用{e.num_uavs} UAV/{e.num_users} UE/{e.num_bs} BS、{cfg.run.num_envs}环境、rollout{cfg.algo.rollout_steps}、{cfg.algo.ppo_epochs} epochs/{cfg.algo.ppo_minibatches} minibatches，训练{manifest["env_steps"]}条，{manifest["updates"]}轮更新。仅验证种子10000/10001；完整配置保留32验证与100测试场景，本轮未使用测试集。

- 主场景状态：{manifest["status"]}，{manifest["env_steps"]}条转移，{manifest["updates"]}轮更新。
- 总时间：{manifest["elapsed_seconds"]:.2f}秒，含本地两个验证场景的周期评估与保存。
- 进程累计峰值RSS：{manifest["peak_rss_mib"] / 1024:.3f} GiB（同一验收进程此前运行过小配置与绘图，并非独立进程内存增量）。
- 回放分配：0字节；原始rollout集合只保留本轮训练所需内容。

这两个更新不能证明收敛、算法优势或鲁棒性收益；训练日志中的当前向量步覆盖不能替代固定验证集全回合均值。

## WSL2 实测

Core Ultra 5 225H，本机CUDA不可用，Numba4线程。每项相同初始场景、16环境、2048条转移、1轮PPO更新（10 epochs×4 minibatches），包括采集、推理、集合处理和更新；不含验证及正式保存。每个线程设置短基准一次，首次PPO更新在计时内。

| PyTorch线程 | 转移/秒 | PPO更新耗时 | 全流程耗时 |
|---|---:|---:|---:|
"""
    for n, r in rows.items():
        text += f"| {n} | {r['transitions_per_second']:.1f} | {r['phase_seconds']['update']:.2f} s | {r['wall_seconds']:.2f} s |\n"
    text += """
线程推荐与训练时间应根据本次benchmark JSON重新计算，不沿用历史场景的速度。短基准不含正式验证与完整保存，也不能保证真实250k速度。本轮未执行正式250k验证。

![CPU基准](../analysis/discrete_mappo_20261013/main_figures/cpu_threads.png)

## 运行与重建

新训练独立保存：

```bash
.venv/bin/python -m uav_research train --config configs/discrete_mappo_wsl_50k.yaml --output runs/wsl_discrete_mappo_paper_scene_250k_seed0
```

原始验收入口 `scripts/validate_discrete_mappo.py`（结果目录必须为空；脚本只训练小配置与4096条主场景，并执行短基准）。报告及图表重建入口 `scripts/report_discrete_mappo.py`，只读取模型做评估/绘图，不更新训练模型。完整续训使用 `resume_XXXXXXXXXXXX` 目录而非last.pt；保持相同动作模式与训练超参数。

本轮未连接服务器、未启动250k、多种子或连续/离散性能对比实验。
"""
    Path("docs/discrete_mappo_validation_20261013.md").write_text(text)


if __name__ == "__main__":
    main()
