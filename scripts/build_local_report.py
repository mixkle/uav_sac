"""Rebuild the local acceptance report and memory plot from saved raw results."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from uav_research.config import load_config
from uav_research.plotting import style, save_figure, plt, COLORS


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def memory_plot(root):
    samples = json.loads((root / "memory/rss_samples.json").read_text())
    style()
    stamps = np.array([s[0] for s in samples])
    rss = np.array([s[2] for s in samples]) / 2**30
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(stamps - stamps[0], rss, color=COLORS[0], lw=1.5)
    for i, phase in enumerate(dict.fromkeys(s[1] for s in samples)):
        ids = [j for j, s in enumerate(samples) if s[1] == phase]
        ax.axvspan(
            stamps[ids[0]] - stamps[0],
            stamps[ids[-1]] - stamps[0],
            color=COLORS[i],
            alpha=0.09,
        )
        ax.text(
            stamps[ids[len(ids) // 2]] - stamps[0], 10.5, phase, ha="center", fontsize=8
        )
    ax.axhline(12, color=COLORS[1], ls="--", label="Local RSS limit")
    ax.set(xlabel="Elapsed time (s)", ylabel="Resident memory (GiB)", ylim=(0, 12.8))
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, root / "figures/resources", "full_replay_memory")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("analysis/beacon_v2_20261013"))
    p.add_argument(
        "--output", type=Path, default=Path("docs/beacon_v2_validation_20261013.md")
    )
    args = p.parse_args()
    root = args.input
    e = load_config(root / "calibration/config.yaml").env
    memory = json.loads((root / "memory/result.json").read_text())
    calibration = read_csv(root / "calibration/summary.csv")
    benchmark = json.loads((root / "benchmark.json").read_text())["results"]
    tests = (root / "logs/tests.log").read_text().strip().splitlines()[-1]
    cleanup = json.loads(Path("analysis/cleanup_20261013.json").read_text())
    accept = json.loads((root / "acceptance.json").read_text())
    assert accept["passed"] and memory["passed"]
    lines = [
        "# 新版双 BS 展开实验：本地验收（2026-10-13）",
        "",
        "已完成 beacon-set-v2 环境、SAC/MAPPO/MATD3、集合回放、分块保存恢复与本地完整性验收。当前默认场景为180 UE四中心截断高斯混合。正式训练、GPU性能与CUDA恢复一致性仍等待训练主机可用后验证。",
        "",
        "本报告由 `scripts/build_local_report.py` 从原始数据生成；全部作业命令与耗时见 `analysis/beacon_v2_20261013/validation_jobs.json`，日志位于其 `logs/` 目录。",
        "",
        "## 场景及参考论文",
        "",
        "1600×1600 m，16 UAV从两角BS各8架出发，240→180 UE；四中心为(480,480)、(1120,480)、(480,1120)、(1120,1120) m，标准差128 m，初始范围[160,1440]²。只对越界坐标重采样，不裁剪、不按服务筛选；初始留白不成为后续运动边界。",
        "",
        "论文第V-A节只提供多中心二维高斯混合这一分布类型，未明确留白、中心或方差。本实验数值为自定参数，依据见 [初始化参考说明](ue_initialization_reference.md)。",
        "",
        f"原始固定观察每UAV91维、编码后124维、全局状态{e.state_dim}维。发现、下行、回传、容量四阶段独立诊断；所有发现UE均输入集合，不按容量截断。",
        "",
        f"清理了重复或旧场景流程结果共 {cleanup['freed_bytes'] / 2**30:.2f} GiB，明细在 `analysis/cleanup_20261013.json`。保留历史故障/性能分析用于溯源，旧训练产物未参与本轮比较。",
        "",
        "## 正确性和保存恢复",
        "",
        f"测试结果：**{tests}**。包含初始化分布与边界、上下行阈值、发现/关联、双BS无隐藏转发、故障上界、空组/变长/置换/重复采样、编码器梯度归属、环形回放副本、600步终止、原子保存失败保护与旧观察检查点拒绝加载。",
        "",
        "三种算法的CPU中断续训与连续训练逐项一致：actor、critic/value、目标网络、优化器、温度/归一化器、Torch RNG及回放数据。小场景完成256条转移→保存→恢复到288条→评估→actor导出→绘图。",
        "",
        "满容量测试使用50,000条合成联合转移，每架UAV都观察全部180 UE，数组使用非零随机数据，按最多256条分块写入和恢复。",
        "",
        "| 指标 | 实测 |",
        "|---|---:|",
        f"| 原始回放 | {memory['replay_bytes'] / 2**30:.3f} GiB |",
        f"| 保存/恢复测试进程峰值RSS | {memory['peak_rss_gib']:.3f} GiB |",
        f"| 保存耗时 | {memory['save_seconds']:.2f} s |",
        f"| 恢复耗时 | {memory['load_seconds']:.2f} s |",
        "| 全部数组SHA256与下一批采样 | 一致 |",
        "",
        "峰值低于12 GiB目标。此压力测试只验收回放I/O，不包括满容量学习器更新；短训内存另列。成功后删除合成检查点，保留result.json和RSS记录。",
        "",
        "![满容量回放内存](../analysis/beacon_v2_20261013/figures/resources/full_replay_memory.png)",
        "",
        "## 参数校准",
        "",
        "独立种子30000–30007，14组参数（默认、3×3上下行门槛、UU/UG各±20%），每组四种布局、每回合600步，共448回合。固定目标布局只用于物理诊断，不是局部策略baseline或最优覆盖上界。默认参数没有因扫描结果自动改变。",
        "",
        "| 布局 | 初始覆盖 | 全任务覆盖 | 终态覆盖 | 全任务发现 | 全任务J | 初始零奖励步数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "static": "出发静止",
        "random": "随机运动",
        "scripted_expansion": "固定目标逐步展开",
        "expanded_static": "展开后静止",
    }
    for mode, label in names.items():
        rows = [
            r for r in calibration if r["variant"] == "default" and r["mode"] == mode
        ]
        lines.append(
            f"| {label} | {mean(rows, 'initial_coverage'):.2%} | {mean(rows, 'coverage'):.2%} | {mean(rows, 'final_coverage'):.2%} | {mean(rows, 'discovery_fraction'):.2%} | {mean(rows, 'robustness'):.2%} | {mean(rows, 'zero_reward_initial_steps'):.1f} |"
        )
    lines += [
        "",
        "![发现与展开校准](../analysis/beacon_v2_20261013/figures/calibration/calibration_expansion.png)",
        "",
        "初始静止布局在这8个校准场景中全程零服务；虽然部分UE可发现，满负载下行干扰使候选不能达标。随机运动首次获得奖励较迟。这是稀疏奖励与协调展开难点，不通过筛选场景消除。展开模板能够服务多数UE，表明默认链路预算具有可行布局，但不是学习成功证据。",
        "",
        "| 布局 | 感知淘汰UE | 下行淘汰UE | 回传淘汰UE | 容量淘汰UE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in names.items():
        rows = [
            r for r in calibration if r["variant"] == "default" and r["mode"] == mode
        ]
        vals = [
            mean(rows, k)
            for k in [
                "sensing_rejected_users",
                "downlink_rejected_users",
                "backhaul_rejected_users",
                "capacity_rejected_users",
            ]
        ]
        lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += [
        "",
        "以上为每步每回合平均人数，四类互斥。均值对应的是瞬时服务资格，不是累计发现人数。新分布形成热点，即使总容量320>180，固定模板也会出现局部容量不足。",
        "",
        "| 范围变体 | 逐步展开覆盖 | 逐步展开J | 回传淘汰UE |",
        "|---|---:|---:|---:|",
    ]
    for variant in ["uu_range_0.8", "uu_range_1.2", "ug_range_0.8", "ug_range_1.2"]:
        rows = [
            r
            for r in calibration
            if r["variant"] == variant and r["mode"] == "scripted_expansion"
        ]
        lines.append(
            f"| {variant} | {mean(rows, 'coverage'):.2%} | {mean(rows, 'robustness'):.2%} | {mean(rows, 'backhaul_rejected_users'):.2f} |"
        )
    lines += [
        "",
        "UU缩短到320 m时，模板中340 m的相邻链路断开，因此性能显著下降；这说明该模板对通信距离敏感，不能把它当作任何参数下都可行的部署。完整门槛/距离扫描表见 `calibration/summary.csv`，未进行学习调参。",
        "",
        "## 主场景短流程",
        "",
        "三种算法均使用2环境、batch64、warmup256、回放4096、训练种子0，最多4096条联合转移；验证只使用10000–10003共4个场景，未使用保留测试集。正式配置的batch256/warmup10000/回放50000均未在此短训中启用。",
        "",
        "| 算法 | 联合转移 | 更新次数 | 完整运行耗时 | 峰值RSS | 最后验证覆盖 | 最佳验证覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ["sac", "mappo", "matd3"]:
        run = root / "main" / algorithm
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "completed" and manifest["env_steps"] == 4096
        last = json.loads((run / "evaluation_last/summary.json").read_text())["mean"]
        best = json.loads((run / "evaluation_best/summary.json").read_text())["mean"]
        lines.append(
            f"| {algorithm.upper()} | {manifest['env_steps']} | {manifest['updates']} | {manifest['elapsed_seconds']:.1f} s | {manifest['peak_rss_mib'] / 1024:.3f} GiB | {last['coverage']:.3%} | {best['coverage']:.3%} |"
        )
    lines += [
        "",
        "小样本确定性验证覆盖较低，不能宣称算法已形成有效展开或超过baseline。随机动作训练采集与确定性验证不同，不能用训练时偶发覆盖替代验证。最佳模型按验证团队回报选择，同时保留并评估最后模型。",
        "",
        "## CPU性能",
        "",
        "本地WSL2、CPU/FP32，256条测量转移，排除启动/Numba编译；每4条转移更新一次，batch64。CUDA未验收。",
        "",
        "| 环境数 | 联合转移/s | 环境阶段 | 推理阶段 | 采样阶段 | 更新阶段 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark:
        phase = row["phase_seconds"]
        lines.append(
            f"| {row['num_envs']} | {row['transitions_per_second']:.2f} | {phase['environment']:.3f} s | {phase['inference']:.3f} s | {phase['sample']:.3f} s | {phase['update']:.3f} s |"
        )
    lines += [
        "",
        "阶段耗时对应同一256条转移；总耗时还包括重置等开销。另保存物理链路、集合打包、原始数据转Tensor、集合编码聚合及小回放保存的独立微基准。独立微基准与端到端阶段不能相加。",
        "",
        "当前CPU网络更新占主要时间，增加环境数只改善部分采集/推理开销。正式预算需要在目标CPU/GPU、batch256和真实发现集合长度下重新测量，不能按本地batch64速度直接估算服务器训练。",
        "",
        "## 重建与剩余验收",
        "",
        "全部图形为PDF/SVG/300 dpi PNG，由保存数据重建。包含初始化、发现—服务、当前/累计发现、初始/中间/终态拓扑、集合长度、逐故障损失、样本/墙钟曲线、门槛敏感性和内存曲线。",
        "",
        "```bash",
        ".venv/bin/python scripts/validate_local.py --output analysis/beacon_v2_20261013",
        ".venv/bin/python scripts/build_local_report.py --input analysis/beacon_v2_20261013",
        "```",
        "",
        "重跑完整验收须使用空的新输出目录，防止覆盖已完成训练。恢复主机后再验收CUDA性能和CUDA中断一致性；正式单种子训练与多种子均等待用户通知。",
        "",
    ]
    memory_plot(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
