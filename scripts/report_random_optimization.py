"""Rebuild the random-initialization and CPU benchmark report from saved data."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("analysis/wsl_random_optimization_20261013")
init = json.loads((root / "initialization.json").read_text())
bench = {
    n: json.loads((root / f"end_to_end_{n}.json").read_text())["results"][0]
    for n in (1, 4, 8)
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].bar(
    ["BS launch", "Uniform"],
    [init[m]["mean"]["coverage"] * 100 for m in ("bs_launch", "uniform")],
    color=["#999999", "#0072B2"],
)
axes[0].set(ylabel="Initial coverage (%)", title="128 paired initialization seeds")
axes[1].bar(
    [str(n) for n in bench],
    [r["transitions_per_second"] for r in bench.values()],
    color="#0072B2",
)
axes[1].set(
    xlabel="PyTorch CPU threads",
    ylabel="Joint transitions / second",
    title="Collection + SAC updates",
)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(root / f"initialization_and_speed.{ext}", dpi=250)
u = init["uniform"]
m = u["mean"]
text = f"""# 随机初始化与 WSL2 CPU 优化（2026-10-13）

默认 UAV 水平坐标在完整 1600×1600 m 区域独立均匀采样，高度100 m。不筛选覆盖、连通性或 BS 可达性。UE 仍为180个四中心截断高斯样本。所有 baseline 和学习方法共用同一分布。

`bs_launch` 保留为历史复现选项；历史配置包含 launch_radius 而无模式字段时按旧模式读取。旧50k结果不能与新任务直接混合比较，新训练使用新目录并从头开始。随机初始化没有证明初始化是旧策略退化的唯一原因，也没有自动解决断网问题。

## 初始物理状态

独立校准种子30000–30127，共128个；仅统计 reset，不训练、不筛选，不使用保留测试集。

| 指标 | BS附近出发 | 全区域随机 |
|---|---:|---:|
| 平均初始覆盖 | 0% | {m["coverage"]:.2%} |
| 平均发现比例 | {init["bs_launch"]["mean"]["discovery_fraction"]:.2%} | {m["discovery_fraction"]:.2%} |
| 平均断网 UAV 数 | 0 | {m["disconnected_uavs"]:.2f}/16 |
| 零覆盖场景数 | 128/128 | {u["zero_initial_coverage"]}/128 |

随机分散降低出发聚集的干扰、增加初始发现，但回传仍可能阻止服务。后续应比较覆盖随时间的变化、断网数量及故障保留；不能仅凭初始奖励上升宣称策略改进。

## CPU 实测与实现变化

本机 Core Ultra 5 225H，PyTorch 2.10.0+cu128；当前 CUDA 不可用。原50k训练耗时32.35分钟，其中参数更新1740.76秒，约89.7%，环境仅29.55秒。主要优化对象是神经网络更新。

1. WSL专用配置将 PyTorch CPU线程从1改为8，Numba保持4；其他配置不强制套用本机设置。
2. SAC标准差日志复用采样时计算的分布，移除重复actor/UE编码器前向。标准差现记录本次更新前、实际采样使用的值；不改变动作采样、损失或随机数顺序。
3. 回放直接批量打包固定字段、UE数组和偏移，避免逐样本创建和校验临时观测对象。仍保存原始集合及副本，不缓存旧编码。
4. Adam foreach 的小幅收益不稳定，本次不启用。保持FP32、batch256、每4条转移一次更新及全部奖励/网络超参数。

新随机场景，16环境、batch256、每种配置256条转移/64次更新；预热不计入，含采集、推理、回放采样和更新，不含周期验证/正式保存。短基准各一次，因此需留意机器负载及后期集合长度变化。

| PyTorch线程 | 联合转移/秒 | 相对单线程 |
|---|---:|---:|
"""
for n, r in bench.items():
    text += f"| {n} | {r['transitions_per_second']:.2f} | {r['transitions_per_second'] / bench[1]['transitions_per_second']:.2f}× |\n"
text += """
在旧训练真实非空回放批次上，优化后单次更新单线程208.6 ms、8线程82.6 ms；这是固定批次微基准，不能等同整次训练加速。随机初始化的UE集合大小与旧出发任务不同，不直接用旧32分钟除以加速比。

按当前短基准估算，50k含10k预热、10k梯度更新，更新及采集约14分钟起，加上验证、保存和运行波动，暂预留 **15–25分钟**。这不是完整50k实测承诺。

![初始化和速度](../analysis/wsl_random_optimization_20261013/initialization_and_speed.png)

## 验证及运行

测试日志见 `analysis/wsl_random_optimization_20261013/tests_final.txt`，覆盖随机分布、零覆盖允许、历史模式解释、回放重复/空集合打包，以及三个算法的采集、更新、保存恢复精确一致、评估和绘图。该验证不证明算法优势。本轮未启动50k或服务器训练。

```bash
.venv/bin/python -m uav_research train --config configs/wsl_50k.yaml --output runs/wsl_sac_random_50k_seed0
```

图表和报告可用 `.venv/bin/python scripts/report_random_optimization.py` 从保存JSON重建。`expansion_distance_mean` 保留日志字段名，其含义是距本回合初始位置的位移，不再特指距BS的展开距离。
"""
Path("docs/wsl_random_optimization_20261013.md").write_text(text)
