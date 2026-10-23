# 中央 2500×2500 m UE 初始化实验

以 C 组（移动 UE、离散 MAPPO、纯覆盖奖励）为基准，从头训练种子 0 的 250000 条联合转移。唯一实验变量为 UE 初始分布：场景仍为 3500×3500 m，150 个 UE 在 `[500,3000]×[500,3000]` m 内独立均匀采样。

配置 `configs/discrete_mappo_central_ue_250k.yaml` 使用 `user_initialization: uniform_interior` 和 `user_initial_margin_fraction: 0.14285714285714285`。原 `uniform` 模式仍在全区域采样，不受 margin 参数影响。回退直接使用原 C 配置。

UAV 的全区域初始化、三个 BS、通信与感知参数、UE 移动、400 步时限、纯覆盖奖励、MAPPO 网络及优化设置（包括原 value clipping）均不变。中央边界仅限制初始化，不限制后续运动。没有继承 critic 校准实验的权重或优化器。

```bash
.venv/bin/python -u -m uav_research train \
  --config configs/discrete_mappo_central_ue_250k.yaml \
  --output runs/mappo_central_ue_250k_seed0

.venv/bin/python scripts/report_mappo_training.py \
  --run runs/mappo_central_ue_250k_seed0 \
  --output analysis/mappo_central_ue_250k_seed0
```

评估 best 与 last，使用原有 32 个固定验证种子与动作采样种子，不使用测试集，不额外运行启发式等 baseline。best 按验证全回合平均覆盖率（与本实验回报等价）选择。

初始用户密度从 12.245 人/km² 变为 24 人/km²，约为原来的 1.96 倍；同时用户离三个角落 BS 更远。因此直接对比原 C 的绝对覆盖率包含场景变化，必须同时报告各自未训练策略与训练后策略的差值。配对种子的 UAV 位置完全一致，UE 位置为原位置的仿射缩放，不能称为相同物理场景。

测试包括原配置兼容、位置上下界与均匀性、配对 UAV 初始化与 UE 运动随机流、初始化边界不约束运动，以及标量/批量/进程环境的一致性。结果与分析保存于 `analysis/mappo_central_ue_250k_seed0/`。

## 本地验收记录

- 初始化、原六组奖励与离散MAPPO流程测试：60项通过。
- 含新增中央初始化的完整初始化测试：15项通过；将向量测试改为两环境以覆盖批量物理计算后，两条路径复测通过。
- 环境、学习、UAV随机初始化与Value Clipping回归：33项通过。
- 新配置相对原C配置，仅 `user_initialization` 与其配套边距改变；测试对整个配置对象作了等价断言。
- 新初始化模式不改变状态维度、观测结构、检查点结构或训练目标；检查点继续保存完整配置。

训练完成后，重新生成对照图与汇总：

```bash
.venv/bin/python scripts/compare_central_ue.py
```

图表与指标见 [独立评估](../analysis/mappo_central_ue_250k_seed0/report.md) 和 [与原C的对照](../analysis/mappo_central_ue_250k_seed0/initialization_comparison.md)。

## 本次执行结果

250000条联合转移、123轮PPO正常完成，训练耗时885.48秒（14.76分钟），峰值RSS1233.09MiB。最佳模型为最后250000步模型；best/last各重新验证32个场景，与训练期记录一致。

全回合覆盖率40.98%，最后100步39.09%，终态38.33%；未训练策略全回合覆盖29.18%，训练增益11.80个百分点。原C组last覆盖38.08%，best覆盖40.53%。新组平均断联24.72%，高于原C last的19.39%。回合101–200步平均覆盖45.13%，后100步下降到39.09%，仍存在部署维持问题。完整图像和拓扑解释见对照报告。

## 冻结策略的价值预测检查

随后仅评估250k last（best同为250k），固定actor与critic，不作MC拟合。采集原32验证场景的完整回合，比较原始尺度V预测与无bootstrap的MC回报。结果全程RMSE22.761，末步预测均值48.959、MC回报0.383；32个场景末步预测全部超过理论上界1。中央初始化没有消除有限时域价值高估问题。

[预测图、误差及方法](../analysis/mappo_central_ue_250k_seed0/critic_audit/report.md)。复现命令：`.venv/bin/python scripts/audit_central_ue_critic.py`。这次没有修改训练设置或权重。
