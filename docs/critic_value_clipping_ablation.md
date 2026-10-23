# Critic Value Clipping 单变量对照

以 C 组“移动 UE＋九动作 MAPPO＋纯覆盖奖励”为基准，从头训练种子 0、250000 条联合转移。实验只将 `algo.value_clipping` 从 `true` 改为 `false`；没有继承 actor、critic 或 Adam。

| 项目 | 基准 | 对照 |
|---|---|---|
| 配置 | `configs/discrete_mappo_ablation_c_mobile_coverage.yaml` | `configs/discrete_mappo_critic_no_value_clip.yaml` |
| 输出目录 | `runs/mappo_ablation_c_mobile_coverage_250k_seed0` | `runs/mappo_critic_no_value_clip_250k_seed0` |
| Value Clipping | 开启（缺省） | 关闭 |
| 其他环境、算法、运行配置 | 完全一致 | 完全一致 |

设 \(v\)、\(v_{\rm old}\)、\(R\) 均为同一组更新后 PopArt 统计量下的归一化值：

\[
L_{\rm base}=\mathbb E\max\{H_{10}(v-R),H_{10}(v_{\rm old}+
\operatorname{clip}(v-v_{\rm old},-0.2,0.2)-R)\},
\qquad L_{\rm no\ clip}=\mathbb E H_{10}(v-R).
\]

保持 Huber δ=10、PopArt（包括输出重标定）、GAE λ=0.95、γ=0.995、actor/critic 学习率5e-4、梯度范数裁剪0.5、**actor PPO clipping=0.2**、熵系数0.01、16环境×128步、10 epochs/4 minibatches不变。奖励始终为联网服务覆盖率，吞吐/鲁棒性/断联权重均为0。

开关缺省 `true`，旧配置与旧检查点保持原语义；禁止切换开关后跨组直接续训。回退使用原 C 组配置即可。

## 运行与复核

正式实验已经完成；以下命令用于复现。已有目录不覆盖，重复实验须换输出目录。

```bash
.venv/bin/python -m uav_research train \
  --config configs/discrete_mappo_critic_no_value_clip.yaml \
  --output runs/mappo_critic_no_value_clip_250k_seed0

.venv/bin/python scripts/report_mappo_training.py \
  --run runs/mappo_critic_no_value_clip_250k_seed0 \
  --output analysis/critic_value_clipping_250k_seed0/no_value_clip

.venv/bin/python scripts/compare_value_clipping.py
```

比较脚本读取既有 C 组报告 `analysis/ue_reward_ablations_250k_seed0/c_mobile_coverage/`，以及新组报告，检查配置只有开关不同、训练均完成250k、报告对应的检查点哈希一致，并保存源文件校验值。若需重建 C 组报告，可对该运行使用同一个 `report_mappo_training.py`。

验证固定32个场景10000–10031，分类动作采样种子880001，环境批次16；保留测试集未使用。分别比较 `last`（相同训练预算）和 `best`（各自全回合验证回报最大）。每组输出损失、奖励、覆盖/感知/断联曲线，以及前两个固定场景的初始/中间/终态拓扑。

额外冻结每个检查点，在相同验证协议下采集完整400步回合，记录原始尺度 \(V(s_t)\)，计算 \(G_t=r_t+\gamma G_{t+1},G_T=0\)。报告全程、前100步、中间100步、最后100步、最后20步、最后一步的偏差、MAE、RMSE；不执行任何训练更新。

两种 loss 定义不同且 PopArt 统计随训练变化，不能仅通过归一化 loss 数值判断哪组 critic 更准。Monte Carlo 回报是单条采样轨迹实现值，不是精确条件期望；不同策略产生不同状态分布，因此本实验检查各自 on-policy 预测误差，不构成同一状态上的 critic 函数比较。单训练种子、单验证动作种子的结果用于定位问题，不证明唯一根因。

结果与图表：[对照报告](../analysis/critic_value_clipping_250k_seed0/report.md)。

## 实施验收（2026-10-20）

- 关闭分支仅删除价值裁剪分支；默认分支与原始公式的数值及梯度完全一致。
- 配置逐项比较确认只有 `algo.value_clipping` 不同；历史 C 组配置仍按缺省 `true` 加载。
- 全套154项测试通过，包括开启/关闭裁剪、1/8线程下的完整训练与精确续训；跨开关续训被拒绝。
- 新组250k训练完成，耗时834.1秒（约13.9分钟）、峰值RSS1172.75 MiB；无回放池。
- best位于210944步，last位于250000步。独立验证与训练时保存的验证结果一致。
