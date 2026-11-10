# Diffusion DPPO vs. Gaussian MAPPO：10M 训练性能差异定位实验计划

> **项目**：`mixkle/uav_sac`  
> **研究对象**：联合 UAV 二维运动与变长 UE 服务评分的多智能体策略  
> **文档版本**：2026-11-10 · 第一版诊断方案  
> **目标**：解释 Diffusion 在已完成的 10M 联合转移实验中为何不及 Gaussian MAPPO；区分**运动生成、评分关联、动作随机性、DPPO 优化机制**的影响，并在证据充分后开展有控制的继续微调。  
> **边界**：本文为**后续实验计划**，不是已执行实验报告。所有尚需编写的脚本、输出文件都标为“建议新增”，不得视为仓库现有实现。

---

## 0. 实验背景与已知证据

### 0.1 固定实验条件

依据仓库中 `analysis/hop_scaling_joint_service_10m_seed0/report.md` 及 `docs/hop_scaling_joint_service_comparison.md`：

- 两组均使用局部观察、共享 Actor、Top-K 服务评分与 UE 自主选择、最短跳数速度缩放、实际位移历史，以及包含时间因子的集中式 Critic；奖励为 `coverage + 0.1 * normalized_throughput`。
- 主场景为 18 架 UAV、150 个 UE、3 个 BS、每 UAV 最大服务 10 个 UE，回合共 400 步。
- Diffusion 为两阶段训练：64,000 条示范、50 轮预训练，再训练 10M 联合转移；Gaussian 从零初始化并训练 10M 联合转移。两组**在线环境交互预算相同，但数据、初始化与计算成本不相同**，不是仅改变策略分布的一因素实验。
- 现有独立验证：冻结 `step0`、`best`、`last`，32 个验证场景 `10000–10031`，每场景 400 步；Best 基于验证回报挑选，不能当未见测试集。报告保留快照、轨迹和 64 次条件采样诊断。

### 0.2 10M 结果概览（已完成，不是本计划的新结果）

| 指标 | Diffusion Best | Gaussian Best | Diffusion Last | Gaussian Last |
|---|---:|---:|---:|---:|
| 平均联网覆盖率 | 60.587% | **69.211%** | 57.662% | 59.322% |
| 最后 100 步覆盖率 | 65.243% | **75.104%** | 59.612% | 66.553% |
| 总吞吐 | 536.961 Mbps | **693.487 Mbps** | 505.563 Mbps | 557.489 Mbps |
| 平均实际速度 | 6.558 m/s | 6.775 m/s | 6.944 m/s | 4.850 m/s |
| 单步位移小于 1 m 的比例 | 4.326% | 2.746% | 3.727% | 19.442% |
| 平均最大单步覆盖跌幅 | 32.500 pp | 26.062 pp | 33.438 pp | 28.542 pp |

主要事实：

1. **Best 的覆盖差距显著大于 Last 的差距**；无法简单地将问题归结为 Diffusion 未收敛或最后回退。
2. Diffusion Last 的平均实际速度较高、近悬停比例较低，但 **Best 时两组平均速度相近**，因此速度只是线索，不是完整解释。
3. 两组策略训练后期均存在回退，提示需要同时排查 PPO 更新稳定性和优势估计，不能先验认定只有 Diffusion 实现有问题。
4. 大幅覆盖跌落不必然等于回传断链，也可能来自 UE 候选、提议、Top-K 或关联变化；必须分解事件来源。

**资料入口（已核对的仓库路径）**：

- [10M 完整分析报告](https://github.com/mixkle/uav_sac/blob/main/analysis/hop_scaling_joint_service_10m_seed0/report.md)
- [两组对照实验说明](https://github.com/mixkle/uav_sac/blob/main/docs/hop_scaling_joint_service_comparison.md)
- [Diffusion Learner](https://github.com/mixkle/uav_sac/blob/main/src/uav_research/diffusion_dppo/learner.py)
- [Diffusion Policy](https://github.com/mixkle/uav_sac/blob/main/src/uav_research/diffusion_dppo/policy.py)

### 0.3 核心假设与优先级

| 编号 | 假设 | 初步证据 | 首选实验 |
|---|---|---|---|
| H1 | 主要差距来自运动策略，而不是评分策略 | Diffusion 部署、服务候选与近悬停行为较弱 | 实验 1 |
| H2 | Diffusion 的条件动作随机性不利于部署维持 | Last 运动较强、反射与覆盖波动较多 | 实验 2 |
| H3 | 逐去噪步 PPO 优化造成更新效率/稳定性问题 | 8 步扩散链，按去噪转移执行 PPO 更新 | 实验 3 |
| H4 | 更保守的 DPPO 更新可改善 Best 保持和继续优化 | 两组均有 Best → Last 回退 | 实验 4 |

**实验执行次序**：`实验 1 + 实验 3 → 实验 2 → 实验 4`。实验 1、2、3 优先离线复用数据与模型；确实缺数据才补充冻结推理。实验 4 是唯一计划中的训练实验。

---

## 1. 统一实验协议与工程准备

### 1.1 数据和模型只读保护

1. 不覆盖原 `analysis/hop_scaling_joint_service_10m_seed0/` 中的快照、训练日志、配置或统计文件。
2. 对照 `snapshot_manifest.json` 和 `raw_data_hashes.json` 核验输入的来源；保留对应训练配置、代码提交 SHA、权重哈希、评估种子与环境状态版本。
3. 新诊断建议写入 `analysis/diffusion_gaussian_diagnosis_20261110/`，不要复用原有报告输出路径。该目录是**建议新增**。
4. 第一轮使用已存的 `best`、`last`，必要时增加 `step0`。先分析相同 32 场景（用于与旧报告一致），**不以其替代真正未见测试集**。
5. 对每个场景尽可能使用相同的 UE 移动随机流和初始 UAV/UE/BS 配置。策略采样随机流应独立存储；不要依赖两个模型恰好消耗相同的 RNG 调用次数。

### 1.2 必须统一的性能指标

对于一组 episode 计算：

- **平均有效联网覆盖率**：`mean_t rho_t`，注意必须要求 UE 所连 UAV 可到达某 BS。
- **最后 100 步覆盖率**：`mean_{t=301..400} rho_t`。
- **回合回报、平均吞吐（Mbps）、联网 UAV 比例**。
- **最坏单步覆盖跌幅**：每场景 `max_t max(0, rho_{t-1}-rho_t)`，再跨场景平均；**不**在场景均值曲线上取最大降幅。
- **≥10 pp 下降事件**：按相邻步定义并记录原始事件索引；必要时对连续下降合并成一次事件，两个定义分别给出。
- **提议—关联流程**：UE 候选数、每机 Top-K 提议数、移动后失效提议数、接受提议数、服务容量利用率、无可行关联 UE 数。
- **运动**：`||Δp_cmd||`、缩放系数、`||Δp_exec||`、反射后 `||Δp_real||`、小于 1m 比例、边界反射、最短跳数分桶。

结果汇总时报告逐场景差值、均值、中位数、配对 bootstrap 95% 区间；此区间仅代表固定训练种子下场景变化。最终论文结论还需要多训练种子验证。

### 1.3 冻结状态与奖励时序核对

- 验证动作从 **同一时刻动作执行前** 的环境状态出发，使用状态 `s_t` 和局部观察 `o_i^t`。
- 新动作执行后使用当前环境的缩放、反射、回传拓扑刷新、SINR 刷新、提议有效性过滤及 UE 关联规则。
- 评分只对时刻 `t` 的 `F_i^t` 有效，**不能把不在候选集里的 UE 评分强行交给协议**。
- 新的历史状态只记录反射后的真实位移，历史跳数/感知人数来自动作前时刻。
- 在 `t=400` 真正终止，不额外 bootstrap。

---

## 2. 实验 1：运动与 UE 评分的分离诊断（最优先）

### 2.1 研究问题

现有总性能差距究竟主要来自 `Δx, Δy` 运动分量，还是来自 UE 服务评分 `q_{i,m}`？两者的交互是否明显？

**假设 H1**：Diffusion 的运动决策导致较差的空间部署；即使采用 Gaussian 评分，也难以完全补偿。

### 2.2 设计：四个组合

| 组别 | 运动动作来源 | UE 评分来源 | 作用 |
|---|---|---|---|
| G/G | Gaussian | Gaussian | 参考控制组 |
| D/D | Diffusion | Diffusion | 当前 Diffusion 控制组 |
| **G/D** | Gaussian | Diffusion | 单独检查 Diffusion 评分能否支持较好运动 |
| **D/G** | Diffusion | Gaussian | 单独检查 Diffusion 运动的损失 |

**重要**：这四组是**人为重组的联合动作**，混合动作可能位于两模型训练分布之外。可用于因果定位（固定状态的动作干预），**不能直接作为第三种公平可部署算法**。

### 2.3 实施步骤

**阶段 A：单步配对反事实（必须完成）**

1. 从 `best/last` 已保存的轨迹中抽取相同的一组完整环境状态。若不能直接恢复，需要先核对快照是否包括完整 UAV/UE 状态、关联、UE 随机游走状态、历史输入和所有随机源；缺失则新增确定性状态导出工具，不要用只包含坐标的“伪相同状态”。
2. 至少从 32 个验证场景中按时间分层抽样，例如 `t ∈ {25, 100, 200, 300, 375}`，并额外选取若干覆盖骤降前1步状态。两模型取样来源都纳入，避免只偏向 Gaussian 常访问的状态。
3. 对同一个 `s_t`，分别从 D、G 两个策略基于**同一现算的局部观察**采样运动和 UE 评分。先确保评分槽位按候选 UE 的真实 ID 对齐，不用向量下标假定两模型顺序相同。
4. 构造 G/G、D/D、G/D、D/G；每种组合用相同的 UE 移动噪声以及单步环境转移规则。采样策略动作时使用独立且记录的 RNG 流。
5. 对各组合运行一步，记录 `r_{t+1}`、`rho_{t+1}`、`tau_{t+1}`、联网 UAV 比例、提议失效与最终接受数；对同一状态重复若干次动作采样（例如 16 次），估计条件均值与分布。
6. 如已有轨迹无法直接导出全局环境快照，应明确标记为**新增冻结推理实验**；不能宣称由原32场景数据即可准确构造所有反事实。

**阶段 B：短程分叉（建议）**

- 在阶段 A 发现明显方向后，从相同 `s_t` 分叉运行 `10/25/50` 步，每一步重新计算局部观察与候选集，并分别采样对应运动/评分。每组使用相同的外生 UE 移动随机流。
- 该测试检验单步差异能否累积为部署差异；短程混合策略同样存在分布外问题，结论必须附条件。

**阶段 C：事件原因分解（推荐与实验 1 同做）**

对所有 `rho_t-rho_{t-1} <= -0.10` 的事件，区分至少三个来源：

1. 回传可达性变化：哪个 UAV 新增断联，其原有服务 UE 有多少受到影响；
2. 无线/候选可行性变化：UE SINR 或发现集合改变导致多少服务失效；
3. 提议与关联决策变化：Top-K 提议不足、UE 重复选择/竞争、移动后提议过滤等导致多少服务下降。

建议在关联流程各阶段记录候选/提议/有效提议/最终关联，并采用**固定次序的反事实分解**；若来源有交互，残差/交互量须单列，不得把所有下降机械地分成互不重叠的三类。

### 2.4 主要图表和产出（建议新增）

- `exp1/paired_one_step.csv`：场景、时间、来源、采样种子、组合、覆盖/吞吐/事件指标。
- `exp1/motion_score_factorial.png`：四组单步回报/覆盖率配对对比。
- `exp1/fork_coverage_curve.png`：短程分叉覆盖曲线（若执行阶段 B）。
- `exp1/drop_attribution.csv` + `drop_case_studies.png`：逐次骤降原因及代表性案例。
- `exp1/summary.md`：结论要同时包含效应大小、区间、失配限制。

### 2.5 判读规则

| 观察结果 | 更支持的解释 | 下一步 |
|---|---|---|
| G/D 明显接近 G/G，而 D/G 接近 D/D | 运动是主要瓶颈 | 优先检查运动分布与运动头学习 |
| D/G 明显改善 D/D，而 G/D 显著弱于 G/G | 评分更可能是瓶颈 | 检查排序、分数尺度、Top-K 与评分头梯度 |
| G/D 和 D/G 都明显退化 | 联合耦合很强或混合分布外 | 查看单步 vs 多步差异，不得下单因素结论 |
| 单步差异很小，多步差异很大 | 长期部署/路径依赖与误差累积 | 优先实验 2、事件分解、Critic 的动作价值排序 |

**完成标准**：至少获得同状态、同外生随机流下的四组配对结果；检查 UE ID 与 Mask 完全一致；明确哪些结论仅适用于离线混合动作干预。

---

## 3. 实验 2：Diffusion 条件动作分布与策略随机性诊断

### 3.1 研究问题

Diffusion 在较好布局下是否仍随机、持续、大幅移动？是动作**方差**更大，还是动作**均值/方向**有偏？Top-K 服务评分是否在相同观察下频繁改变排序？

### 3.2 数据利用策略

现有报告声称已保存 **64 次条件动作采样统计**及 `best/last` 动作方差图。先检查这些文件包含的是逐动作样本还是仅汇总统计：

- **有逐样本**：优先纯离线复算；
- **只有均值/方差**：可以做描述性统计，但若需要方向、相关性、Top-K 一致性，则使用冻结 `best/last` 权重新增条件采样，不做训练。

采样对象应包括 Diffusion `step0/best/last` 和 Gaussian `best/last`；对同一个观察重复 `K=64` 次，用独立动作 RNG，固定环境状态，不执行 `env.step()`。

### 3.3 测量指标

**运动输出分布**：

- 指令位移 `Δp_cmd` 的均值、标准差、协方差特征值及运动方向圆形统计；
- 实际执行前的缩放位移 `Δp_exec=α(h)Δp_cmd`，以及若执行单步后 `Δp_real` 的差异；
- 悬停概率：`P(||Δp_exec|| < 1m)`，同时记录**同阈值的实际位移**以和旧报告一致；
- 条件分桶：`hop=1 / 2–3 / 4+ / disconnected`、回合阶段、当前覆盖水平、用户候选数；
- 高覆盖状态下的运动方向稳定性：同一观察反复采样是否多方向摆动。

**评分与选择分布**：

- 对每个 UE 评分 latent 和 sigmoid 分数的均值、方差；
- 同一观察下 Top-K 提议集合的 Jaccard 相似度，以及跨样本 UE 第一选择的一致率；
- 有效候选数 `|F_i|` 分桶、容量饱和（`|F_i|>10`）分桶；
- 空候选 UAV 只统计运动，评分统计按 Mask 排除，不以填零作为真实评分。

**与部署效果的关联**：

- 在相同状态上，从每组策略抽样动作后运行一小步，衡量 `movement_variance` / `ranking_instability` 与即刻覆盖变化的关系；
- 根据实验 1 中典型的覆盖骤降前状态，专门检验是否有高随机性或方向选择失误。

### 3.4 额外对照：降低推理噪声（仅作为诊断）

如当前实现支持固定噪声流或控制各去噪转移方差，可设计小幅降噪推理，对照原始随机采样；同时允许对 Gaussian 以相近方式调整采样方差作为参照。

**警告**：改变扩散采样方差会改变策略分布；对未重新训练的模型，这不是原始 on-policy 性能的公平比较。不能因为降噪后一次验证变好，就推断训练中的“随机性是唯一原因”。**不得直接假定 deterministic 模式具备合法的策略 log-prob**。

### 3.5 输出（建议新增）

- `exp2/conditional_samples.parquet`（或压缩 NPZ，仅在样本量可控时存完整动作）。
- `exp2/motion_magnitude_by_hop.png`、`motion_covariance.png`。
- `exp2/topk_consistency.png`、`speed_vs_coverage_state.png`。
- `exp2/best_last_distribution_shift.md`。

### 3.6 判读规则

- **方差升高且覆盖下降**：支持探索噪声/动作随机性问题；进一步核对去噪末步方差和 PPO 更新。
- **方差未升高，但均值运动幅度或方向改变**：更可能是策略均值漂移或长期价值学习不足。
- **运动稳定，但 Top-K 排序不稳定**：优先检查评分头、候选 Mask、联合高维动作的更新权重。
- **Best/Last 分布类似但价值差异显著**：转向状态分布差异与 Critic 对“停留 vs 运动”的估值。

**完成标准**：在固定观察和 Hop 分桶下给出运动方差、均值、悬停比例、关联排序一致性，并明确区分**命令运动、缩放运动、真实运动**。

---

## 4. 实验 3：DPPO 逐去噪步 PPO 更新机制诊断

### 4.1 研究问题

Diffusion 的 8 步随机去噪链是否存在：某些去噪步骤更新过强、Clip 过频、动作维度放大 log-prob、Actor 梯度被评分头主导，或者 Actor 优化几乎停滞？

### 4.2 代码与假设

已核对代码入口：`src/uav_research/diffusion_dppo/learner.py` 的更新遵循**per-agent、per-denoising-transition clipped PPO** 逻辑，`src/uav_research/diffusion_dppo/policy.py` 生成带有效 UE Mask 的去噪动作。对照 Gaussian 单步联合动作 log-prob 时，不能把单步 KL 与 Diffusion 每个去噪步的 KL 直接当成同一个量。

- Gaussian：对一次有效联合动作计算 log-prob 和一次 PPO ratio；
- Diffusion：对每个去噪转移计算 log-prob 和 ratio，并聚合去噪步骤的策略损失。

此处先做**日志与梯度诊断**，不预设逐步 PPO 就是错误。

### 4.3 离线提取现有日志

先从快照目录中定位 `metrics.csv`（**具体嵌套路径以仓库快照清单为准**），针对训练步数分段：

- `0–1M`：早期（包括 Gaussian Best 附近）；
- `1–5M`：中期；
- `5–7.5M`：Diffusion Best 之前；
- `7.5–10M`：Diffusion Best 之后。

优先检查**存在的字段**（实际字段名以 CSV 表头为准，若缺失则需要补日志）：

| 类别 | 建议提取字段/指标 | 重点判定 |
|---|---|---|
| 更新大小 | `ppo_approx_kl`、`ppo_clip_fraction` | 是否突然升高或长期几乎为0 |
| 分步更新 | `denoise_1_kl`～`denoise_8_kl`、对应 clip | 哪些去噪步骤贡献最大 |
| 梯度 | `actor_grad_norm`、运动头/评分头梯度 | 梯度失衡或爆炸/消失 |
| 学习信号 | Advantage 均值/方差、GAE、value loss | 优势近零或异常偏置 |
| 动作复杂度 | 有效评分维数、候选超过容量的比例 | log-prob 求和是否随维数增大 |
| 训练过程 | Actor loss、Critic loss、学习率、optimizer step | Best → Last 是否伴随更新状态变化 |

**注意**：日志若只记录策略当前旧参数下的近似 KL，应明确这一点。分步 KL 的求和/均值并不自动等于最终运动/关联动作分布的 KL；可补充相同观察下的动作行为变化作为外部验证。

### 4.4 建议增加的细粒度检查（如现有日志不足）

1. 对每个 denoise step ℓ（按代码正反步序明确编号），统计 `logp_new-logp_old` 的分位数（5/50/95%），ratio clipping 概率和高斯均值漂移；
2. 按 `|F_i|` 分桶，比较 **运动2维 log-prob** 与 **评分 `|F_i|` 维 log-prob** 的绝对量级、方差及相应梯度贡献；
3. 记录 mini-batch 内是否出现 NaN/Inf，Mask 空集是否导致非法统计；
4. 逐步比较 Actor 参数更新后，在固定观察集合上的最终动作均值、方差和 Top-K 一致率；
5. 如可行，记录不启用 clip 的 ratio 分布作为**离线诊断**，不要在此阶段直接改变训练公式。

### 4.5 分析方法

- 绘制每个 denoise step 的 KL 与 clip 随在线训练步数变化的热图。
- 绘制 Actor/运动头/评分头的梯度范数时间曲线，叠加 Best 的时间位置。
- 绘制按有效评分维数分桶的 log-prob 数值尺度，以及近似 KL/clip 关联。
- 结合实际验证曲线确定是**性能回退时更新变强**，还是**始终难以有效改进**。

### 4.6 输出（建议新增）

- `exp3/metrics_field_audit.md`：可用/缺失字段清单与指标含义。
- `exp3/ppo_stepwise_kl_clip.png`。
- `exp3/motion_vs_score_gradient.png`。
- `exp3/logprob_by_candidate_count.png`。
- `exp3/diagnosis.md`：明确该证据能否支持“PPO 更新过强”。

### 4.7 判读规则

| 观察 | 可能性 | 是否足以直接改超参？ |
|---|---|---|
| 某些 denoise step KL/clip 持续偏高 | 去噪步局部更新过强 | 可以做实验4的小预算对照 |
| 所有步 KL 很小、clip 很低，覆盖提升慢 | 更新过弱或策略/优势信号无效 | **不应盲目再降低学习率** |
| 评分梯度远高于运动梯度 | 联合动作尺度/任务竞争问题 | 需要检查损失权重、有效评分数 |
| Critic loss 稳定但动作排序差 | Critic 均方误差不足以衡量决策质量 | 增加固定状态下 `stay vs move` 的回报排序评估 |

**完成标准**：给出「是否存在异常更新」「异常是否集中于某些 denoise step」「是否随评分维数变化」的有据结论。若无异常，实验4应改成针对发现的瓶颈，而不是机械微调。

---

## 5. 实验 4：从 Diffusion Best 出发做保守 DPPO 微调

> **启动门槛**：实验 1–3 至少提供一条证据支持“现有 PPO 更新幅度可能过强或不稳定”。如果诊断显示根源是评分排序/策略结构，优先修正根源，本实验可暂缓。

### 5.1 目标

验证更保守的策略更新能否 **(a) 减少 Best → Last 回退，(b) 提升长期覆盖稳定性，(c) 在受控成本内继续改善平均覆盖/吞吐**。

### 5.2 起始 checkpoint 选择

首选 Diffusion 在约 `7,501,824` 联合转移时的 Best 权重。开始前检查是否拥有完整可恢复训练状态，包括：

- Actor、Critic、两类优化器、Value Normalizer、Scheduler（如存在）；
- 全部随机状态（Python/NumPy/PyTorch/环境/向量采样）、当前 step 和 PPO 更新计数；
- 与新方案完全一致的观察版本、Hop 缩放配置、UE ID 映射与 Diffusion 调度。

若 `best` 只有冻结推理权重、没有完整训练状态：

1. **优先选择有完整状态且靠近 Best 的最近 checkpoint**，并标记其实际步数；或者
2. 使用 Best 的 Actor 权重进行**新启动微调**，重新初始化缺失的优化器/状态，同时控制组也必须使用**同一权重、同一新初始化策略**；不要称为严格无缝续训。

训练起点的 Critic 不可悄悄混用 Last 或随机权重；必须明确记录和两组对齐。

### 5.3 保守化候选配置

原始已记录配置（以 checkpoint/配置副本核对为准）与建议筛查方案：

| 参数 | 原始（约定） | 保守候选 |
|---|---:|---:|
| Actor learning rate | `1e-4` | `3e-5` |
| PPO epochs | `10` | `4` |
| PPO clip | `0.10` | `0.05` |
| Critic learning rate | `5e-4` | 初步不变 |
| Diffusion denoise steps | `8` | 不变 |
| Hop 缩放 & 真实位移历史 | 开启 | 不变 |
| 网络结构、奖励、Top-K、训练数据格式 | 当前版本 | 不变 |
| 额外交互预算 | — | 首轮 `0.2–0.5M` 筛查，确认后 `1–2M` |

**先进行 A/B 筛查**：

- **A 组（控制）**：起点相同，继续用原 LR / epochs / clip；
- **B 组（保守组合）**：同时使用较低 LR、epochs 和 clip；
- **可选 C1/C2/C3 消融**：在 B 有效果后，分别仅改 LR / epochs / clip，识别有效因素。B 只是组合干预，不能据此声称某个参数独立起作用。

### 5.4 公平性要求

- 从同一初始 checkpoint 派生两组，冻结相同评估种子、环境参数、episode 长度、吞吐归一化、缩放函数。
- 可使用相同**预先定义的外生环境随机流**，但策略动作随机数不能靠消耗次数一致来冒充配对。
- 先一个训练种子筛查是否值得继续；有改善后至少增加其他独立训练种子，再做结论。
- 在新旧方案中统一评估频率（例如每 `50k–100k` 联合转移）及评估场景。**选择 Best 用验证集，最终结论用未见测试集**。
- 同时记录实际 wall-clock、在线交互数、优化器步数；减少 epoch 也减少参数更新次数，必须在报告中明确，以免把“更少更新”与“更小更新”混为一谈。

### 5.5 记录、早停与安全回退

- 每次验证：平均覆盖、后100步覆盖、5%分位覆盖、吞吐、最大单步下降、≥10pp下降事件、联网率、动作实际速度、Top-K 接受率；
- 每次优化：分步 KL/clip、Actor 梯度、Critic 的累计价值 RMSE 与解释方差、有效评分维数；
- 出现 NaN/Inf、持续高 clip 或覆盖显著连续退化时可以暂停并保留 checkpoint，**不覆盖现有 Best**；停止条件及选择规则事先记录。

### 5.6 成功/失败标准

**初步成功**：在同预算 A/B 对照下，B 相比 A 的平均覆盖或后100步覆盖有改善趋势，同时高跌幅事件不显著恶化；区间/多种子检验随后确认。  
**更强成功**：B 在多个训练种子、未见测试环境中，显著提升长期覆盖与稳定性，并且不会以更大的训练总成本或吞吐损失换取有限收益。  
**失败或结论不清**：只在单个评估种子、只在 Best checkpoint 或只在某个场景改善；此时不能声称稳定化有效。

### 5.7 输出（建议新增）

- `exp4/configs/`：控制与保守组合配置、恢复元数据。
- `exp4/checkpoint_origin.json`：起点哈希、实际步数与优化器恢复策略。
- `exp4/learning_curve.png`、`stable_performance.png`、`gradient_kl.png`。
- `exp4/paired_summary.csv`、`exp4/summary.md`。

---

## 6. 实施接口、数据字段与最小伪代码

以下是**实现指导，不是声称仓库已有的可执行 API**。函数名 `restore_full_state`、`sample_joint_action`、`step_with_fixed_exogenous_rng` 都需要根据真实代码补齐。

### 6.1 实验 1 单步反事实伪代码

```python
for snapshot in selected_states:
    for repeat in range(num_repeats):
        # 同一物理状态，候选集合由真实环境状态重建
        o = make_local_observations(snapshot)
        a_g = gaussian.sample_joint_action(o, seed=seed_for("G", snapshot, repeat))
        a_d = diffusion.sample_joint_action(o, seed=seed_for("D", snapshot, repeat))

        for name, motion, scores in [
            ("G/G", a_g.motion, a_g.scores),
            ("D/D", a_d.motion, a_d.scores),
            ("G/D", a_g.motion, a_d.scores),
            ("D/G", a_d.motion, a_g.scores),
        ]:
            env = restore_full_state(snapshot)
            # 将 score 与候选真实UE ID对齐；环境内部执行Hop缩放、反射和Top-K
            result = env.step_with_fixed_exogenous_rng(
                motion=motion, scores_by_ue_id=scores,
                ue_noise=ue_noise_for(snapshot, repeat)
            )
            save_one_step_result(snapshot, repeat, name, result)
```

### 6.2 实验 3 日志审计建议

可先用以下**通用**命令在仓库中定位现有 CSV 表头，实际 CSV 位置需经快照清单确认：

```bash
find analysis/hop_scaling_joint_service_10m_seed0 -type f \
  \( -name 'metrics.csv' -o -name '*metrics*.csv' \) -print

# 替换为找到的真实文件路径
head -n 1 /path/to/metrics.csv | tr ',' '\n' | grep -E \
  'kl|clip|grad|actor|critic|denoise|advantage|candidate'
```

对真实日志文件完成字段确认后再写解析脚本，不能假设所有推荐字段都已记录。

### 6.3 输出目录建议（全为拟新增）

```text
analysis/diffusion_gaussian_diagnosis_20261110/
├── README.md
├── manifest.json
├── exp1_motion_vs_scoring/
│   ├── paired_one_step.csv
│   ├── fork_coverage_curve.png
│   ├── drop_attribution.csv
│   └── summary.md
├── exp2_action_distribution/
│   ├── conditional_samples.parquet
│   ├── motion_magnitude_by_hop.png
│   ├── topk_consistency.png
│   └── summary.md
├── exp3_dppo_update/
│   ├── metrics_field_audit.md
│   ├── ppo_stepwise_kl_clip.png
│   ├── motion_vs_score_gradient.png
│   └── summary.md
└── exp4_conservative_finetuning/
    ├── configs/
    ├── checkpoint_origin.json
    ├── learning_curve.png
    ├── paired_summary.csv
    └── summary.md
```

---

## 7. 结果整合：一张决策表决定下一轮改什么

| 最终观察 | 优先修改方向 | 不推荐的盲目行动 |
|---|---|---|
| Diffusion 运动差、评分相近 | 运动头监督/运动动作分布/时间价值学习 | 无依据增大 UE 评分网络 |
| Diffusion 评分差、运动相近 | 评分头、Mask/UE ID、动作维数归一与排序学习 | 只调飞行速度缩放 |
| 高覆盖状态下 Diffusion 条件运动方差过大 | 探索噪声、去噪方差、策略稳定化 | 直接增加 10M 训练 |
| 分步 KL/clip 过高且性能回退相关 | 保守 PPO + 分步更新控制 | 把所有问题归因于 Critic |
| KL/clip 正常、方差也正常，仍表现弱 | 策略表达与训练目标不匹配、Critic 排序能力、场景动作复杂度 | 继续降低学习率 |
| 四项都没有定位 | 增加同起点动作回报排序实验、扩大状态覆盖、检查关联事件 | 根据单次观察给出确定结论 |

额外建议：如果现有事件分析发现大幅覆盖跌幅并非主要来自 UAV-BS 回传断联，而是来自 Top-K 提议变化或移动后有效性过滤，则论文和模型优先应关注**关联协议稳定性**；Hop 缩放无法直接解决 UE 服务评分跳变。

---

## 8. 最终验收清单

- [ ] 实验输入能够追溯到模型哈希、训练配置、代码版本和场景/动作种子。
- [ ] 四个组合的单步反事实来自同一完整环境快照，UE ID / Mask 正确对齐。
- [ ] 大幅覆盖骤降有回传变化、无线可行性变化、关联协议变化的分解分析。
- [ ] Diffusion 的运动均值/方差、近悬停与评分 Top-K 一致性在 Best/Last 上完成对照。
- [ ] 逐 denoise step 的 PPO KL / clip / 梯度随训练变化得到可复核的统计；缺字段已明确列出。
- [ ] 只有在前三个实验出现优化强度证据时，才启动实验 4。
- [ ] 实验 4 对照组和保守组的起点状态、总预算与评估协议一致；优化器恢复方式有明确记录。
- [ ] 验证集用于诊断/选模，不将其置信区间误称为训练种子不确定性；最终论文另做独立测试。
- [ ] 每个实验提交 `summary.md`，同时列出支持与反对的证据、尚未排除的解释以及下一步决定。

## 9. 来源与计划性质

**来源**：以上背景数值、现有评估协议、仓库路径和两组训练设置取自 2026-11-10 的现有报告及对照实验说明；其余分叉反事实、条件分布诊断、细粒度更新审计和保守微调为本文件提出的**新实验方案**。现有仓库原始数据是否包含每一种所需环境快照与明细列，必须在实际实现前核查，不应假设必然齐全。

- https://github.com/mixkle/uav_sac/blob/main/analysis/hop_scaling_joint_service_10m_seed0/report.md
- https://github.com/mixkle/uav_sac/blob/main/docs/hop_scaling_joint_service_comparison.md
- https://github.com/mixkle/uav_sac/blob/main/src/uav_research/diffusion_dppo/learner.py
- https://github.com/mixkle/uav_sac/blob/main/src/uav_research/diffusion_dppo/policy.py
