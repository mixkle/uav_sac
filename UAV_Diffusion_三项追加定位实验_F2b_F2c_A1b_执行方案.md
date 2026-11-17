# UAV Diffusion Actor：F2b、F2c 与 A1b 三项追加定位实验执行方案

> **版本**：2026-11-13 · 研究用途 / 实现指导  
> **项目**：[mixkle/uav_sac](https://github.com/mixkle/uav_sac)  
> **依据**：已完成的 1M 系统消融 A–F 的 `analysis.md`、`comparison_analysis.md`、`report(3).md` 与 `unified_best_last.csv`；旧版 10M Diffusion/Gaussian 诊断结果。  
> **性质**：**诊断性、受控消融**；不以单种子 1M 结果直接声称算法通用优势。

---

## 0. 为什么只做这三项？

已有 1M 消融中，Diffusion 从零训练 A1 几乎没有收益，使用相同教师示范预训练的 A3 则能达到很好的覆盖；与此同时，运动专用 Diffusion F2a 明显优于联合 Diffusion A3。我们需要把“执行评分的影响”“去噪网络的结构耦合”“随机初始化动作尺度”拆开，而不是又同时更换服务协议、动作结构与优化参数。

### 0.1 已有基准结果（1M Last，32 验证场景 × 3 动作采样种子）

| 实验 | 定义 | 后100步覆盖率 | 后100步吞吐 | 后100步时间 σ | 后100步 ≥10pp 跌落/回合 |
|---|---|---:|---:|---:|---:|
| A1 | 随机 Diffusion；联合运动+评分；无 BC | 28.93% | 216.34 Mbps | 3.27 pp | 0.83 |
| G0 | Gaussian；联合运动+评分；指数 Hop Scaling | 72.47% | 702.30 Mbps | 5.14 pp | 2.16 |
| A3 | Diffusion；联合运动+评分；BC 后 1M 在线微调 | 72.10% | 736.60 Mbps | 3.25 pp | 1.03 |
| A5 | Gaussian；同一示范 BC 后 1M 在线微调 | 74.02% | 744.04 Mbps | 4.21 pp | 1.99 |
| F2a | Diffusion 仅运动；SINR 关联；运动 BC 后 1M 在线微调 | 78.17% | 815.99 Mbps | 2.85 pp | 0.72 |

A3 的 Step 0/Best/Last 后100步覆盖分别为 **73.02% / 75.30% / 72.10%**；F2a 为 **75.57% / 77.69% / 78.17%**。两者的初始能力不同，最后 6.06pp 的差距不能全算成优化效果。A1 的随机初始化运动 latent 均值模长约 **204.77**，物理指令几乎最大幅度，评分饱和比例约 **95.05%**；这不是一个正常的 Gaussian vs Diffusion 从零样本效率对照。

### 0.2 三项实验的因果定位

| ID | 只改变什么？ | 优先回答的问题 | 主要比较对象 |
|---|---|---|---|
| **F2b** | **只在执行层**把 learned UE scores 换成 local SINR scores；仍采样全部联合去噪变量并照旧算联合 PPO | 差距来自**学习评分实际执行/服务切换**，还是来自**评分分支的训练结构**？ | A3；辅看 F2a |
| **F2c** | 保留联合评分及执行；**只切断 noisy-score → motion denoising head** 的信息路径 | 评分噪声对运动去噪的直接条件耦合是否有害？ | A3；辅看 F2b |
| **A1b** | 不用 BC；采用**尺度稳定、概率一致的 Diffusion 初始化** | A1 从零学习失败，是否主要因为反向链放大导致执行动作饱和？ | A1、G0 |

重要：**F2b ≠ F2a；F2c ≠ F2a；A1b ≠ BC 预训练。** 必须保留各项真正要隔离的对照边界。

---

## 1. 通用训练、验证及复现协议

### 1.1 必须沿用的环境与算法约定

- 场景：18 UAV、150 UE、3 个 gateway BS，UAV 容量10，400步/回合；奖励仍为 `coverage + 0.1 × normalized_throughput`。
- 所有追加实验沿用最新局部观察（包括共享 UE Set Encoder、UE ID/Mask、邻居状态、**缩放+反射后真实位移历史**）。
- 运动执行使用当前指数型 Hop Scaling（默认 `alpha_min=0.25`、`k_h=0.4`），不额外加入回传安全约束、服务切换迟滞、奖励惩罚等功能。
- UE 服务协议沿用当前 Top-K 提议、移动后拓扑检查、UE 本地选择、容量限制及冲突解决时序；**F2b 才改变送入这个协议的评分来源**。
- 扩散反向步数、噪声日程、后验方差/标准差下限、Masked log-prob、DPPO-MAPPO、Critic、PPO/GAE、Rollout 和优化器设置，原则上完全复制 A3（A1b 复制 A1）；**不要趁机同时调 LR、Epoch、Clip 或熵系数**。
- 冻结的模型、已有 A3 与 A1 原始训练数据和代码不得覆盖修改。所有新运行使用新 ID/新目录。

> **仓库接口说明**：下文 `scoring_source`、`disable_noisy_score_to_motion`、`diffusion_init_mode` 等是**建议的配置字段名/伪代码接口**，不是声称仓库现在已经具有这些字段；实现前应核对实际 YAML schema、Actor 和环境的接口。

### 1.2 统一随机性、预算和选模

| 项目 | 规则 |
|---|---|
| 在线预算 | 每项完整训练 **1,000,000 条联合环境 transition**；示范收集和 BC 不计入在线预算，必须另记成本 |
| 教师与 BC 数据 | F2b/F2c 使用 **与 A3 相同**的 G0 Best 教师、64K 示例及 UE ID/Mask；A1b 无 BC |
| 独立训练种子 | 第一阶段复用 A3/A1 对应训练种子用于筛查；关键结论随后至少 **3 个独立训练种子**确认 |
| 在线初始化 | F2b **从 A3 的 BC Step 0 Actor 完整复制**；F2c 从同样教师/数据重新执行结构受控的 BC；A1b 随机初始化 |
| Critic / 归一化 / Adam | 依照 A3/A1 的真实启动语义重置或加载，写入 manifest；F2b 与 A3 做到一致 |
| 验证间隔 | 与原 1M 消融相同；必要时目标约每 50K 验证，注意不改变总训练预算 |
| Best 选取 | 原规则：固定验证场景10000–10031、固定选模动作种子 `880001`、**后100步平均覆盖率最大**；精确数值平局才以后100步吞吐决定 |
| Last 选取 | 1M 最末 checkpoint，不做优化后挑选 |
| 冻结复评 | 对 **Step 0 / Best / Last** 用32个相同场景 × 3个动作采样种子 `880001/880002/880003` 评估，不用复评结果再选择 Best |
| 置信区间 | 场景配对 bootstrap 2,000次；**不是跨训练种子的置信区间**；最后独立测试集在算法定稿后使用 |

注意：三个实验的“Best”均由训练过程的**单一固定选模种子**决定，三种子复评的 Last 偶尔高于 Best 是正常的，不能事后重选。比较同在线预算的 Last 为第一主分析，Best 为学习与回退分析。

### 1.3 共同主指标

设第 `s` 个场景第 `t` 步执行后的有效联网覆盖为 `rho[s,t]`、吞吐为 `R[s,t]`：

1. `last100_coverage`：场景内 `t=301..400` 求均值，再在场景与动作重复上求均值，**越高越好**。
2. `last100_temporal_std`：**先每条轨迹的 t=301..400 计算时间总体标准差**，再平均，单位 pp；**越低越好但必须与覆盖均值联读**。
3. `final_coverage`：第400步执行后覆盖，再平均。
4. `last100_throughput_mbps`：最后100步总吞吐均值，再平均。
5. `final_throughput_mbps`：第400步总吞吐，再平均。
6. `episode_return`：原完整400步奖励求和，再平均。
7. `formation_step_90pct`：首次达到**该场景后100步均值的90%**的步数；需同时记录是否达标及固定绝对覆盖阈值（例如60%连续10步），避免低覆盖策略看似“形成快”。

诊断性次指标：后100步 P05 覆盖、≥10pp 单步跌落次数、最大跌幅、联网 UAV 比例、真实运动速度/悬停比例、UE发现数和可服务候选数、Direct Switches、Service Interruptions、Top-K 决策变化率、评分饱和比例、运动 latent 与物理动作分布、分步 KL/Clip/梯度。后两组分别在 A1b/F2b/F2c 重点增加。

### 1.4 数据和运行目录建议

```text
analysis/diffusion_followup_f2b_f2c_a1b_20261113/
├── README.md
├── manifest.json                  # commit、配置、数据、seed、checkpoint、hash
├── metrics/
│   ├── unified_step0_best_last.csv
│   ├── per_scene_action.csv
│   ├── paired_differences.csv
│   ├── online_gains.csv
│   └── internal_diagnostics.csv
├── F2b_score_execution/
│   ├── frozen_A3_replay/
│   ├── train_1m/
│   └── figures/
├── F2c_remove_score_motion_coupling/
│   ├── pretrain/
│   ├── train_1m/
│   └── figures/
├── A1b_stable_initialization/
│   ├── init_audit/
│   ├── train_1m/
│   └── figures/
└── report.md
```

保存每个模型 Step 0、Best、Last **权重和规范化统计量**。如果要精确续训，另保存完整 Adam、Critic、环境/RNG及训练计数状态；轻量 Best 权重不能声称可无缝恢复在线轨迹。

---

## 2. F2b：联合 Diffusion 保留，执行时改为 SINR 评分

### 2.1 假设与目标

- **H-score-exec**：A3 的 learned UE scores 导致了不理想的分配、服务中断或布局演化；只改变实际执行评分就能改善后期表现。
- **对立解释**：A3 的主要瓶颈不在评分执行，而在去噪评分随机维度、联合 PPO 或运动-评分耦合，即使执行换成 SINR 也很难改善。

核心原则：**同一个联合 Diffusion Actor、同一批有效评分 latent、同一份联合 log-prob、同样的训练 Loss，只替换进入关联协议的分数**。

### 2.2 最小代码改动

在“提交 Top-K 服务提议”的边界设置可配置评分来源，尽量调用已在 **D3** 中验证过的 SINR 评分函数；不得另写一套行为不一致的规则。

```python
# 仅为概念伪代码；根据仓库实际接口适配
motion_latent, learned_score_latents, denoise_chain = diffusion_actor.sample(local_obs)
learned_scores = latent_to_scores(learned_score_latents)

if cfg.scoring_source == "local_sinr":   # F2b
    executed_scores = local_sinr_score(local_obs, candidate_ids, valid_mask)
else:                                      # A3
    executed_scores = learned_scores

# 两组共用同一个 Top-K / 选择 / 容量 / 移动后过滤实现
next_state = env.step(motion_latent, executed_scores, candidate_ids, valid_mask)

# 重要：F2b 仍按 motion + learned_score_latents 的整条联合去噪轨迹算 DPPO
logprob = actor.joint_denoising_logprob(denoise_chain, valid_mask)
```

**必须遵守**：

- SINR 必须是当前 UAV **实际可用的本地测量值**，不能悄悄使用 UE/BS 的全局真值；采用 D3 已定义的 SINR 方向、归一化、非法候选与 tie-break 规则。
- Top-K 使用动作提出时的本地候选集合；移动后仍按原逻辑过滤，不在动作后“偷看”重新补满提议。
- 保留 score head、评分采样、反向链、评分 Mask、评分 log-prob、联合 PPO Clip/梯度，**即使评分对环境执行已无作用**；这是刻意的诊断组而非部署推荐。
- 不改奖励，不加入新的服务切换惩罚/迟滞；只把切换与中断列为评估指标。
- F2b 一定仍具备 A3 **完全相同的网络参数量、去噪坐标数及动作潜变量维度**。

### 2.3 两阶段执行

**阶段 F2b-0：零训练成本的冻结评估（优先执行）**

1. 读取 A3 的 `Step0 / Best / Last` 各 checkpoint，保持 Actor、Critic和归一化冻结。
2. 在相同场景和动作 RNG 下，分别运行 `scoring_source=learned` 与 `scoring_source=local_sinr`，对两组各做完整400步验证。保存逐步结果。
3. 对相同**完整环境快照**再做一步反事实试验：相同 UAV 运动 latent、UE随机流，仅替换关联分数，比较立即覆盖/吞吐、Top-K 差异、切换和中断。多步两条分支状态会分化，不应误认为其每个后续动作仍一致。
4. 建立 `frozen_delta` 表：A3 同权重、同状态/随机性、只更改关联评分的配对结果；**这个阶段用于判定即时执行评分的影响，不用于判定重新学习后的上限**。

**阶段 F2b-1：相同 BC 初始化后在线训练1M**

1. **直接复制 A3 BC 后 Step0 Actor**（不从 A3 Best/Last 继续训练）；恢复相同预训练归一化与去噪计划，记录源 SHA256。
2. 按 A3 新在线阶段的规则重新初始化 Critic/Adam/GAE 统计，保持配置与训练预算一致；训练和验证都使用 SINR 执行评分。
3. 保存 Step0/Best/Last；训练中收集完整去噪轨迹、motion/score 分支 log-prob、KL/Clip、梯度和 UE 服务事件。
4. A3 原始训练不变；用同训练种子和相同评价随机流做配对比较。**F2b 的 Step0 BC 权重应与 A3 完全相同，但环境规则不同，环境表现可以不同。**

### 2.4 特定诊断输出

- 同权重冻结的 `ΔLast100 Coverage`、`ΔThroughput`、`ΔTemporal σ`；逐步 Top-K 提议集合 Jaccard/变更率、因评分改变而变化的 UE 归属比例。
- `direct_switches_last100`、`service_interruptions_last100`：严格沿用已有 CSV 的事件定义，并写出单位（次/回合或次/UE-回合）。
- 关联层漏斗：本地发现 → 下行可行 → 回传可行 → 提议 → 被接受 → 最终覆盖；同一个可服务集合内对照评分规则。
- 去噪训练：score head 梯度与 conditional KL，特别检查虽不执行评分，评分分支仍被联合 PPO 优化的情况。

### 2.5 判读决策

| 观察 | 能支持什么？ | 下一步 |
|---|---|---|
| F2b 接近 F2a，明显优于 A3 | **评分执行/布局反馈**是主要候选因素 | 查关联竞争、切换、中断；考虑保留固定 SINR 或分离执行评分 |
| F2b 接近 A3，明显弱于 F2a | **评分去噪/联合 PPO/运动耦合**可能比执行评分更重要 | 优先看 F2c 和后续概率比隔离 |
| F2b 介于 A3 与 F2a 之间 | 执行与联合表示都可能贡献差距 | 冻结因果结果 + F2c 联合分析 |
| 冻结替换有效但在线1M无增益 | 策略随新协议重新访问状态，或优化没有有效利用新反馈 | 看 Step0→Last 的收益与轨迹分布 |
| 冻结替换无效但在线1M改善 | 主要是新执行协议改变了长期优化结果 | 检查学习曲线及多步空间布局 |

---

## 3. F2c：切断 noisy-score → motion 去噪的直接条件耦合

### 3.1 假设与目标

- **H-coupling**：运动去噪头依赖评分噪声变量/其聚合特征，导致对 UE 候选规模和无关评分随机性的敏感性，影响运动策略的学习。
- **对立解释**：即使运动头不读取 noisy-score，联合评分 log-prob、score head 和共享编码参数仍然会影响 PPO，性能差距来自其余部分。

**唯一指定的结构改动：保留 A3 的全部运动及评分 latent、score head、条件掩码、执行学到的评分与联合概率，只禁用由 noisy score 聚合进入运动 denoising head 的直接路径。**

### 3.2 受控结构实现

推荐保留网络输入维度和原层参数形状，以一个**固定配置 Gate** 令该分支输入为零，避免“移除一条连接的同时改变一大段网络容量”。

```python
# 概念伪代码；具体张量形状以仓库 policy.py 为准
z_motion_t, z_scores_t = noisy_latents_at_t
score_summary_t = masked_score_aggregate(z_scores_t, candidate_mask)

if cfg.disable_noisy_score_to_motion:  # F2c
    motion_score_condition = torch.zeros_like(score_summary_t)
else:                                  # A3
    motion_score_condition = score_summary_t

motion_eps = motion_head(concat(context, z_motion_t, motion_score_condition, t_embed))
score_eps = score_head( ... )         # 与 A3 完全相同；仍可条件于 z_motion_t
```

需要检查仓库实际实现是否还有**其他 noisy-score → motion 的直接或间接路径**（共享注意力、共享 MLP 的拼接等）。本实验只切断指定路径，固定特征中的 UE 聚合上下文继续保留。禁止错误地把所有 UE 信息从运动输入中删除。

特别注意：

- motion 去噪头仍接受本机状态、邻居与 UE set context、当前运动 noisy latent 和去噪时间嵌入。
- score head 仍输出全部候选 UE 的噪声预测，使用原 Mask；学习评分照常进入 Top-K。
- 保留每一步完整转移分布的 log-prob 和 PPO ratio；不要仅对 motion log-prob 训练。
- 固定 Gate 应同时作用于 **BC 预训练、在线采样、log-prob 重算、冻结验证及模型恢复**；不能训练时关、推理时开。
- 因为模型条件结构改变，不能只把 A3 预训练权重加载后关闭输入，就把它称为“同等初始化训练”；**正式实验必须在该结构下重新做 BC**。可以额外做冻结 A3 的即时 Gate 关断测试，但只用于机制诊断。

### 3.3 训练流程

1. 复用 **A3 的教师 G0 Best、64K 联合示范数据、采集环境与动作标签**；运动监督仍是**缩放前原始 latent**，不是缩放后的物理位移。
2. 从与 A3 对应的随机初始化与训练 seed 出发，在 `Gate=0` 结构下用**相同的50轮 BC 和损失权重**重新预训练。记录运动/评分去噪 loss 与实际执行 Step0 性能。
3. 以新的 BC Actor 初始化在线模型，Critic 与 Adam 的初始化语义同 A3；同一环境、同一奖励、同一 Hop、同一 Top-K，在线训练1M。
4. 和 A3 一样选择 Best、保存 Last，并对 Step0/Best/Last 做32场景×3采样种子冻结复评。
5. 额外对同一批验证观察进行“运动 latent 对评分噪声扰动的敏感性”审计：固定局部观察与运动噪声，扰动评分噪声，测量运动头预测变化（在 F2c 中指定 Gate 贡献应严格为0）。同时检查 score head、Mask 与概率密度仍正常。

### 3.4 关键比较与边界

| 比较 | 主要目标 | 必须同时报告 |
|---|---|---|
| F2c vs A3 | 评分到运动去噪的直接路径是否影响在线结果 | 两组 BC Step0、在线净增益 `Last−Step0`、Best、Last |
| F2c vs F2b | 去噪结构变化 vs 执行 SINR 评分替换的影响 | 两者执行的服务规则不同，**不是纯单因素比较** |
| F2c vs F2a | 保留联合去噪、执行 learned scores vs 彻底去掉 score 分支 | 结构和执行均有差异，适用于界定差距，不可唯一归因 |

若 F2c 改善，还需判断是 **BC Step0 更好**还是 **在线训练净增益更大**；两者不能混为一谈。若 F2c 没有改善，不代表其他类型的运动—评分耦合均无效，只能说明当前被切断的路径不是本设置下的显著性能瓶颈。

---

## 4. A1b：无 BC 的 Diffusion 初始反向链尺度校准

### 4.1 假设与目标

- **H-init**：A1 从零训练几乎无法学习，主要因为随机 epsilon 网络使8步余弦 DDPM 反向链在末端高噪声区出现 latent 放大，导致运动最大幅度/评分 sigmoid 饱和，削弱有效探索与可分辨的动作后果。
- **对立解释**：即使校正初始动作尺度，DPPO 的高维策略梯度估计或信号仍不足，1M内依然学不会。

A1b **不使用教师、示范数据或 BC**，不改变 UE 观察/Top-K/Hop/奖励/在线预算。唯一变更为一种**概率一致的网络初始化参数化**；禁止为了让结果好看而直接对采样 latent 进行训练前后不一致的裁剪/缩放。

### 4.2 推荐的单一初始化方案：标准正态参考残差（优先）

当前 DDPM 的噪声预测记作 `epsilon_theta(z_t, o, t)`。以标准正态 latent 作为初始化参考先验，将网络重写成：

\[
\widehat\epsilon_\theta(z_t,o,t)
 =\sqrt{1-\bar\alpha_t}\,z_t+\Delta_\theta(z_t,o,t).
\]

- `\bar alpha_t` 使用**同一实际训练噪声日程**；`z_t` 是当前被去噪的动作 latent（运动及被 Mask 的有效评分坐标）。
- `Delta_theta` 保持原有上下文、UE集合和去噪网络表达能力；最后输出投影层初始化为**全零权重/偏置**，仅使初始残差为0，后续正常可训练。
- 对无效评分槽位保持原 Mask 语义；未观察 UE 不允许进入 Loss、PPO log-prob 或物理执行。
- 这样初始化时 epsilon 预测会近似 `sqrt(1−bar alpha_t)·z_t`，对于**标准正态参考 latent**，对应更温和的反向均值 `sqrt(alpha_t)·z_t`，从而避免未训练 epsilon≈0 时的极端放大。
- **保留原 DDPM 反向方差/标准差下限、8个步骤和 PPO 密度公式**。由于原实现可能使用后验方差下限而不是精确逆向稳态方差，**最终输出分布只是预期接近合理尺度，不保证严格标准正态**；以初始化审计实测为准。

仅供实现理解的伪代码：

```python
# 需要核对 epsilon-prediction/DDPM 后验均值的当前实现，不能生搬硬套
class StableEpsilonPredictor(nn.Module):
    def __init__(self, original_delta_network):
        super().__init__()
        self.delta_network = original_delta_network
        zero_initialize_final_output_projection(self.delta_network)

    def forward(self, z_t, cond, t, valid_mask):
        alpha_bar_t = scheduler.alpha_bar_at(t)
        ref_noise = (1 - alpha_bar_t).sqrt() * z_t
        residual = self.delta_network(z_t, cond, t, valid_mask)
        out = ref_noise + residual
        return mask_invalid_score_dims(out, valid_mask)
```

此参数化不是“无成本修复”，而是**明确改变随机初始化时的反向策略先验**；优化时仍须对正确的随机转移均值及方差计算 log-prob。若当前策略输出不是 epsilon 预测而是 `x0`/`v`，不得使用此公式，应重新推导等价的稳定初始化。

### 4.3 训练前必须通过初始化审计（Gate）

在固定训练初始观察和另外一份固定未训练观察池上，对 A1/G0/A1b 随机初始化策略各采样 ≥64 次动作/观察（观察池建议≥128个独立状态，不把验证回报作为校准目标）。至少保存：

| 指标 | A1 已知异常 | A1b 审计目标 |
|---|---:|---|
| 运动 latent 模长均值/P50/P90/P99 | 均值约204.77 | 显著不再百量级；分布量级接近可用高斯策略，数值以实测和预设门槛为准 |
| 归一化圆盘指令模长均值与 P95 | 接近1.0 | 不再几乎全部贴近动作上限 |
| 评分 sigmoid 饱和比例 | 约95.05% | 大幅降低；分别报告小于0.01或大于0.99、以及0.05/0.95阈值下比例 |
| 关联提议有效率与动作变化 | 可能退化 | 不应由数值异常触发候选错位/非法操作 |
| 反向步骤的 `z_t` 均值/方差、每步运动/评分 KL 和 log-prob | 需要重算 | 所有步 finite，不溢出、不出现异常饱和 |

**审计达标前不要正式开跑1M**。不可根据验证集覆盖选多个初始化再挑最优；若调整门槛或残差设计，应标注为新候选并保留所有尝试记录。

### 4.4 必备数学/实现回归检查

1. 对任一固定去噪链，采样时记录的总 log-prob 与使用**相同参数**重算的结果在浮点误差内一致；同一个 Mask 下有效维的运动+评分 log-prob 求和严格一致。
2. 初始 PPO ratio 应近似1，更新前 clip fraction≈0；**不允许在物理动作映射后额外把 latent 数值裁剪，再假装概率密度未变**。
3. 初始执行动作 `finite`，满足运动幅度、容量、UE Mask、Top-K 等物理约束；同 RNG 多次运行结果可复现。
4. 标准正态参考残差必须对**所有去噪时间 t**保持一致；最后一层零初始化不能错误地把整个网络永久冻结。
5. 不改动 A1 的原有奖励、训练预算或学习率；否则无法判断究竟是初始分布改善还是更强的优化器带来结果。

### 4.5 正式训练及对照

1. 固定 A1 原种子及同样的 Gaussian G0 参照，采用新初始化 **无 BC** 在线训练1M。
2. 评估 Step0/Best/Last，记录在线增益与训练曲线，尤其是前0–200K的性能变化；动作尺度审计在 Step0、100K、Best、Last 重复。
3. A1b vs A1 直接比较是否从“运动饱和+几乎无在线收益”变成“有可观在线增长”；与 G0 比较从零学习样本效率时必须明确两者仍是不同策略族和初始化先验。
4. 若 A1b 已显著改善却尚未达到 G0，不要马上断言 DPPO 无效；在确认无饱和后，才按阶段诊断去噪步 KL/Clip、优势方差、motion/score 梯度信用分配。

---

## 5. 三组统一的测试清单与成功标准

### 5.1 实现前的回归测试（建议自动化）

| 检查 | F2b | F2c | A1b |
|---|:---:|:---:|:---:|
| 原 A3/A1 配置和 checkpoint 仍可重放，不发生兼容性破坏 | ✓ | ✓ | ✓ |
| Mask/UE ID 对齐、可服务集合及 Top-K 不因新增配置错位 | ✓ | ✓ | ✓ |
| 原 Actor/新 Actor 具有预期的动作维度和随机变量数目 | ✓ | ✓ | ✓ |
| 相同参数下 `sample logp` 与 `recompute logp` 一致 | ✓ | ✓ | ✓ |
| 在训练时与评估时使用**相同**语义和配置 | ✓ | ✓ | ✓ |
| 无 NaN/Inf、PPO ratio/Clip 回归正常 | ✓ | ✓ | ✓ |
| 训练预算、seed、模型保存及摘要字段完整 | ✓ | ✓ | ✓ |

专有测试：

- **F2b**：同一 A3 Actor、同一完整状态与随机流下，替换评分不能改变本步 Actor 的运动 latent/去噪链；能够改变服务提议及最终 UE 分配；两模式使用完全相同的服务后处理。
- **F2c**：运动头对 noisy-score 聚合输入的局部敏感度为零，但对固定 UE context 的输入仍可响应；score head 和全部评分密度仍参与策略更新。
- **A1b**：高噪声步骤在残差初始化下没有百量级异常放大；残差权重在优化一步后确实获得梯度与更新。

### 5.2 完成标准

- 每项明确存在 **Step 0 / Best / Last** 模型、配置、在线预算、原始逐场景×动作种子轨迹、训练诊断以及可重建报告。
- 对 F2b 冻结 A3 评分替换与其1M重训**分开报告**，不混写为“评分机制改进收益”。
- 对 F2c 报告 BC后的 **Step0差值**与 **在线净收益差值**，并写清由于结构改变不能做到完全相同起点。
- 对 A1b 先报告初始动作校准是否成功，再报告1M训练；若审计失败明确写“初始化方案未通过”，不能把它当作有效反证。
- 统一导出后100步覆盖/吞吐、时间σ、终态、回报、形成时间、P05、drop、联网、服务切换/中断、动作尺度与 DPPO 诊断。
- 主分析比较 Last，同步分析 Best 与 `Last−Step0`；对每个关键对照给出32场景配对区间，但不据此声称跨训练种子显著性。

---

## 6. 最终判读矩阵：什么结果指向哪种根因？

| F2b（执行换SINR） | F2c（切断评分→运动） | A1b（稳定初始化） | 优先解释 |
|---|---|---|---|
| 明显改善 | 无改善 | 任意 | **评分执行与环境反馈**可能是主要问题；先分析服务切换/中断 |
| 无改善 | 明显改善 | 任意 | **评分噪声对运动去噪的直接条件耦合**可能是重要因素 |
| 无改善 | 无改善，但 F2a 很好 | 任意 | 继续隔离**联合评分 log-prob/PPO 更新**及 Score Head 参数共享；必要时追加保留Score Head但分支独立的实验 |
| 二者都改善 | 二者都改善 | 任意 | 两种机制都有可能有害；若需拆交互，再做 F2b+F2c 组合（**非本轮必做**） |
| 任意 | 任意 | 从零训练明显改善 | A1 失败可部分归因于**初始化动作饱和**，再检验剩余样本效率差 |
| 任意 | 任意 | 动作尺度修复但训练无改善 | 初始化异常不是唯一瓶颈；转向 DPPO 优势/去噪信用分配及梯度分析 |

**重要证据边界**：F2b 与 A3 执行协议不同，F2c 与 A3 的 BC后起点可能不同，A1b 改变了扩散先验；每一个对照最多确认当前特定实现与1M预算下的作用，不能把结果推广为“所有 Diffusion 策略都如此”。

---

## 7. 建议执行顺序与工程交付

### 阶段一：最快验证，尽量不花 GPU 在线训练

1. 先完成 **F2b 的冻结 A3 关联替换评估**，做单步完整快照干预和全400步回合复评。
2. 同时完成 **A1b 初始化审计**，检查 DDPM概率一致性、动作 latent/评分饱和以及合法性。
3. 评估 **F2c 的结构 Gate 即时敏感性**，验证切断指定路径没有误删 UE context。

### 阶段二：三项正式1M训练

1. **F2b 1M**：完全复制 A3 BC Step0 Actor，再用 SINR 关联在线优化。
2. **F2c 50轮 BC＋1M**：复用同一教师与示范，Gate=0 全程一致。
3. **A1b 1M**：前置初始化审计通过后，从零无 BC 训练。

### 阶段三：统一报告和决策

1. 重建统一 A1/A3/F2a/G0/F2b/F2c/A1b 的 Step0/Best/Last 对照；必要时加入 A5。
2. 展示相同指标的 Last 主表、Best 主表、在线 `Last−Step0` 表、覆盖/吞吐随400步的曲线、训练过程性能曲线和内部动作/去噪诊断。
3. 对最有信息量的 **A3–F2b、A3–F2c、A1–A1b** 三个对照追加独立训练种子，最后使用未参与选模的保留测试场景报告性能。

### 7.1 建议实现任务拆分

```text
[ ] 冻结原 A3/A1/G0/F2a 基线结果和 SHA256
[ ] 复用 D3 的 local SINR 评分执行实现，新增 F2b switch
[ ] 编写 F2b frozen scoring-swap + single-step state replay
[ ] F2b Step0/Best/Last 1M 训练与评价
[ ] 新增 F2c noisy-score→motion gate；BC/PPO/logp/采样一致
[ ] F2c 使用原64K示范做50轮BC；训练1M
[ ] 新增 A1b 标准正态残差初始化；验证重算logp
[ ] A1b 动作初始尺度审计；满足后启动1M
[ ] 统一指标、配对差值、三个实验的解释边界及报告
[ ] 对关键结论补多训练种子/保留测试集
```

### 7.2 运行命令说明

具体训练入口、YAML 文件路径和 CLI 参数必须根据仓库当前 `scripts/`、`configs/` 与已有 A3/F2a 的 manifest 核对。本文件**不虚构未经核实的可执行命令**。建议先复制原始 A3 与 A1 的真实启动命令，仅增加前述字段；将最终完整命令、Git commit、Python/CUDA/PyTorch 版本、模型 SHA256 和关键环境变量写入各自 `manifest.json`，确保服务器与本地评估一致。

---

## 8. 实验执行后报告最少回答的六个问题

1. **F2b 冻结 A3**：仅替换评分执行，单步/长期覆盖、吞吐、切换和中断实际改善了吗？
2. **F2b 重训**：保持联合去噪 PPO 后，能否获得接近 F2a 的性能？净在线提升是多少？
3. **F2c**：移除 noisy-score 对运动的直接耦合后，BC起点和在线增益分别怎样改变？
4. **A1b**：随机 Diffusion 的 200 量级 latent 和评分饱和问题是否消失？从零训练是否真正开始学？
5. **机制归因**：哪个对照有最强的同条件、单变量证据？哪些现象仍可能来自初始化、优化器和状态分布差异？
6. **论文方向**：是否值得保留联合 Diffusion；还是运动 Diffusion + 简单关联更稳健且证据更强？

> **最终原则**：先证实机制，再谈性能提升；先对齐 Step0 与训练协议，再分析 Last；所有代码改动必须能回归到原 A3/A1 行为。
