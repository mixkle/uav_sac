# Baseline 10M best、DPPO 10M best与新teacher验证

仅比较用户指定三组。红色：baseline 10M best；蓝色：DPPO 10M best；橙色：baseline 10M teacher。

32个验证场景10000–10031，400步真正终止，采样动作种子880001，批次16，CUDA FP32。正式保留测试集未使用。
场景、容量10、中心UE初始化、移动UE、观察历史和物理参数一致。奖励为覆盖率＋0.1×归一化吞吐，归一化分母750 Mbps；Mbps为全系统总吞吐。

## 模型与teacher定义

baseline完成10M从零训练，best实际位于3,821,568步；DPPO完成10M在线训练，best实际位于9,550,656步。两组均按32场景全回合任务回报选模。
teacher组直接冻结上述baseline best的同一个运动actor，局部评分为sigmoid((移动前下行SINR−门槛)/10)，使用移动前Top-K、移动后门控；不补位、不二次推理。初始关联沿用baseline。它没有独立训练。
**本次新teacher用于后验协议比较；原DPPO示范仍来自旧1M baseline的671,744步模型，不是本次10M teacher。没有重新采集示范或训练DPPO。**

| 模型 | 全回合覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 |
|---|---:|---:|---:|---:|---:|
| Baseline 10M best | 61.476% | 68.841% | 67.146% | 516.189 | 273.4338 |
| DPPO 10M best | 63.894% | 66.338% | 66.771% | 584.213 | 286.7338 |
| Baseline 10M teacher | 57.970% | 65.839% | 65.021% | 509.934 | 259.0771 |

## 核心解读

同为10M训练预算，DPPO best相对baseline best：平均覆盖变化+2.418个百分点，吞吐变化+68.023Mbps（+13.18%），任务回报变化+13.3000。相比此前1M baseline，预算差异已缩小；本次仅报告新10M对照。
同一个baseline actor换成SINR评分Top-K闭环执行后：平均覆盖变化-3.506个百分点，吞吐变化-6.255Mbps，任务回报变化-14.3567。该差值反映协议与随之发生的闭环状态变化。
DPPO相对新teacher：平均覆盖变化+5.924个百分点，吞吐变化+74.279Mbps，任务回报变化+27.6567。它不能单独分离扩散表示、学习评分和运动策略的贡献。
**DPPO对baseline的全回合覆盖差值95%场景区间为[-0.029, +4.953]个百分点，包含零；目前不能据此声称覆盖提升具有稳健的统计证据。** 吞吐与回报的配对区间见下表；best在同一验证集上选择，区间还有选模偏差。
**后期覆盖没有超过baseline。** 后100步分别为baseline 68.841%、DPPO 66.338%，终态分别为67.146%、66.771%。DPPO较高的全程覆盖伴随更快的早期展开，不能据此认定部署维持已优于baseline。
按全部服务用户时隙汇总，每个已服务UE的平均数据率为baseline 5.598Mbps、DPPO 6.096Mbps、teacher 5.864Mbps。该比值为总吞吐之和除以服务UE时隙之和；说明数据率差异不只来自服务人数，但不能直接归因于学习评分。

## 配对场景差值

差值为左减右；2000次配对场景bootstrap、种子950001。区间仅描述这32个验证场景，不是训练种子置信区间；同一验证集选best存在选择偏差。

| 比较 | 指标 | 差值 | 95%场景区间 |
|---|---|---:|---:|
| DPPO - Baseline | coverage | +2.418 个百分点 | [-0.029, +4.953] |
| DPPO - Baseline | throughput_bps | +68.023 Mbps | [+39.793, +96.631] |
| DPPO - Baseline | return | +13.300 | [+2.083, +24.740] |
| DPPO - Baseline | last100_coverage | -2.503 个百分点 | [-6.342, +1.465] |
| DPPO - Baseline | maintain_coverage | -10.748 个百分点 | [-15.508, -6.035] |
| Teacher - Baseline | coverage | -3.506 个百分点 | [-5.007, -2.027] |
| Teacher - Baseline | throughput_bps | -6.255 Mbps | [-25.115, +10.876] |
| Teacher - Baseline | return | -14.357 | [-21.240, -7.590] |
| Teacher - Baseline | last100_coverage | -3.002 个百分点 | [-5.515, -0.386] |
| Teacher - Baseline | maintain_coverage | +0.622 个百分点 | [-2.202, +3.570] |
| DPPO - Teacher | coverage | +5.924 个百分点 | [+3.719, +8.154] |
| DPPO - Teacher | throughput_bps | +74.279 Mbps | [+51.168, +97.783] |
| DPPO - Teacher | return | +27.657 | [+17.728, +37.805] |
| DPPO - Teacher | last100_coverage | +0.499 个百分点 | [-2.637, +3.730] |
| DPPO - Teacher | maintain_coverage | -11.371 个百分点 | [-16.014, -6.811] |

## 部署维持与服务瓶颈

| 模型 | 第51–150步覆盖 | 后期−前期覆盖 | 覆盖曲线峰值时刻 | 断联比例 | 边界反射 |
|---|---:|---:|---:|---:|---:|
| Baseline 10M best | 57.750% | +11.092个百分点 | 228 | 9.316% | 0.129% |
| DPPO 10M best | 65.995% | +0.344个百分点 | 258 | 8.168% | 3.286% |
| Baseline 10M teacher | 54.125% | +11.714个百分点 | 320 | 10.003% | 0.117% |

| 模型 | 未感知UE | 下行不足UE | 回传不足UE | 容量/协议未服务UE |
|---|---:|---:|---:|---:|
| Baseline 10M best | 20.15 | 18.19 | 12.36 | 7.09 |
| DPPO 10M best | 19.43 | 16.36 | 6.82 | 11.55 |
| Baseline 10M teacher | 20.79 | 18.68 | 13.12 | 10.45 |

最后一列不能全部解释为容量浪费；Top-K组还包含重复提议、移动后失效和新候选等待造成的缺口。teacher换协议会改变后续本机服务标记与动作，因此是闭环协议对照，不是固定运动轨迹下的纯关联求解比较。

## 覆盖骤降与固定场景

相邻覆盖变化从动作后状态t1–t400计算，不含初始t0→t1；大幅下降阈值为10个百分点。

| 模型 | 每回合最大单步跌幅均值 | 最大跌幅95分位 | 每回合大幅下降次数 | 终态低于30%场景数 |
|---|---:|---:|---:|---:|
| Baseline 10M best | 32.58个百分点 | 58.10个百分点 | 12.69 | 1/32 |
| DPPO 10M best | 40.04个百分点 | 69.50个百分点 | 11.41 | 0/32 |
| Baseline 10M teacher | 33.71个百分点 | 56.47个百分点 | 7.94 | 0/32 |

DPPO的平均断联比例较低，但每回合最大单步覆盖损失仍很大；平均联网率不能替代严重服务崩落的评估。上述统计未将事件全部归因为回传失效，还可能包含下行、容量及提议时序影响。

场景10000的终态：
- Baseline 10M best：覆盖64.00%，联网18/18，感知131人，可服务候选106人，提议不适用条。
- DPPO 10M best：覆盖63.33%，联网18/18，感知117人，可服务候选108人，提议98条。
- Baseline 10M teacher：覆盖52.00%，联网16/18，感知135人，可服务候选80人，提议100条。

场景10001的终态：
- Baseline 10M best：覆盖68.00%，联网18/18，感知117人，可服务候选105人，提议不适用条。
- DPPO 10M best：覆盖74.67%，联网17/18，感知133人，可服务候选124人，提议116条。
- Baseline 10M teacher：覆盖67.33%，联网18/18，感知138人，可服务候选111人，提议113条。

## Critic与训练成本

仅对原关联baseline和DPPO自己的on-policy轨迹检查MC；teacher切换协议后原critic不是该teacher的价值估计，因此不将其误差作为teacher预测准确性。

| 模型 | 累计价值RMSE | 标准差归一化RMSE |
|---|---:|---:|
| Baseline 10M best | 9.2758 | 0.2659 |
| DPPO 10M best | 7.4178 | 0.2038 |

Baseline 10M best在线训练总墙钟14.87小时，其中验证7.06小时。

DPPO 10M best在线训练总墙钟33.97小时，其中验证15.22小时。

## 归因边界

在线预算已统一为10M；DPPO仍多出示范和预训练，actor容量、观察无线特征、学习率、PPO阈值、熵机制及关联协议不同。该比较支持系统级性能判断，不能把全部差值单独归因于扩散模型。
teacher与baseline同一actor但关联协议不同；DPPO与teacher的差异同时包含运动策略、学习评分与训练流程。只有一个训练种子；本报告不增加训练、调参或测试集评估。

图表均提供PDF/SVG/PNG；重建只读取已保存数据，无需重新采样。

- [任务曲线](figures/task_curves.png)
- [性能汇总](figures/performance.png)
- [奖励分量](figures/reward_components.png)
- [部署与断联](figures/deployment.png)
- [骤降与尾部稳定性](figures/stability.png)
- [服务瓶颈](figures/service_bottlenecks.png)
- [协议诊断](figures/protocol.png)
- [训练验证曲线](figures/validation_learning.png)
- [Loss及策略诊断](figures/loss_and_policy_diagnostics.png)
- [Critic与MC](figures/critic_mc.png)
- [场景10000拓扑](figures/topology_best_10000.png)
- [场景10001拓扑](figures/topology_best_10001.png)
