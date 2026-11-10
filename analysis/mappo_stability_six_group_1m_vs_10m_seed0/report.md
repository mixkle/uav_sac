# 稳定性奖励三组与10M三组：六组best对照

本报告主图只比较六组best。新增三组各训练1M；原baseline、DPPO各训练10M，teacher冻结baseline 10M best，没有独立训练。新增三组last仅作补充表，不增加主图实验组。
统一32验证场景10000–10031、400步真正终止、采样种子880001、批次16、CUDA FP32。保留测试集未使用。新增best/last复验与各自训练日志一致。

训练目标：原三组为rho＋0.1tau；新增组分别加−0.1D、＋0.5J、＋0.5J−0.1D。主表共同回报始终按rho＋0.1tau计算，不能把含鲁棒奖励的较大训练回报直接视为更好的服务。
新增best按各自训练目标选择，原best按原任务回报选择；单训练种子、验证集选模。预算、选模目标及训练机制不同，六组比较不能单独识别奖励项的因果贡献。

## 主要任务表现

| 组别 | best实际步数 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 共同回报 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 10M best | 3,821,568 | 61.476% | 68.841% | 67.146% | 516.19 | 273.434 |
| DPPO 10M best | 9,550,656 | 63.894% | 66.338% | 66.771% | 584.21 | 286.734 |
| Baseline 10M teacher | 3,821,568 | 57.970% | 65.839% | 65.021% | 509.93 | 259.077 |
| 1M + disconnection | 620,544 | 62.284% | 65.338% | 64.833% | 524.11 | 277.086 |
| 1M + robustness | 610,304 | 64.096% | 68.562% | 67.792% | 542.63 | 285.323 |
| 1M + both | 831,488 | 62.030% | 66.714% | 67.458% | 523.14 | 276.021 |

## 主要发现

鲁棒性组best平均覆盖64.096%，与DPPO的63.894%接近；但吞吐分别为542.63与584.21Mbps。不能只用覆盖或加入J后的训练回报判断优劣。
新增三组best相对baseline 10M的最大单步跌幅均值变化分别为+3.08、+11.33、+17.10个百分点；当前六组比较中，没有看到新增项让该严重度指标低于baseline 10M。预算不同，不能直接解释为奖励项导致恶化。
断联惩罚组和鲁棒性组平均断联比例分别为8.03%、7.93%，低于baseline的9.32%；与此同时严重覆盖损失仍存在。平均联网与服务尾部稳定性不是同一个目标。
鲁棒性组绝对J为58.196%，baseline为56.443%；相应保留比例为90.676%与91.789%。绝对J提高没有表现为更高的相对服务保留。

## 覆盖骤降

从动作后t1–t400计算，不含reset→首动作。先逐场景计算最大跌幅，再跨场景平均；不是对平均覆盖曲线求跌幅。5步是端点差；阈值10个百分点另附5/20个百分点敏感性数据。

| 组别 | 最大单步跌幅均值 | 单步最大跌幅的场景95分位 | 最大5步跌幅均值 | 每回合≥10pp次数 | 最低20步覆盖均值 |
|---|---:|---:|---:|---:|---:|
| Baseline 10M best | 32.58pp | 58.10pp | 33.04pp | 12.69 | 27.49% |
| DPPO 10M best | 40.04pp | 69.50pp | 40.88pp | 11.41 | 33.41% |
| Baseline 10M teacher | 33.71pp | 56.47pp | 36.79pp | 7.94 | 24.98% |
| 1M + disconnection | 35.67pp | 67.80pp | 36.40pp | 14.91 | 30.53% |
| 1M + robustness | 43.92pp | 72.30pp | 44.62pp | 17.12 | 31.86% |
| 1M + both | 49.69pp | 70.67pp | 49.92pp | 17.28 | 29.55% |

## 对三个原参考的配对差值

差值为新增组减参考；骤降指标负值表示更少，但必须结合绝对覆盖和吞吐判断，防止“低覆盖伪稳定”。2000次场景配对bootstrap，种子950001；不是训练种子置信区间，未调整多重比较。

| 新增组 | 参考 | 指标 | 差值 | 95%场景区间 |
|---|---|---|---:|---:|
| 1M + disconnection | Baseline 10M best | coverage | +0.808 | [-1.562, +3.167] |
| 1M + disconnection | Baseline 10M best | throughput_bps | +7.916 | [-18.174, +34.158] |
| 1M + disconnection | Baseline 10M best | max_drop_1 | +3.083 | [-5.375, +11.271] |
| 1M + disconnection | Baseline 10M best | drop_events_10pp | +2.219 | [-2.250, +6.438] |
| 1M + disconnection | DPPO 10M best | coverage | -1.610 | [-4.595, +1.080] |
| 1M + disconnection | DPPO 10M best | throughput_bps | -60.107 | [-92.127, -29.086] |
| 1M + disconnection | DPPO 10M best | max_drop_1 | -4.375 | [-15.377, +5.834] |
| 1M + disconnection | DPPO 10M best | drop_events_10pp | +3.500 | [-0.312, +7.562] |
| 1M + disconnection | Baseline 10M teacher | coverage | +4.313 | [+1.911, +6.638] |
| 1M + disconnection | Baseline 10M teacher | throughput_bps | +14.171 | [-10.117, +39.866] |
| 1M + disconnection | Baseline 10M teacher | max_drop_1 | +1.958 | [-5.440, +9.042] |
| 1M + disconnection | Baseline 10M teacher | drop_events_10pp | +6.969 | [+2.655, +11.532] |
| 1M + robustness | Baseline 10M best | coverage | +2.620 | [-0.010, +5.329] |
| 1M + robustness | Baseline 10M best | throughput_bps | +26.441 | [-2.262, +54.074] |
| 1M + robustness | Baseline 10M best | max_drop_1 | +11.333 | [+2.166, +20.626] |
| 1M + robustness | Baseline 10M best | drop_events_10pp | +4.438 | [+0.499, +8.375] |
| 1M + robustness | DPPO 10M best | coverage | +0.202 | [-2.423, +2.685] |
| 1M + robustness | DPPO 10M best | throughput_bps | -41.582 | [-73.817, -11.309] |
| 1M + robustness | DPPO 10M best | max_drop_1 | +3.875 | [-5.400, +13.148] |
| 1M + robustness | DPPO 10M best | drop_events_10pp | +5.719 | [+1.310, +10.063] |
| 1M + robustness | Baseline 10M teacher | coverage | +6.126 | [+3.723, +8.729] |
| 1M + robustness | Baseline 10M teacher | throughput_bps | +32.696 | [+5.295, +62.254] |
| 1M + robustness | Baseline 10M teacher | max_drop_1 | +10.208 | [+3.374, +16.730] |
| 1M + robustness | Baseline 10M teacher | drop_events_10pp | +9.188 | [+5.968, +12.375] |
| 1M + both | Baseline 10M best | coverage | +0.554 | [-1.343, +2.477] |
| 1M + both | Baseline 10M best | throughput_bps | +6.948 | [-16.069, +29.714] |
| 1M + both | Baseline 10M best | max_drop_1 | +17.104 | [+9.270, +23.690] |
| 1M + both | Baseline 10M best | drop_events_10pp | +4.594 | [+1.094, +8.031] |
| 1M + both | DPPO 10M best | coverage | -1.864 | [-3.660, -0.140] |
| 1M + both | DPPO 10M best | throughput_bps | -61.076 | [-82.730, -38.473] |
| 1M + both | DPPO 10M best | max_drop_1 | +9.646 | [-0.960, +19.105] |
| 1M + both | DPPO 10M best | drop_events_10pp | +5.875 | [+2.781, +9.344] |
| 1M + both | Baseline 10M teacher | coverage | +4.060 | [+2.004, +6.013] |
| 1M + both | Baseline 10M teacher | throughput_bps | +13.203 | [-11.261, +36.103] |
| 1M + both | Baseline 10M teacher | max_drop_1 | +15.979 | [+9.935, +22.312] |
| 1M + both | Baseline 10M teacher | drop_events_10pp | +9.344 | [+5.967, +12.876] |

## 联网、鲁棒性与回合内保持

| 组别 | 断联UAV比例 | 故障保留覆盖J | 保留比例J/rho | 第51–150步覆盖 | 后期−前期覆盖 | 平均曲线峰值时刻 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 10M best | 9.316% | 56.443% | 91.789% | 57.750% | +11.092pp | 228 |
| DPPO 10M best | 8.168% | 58.195% | 91.047% | 65.995% | +0.344pp | 258 |
| Baseline 10M teacher | 10.003% | 53.302% | 91.900% | 54.125% | +11.714pp | 320 |
| 1M + disconnection | 8.034% | 56.964% | 91.409% | 63.771% | +1.566pp | 146 |
| 1M + robustness | 7.934% | 58.196% | 90.676% | 65.656% | +2.906pp | 327 |
| 1M + both | 8.774% | 56.211% | 90.590% | 61.734% | +4.980pp | 221 |

保留比例先按每场景sum(J)/sum(rho)计算，再平均。最低20步覆盖可能发生在初期展开，需与后期绝对覆盖一同判断。新增三组中，单步覆盖下降≥10个百分点的事件均同时出现净断联UAV比例增加；这是一致性证据，不代表所有服务损失都只能由回传解释。

断联惩罚考察当前联网比例；J是平均假想单UAV故障后的固定关联服务保留。两者不直接约束连续动作造成的多链路失效。鲁棒项同时包含覆盖偏好：rho＋lambda J＝(1＋lambda)rho−lambda(rho−J)，因此提高J不等于单独学到冗余。

## 新增组last补充

| 组别 | 平均覆盖 | 后100步覆盖 | 吞吐Mbps | 最大单步跌幅均值 | 每回合≥10pp次数 |
|---|---:|---:|---:|---:|---:|
| 1M + disconnection last | 59.554% | 62.763% | 500.78 | 35.67pp | 12.22 |
| 1M + robustness last | 58.188% | 60.065% | 482.28 | 48.00pp | 28.12 |
| 1M + both last | 60.924% | 66.255% | 512.38 | 43.15pp | 20.88 |

鲁棒性组last相对best平均覆盖变化-5.91个百分点，≥10pp骤降次数由17.12增至28.12次/回合。继续训练到预算终点没有保持住best表现；本轮不自动延长训练。

## 固定场景终态

| 组别 | 场景10000覆盖/联网 | 场景10001覆盖/联网 |
|---|---|---|
| Baseline 10M best | 64.00% / 18/18 | 68.00% / 18/18 |
| DPPO 10M best | 63.33% / 18/18 | 74.67% / 17/18 |
| Baseline 10M teacher | 52.00% / 16/18 | 67.33% / 18/18 |
| 1M + disconnection | 81.33% / 18/18 | 48.67% / 15/18 |
| 1M + robustness | 71.33% / 18/18 | 76.67% / 18/18 |
| 1M + both | 64.67% / 17/18 | 66.67% / 18/18 |

## 固定场景10000：覆盖骤降前后

以下时刻是该固定场景内的最大单步损失，属于明确挑选的事件诊断；完整场景分布仍以上表为准。

| 新增组 | 动作后时刻 | 覆盖前→后 | 联网UAV前→后 | 失去服务UE | 其中原服务UAV断联 | 丢失UU/UG链路数 |
|---|---:|---|---|---:|---:|---|
| 1M + disconnection | 218 | 65.33%→18.67% | 18→8 | 71 | 69 | 2/0 |
| 1M + robustness | 370 | 72.00%→18.67% | 18→4 | 80 | 79 | 3/1 |
| 1M + both | 214 | 65.33%→12.67% | 18→9 | 80 | 79 | 4/0 |

特别是鲁棒性组：第369步覆盖72.00%、J=66.93%，假想单UAV故障的最坏保留覆盖仍为58.67%；但下一步同时丢失3条UU与1条UG链路，覆盖降至18.67%。这说明本例的静态单节点故障指标未覆盖实际运动导致的同时多链路失效，不能仅靠提高平均J判断动态部署可靠性。

新增三组仍使用原集中式关联，没有移动前Top-K提议和新候选等待。它们同样发生大规模断联，因此本轮观察到的服务崩落不依赖Top-K时序；Top-K可以影响其他组的服务恢复，但不能作为新增三组崩落的解释。

## 来源与结论边界

原三组复用已校验的GPU评估，新三组在同设备与采样口径重新验证；模型、日志、原始结果与评估文件的来源SHA256保留。teacher仍是新baseline 10M best＋SINR Top-K闭环，不是DPPO预训练时使用的旧教师。
该比较不含同预算原1M baseline；因此六组结果可比较当前最终表现，若要严格估计新增奖励的作用，仍应回到1M对1M对照。没有追加训练或修改物理与算法设置。

图表提供PDF/SVG/PNG；红/蓝/橙保留原三组，新三组使用黑/浅蓝/灰，并增加线型与标记，避免紫色及红绿对照。

- [覆盖、吞吐与共同奖励](figures/task_curves.png)
- [覆盖骤降](figures/coverage_drops.png)
- [稳定性分布](figures/stability_distribution.png)
- [联网与鲁棒性](figures/deployment.png)
- [性能汇总](figures/performance.png)
- [新三组验证曲线](figures/new_runs_validation.png)
- [新三组loss](figures/new_runs_loss.png)
- [场景10000拓扑](figures/topology_best_10000.png)
- [场景10001拓扑](figures/topology_best_10001.png)
- [场景10000逐步覆盖与断联](figures/fixed_scene_events_10000.png)
- [场景10001逐步覆盖与断联](figures/fixed_scene_events_10001.png)
