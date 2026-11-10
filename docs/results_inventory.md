# 结果目录与清理状态

更新日期：2026-11-10。按本地实际文件和manifest整理；主要入口见[项目导航](project_structure.md)。

## 当前主实验：跳数缩放联合控制

新增[Diffusion / Gaussian 10M 定位实验](../analysis/diffusion_gaussian_diagnosis_20261110/report.md)：同状态运动/评分交换、50步分叉、条件动作采样、关键断链干预、分步PPO审计及各20万步配对微调。原10M模型和结果保留；诊断运行不混入下面的正式运行目录统计。[执行说明](diffusion_gaussian_diagnosis.md)。

两组均完成在线10M，奖励为覆盖＋0.1归一化吞吐，容量10；actor同时输出运动和UE评分，采用Top-K。扩散另有64,000条示范和50轮预训练，Gaussian从零训练。

- [完整验证报告](../analysis/hop_scaling_joint_service_10m_seed0/report.md)。
- [训练验证趋势](../analysis/hop_scaling_joint_service_10m_seed0/figures/validation_trend.png)。
- [实际部署与启动说明](hop_scaling_joint_service_comparison.md)。

| 模型 | 保存步数 | 平均覆盖 | 后100步覆盖 | 吞吐 Mbps |
|---|---:|---:|---:|---:|
| diffusion last | 10,000,000 | 57.662% | 59.612% | 505.563 |
| gaussian last | 10,000,000 | 59.322% | 66.553% | 557.489 |
| diffusion best | 7,501,824 | 60.587% | 65.243% | 536.961 |
| gaussian best | 970,752 | 69.211% | 75.104% | 693.487 |

本地保留上述两组的step0/best/last副本、配置、训练日志和完整评估数据，位于`analysis/hop_scaling_joint_service_10m_seed0/snapshots/`及各评估子目录。服务器原完整训练目录为`/home/dym/uav_sac/runs/diffusion_dppo_hop_scaling_capacity10_10m_seed0/`和`gaussian_mappo_topk_hop_scaling_capacity10_10m_seed0/`；本地未伪装成已经下载完整恢复目录。

## 本地正式训练运行：20个，全部保留

目录名里的旧预算可能与实际步数不同。表中步数来自manifest；SAC本地轻量模型不足以替代其服务器完整回放恢复文件。

| 实验 | 实际步数 | best步数（manifest提供时） | 运行目录 |
|---|---:|---:|---|
| 独立actor集中式PPO，纯覆盖，历史容量20 | 1,200,000 | 见检查点/验证日志 | [continuous_mappo_coverage_independent_central_history_100k_seed0](../runs/continuous_mappo_coverage_independent_central_history_100k_seed0/) |
| 共享连续PPO，纯覆盖，历史容量20 | 1,000,000 | 见检查点/验证日志 | [continuous_mappo_coverage_shared_central_history_resumed_400k_seed0](../runs/continuous_mappo_coverage_shared_central_history_resumed_400k_seed0/) |
| 原扩散DPPO，无跳数缩放，10M续训链 | 10,000,000 | 9,550,656 | [diffusion_dppo_capacity10_coverage_throughput_10m_seed0](../runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0/) |
| 原扩散DPPO，无跳数缩放，1M续训链 | 1,000,000 | 40,960 | [diffusion_dppo_capacity10_coverage_throughput_1m_seed0](../runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/) |
| 原扩散DPPO，无跳数缩放，3M续训链 | 3,000,000 | 2,792,000 | [diffusion_dppo_capacity10_coverage_throughput_3m_seed0](../runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0/) |
| 原扩散DPPO，无跳数缩放，5M续训链 | 5,000,000 | 4,601,536 | [diffusion_dppo_capacity10_coverage_throughput_5m_seed0](../runs/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/) |
| 容量10，4邻居观测，覆盖＋数据率 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_neighbors4_1m_seed0](../runs/mappo_capacity10_neighbors4_1m_seed0/) |
| 容量10，5邻居观测，覆盖＋数据率 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_neighbors5_1m_seed0](../runs/mappo_capacity10_neighbors5_1m_seed0/) |
| 容量10，6邻居观测，覆盖＋数据率 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_neighbors6_1m_seed0](../runs/mappo_capacity10_neighbors6_1m_seed0/) |
| 容量10，7邻居观测，覆盖＋数据率 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_neighbors7_1m_seed0](../runs/mappo_capacity10_neighbors7_1m_seed0/) |
| 容量10，8邻居观测，覆盖＋数据率 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_neighbors8_1m_seed0](../runs/mappo_capacity10_neighbors8_1m_seed0/) |
| 容量10，鲁棒性0.5＋断联惩罚0.1 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_combined_1m_seed0](../runs/mappo_capacity10_stability_combined_1m_seed0/) |
| 容量10，断联惩罚0.1 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_disconnection_1m_seed0](../runs/mappo_capacity10_stability_disconnection_1m_seed0/) |
| 容量10，断联惩罚0.5 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_disconnection_mu05_1m_seed0](../runs/mappo_capacity10_stability_disconnection_mu05_1m_seed0/) |
| 容量10，断联惩罚2 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_disconnection_mu2_1m_seed0](../runs/mappo_capacity10_stability_disconnection_mu2_1m_seed0/) |
| 容量10，断联惩罚5 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_disconnection_mu5_1m_seed0](../runs/mappo_capacity10_stability_disconnection_mu5_1m_seed0/) |
| 容量10，鲁棒性奖励0.5 | 1,000,000 | 见检查点/验证日志 | [mappo_capacity10_stability_robustness_1m_seed0](../runs/mappo_capacity10_stability_robustness_1m_seed0/) |
| 容量10原baseline，集中式关联，10M预算对照 | 10,000,000 | 见检查点/验证日志 | [mappo_coverage_throughput_capacity10_10m_seed0](../runs/mappo_coverage_throughput_capacity10_10m_seed0/) |
| 容量10原baseline；best 671,744步为运动教师 | 1,000,000 | 见检查点/验证日志 | [mappo_coverage_throughput_capacity10_1m_seed0](../runs/mappo_coverage_throughput_capacity10_1m_seed0/) |
| 共享连续SAC，纯覆盖，历史对照 | 1,000,000 | 见检查点/验证日志 | [sac_coverage_time_factor_relative_central_history_server_1m_seed0](../runs/sac_coverage_time_factor_relative_central_history_server_1m_seed0/) |

原扩散1M/3M/5M是10M续训链中的正式实验切片，用于说明学习过程，继续保留；1M下的示范和预训练也保留。当前跳数缩放扩散重新采集了示范，两套示范不能混用。所有正式运行路径未重命名。

## 保留的完整分析与论文材料

| 用途 | 报告 |
|---|---|
| 当前跳数缩放双组10M完整验证，优先阅读 | [hop_scaling_joint_service_10m_seed0](../analysis/hop_scaling_joint_service_10m_seed0/report.md) |
| 原baseline 10M／原DPPO 10M／新教师三组 | [baseline_dppo_10m_best_teacher_seed0](../analysis/baseline_dppo_10m_best_teacher_seed0/report.md) |
| 原无缩放扩散10M与原baseline | [diffusion_dppo_capacity10_coverage_throughput_10m_seed0](../analysis/diffusion_dppo_capacity10_coverage_throughput_10m_seed0/report.md) |
| 原扩散5M历史切片 | [diffusion_dppo_capacity10_coverage_throughput_5m_seed0](../analysis/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/report.md) |
| 原扩散3M历史切片与收敛诊断 | [diffusion_dppo_capacity10_coverage_throughput_3m_seed0](../analysis/diffusion_dppo_capacity10_coverage_throughput_3m_seed0/report.md) |
| 原扩散1M历史切片与退化诊断 | [diffusion_dppo_capacity10_coverage_throughput_1m_seed0](../analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/report.md) |
| 容量10、覆盖＋数据率原baseline 1M | [mappo_coverage_throughput_capacity10_1m_seed0](../analysis/mappo_coverage_throughput_capacity10_1m_seed0/analysis_report.md) |
| 邻居3–8的1M best与原DPPO 10M | [mappo_neighbors3to8_1m_vs_dppo10m_seed0](../analysis/mappo_neighbors3to8_1m_vs_dppo10m_seed0/report.md) |
| 邻居3／4／5，包含1M last与10M对照 | [mappo_neighbors_1m_vs_10m_seed0](../analysis/mappo_neighbors_1m_vs_10m_seed0/report.md) |
| 邻居3／4／5冻结策略的UE／UAV数量泛化 | [mappo_neighbors_population_generalization_1m_seed0](../analysis/mappo_neighbors_population_generalization_1m_seed0/report.md) |
| 断联权重0／0.1／0.5／2／5对照 | [mappo_disconnection_weight_sweep_1m_seed0](../analysis/mappo_disconnection_weight_sweep_1m_seed0/report.md) |
| 鲁棒性／断联奖励消融与10M参考 | [mappo_stability_six_group_1m_vs_10m_seed0](../analysis/mappo_stability_six_group_1m_vs_10m_seed0/report.md) |
| 冻结原baseline的跳数减速诊断 | [baseline3_best_hop_speed_025_k04](../analysis/baseline3_best_hop_speed_025_k04/report.md) |
| 跳数减速＋5m局部链路安全诊断 | [baseline3_best_hop_speed_link_safety_delta5](../analysis/baseline3_best_hop_speed_link_safety_delta5/report.md) |
| 最新容量10教师的完整固定布局Top-K诊断 | [topk_capacity10_coverage_throughput_best_diagnostic](../analysis/topk_capacity10_coverage_throughput_best_diagnostic/report.md) |
| 原布局容量20／10反事实诊断 | [topk_capacity10_shared_best_diagnostic](../analysis/topk_capacity10_shared_best_diagnostic/report.md) |
| 历史共享／独立actor同预算1M对照 | [mappo_actor_sharing_coverage_1m_seed0](../analysis/mappo_actor_sharing_coverage_1m_seed0/report.md) |
| 历史独立1.2M与共享1M，预算不同 | [mappo_actor_sharing_independent_1p2m_vs_shared_1m_seed0](../analysis/mappo_actor_sharing_independent_1p2m_vs_shared_1m_seed0/report.md) |
| 研究汇报图表与演示素材 | [progress_presentation_20261103](../analysis/progress_presentation_20261103/README.md) |

另外保留`analysis/diffusion_dppo_baseline_reference/`的旧模型复现数据和`analysis/diffusion_dppo_server_deployment_20261105/`的部署记录。未列出的旧低预算实验若已在2026-10-29清理，不再假设其原始数据存在。

## 本次删除内容

只清理本地，服务器未修改。删除8个已完成短验收目录及4个临时传输包，合计约983 MiB；保留70份摘要约1.4 MiB。

| 已删除目录 | 性质 |
|---|---|
| `runs/diffusion_dppo_local_acceptance/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `analysis/diffusion_dppo_local_acceptance/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `analysis/hop_scaling_local_acceptance/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `output/hop_scaling_acceptance/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `output/topk_capacity_short_check/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `output/topk_capacity10_short_check/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `output/topk_capacity10_coverage_throughput_best_short/` | 已完成实现验收或短完整性核查，正式结果另存 |
| `analysis/diffusion_dppo_3m_continuation_validation/` | 已完成实现验收或短完整性核查，正式结果另存 |

临时传输包：`.cache/dppo_final_sync.tar.gz`、`.cache/diffusion_dppo_server_upload.tar.gz`、`.cache/hop_scaling_upload.tar.gz`、`.cache/hop_server_acceptance.tar.gz`。部署哈希记录及源码保留。

摘要入口见[acceptance_records](acceptance_records/README.md)。内部原模型和轨迹路径属于历史来源，已清理部分不能继续加载或重画；不把这些摘要当成正式训练证据。

## 核查与历史记录

- 20个正式运行保留；415个受保护的模型、配置、训练日志和最新原始结果SHA256在清理前后相同。
- 本次清单与保留摘要哈希：[cleanup_20261110.json](cleanup_20261110.json)。
- 上一轮只清理两次中途评估：[最新结果的cleanup_manifest.json](../analysis/hop_scaling_joint_service_10m_seed0/cleanup_manifest.json)。
- 2026-10-29清理记录：[cleanup_20261029.json](cleanup_20261029.json)。它对应更早的一轮低预算实验删除，不代表当前磁盘占用。
- 参考论文、建模、源码、测试、正式与短验收配置、环境和论文草稿保留。新入口与历史用途见[项目导航](project_structure.md)。
