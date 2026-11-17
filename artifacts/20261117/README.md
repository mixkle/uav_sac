# 2026-11-17：代码与实验结果同步

本次更新包含当前训练/评估代码、配置、服务器脚本、测试、实验计划与验收记录，以及以下结果的报告、汇总表、部分逐场景表、图表与来源记录。

| 实验 | 报告 | 主要表格 |
|---|---|---|
| 系统消融 A–F | [报告](../../analysis/diffusion_gaussian_ablation_1m/report.md) | [best/last](../../analysis/diffusion_gaussian_ablation_1m/metrics/unified_best_last.csv) |
| F2b/F2c/A1b 追加实验 | [分析](../../analysis/diffusion_followup_f2b_f2c_a1b_20261113/analysis.md) | [指标目录](../../analysis/diffusion_followup_f2b_f2c_a1b_20261113/metrics/) |
| T1–T3 关联时序 | [分析](../../analysis/t1_t2_t3_post_motion_association/analysis.md) | [汇总](../../analysis/t1_t2_t3_post_motion_association/summary/performance.csv) |
| S1/S2 评分与规则诊断 | [报告](../../analysis/score_rule_s1_s2_20261116/report.md) | [规则对比](../../analysis/score_rule_s1_s2_20261116/s2/comparison.csv) |
| 仅运动 M1/M2 10M | [报告](../../analysis/motion_only_sinr_topk_10m/report.md) | [best/last](../../analysis/motion_only_sinr_topk_10m/metrics/unified_best_last.csv) |
| 历史结果后期指标复评 | [A–F 演进复评](../../analysis/uav_evolution_af_reevaluation_20261110/report.md) | [中文指标](../../analysis/uav_evolution_af_reevaluation_20261110/全部指标_中文.csv) |

## 范围与完整性

[results_manifest.json](results_manifest.json)记录本次新增入库的结果文件大小和SHA256，并列出未选入的可见明细。CSV主要选择不超过2 MiB的表格，另保留报告、来源JSON、分析脚本及主图PNG/PDF/SVG。

本次没有上传新的模型权重、完整恢复检查点、NPZ/NPY/Parquet原始数组或大型逐步CSV；这些数据仍在本地或服务器。本次也没有创建新的Release原始数据附件。此前2026-11-10的Release和快照保持不变。依赖原始数组或权重的重建/重新评估需要另行取得对应数据，不能仅凭本快照完整恢复训练。

认证文件、虚拟环境、缓存和Windows下载元数据均不入库。M2-NoHop/N5/N7/N9仅包含实现、配置和验收记录，未自动启动训练。

## 提交前验证

新增及相关回归测试：130 passed（59.33 s）。日志见[tests.txt](tests.txt)。结果图表来自既有评估，本次同步没有重新训练或重新采样全部实验。
