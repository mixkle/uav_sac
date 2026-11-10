# 已完成验收的保留摘要

2026-11-10清理短测模型、原始轨迹和重复图表时，将70份文字报告、数值摘要和运行信息保存在`20261110/`下，子路径与原项目路径对应。

这些文件只记录实现验收，不能作为正式训练性能证据。其内部出现的原模型、轨迹和图表路径可能已清理，不再承诺从这些摘要重建短测图或续训；正式实验的模型和原始数据保留在`runs/`、`analysis/`。

| 原短测 | 保留入口 |
|---|---|
| 原扩散DPPO端到端短验收 | [报告](20261110/analysis/diffusion_dppo_local_acceptance/report.md)、[训练manifest](20261110/runs/diffusion_dppo_local_acceptance/online/manifest.json) |
| 跳数缩放双组短验收 | [报告](20261110/analysis/hop_scaling_local_acceptance/report.md)、[对比摘要](20261110/analysis/hop_scaling_local_acceptance/comparison.json) |
| 跳数缩放Gaussian 2048步 | [acceptance.json](20261110/output/hop_scaling_acceptance/gaussian/acceptance.json) |
| 跳数缩放Diffusion 2048步 | [acceptance.json](20261110/output/hop_scaling_acceptance/diffusion/acceptance.json) |
| 原3M续训实现核查 | [delivery.json](20261110/analysis/diffusion_dppo_3m_continuation_validation/delivery.json) |

详细来源和SHA256见[清理清单](../cleanup_20261110.json)。原验收结论及GPU性能数据仍见[本地DPPO验收](../diffusion_dppo_local_validation_20261105.md)、[跳数缩放验收](../hop_scaling_validation_20261108.md)及`../hop_scaling_server_acceptance_20261108/`。
