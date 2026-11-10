# 10M 两策略定位实验数据

主入口：[结果解释](report.md) · [逐项统计](detailed_statistics.md) · [命令与设计](../../docs/diffusion_gaussian_diagnosis.md) · [回归限制](regression_notes.md)

- `manifest.json`：计划、原git版本、67个原始输入SHA256及完成状态。
- `states/`：四组完整环境快照（随机流、关联、历史俱全）；factual文件保存事实动作及恢复验证所需上下文。`t`从0计，对应动作执行后的时刻为`t+1`。
- `exp1_motion_vs_scoring/`：4组合×16次单步、全骤降事件、18个单机保持干预；128起点×4组合×50步分叉；固定物理快照、逐场景bootstrap、拓扑图。
- `exp2_action_distribution/`：5策略×64条件样本，保留ragged评分、latent、UE ID/offset；几何回传风险、Top-K一致性和0.8噪声干预。`incumbent_users_exposed`不是实际未服务人数。
- `exp3_dppo_update/`：4阶段历史更新统计、分步KL/clip、分支梯度口径、概率重算、日程敏感度及缺失字段审计。没有伪造历史逐minibatch明细。
- `exp4_conservative_finetuning/`：门控依据，两组各200k新微调配置/初始化/完整恢复/冻结best-last评估/学习曲线；来源为同一D best，两组均重置Adam，非无缝续训。
- `code_snapshot/`：最终诊断代码副本及SHA256。不同阶段上传的最初脚本哈希见运行记录；最终副本包含后续cases/noise/几何重算扩展，不能冒充全部采集时已存在。

原始67个源文件不在此目录复制覆盖，位于`../hop_scaling_joint_service_10m_seed0/`。本轮只保护这些输入哈希，未声称复制原服务器全部历史完整训练状态。服务器输出根与本地同相对路径。

重建图表不需要采样，遵循docs中的顺序。`report_diffusion_gaussian_diagnosis.py report`生成基础统计页，`finalize_diffusion_gaussian_diagnosis.py`生成最终解释页。所有图输出PNG/PDF/SVG，绿色与蓝色配合线型；灰色用于不同失效环节。

保留测试集未参与。本轮区间为场景配对，不是训练种子区间。20万步微调是有限筛查，未自动追加1–2M或多种子。
