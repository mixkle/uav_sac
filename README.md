# UAV 多跳覆盖与联合服务关联研究

当前工作是**跳数速度缩放下的集合条件化Diffusion DPPO与Gaussian评分MAPPO对照**，两组在线10M训练均已完成。两组都输出连续运动和UE服务评分，采用相同Top-K关联；扩散另有教师示范和去噪预训练。

- [最新10M完整验证](analysis/hop_scaling_joint_service_10m_seed0/report.md)：step0／best／last、拓扑、MC价值误差与稳定性诊断。
- [当前实验与启动说明](docs/hop_scaling_joint_service_comparison.md)：服务器准备、启动、恢复及报告命令。
- [项目文件导航](docs/project_structure.md)：建模、配置、代码、脚本、文档和结果的对应关系。
- [完整结果索引](docs/results_inventory.md)：本地实际保留的正式运行与主要报告。
- [模型及完整结果下载](artifacts/20261110/README.md)：同仓库Release分片、SHA256清单及恢复方法。

## 场景与方法版本

场景为3500×3500 m、18 UAV、150移动UE、三个角落BS、每架容量10；UE在中心2500×2500 m初始化。使用相对坐标和一步历史、400步真正终止、连续圆盘动作及镜面反射。当前任务奖励为 `coverage + 0.1 * normalized_throughput`。

| 版本 | 策略与关联 | 主要入口 |
|---|---|---|
| 原容量10 baseline | Gaussian MAPPO控制运动，集中式关联 | [1M配置](configs/mappo_coverage_throughput_capacity10_1m.yaml)、[10M配置](configs/mappo_coverage_throughput_capacity10_10m_server.yaml) |
| 原扩散DPPO | 扩散运动＋评分、Top-K，不启用跳数缩放 | [实验说明](docs/diffusion_dppo_experiment.md)、[10M报告](analysis/diffusion_dppo_capacity10_coverage_throughput_10m_seed0/report.md) |
| 当前跳数缩放对照 | 两类actor均控制运动＋评分、Top-K、实际位移历史 | [扩散配置](configs/diffusion_dppo_hop_scaling_capacity10_10m_server.yaml)、[Gaussian配置](configs/gaussian_mappo_topk_hop_scaling_capacity10_10m_server.yaml) |

原baseline的1M best来自671,744步，仍是扩散运动教师。原baseline与最新Gaussian评分策略属于不同关联版本，比较时应明确区分。完整方法见[跳数缩放建模](ENVIRONMENT_MODELING_Diffusion_DPPO_Hop_Scaling.md)。

## 数据与清理

正式运行保存在 `runs/`，验证与论文图保存在 `analysis/`，开发临时输出保存在 `output/` 或 `.cache/`。最新两组的本地模型副本位于 `analysis/hop_scaling_joint_service_10m_seed0/snapshots/`；完整训练恢复目录仍在服务器。

2026-11-10清除了已经完成验收的短训模型、短测轨迹和临时传输包，保留20个本地1M及以上正式运行、示范与预训练、完整诊断及最新10M结果。验收摘要移入 [acceptance_records](docs/acceptance_records/README.md)，删除范围及哈希核对见 [清理记录](docs/cleanup_20261110.json)。服务器未执行本次清理。

运行目录名可能仍含 `100k`、`400k` 等早期预算，实际完成步数以 `manifest.json` 为准；保留原路径以支持检查点来源核对。

## 环境与验证

本地扩散路径使用 `.venv-dppo-compat/`，依赖见 [CPU锁定文件](requirements/dppo-cpu.lock.txt)；服务器使用 `.venv-dppo/` 和 [CUDA11.8依赖](requirements/dppo-server-cu118.txt)。旧WSL环境 `.venv/` 继续用于原算法；不互相复制虚拟环境。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv-dppo-compat/bin/python analysis/hop_scaling_joint_service_10m_seed0/build_report.py
```

以上命令从保存数据重建最新报告，不重新训练。算法回归入口为 `python -m pytest`；旧配置测试样本在 `tests/fixtures/configs/`。参数与历史实验说明见 [训练参数文档](docs/training_workflows_and_parameters.md)，实现语义见 [模型约定](docs/model_contract.md)。
