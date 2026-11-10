# 项目文件导航

更新日期：2026-11-10。入口与路径按当前实际文件整理；原实验目录不重命名，以保留模型来源和配置引用。

## 当前研究入口

| 内容 | 文件 |
|---|---|
| 最新建模：跳数缩放、运动与服务评分 | [ENVIRONMENT_MODELING_Diffusion_DPPO_Hop_Scaling.md](../ENVIRONMENT_MODELING_Diffusion_DPPO_Hop_Scaling.md) |
| 当前双组准备、启动、恢复、报告 | [hop_scaling_joint_service_comparison.md](hop_scaling_joint_service_comparison.md) |
| 完整10M结果 | [最终报告](../analysis/hop_scaling_joint_service_10m_seed0/report.md) |
| 所有正式运行及诊断结果 | [results_inventory.md](results_inventory.md) |
| TVT论文草稿 | [中英Introduction与Related Work](paper_tvt_introduction_related_work_bilingual.md) |
| 汇报材料 | [研究进展说明](progress_presentation_diffusion_allocation_20261103.md) |

## 目录职责

| 目录 | 内容与使用原则 |
|---|---|
| `src/uav_research/` | 物理环境、局部观察、原MAPPO/SAC、运行管理；本次整理没有改训练逻辑 |
| `src/uav_research/diffusion_dppo/` | 新服务评分环境、扩散与Gaussian评分actor、示范/预训练、在线训练、评估 |
| `configs/` | 正式训练及可重复短测配置；当前主入口见下表 |
| `scripts/` | 准备/启动/恢复、诊断、报告和性能核查；历史诊断脚本保留以便追溯 |
| `tests/` | 回归测试；`fixtures/configs/`中的旧配置是测试输入 |
| `requirements/` | 原环境与隔离DPPO环境的CPU/CUDA依赖和锁定文件 |
| `runs/` | 正式训练模型、日志、完整恢复检查点；按manifest实际步数识别实验 |
| `analysis/` | 正式评估、完整关联诊断、论文图及其可重建原始数据 |
| `docs/` | 当前操作说明、实验记录、论文草稿、清理与验收记录 |
| `docs/acceptance_records/` | 已清理短测的文字报告、数值摘要与来源记录，不含短测模型和轨迹 |
| `deployment/` | 历次部署与下载的SHA256清单，作为来源核对依据 |
| `output/` | 开发生成物；保留现有汇报和绘图素材，本次移除了已完成短测 |
| `.cache/` | 临时脚本、缓存和下载核对信息；已删除失去用途的传输包 |

## 配置分组

| 用途 | 配置 |
|---|---|
| 当前扩散DPPO，跳数缩放，10M | [diffusion_dppo_hop_scaling_capacity10_10m_server.yaml](../configs/diffusion_dppo_hop_scaling_capacity10_10m_server.yaml) |
| 当前Gaussian服务评分MAPPO，跳数缩放，10M | [gaussian_mappo_topk_hop_scaling_capacity10_10m_server.yaml](../configs/gaussian_mappo_topk_hop_scaling_capacity10_10m_server.yaml) |
| 原容量10 baseline，1M；671,744步best为运动教师 | [mappo_coverage_throughput_capacity10_1m.yaml](../configs/mappo_coverage_throughput_capacity10_1m.yaml) |
| 原容量10 baseline，10M；集中式关联 | [mappo_coverage_throughput_capacity10_10m_server.yaml](../configs/mappo_coverage_throughput_capacity10_10m_server.yaml) |
| 原扩散版本，不启用跳数缩放，10M | [diffusion_dppo_capacity10_coverage_throughput_10m_server.yaml](../configs/diffusion_dppo_capacity10_coverage_throughput_10m_server.yaml) |
| 邻居数量消融 | `mappo_capacity10_neighbors{4,5,6,7,8}_1m_server.yaml`；3邻居为原baseline |
| 鲁棒性与断联权重消融 | `mappo_capacity10_stability_*_1m_server.yaml` |
| 历史actor共享、SAC对照 | `mappo_coverage_shared_1m.yaml`、`mappo_coverage_independent_1p2m.yaml`、`sac_coverage_*_1m.yaml` |
| 实现验收 | 名含`local_smoke`、`local_probe`的配置；允许重跑，但输出需与正式运行隔离 |

原1M/3M/5M扩散配置用于既有续训链和曲线复现，继续保留。历史容量20或四项奖励配置不代表当前容量10、两项奖励设置，使用时核对展开配置。

## 当前代码与脚本

| 步骤 | 入口 |
|---|---|
| 新环境与评分Top-K执行 | [environment.py](../src/uav_research/diffusion_dppo/environment.py) |
| 跳数缩放公共逻辑 | [motion.py](../src/uav_research/motion.py) |
| 扩散actor | [policy.py](../src/uav_research/diffusion_dppo/policy.py) |
| Gaussian服务评分actor | [gaussian.py](../src/uav_research/diffusion_dppo/gaussian.py) |
| 示范采集和扩散预训练 | [data.py](../src/uav_research/diffusion_dppo/data.py) |
| Learner和在线运行 | [learner.py](../src/uav_research/diffusion_dppo/learner.py)、[runner.py](../src/uav_research/diffusion_dppo/runner.py) |
| 服务器环境 | [dppo_server_env.sh](../scripts/dppo_server_env.sh) |
| 扩散准备 | [prepare_hop_scaling_diffusion_server.sh](../scripts/prepare_hop_scaling_diffusion_server.sh) |
| 双组tmux启动 | [launch_hop_scaling_comparison_server.sh](../scripts/launch_hop_scaling_comparison_server.sh) |
| 单组训练和恢复 | [train_hop_scaling_group_server.sh](../scripts/train_hop_scaling_group_server.sh) |
| 通用双组报告 | [report_hop_scaling_comparison.py](../scripts/report_hop_scaling_comparison.py) |
| 本次最终中文报告重建 | [build_report.py](../analysis/hop_scaling_joint_service_10m_seed0/build_report.py)，从仓库根目录运行 |

## 建模及历史记录

`ENVIRONMENT MODELING.md`与`ENVIRONMENT MODELING DISCRETE MAPPO.md`保留原基础建模；`ENVIRONMENT MODELING Diffusion_DPPO.md`对应不带跳数缩放的联合控制版本。最新缩放实验优先读取本页首项建模和当前配置。

带日期的部署、验收文档是对应日期的历史快照；“尚未训练”“等待上传”等描述不代表当前状态。最新两组10M已完成，结果与限制以最终报告为准。

## 本次清理边界

只处理本地的8个已完成短测目录和4个临时传输包。验收文字与数值记录另存；所有20个1M及以上正式运行、原1M/3M/5M扩散链、完整Top-K诊断、消融、泛化、速度与链路安全分析均保留。

最新跳数缩放两组本地保留step0/best/last及评估数据，完整10M训练目录位于服务器`/home/dym/uav_sac/runs/`。本次未连接服务器或更改服务器文件。详细删除清单、保留模型哈希与核对结果见[cleanup_20261110.json](cleanup_20261110.json)。
