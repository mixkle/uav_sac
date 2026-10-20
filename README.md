# 多 BS 随机部署、发现与服务实验

连续SAC场景（**beacon-set-v2**）：1600×1600 m，16 UAV 在整个区域独立均匀随机初始化，通过上行信标发现180个移动UE（四中心截断高斯混合，初始距边界至少160 m）。SAC、MAPPO、MATD3均使用共享UE集合编码器和局部actor；critic使用全局状态。静止、随机和启发式共用随机初始化分布。

已有训练产物和分析报告已清理。默认参数见 [main.yaml](configs/main.yaml)，完整流程及全部参数见 [SAC / MAPPO训练说明](docs/training_workflows_and_parameters.md)，环境语义见 [模型实现约定](docs/model_contract.md)。

## 安装

推荐Python3.11，支持3.10–3.12；本地验收使用3.10 CPU。项目放在WSL Linux文件系统。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pytest -q
```

服务器恢复可用后用 `requirements/cuda.txt`，重新创建虚拟环境，不复制WSL环境。CUDA性能与恢复一致性尚未验收。

## 本地校准与短流程

```bash
.venv/bin/python -m uav_research check-env --config configs/main.yaml --output runs/beacon_v2/check.json
.venv/bin/python -m uav_research calibrate --config configs/main.yaml --output runs/beacon_v2/calibration
.venv/bin/python -m uav_research plot --calibration runs/beacon_v2/calibration --output runs/beacon_v2/calibration/figures
.venv/bin/python -m uav_research train --config configs/smoke.yaml --algo sac --output runs/beacon_v2/smoke_sac
.venv/bin/python -m uav_research train --config configs/local_validation.yaml --algo sac --output runs/beacon_v2/local_sac
```

将 `--algo` 换为 `mappo` 或 `matd3` 可验证其流程。smoke为4 UAV/24 UE/32步小任务，256条转移；local_validation保留主物理场景，明确覆盖为2环境、batch64、warmup256、回放4096、总4096条转移、4个验证场景。二者都是完整性验证，不是论文效果实验。

校准使用独立种子30000–30007，扫描当前感知门槛±5 dB、下行门槛±3 dB及两种回传距离±20%，默认值不会自动修改。对比初始静止、随机运动、固定目标逐步展开、已展开静止。后两种为物理诊断，不是可部署局部策略baseline。保存完整CSV、代表轨迹和可重建图表。

```bash
.venv/bin/python -m uav_research benchmark --config configs/local_validation.yaml --output runs/beacon_v2/benchmark.json --counts 1 2 4 8 --backends batch --devices cpu --steps 256
.venv/bin/python scripts/check_full_replay_checkpoint.py --output runs/beacon_v2/memory
```

满容量测试会临时占用约8 GiB内存和8 GiB磁盘；生成合成50k满可见回放，校验保存、恢复和下一批采样，成功后删除其生成的合成检查点，保留结果与RSS记录。测试不会训练策略。

## 保存、恢复与导出

`env_steps`是一条完整联合环境转移，不乘UAV数，预算需整除环境数。每次运行保存配置、源码哈希、依赖、设备和随机流。

`best.pt`按验证平均团队回报选择，`last.pt`在验证和完整保存前写入。完整恢复是 `resume_XXXXXXXXXXXX/` **目录**，内含metadata、SHA256索引及最多256条转移的Tensor分块；保留最近两份。只复制best/last无法续训。

```bash
.venv/bin/python -m uav_research train --config runs/beacon_v2/smoke_sac/config.yaml --output runs/beacon_v2/smoke_sac --resume runs/beacon_v2/smoke_sac/resume_000000000256 --steps 288
.venv/bin/python -m uav_research evaluate --checkpoint runs/beacon_v2/smoke_sac/last.pt --output runs/beacon_v2/smoke_sac/evaluation --device cpu
.venv/bin/python -m uav_research export-actor --checkpoint runs/beacon_v2/smoke_sac/best.pt --output runs/beacon_v2/smoke_sac/actor.pt
```

恢复要求环境、算法、并行采集和随机流配置一致。回滚到更早步时使用新输出目录，避免拼接不同分支。actor导出包含UE编码器、策略主干、缩放配置，不需要critic或全局UE数量；推理接口为 `DeployedActor(path).act(raw_obs)`。

## 诊断、评估和绘图

```bash
.venv/bin/python -m uav_research evaluate --config configs/local_validation.yaml --baseline heuristic --output runs/beacon_v2/heuristic
.venv/bin/python -m uav_research plot --runs runs/beacon_v2/local_sac --evaluations runs/beacon_v2/smoke_sac/evaluation --traces runs/beacon_v2/smoke_sac/evaluation/trace_10000.npz --output runs/beacon_v2/figures
```

每步诊断包括发现、下行、回传、容量筛选，覆盖/吞吐/J、集合长度、编码器梯度、边界裁剪、展开距离和回放字节数。评估给出任务平均、终态和最后100步表现，以及首次服务、零奖励持续时间。累计发现明确标记为历史诊断；不进入观察或奖励。

图形输出PDF/SVG/300 dpi PNG，包含发现—服务、轨迹拓扑、故障损失、训练样本/墙钟曲线。训练单种子不画训练种子置信区间。未来正式验证默认32场景，保留测试100场景；校准与本地短流程不使用测试集。

| 数据 | 用途 |
|---|---|
| config.yaml / metadata.json / sessions.json | 展开配置、源码、依赖、设备、随机流、恢复记录 |
| metrics.csv / validation.csv | 训练诊断、资源、阶段耗时和验证历史 |
| manifest.json | 当前阶段、停止原因、完整恢复路径 |
| episodes.csv / summary.json | 逐场景全任务、终态、最后100步评估 |
| trace_*.npz | 位置、动作、A/B/C/Y、发现矩阵、上行SNR、逐故障结果 |
| actor.pt | 不含critic的可部署策略 |

所有相对输出在AutoDL自动放到 `/root/autodl-tmp/uav_sac/`，也可设 `UAV_DATA_ROOT`；显式绝对路径优先。当前不连接服务器。运行速度需在实际训练设备重新测量。

完整本地验收可一键重建（包括满容量内存测试、校准、三种算法短训、恢复、评估和绘图；不会SSH或运行正式训练）：

```bash
.venv/bin/python scripts/validate_local.py --output analysis/beacon_v2_20261013
```

输出目录必须未包含同名训练/内存任务产物，重跑请使用新目录。UE分布依据和自定参数见 [初始化参考说明](docs/ue_initialization_reference.md)。

## WSL2 随机初始化 50k

使用新输出目录从头训练，旧出发场景检查点不用于本轮续训：

```bash
.venv/bin/python -m uav_research train --config configs/wsl_50k.yaml --output runs/wsl_sac_random_50k_seed0
```

初始化、CPU线程及算法参数见 [训练说明](docs/training_workflows_and_parameters.md)。

## 九动作离散 MAPPO 版本

连续版本继续使用原配置。离散版本采用论文规模3500×3500 m、18 UAV、150 UE、三个角落BS（左下、左上、右下）。UAV与UE全区域独立均匀初始化；世界坐标八方向＋悬停，30 m/步、400步任务、100 m高度。保留下行SINR与独立信标，校准为UU650 m/UG1500 m、发现20 dB/下行−3 dB；鲁棒性权重0。共用参数化物理实现与UE集合编码，MAPPO学习参数保持原值。

- [完整离散建模](<ENVIRONMENT MODELING DISCRETE MAPPO.md>)
- [主配置](configs/discrete_mappo.yaml) · [WSL 250k配置](configs/discrete_mappo_wsl_50k.yaml) · [零鲁棒性兼容入口](configs/discrete_mappo_no_robustness.yaml)
- [训练流程与全部参数](docs/training_workflows_and_parameters.md)

```bash
.venv/bin/python -m uav_research train --config configs/discrete_mappo_wsl_50k.yaml --output runs/wsl_discrete_mappo_paper_scene_250k_seed0
```

该命令训练250000条联合转移；配置文件名保留50k以兼容旧命令，实际预算以total_steps为准。连续与离散检查点不能跨模式续训；旧离散场景的检查点也不能用于新场景，须从头训练并使用新输出目录。
