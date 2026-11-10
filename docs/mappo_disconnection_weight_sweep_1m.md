# 3邻居baseline：断联惩罚权重0.5／2／5

以 `configs/mappo_coverage_throughput_capacity10_1m.yaml` 为基准，新增三组种子0、从零训练1,000,000条联合转移的实验。保持3邻居、共享连续Gaussian MAPPO、一步历史、相对坐标和时间因子critic。旧baseline及原断联权重0.1实验全部保留。正式训练由用户启动。

## 实验矩阵

令 `D` 为无法通过任意多跳路径接入任意BS的UAV数量除以18；它不是断联UE比例，也不是断链次数。

`r = coverage + 0.1 * tau - mu * D`

`tau = total_throughput / (150 * 5 Mbps)`。鲁棒性与边界惩罚权重均为0。不改无线、关联或运动实现，不增加独立的断联loss。

| 组别 | mu | 单架UAV断联单步成本 | 相当于UE覆盖贡献（忽略吞吐） | 默认GPU | tmux会话 |
|---|---:|---:|---:|---:|---|
| 原baseline（已有） | 0 | 0 | 0 | — | — |
| 原断联实验（已有） | 0.1 | 0.00556 | 0.83 UE | — | — |
| disconnection_mu05 | 0.5 | 0.02778 | 4.17 UE | 0 | mappo-mu05 |
| disconnection_mu2 | 2 | 0.11111 | 16.67 UE | 1 | mappo-mu2 |
| disconnection_mu5 | 5 | 0.27778 | 41.67 UE | 2 | mappo-mu5 |

这些是明显不同的取舍强度，不是只改变网络连通诊断。单架最多服务10 UE，因此mu=2／5时，单架断联成本已经超过10 UE的覆盖贡献，但中继仍可能影响多台UAV。强惩罚可能鼓励接近BS或聚集；结果必须同时比较覆盖、吞吐和连接稳定性，不能只看平均断联下降。允许训练奖励为负，不裁剪奖励或critic目标。

## 控制变量

3500×3500m、18 UAV、150 UE、3个角落BS；容量10；UAV全场均匀初始化，UE中心2500×2500m均匀初始化并随机游走；400步回合、30m/步、镜面反射、UU650m、UG三维1500m。保持原全部通信参数、观察、策略结构、PPO／GAE／PopArt／Value Clipping设置。

相对原baseline，配置有效差异只有 `env.disconnection_weight` 与服务器设备 `run.device: cuda:0`。三组之间只改变断联权重。CPU8线程、Numba4线程、16环境×128步、489轮PPO；最后576条联合转移，共19560次actor和critic优化。旧3邻居baseline在CPU训练，新组使用CUDA，跨训练设备的数值及采样随机流差异仍需记录。

best仍按各组自身全回合任务回报选择，即包含断联惩罚，包含step0。主要诊断同预算last，best作为补充。跨组比较另看共同服务回报 `coverage + 0.1*tau`，不能直接比较权重不同的训练回报。沿用验证场景10000–10031、动作采样种子880001、批次16，不使用保留测试集。

## 服务器启动

代码、环境、临时文件、日志、结果均在 `/home/dym/uav_sac/` 下。复用已有 `.venv-dppo`，每组单卡，三组可并行。

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_disconnection_sweep_server.sh
```

默认GPU0／1／2，CPU亲和范围0–7／8–15／16–23，创建上表三个独立tmux会话。只检查、不训练：

```bash
bash scripts/launch_mappo_disconnection_sweep_server.sh --check
```

换卡：

```bash
UAV_DISCONNECTION_GPUS=1,2,3 bash scripts/launch_mappo_disconnection_sweep_server.sh
```

CPU范围可以通过 `UAV_DISCONNECTION_CPUSETS='0-7|8-15|16-23'` 指定。脚本拒绝已有非空结果目录或同名tmux会话，不覆盖、不续训。查看一组：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-mu05
```

其他组替换为 `mappo-mu2`／`mappo-mu5`。按Ctrl+B再按D脱离，训练继续。单独启动：

```bash
CUDA_VISIBLE_DEVICES=0 UAV_CPU_AFFINITY=0-7 \
bash scripts/train_mappo_stability_rewards_server.sh disconnection_mu05
```

配置：`configs/mappo_capacity10_stability_disconnection_mu{05,2,5}_1m_server.yaml`。
结果：`runs/mappo_capacity10_stability_disconnection_mu{05,2,5}_1m_seed0/`。
日志：`logs/mappo_stability_disconnection_mu{05,2,5}_1m_日期.log`。
完整恢复机制不变，保留最近两份完整恢复目录，轻量best／last不是完整恢复入口。

## 验收与训练后判断

本地复用既有奖励实验测试，核对有效配置只改变权重和设备、同种子初始网络一致、固定动作物理状态一致、奖励贡献求和、负奖励可训练、保存恢复与连续执行一致、导出与评估。上传后仅预检查，不启动正式训练。

训练完成后比较原mu=0／0.1与本次0.5／2／5，重点看：

- 平均、后100步、终态覆盖及吞吐，共同服务回报与各自训练回报。
- 最大单步／5步覆盖跌幅，≥10pp骤降次数及比例，最差20步覆盖。
- 断联率、网关接入、故障保留、聚集与靠边，解释是否为了联网牺牲展开。
- best与last差异，避免只比较各组不同奖励所选择的best。

单种子仅用于定位权重取舍，不预设强惩罚必然改善覆盖或消除关键链路风险。原四组稳定性报告矩阵保持不变，本次配置不会自动混入旧报告。

### 本次交付验收

本地35项测试通过；服务器8个上传文件SHA256一致，三个单卡预检查全部通过，GPU0／1／2检查时空闲，空闲磁盘约62GiB。未启动正式训练、未创建新的训练输出目录。验收来源记录：`docs/mappo_disconnection_sweep_deployment_20261108.json`。

## 已完成训练与验证（2026-11-08）

三组均完成1M训练，best实际步数分别为911,360／821,248／1,000,000。完整运行、恢复文件及best／last验证数据已下载；396文件、679,354,833字节通过逐文件SHA256核对，清单为 `deployment/disconnection_sweep_results_sha256.json`。

[统一五组报告](../analysis/mappo_disconnection_weight_sweep_1m_seed0/report.md)比较mu=0／0.1／0.5／2／5，仅采用原始策略执行，没有使用后续跳数减速或链路安全约束。新三组best／last的覆盖、吞吐和自身训练回报均复现服务器CUDA验证日志，最大绝对差小于1e-6。原两组使用已校验的同口径CUDA缓存。

主要比较同预算last：

| mu | 平均覆盖 | 后100步 | 吞吐Mbps | 断联率 | 最大单步跌幅均值 |
|---|---:|---:|---:|---:|---:|
| 0 | 56.708% | 59.933% | 471.22 | 13.662% | 48.875pp |
| 0.1 | 59.554% | 62.763% | 500.78 | 9.247% | 35.667pp |
| 0.5 | 54.960% | 57.266% | 451.25 | 9.306% | 28.042pp |
| 2 | 43.307% | 37.296% | 334.29 | 4.938% | 24.000pp |
| 5 | 46.295% | 41.445% | 349.20 | 5.756% | 18.875pp |

当前单种子结果中mu=0.1服务与稳定性取舍最好；0.5降低最大跌幅但覆盖、吞吐不及0.1；2／5更少断联，但感知、服务覆盖和吞吐下降，聚集及后期退化明显。场景10000的mu=2／5终态都有18架联网，覆盖却仅36%／37.33%，说明当时低覆盖并非断联所致。best结果、共同服务回报、奖励分解、固定拓扑、loss和配对场景区间均在报告中保留。

本地重建不重新采样：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python scripts/report_disconnection_weight_sweep.py --rebuild
```

报告不改变原四组稳定性实验的定义，也未加入baseline 10M或DPPO。区间基于配对场景，不是训练种子置信区间；不能将0.1当作已证明的普适最优权重。
