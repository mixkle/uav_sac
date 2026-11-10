# 容量10 baseline：6／7／8邻居的1M对照

沿用 `configs/mappo_coverage_throughput_capacity10_1m.yaml`，新增三组从零训练：种子0、各1,000,000条联合转移。UAV总数仍为18、UE仍为150，场地3500×3500 m，UE中心2500×2500 m初始化并持续移动，保持3 BS、400步回合、30 m/步及边界反射。

只增加650 m直连范围内可观察邻居的最大数量，不扩大通信范围，不改变关联。奖励仍为 `coverage + 0.1 * normalized_throughput`，鲁棒性、断联和边界惩罚权重均为0。三个配置与原baseline的有效差异仅为 `env.neighbor_slots` 和设备 `cuda:0`。

| 邻居数 | 固定观测（含历史） | 编码后actor输入 | Actor参数量 | 默认GPU | 默认CPU | tmux会话 |
|---|---:|---:|---:|---:|---|---|
| 6 | 177 | 210 | 122,052 | 0 | 0–7 | `mappo-neighbors6` |
| 7 | 201 | 234 | 128,196 | 1 | 8–15 | `mappo-neighbors7` |
| 8 | 225 | 258 | 134,340 | 2 | 16–23 | `mappo-neighbors8` |

每组使用一张GPU、PyTorch8线程、Numba4线程，三个任务分别更新各自的模型。保持16环境×128步、时间因子critic6292维、共享连续高斯圆盘actor、相对坐标和一步历史，全部学习参数保持baseline。邻居按距离和编号排序，不足补零；因此设置8槽不意味着每时刻都有8架可见邻居。

初始网络沿用4／5邻居实验的控制方式：按原3邻居未训练网络构建，将新增邻居输入列初始化为零，保留其余参数及critic初值和随机流，新增列正常参与训练。既不加载旧权重，也不继承此前短测；比较时需考虑输入层参数量和单种子随机性。

## 一键启动

在服务器执行：

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_neighbors678_server.sh
```

该命令分别启动三个tmux会话，从零训练至1M。终端关闭不会中断tmux中的训练。只进行预检查、不启动训练：

```bash
bash scripts/launch_mappo_neighbors678_server.sh --check
```

需要换卡时指定三个不同GPU：

```bash
UAV_NEIGHBOR_GPUS=1,2,3 bash scripts/launch_mappo_neighbors678_server.sh
```

查看一组：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-neighbors6
```

另外两组将会话名中的6换为7／8。按 `Ctrl+B`，再按 `D` 脱离会话。

## 文件与单组命令

配置：

- `configs/mappo_capacity10_neighbors6_1m_server.yaml`
- `configs/mappo_capacity10_neighbors7_1m_server.yaml`
- `configs/mappo_capacity10_neighbors8_1m_server.yaml`

结果分别保存在 `/home/dym/uav_sac/runs/mappo_capacity10_neighbors{6,7,8}_1m_seed0/`；控制台日志在项目 `logs/mappo_neighbors{6,7,8}_1m_日期.log`。完整保存和best／last选模沿用baseline，拒绝覆盖非空结果目录，不自动续训。

单独启动一组：

```bash
CUDA_VISIBLE_DEVICES=0 UAV_CPU_AFFINITY=0-7 \
bash scripts/train_mappo_neighbors_server.sh 6
```

原4／5邻居的启动方式保留；通用启动器默认仍只处理4／5组。实现与部署阶段只做本地小配置测试、服务器预检查和无更新的GPU前向验收，正式训练由用户启动。

## 部署验收

本地21项相关测试通过，覆盖4–8邻居的排序、补零、初始化、梯度、训练／保存／恢复和导出。服务器三个配置预检查通过，16环境的CUDA前向验收通过：初始动作最大差不超过5.96e-8，critic和随机状态与原baseline一致，固定动作的物理结果一致，优化更新次数为0。

部署来源及验收记录保存为 `docs/mappo_neighbors678_deployment_20261108.json`。未创建正式运行目录，启动命令由用户执行。

## 训练完成后的验证（2026-11-08）

三组均由用户完成1M训练，489轮PPO。完整结果及恢复文件已下载，连同新验证数据共372文件通过服务器SHA256核对；清单为 `deployment/neighbors678_results_sha256.json`。

冻结best，在32个验证场景10000–10031、动作种子880001、批次16、CUDA FP32下复验，覆盖／回报／吞吐与训练期验证一致（最大绝对误差小于1e-6）。未训练或更新模型，未使用保留测试集。

| 邻居数 | best实际步数 | 平均覆盖 | 后100步覆盖 | 吞吐Mbps |
|---|---:|---:|---:|---:|
| 6 | 540,672 | 62.313% | 64.808% | 511.60 |
| 7 | 940,032 | 59.362% | 62.718% | 490.33 |
| 8 | 890,880 | 63.054% | 66.461% | 520.82 |

[七组对照报告](../analysis/mappo_neighbors3to8_1m_vs_dppo10m_seed0/report.md)只比较baseline 1M的3–8邻居best和DPPO 10M best。原3邻居在CPU训练，本轮统一GPU评估使用56.438%的覆盖值，避免混入其历史CPU验证58.631%的不同采样结果。七组单训练种子、训练预算及actor输入参数量的差异均在报告中说明。

仅从保存数据重建报告：

```bash
.venv-dppo-compat/bin/python scripts/report_mappo_neighbors_3to8.py --rebuild
```

单独复验（在具有CUDA环境的服务器项目目录执行，将6改为7／8即可）：

```bash
source scripts/dppo_server_env.sh
.venv-dppo/bin/python scripts/report_mappo_neighbors_3to8.py --evaluate neighbors6
```

图表同时导出PNG／PDF／SVG，采用不同颜色、线型和标记区分七组，不使用紫色。
