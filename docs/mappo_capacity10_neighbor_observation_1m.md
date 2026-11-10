# 容量10 baseline：4／5邻居观测对照

## 实验定义

以 `configs/mappo_coverage_throughput_capacity10_1m.yaml` 为来源，两组均从零训练种子0、1,000,000条联合转移，正式训练由用户启动。沿用共享连续高斯圆盘 actor、相对坐标、一步历史和时间因子 critic；不继承旧模型或优化器。

| 项目 | 原 baseline | 邻居4组 | 邻居5组 |
|---|---:|---:|---:|
| `env.neighbor_slots` | 3 | 4 | 5 |
| 当前固定特征 | 101 | 125 | 149 |
| 含一步历史的固定观测 | 105 | 129 | 153 |
| 集合编码后的 actor 输入 | 138 | 162 | 186 |
| Actor 参数量 | 103,620 | 109,764 | 115,908 |
| Critic 输入 | 6292 | 6292 | 6292 |
| Critic 参数量 | 1,677,057 | 1,677,057 | 1,677,057 |

每个邻居包含24维特征，只读取650 m直连范围内的 UAV，按距离、编号排序，数量不足时补零。增加槽位不会扩大通信距离，不会让 UAV 看到范围外的节点。

奖励保持

\[
r_t=\rho_t+0.1\tau_t,\qquad
\tau_t=\frac{\sum_m R_m^t}{150\times5\,\mathrm{Mbps}}.
\]

鲁棒性、断联和边界惩罚权重均为0。3500×3500 m、18 UAV、150 UE、3 BS、容量10、UE中心2500×2500 m初始化及后续运动、400步回合、30 m/步、镜面反射全部保持不变。

学习参数保持 baseline：actor／critic 学习率均5e-4，折扣0.995、GAE 0.95、PPO clip 0.2、价值裁剪开启、Huber delta 10、熵系数0.01、10 epochs、4 minibatches、梯度裁剪0.5。16环境×128步，每组489轮PPO，末轮576条联合转移，每个优化器19,560次更新。CPU8线程、Numba4线程，每组单GPU。

## 初始化控制与兼容

按原3邻居 baseline 的顺序、种子生成**未训练** actor 和 critic。随后在 actor 第一层插入新增邻居特征列，初始权重为零，其余权重保持对应位置；新增列正常参与训练。历史、UE编码和人数特征的列随之右移。这样初始策略函数在数值容差内与3邻居 baseline 一致，critic 初值及后续随机流保持一致。

每增加一个邻居，actor 第一层增加6144个可训练权重。实验同时改变可用邻居信息和输入层容量，不能单独将改善归因于信息量。相同初值函数也不保证不同宽度的浮点矩阵运算产生逐位相同的完整采样轨迹。

原3邻居路径不变；快照与 actor 导出记录完整配置和输入维度。跨邻居数量直接续训会被拒绝。回退使用原 baseline 配置即可，旧结果保留。

## 服务器启动

项目、环境、数据和缓存均位于 `/home/dym/uav_sac/`。新增配置：

- `configs/mappo_capacity10_neighbors4_1m_server.yaml`
- `configs/mappo_capacity10_neighbors5_1m_server.yaml`

它们与原 baseline 的差异仅为邻居数量和设备 `cuda:0`。默认同时启动两个独立训练任务，4邻居用物理GPU0／CPU0–7，5邻居用物理GPU2／CPU12–19，以匹配服务器NUMA拓扑。每个进程内部只看到一张卡，因此配置均使用 `cuda:0`。

只检查环境、配置、磁盘和输出目录，不启动训练：

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_neighbors_server.sh --check
```

由用户执行，分别在两个tmux会话从零启动1M训练：

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_neighbors_server.sh
```

指定其他两张卡／CPU集合：

```bash
UAV_NEIGHBOR_GPUS=1,3 UAV_NEIGHBOR_CPUSETS='0-7|12-19' \
bash scripts/launch_mappo_neighbors_server.sh
```

查看训练：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-neighbors4
# 或
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-neighbors5
```

按 `Ctrl+B`，再按 `D` 脱离会话，训练继续。也可单独启动一组：

```bash
CUDA_VISIBLE_DEVICES=0 UAV_CPU_AFFINITY=0-7 \
bash scripts/train_mappo_neighbors_server.sh 4
```

输出分别为：

- `/home/dym/uav_sac/runs/mappo_capacity10_neighbors4_1m_seed0/`
- `/home/dym/uav_sac/runs/mappo_capacity10_neighbors5_1m_seed0/`

控制台日志写入项目 `logs/mappo_neighbors{4,5}_1m_日期.log`。脚本拒绝覆盖已有非空运行目录，不自动续训，也不加载旧权重。

## 评估口径

沿用32个验证场景10000–10031、动作采样种子880001、批次16、约每10k完整更新边界验证。best按原两项任务回报选择并包含step 0。主要比较相同1M预算的last，best作为补充；与10M baseline的差异应另列预算影响。

重点检查全回合／最后100步／终态覆盖和吞吐、覆盖骤降、断联、部署维持、靠边与聚集；补充每架UAV实际可见邻居数及第4／5槽位的非空率，确认新增信息在访问的布局中是否确实存在。奖励中的鲁棒性和断联权重为0不妨碍记录这些诊断。

验证单个检查点示例（在服务器项目目录执行）：

```bash
source scripts/dppo_server_env.sh
.venv-dppo/bin/python -m uav_research evaluate \
  --checkpoint runs/mappo_capacity10_neighbors4_1m_seed0/last.pt \
  --device cuda:0 --split validation \
  --output analysis/mappo_capacity10_neighbors4_1m_seed0/last
```

5邻居组及best模型替换路径即可。保留测试集不用于本轮开发诊断；单种子结论不能替代多训练种子验证。

## 本地与部署验收

`tests/test_neighbor_observation_ablation.py`覆盖配置差异、4／5邻居排序和补零、NumPy／Numba、固定动作物理一致性、初始化和随机状态配对、新列及UE编码器梯度、小配置训练／恢复／导出／评估，以及跨模式恢复拒绝。

本地新增及相关测试21项通过，桥接、历史观测、时间因子、独立actor及SAC历史回归38项通过，共59项。服务器两组均通过16环境的CUDA初始化／前向验收：初始动作最大绝对差为5.96e-8，critic及随机状态一致，固定动作下物理结果一致，优化更新次数为0。

可重新运行无训练验收（物理GPU由环境变量选择）：

```bash
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=0 .venv-dppo/bin/python scripts/check_mappo_neighbors.py \
  --neighbors 4 --output deployment/mappo_neighbors4_cuda_check.json
```

部署记录另存 `docs/mappo_neighbors_deployment_20261108.json`。服务器只运行预检查和无优化的CUDA初始化／前向验收，不启动两组正式1M训练。

## 训练完成后的统一验证（2026-11-08）

两组均完成1M，完整结果和恢复目录已下载，232个文件逐一SHA256校验。五组比较见[验证报告](../analysis/mappo_neighbors_1m_vs_10m_seed0/report.md)：仅包含3／4／5邻居baseline 1M、baseline 10M best、DPPO 10M best。

本轮重新执行1M三组的best和last共6次32场景CUDA验证；10M两组复用同口径且模型SHA256一致的保存评估。原3邻居1M训练及历史验证在CPU；本轮主表统一使用CUDA，其best平均覆盖56.438%，与历史CPU口径58.631%不同，同一整数种子在两个设备上不产生相同高斯样本。两份数据均保留并明确标注。

4／5邻居best分别为620,544／980,992步，CUDA验证覆盖60.276%／60.680%，吞吐504.83／506.66 Mbps；last覆盖59.364%／59.405%。新增邻居信息具有改善服务能力的迹象，但5邻居的极端骤降仍未消除。当前结果为单训练种子，且3邻居与新增组训练设备、随机流和输入层容量不同，不能将全部变化因果归于邻居数量。

只读重建五组报告，不重新采样：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python scripts/report_mappo_neighbors.py --rebuild
```

报告入口也支持 `--evaluate neighbors4 --checkpoint best` 等独立冻结验证。默认输出 `analysis/mappo_neighbors_1m_vs_10m_seed0/`，保留原始轨迹、逐步指标、MC标签、来源哈希、场景级配对区间和PDF／SVG／PNG图表。
