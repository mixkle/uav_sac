# Baseline 1M：断联惩罚与鲁棒性奖励对照

目的是检查新增奖励能否降低覆盖骤降，同时保留平均覆盖、后期覆盖和数据率。复用原容量10、覆盖＋数据率1M baseline，新增三组种子0、从零训练1,000,000条联合转移的实验。正式训练由用户启动。

## 实验矩阵

令`rho`为实际联网服务覆盖，`tau`为总吞吐除以`150 × 5 Mbps`，`D`为无法接入任意BS的UAV比例，`J`为固定UE关联、分别移除每架UAV后得到的覆盖率的平均值。

| 组别 | 训练奖励 | 断联权重mu | 鲁棒权重lambda |
|---|---|---:|---:|
| 原baseline 1M（复用） | rho + 0.1 tau | 0 | 0 |
| disconnection | rho + 0.1 tau − 0.1 D | 0.1 | 0 |
| robustness | rho + 0.1 tau + 0.5 J | 0 | 0.5 |
| combined | rho + 0.1 tau + 0.5 J − 0.1 D | 0.1 | 0.5 |

权重为首轮诊断选择，不自动扫描或调整。mu=0.1时，18架中的一架断联产生约0.00556单步成本，约等于0.83个UE的覆盖奖励。lambda=0.5沿用原完整奖励的鲁棒权重；`0 ≤ J ≤ (17/18) rho`，最大额外贡献为约0.4722 rho。

鲁棒项本身与覆盖相关：`rho + lambda J = (1+lambda) rho − lambda(rho−J)`。因此奖励该项同时加强服务覆盖与故障保留，不能将所有收益解释为单独改善冗余结构。假想的单UAV故障鲁棒性也不保证实际移动导致的多链路失效得到解决。

## 控制变量与选模

以`configs/mappo_coverage_throughput_capacity10_1m.yaml`为原配置。三份新配置只改变两个新增奖励权重及设备`cuda:0`，其余设置相同：3500×3500m、18 UAV、150 UE、3 BS、容量10、中心2500×2500m UE初始化、移动UE、400步回合、30m/步、反射边界、共享连续高斯actor、相对坐标与一步历史、时间因子critic。

维持原PPO、GAE、PopArt、价值裁剪、熵正则、学习率和16环境×128步设置。CPU8线程、Numba4线程。每组489轮PPO，末轮576条联合转移，每个actor与critic共19560次优化。不读取旧baseline或其他实验权重、Adam或环境状态。

环境已有的完整奖励实现直接用于本轮，新增项通过GAE和critic训练目标影响actor；不额外添加不可微的“断联loss”。固定动作时，新增权重不改变无线、位置、关联、观察或UE随机流。

best仍按各组自身训练奖励的全回合验证回报选择，包含step0。**主要比较相同1M预算last，best为补充**。跨组比较统一保留`rho + 0.1 tau`的共同回报，不能直接比较各组不同定义的训练回报。验证场景10000–10031，采样种子880001；保留测试集不参与本轮诊断。

历史baseline在WSL CPU训练，新组在服务器CUDA训练；冻结模型评估统一在同一设备与采样口径下进行，不混用历史CPU报告值。初始网络参数仍用种子0配对，跨CPU/CUDA不承诺逐位相同训练轨迹。

## 三卡并行启动

服务器项目、虚拟环境、缓存、临时文件、结果和日志均位于`/home/dym/uav_sac/`，不安装新的系统环境。使用已有`.venv-dppo`。

登录服务器后执行以下一条命令即可创建三个独立的tmux会话，并行从零训练：

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_stability_rewards_server.sh
```

| 组别 | 默认物理GPU | CPU亲和范围 | tmux会话 |
|---|---:|---|---|
| disconnection | 0 | 0–7 | mappo-stab-disconnection |
| robustness | 1 | 8–15 | mappo-stab-robustness |
| combined | 2 | 16–23 | mappo-stab-combined |

这三组是独立进程，不是多卡数据并行；每组内部仅可见一张GPU，逻辑设备均为`cuda:0`。默认CPU范围按已检查的服务器映射为互不共享物理核的8核组；第二组跨CPU插槽，可能影响运行速度，不改变采样预算和学习参数。

可改GPU映射，仍要求三张不同显卡：

```bash
UAV_STABILITY_GPUS=1,2,3 bash scripts/launch_mappo_stability_rewards_server.sh
```

可通过`UAV_STABILITY_CPUSETS='0-7|8-15|16-23'`显式指定CPU列表。脚本先对三个配置、GPU、CPU范围和新输出目录全部预检查，再启动训练。CPU范围不可重叠；若同名会话或已有结果存在则拒绝启动，不自动覆盖或续训。

只做预检查、不启动任何训练：

```bash
bash scripts/launch_mappo_stability_rewards_server.sh --check
```

查看某组：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-stab-disconnection
```

按`Ctrl+B`再按`D`离开，训练继续。另两组替换会话名即可。单独启动某组可执行：

```bash
CUDA_VISIBLE_DEVICES=0 UAV_CPU_AFFINITY=0-7 \
bash scripts/train_mappo_stability_rewards_server.sh disconnection
```

各组输出为`runs/mappo_capacity10_stability_<组名>_1m_seed0/`，日志为`logs/mappo_stability_<组名>_1m_*.log`。完整恢复机制沿用baseline，保留最近两份完整检查点；轻量best/last不能直接用于恢复。

## 训练后的统一比较

新增报告入口只接受原1M baseline及上述三组；默认评估last、best：

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=0 .venv-dppo/bin/python -u scripts/report_mappo_stability_rewards.py \
  --output analysis/mappo_capacity10_stability_rewards_1m_seed0
```

不重新采样，直接由保存的评估数据重建报告：

```bash
.venv-dppo/bin/python scripts/report_mappo_stability_rewards.py \
  --output analysis/mappo_capacity10_stability_rewards_1m_seed0 --rebuild
```

预先固定的主要指标：

- 每回合最大单步覆盖跌幅；最大5步端点跌幅；峰值到之后任一状态的最大回撤。
- 单步跌幅≥10个百分点的次数与发生场景比例；另报5、20个百分点阈值。
- 全回合、后100步和终态覆盖，最低20步滑动平均覆盖，完整吞吐与共同任务回报。
- 平均断联、故障保留覆盖，以及早期51–150步与后100步的绝对覆盖及其差值。

统计使用动作后t1–t400，排除随机初始化到首动作的t0→t1。下降越少必须与覆盖、数据率一起判断；低覆盖或停止展开可能让骤降数字看起来更小。

每场景生成一行指标，以场景为单位做2000次配对bootstrap（种子950001），不是训练种子置信区间，不把时刻当作独立样本。单种子仅用于定位问题。常规报告使用灰色baseline与红／蓝／橙三组，不使用绿紫对照。

图表、逐场景指标、共同回报、各训练奖励贡献和来源SHA256均保存。当前代码交付阶段只做小配置正确性、保存恢复与CUDA前向验收，不运行正式1M。

## 已完成训练后的六组比较（2026-11-08）

三组现均完成1M训练；完整结果和恢复文件已下载。新报告按后续指令仅在主图比较这三组best与baseline 10M best、DPPO 10M best、baseline 10M teacher，新增三组last单独补充。新旧预算和选模目标不同，不能把六组差值单独解释为奖励项的因果效果。

报告见[六组验证](../analysis/mappo_stability_six_group_1m_vs_10m_seed0/report.md)。新增best/last在服务器以同样32场景、采样种子880001、批次16、CUDA FP32复验，训练回报与日志一致；原三个参考复用来源校验一致的GPU评估。覆盖、吞吐和跨组共同回报按原两项指标计算，训练奖励另行保留。

本地只读取保存数据重建六组报告：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python scripts/report_stability_six_group_comparison.py \
  --output analysis/mappo_stability_six_group_1m_vs_10m_seed0
```

原四组1M配对报告入口仍保留；本次六组主报告未将原1M baseline额外画成第七组。
