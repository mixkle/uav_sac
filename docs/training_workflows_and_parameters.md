# 当前训练入口与参数（2026-11-03）

## 研究 baseline

当前基准配置：[`configs/mappo_coverage_throughput_capacity10_1m.yaml`](../configs/mappo_coverage_throughput_capacity10_1m.yaml)。容量10、覆盖＋0.1归一化数据率奖励，共享连续MAPPO＋时间因子critic＋一步历史。种子0的1M训练已完成；best来自671,744步，32场景平均覆盖58.63125%。结果见[验证报告](../analysis/mappo_coverage_throughput_capacity10_1m_seed0/analysis_report.md)。后续固定布局关联诊断以此模型为准。

`mappo_full_reward_baseline_1m.yaml` 继续保留为历史容量20、四项完整奖励配置，文件名不再表示当前工作基准；其模型未用于本次新诊断。下表列出当前容量10、两项奖励基准。

### 场景与无线

|设置|数值|
|---|---|
|区域、UAV、UE|3500×3500 m；18；150|
|BS|(0,0,0)、(0,3500,0)、(3500,0,0)|
|高度、UAV初始化|100 m；全场独立均匀，不按连通性筛选|
|UE初始化|中心[500,3000]²均匀；margin fraction=1/7|
|UE移动|每秒独立高斯幅度std=1，截断±3 m/s，均匀方向，边界反射|
|回合、时间步、UAV速度|400步真终止；1 s；最大30 m/s|
|UAV边界|反射；连续高斯＋圆盘映射|
|机间／UAV–BS距离|650 m／三维1500 m|
|频率／总带宽／总功率|2.4 GHz／20 MHz／30 dBm每UAV|
|容量|10 UE/UAV；子带宽度2 MHz、每子带功率20 dBm|
|噪声谱密度／噪声系数|−174 dBm/Hz／15 dB|
|LoS参数|9.61、0.16；LoS/NLoS附加损耗1/20 dB|
|下行门槛|SINR≥−3 dB；满负载同频干扰|
|发现信标|26.0205999133 dBm；180 kHz；NF15 dB；SNR≥20 dB|
|速率归一化参考|5 Mbps/UE|
|邻居|最近三个直连UAV，按距离、编号排序|
|观察|relative＋one_step；固定105维，集合编码后138维|
|全局状态|6292维，包含t/400；不含actor历史|

### 当前基准奖励

\[
r=\rho+0.1\tau.
\]

`rho`为实际获得联网服务的UE比例；`tau=总吞吐/(150×5 Mbps)`。当前并非纯覆盖奖励，吞吐也不是未经缩放的bps。鲁棒性权重0、断联惩罚权重0、边界惩罚权重0；故障保留、断联和感知仍作诊断记录。历史完整奖励配置则为 `rho+0.1*tau+0.5*J_rob-0.1*D`，不应混入当前基准的奖励统计。

### 网络与优化

|参数|设置|
|---|---|
|Actor|18 UAV共享；UE编码器4→32→32、均值聚合＋人数；主干256×256 ReLU|
|动作|状态相关二维均值/log std；log std截到[-5,2]；径向圆盘变换|
|Critic|6292→256→256→1；time_factor；V=H_t×预测平均任务奖励|
|学习率|actor/critic均5e-4；Adam eps=1e-5|
|折扣／GAE lambda|0.995／0.95|
|PPO clip／Value Clipping|0.2／开启，在累计价值归一化空间执行|
|价值损失|Huber delta=10；两套归一化及平均奖励PopArt|
|熵系数|0.01；只加入actor损失，critic标签不含熵|
|epochs／minibatches|10／4|
|梯度裁剪|0.5；优势归一化开启|
|环境／rollout|16／128；通常每2048条联合转移更新|
|预算|1,000,000条联合转移，种子0；489轮PPO、19560次actor与critic优化|
|最后rollout|剩余576条转移＝36个向量步；不补足2048|
|执行|CPU，PyTorch8线程，Numba4线程，batch向量环境|
|验证|场景10000–10031；采样策略；动作种子880001；每约10k的更新完成边界|
|best|验证全回合task_return，即覆盖＋0.1归一化数据率之和；包含step0|
|保存|轻量best/last；约每10分钟完整恢复目录，保留最近两份；RSS限制12 GiB|
|保留测试|20000–20099；不参与训练选模|

MAPPO不使用SAC的warmup、回放池、温度或目标熵。新配置省略了这些不生效的算法字段。

## 容量10的独立对照配置

新增 [`configs/mappo_full_reward_capacity10_1m.yaml`](../configs/mappo_full_reward_capacity10_1m.yaml)，有效配置与完整奖励baseline相比仅将 `env.capacity` 从20改为10。仍使用共享连续MAPPO、时间因子critic、完整四项奖励、种子0和1M联合转移，独立输出；正式训练由用户启动。

沿用现有模型中容量与资源的绑定：每UE子带宽度从1 MHz变为2 MHz，每子带功率从约16.99 dBm变为20 dBm，子带噪声功率也翻倍。同一物理布局下，由于信号、满负载干扰和噪声同步翻倍，下行SINR保持不变，候选链路速率翻倍。总带宽20 MHz、总功率30 dBm、速率参考5 Mbps和奖励权重均保持原值。因此这不是仅收紧接纳名额、同时保持每链路速率不变的消融；吞吐项的数值尺度也会受到影响。

服务人数、历史感知人数和集合编码器人数特征的容量尺度同步从20变为10，固定观察105维、全局状态6292维保持不变。正式比较应从头训练，不继承容量20模型或Adam。若只在原固定物理快照上诊断容量10的匹配损失，可以不重新训练，但结论只代表这些已有布局；不能当作容量10策略的训练结果。固定布局诊断支持用 `--reuse-trajectories` 指定原诊断目录、用 `--audit-config` 指定容量10配置；这种重审计不运行actor，因此也不把新容量观察输入旧策略。具体命令见 [容量诊断说明](topk_capacity_validation.md)。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python -u -m uav_research train \
  --config configs/mappo_full_reward_capacity10_1m.yaml \
  --output runs/mappo_full_reward_capacity10_1m_seed0
```

18架UAV总容量由360降到180，仍大于150名UE；是否形成足够的局部竞争，需要重新测量每机超容量比例和多候选UE比例，不能预设评分分支一定变得可学习。

### 容量10＋覆盖和数据率（已完成，当前基准）

配置：[`configs/mappo_coverage_throughput_capacity10_1m.yaml`](../configs/mappo_coverage_throughput_capacity10_1m.yaml)。沿用容量10完整奖励配置的场景、网络、优化、种子0和1M步预算，仅将 `robustness_weight`、`disconnection_weight` 均置为0；吞吐权重保持0.1，边界惩罚保持0。环境奖励为：

\[
r=\rho+0.1\tau,\qquad \tau=\frac{\text{总吞吐}}{150\times5\,\mathrm{Mbps}}.
\]

本实验采用共享连续高斯圆盘MAPPO与时间因子critic，UE关联仍为原集中式规则。仅改变奖励权重，不加入扩散actor或学习评分。Critic、GAE和best选模都使用这份两项奖励；鲁棒性、断联继续作为诊断记录。容量10的每链路2 MHz及功率/速率联动保持前述定义。

该运行已由用户从头完成训练，未加载容量20、容量10完整奖励或历史纯覆盖权重/Adam。以下是原始启动命令；现有输出目录已包含正式结果，不应覆盖，如需重复实验应另选输出目录：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python -u -m uav_research train \
  --config configs/mappo_coverage_throughput_capacity10_1m.yaml \
  --output runs/mappo_coverage_throughput_capacity10_1m_seed0
```

原 `mappo_full_reward_capacity10_1m.yaml` 继续表示容量10的四项奖励对照，两份配置使用不同输出目录。

## 手动训练与恢复

建议在tmux中运行，关闭终端不会终止会话：

```bash
tmux new -s mappo_baseline
cd /home/mixkle/uav_sac
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python -u -m uav_research train \
  --config configs/mappo_full_reward_baseline_1m.yaml \
  --output runs/mappo_full_reward_baseline_1m_seed0
```

`Ctrl+B`后按`D`退出tmux界面，`tmux attach -t mappo_baseline`重新连接。已有输出目录不能不带resume直接重新开始。

完整恢复使用最新完整目录，不使用轻量last.pt；下例需替换实际目录：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python -u -m uav_research train \
  --config runs/mappo_full_reward_baseline_1m_seed0/config.yaml \
  --output runs/mappo_full_reward_baseline_1m_seed0 \
  --resume runs/mappo_full_reward_baseline_1m_seed0/resume_XXXXXXXXXXXX \
  --steps 1000000
```

`--steps`为累计目标，不是额外增加的步数。不要使用旧纯覆盖检查点续训完整奖励baseline。

## 验证与绘图

```bash
.venv/bin/python -m uav_research evaluate \
  --checkpoint runs/mappo_full_reward_baseline_1m_seed0/last.pt \
  --device cpu --split validation \
  --output analysis/mappo_full_reward_baseline_1m_seed0/last

.venv/bin/python -m uav_research evaluate \
  --checkpoint runs/mappo_full_reward_baseline_1m_seed0/best.pt \
  --device cpu --split validation \
  --output analysis/mappo_full_reward_baseline_1m_seed0/best

.venv/bin/python -m uav_research plot \
  --runs runs/mappo_full_reward_baseline_1m_seed0 \
  --evaluations analysis/mappo_full_reward_baseline_1m_seed0/last analysis/mappo_full_reward_baseline_1m_seed0/best \
  --traces analysis/mappo_full_reward_baseline_1m_seed0/last/trace_10000.npz \
  --output analysis/mappo_full_reward_baseline_1m_seed0/figures
```

以同预算last为主要结论，best为补充；分别检查覆盖、吞吐、故障保留、断联与完整回报，避免用总奖励提升代替覆盖提升。

## 其他保留配置

- `mappo_coverage_shared_1m.yaml`：复现已有纯覆盖共享组。
- `mappo_coverage_independent_1p2m.yaml`：复现已有纯覆盖独立组。
- `sac_coverage_time_factor_relative_central_history_1m.yaml`：纯覆盖SAC，CPU。
- `sac_coverage_time_factor_relative_central_history_server_1m.yaml`：同实验CUDA版本。

它们与完整奖励baseline的任务权重不同；结果不能作为同奖励的严格算法对照。当前baseline采用共享连续actor，不是独立actor，也不是九动作版本。


## 集合条件化扩散DPPO（2026-11-05新增）

独立主配置 `configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml`，环境全部字段与当前容量10 baseline相同，奖励为覆盖＋0.1归一化吞吐。新算法采用5维UE集合、共享8步DDPM、Top-K服务提议与UE选择，先教师示范去噪预训练，再在线1M联合转移；旧MAPPO/SAC继续保留。

| 参数 | 扩散DPPO设置 |
|---|---|
| 观察/状态 | 固定105维＋5维UE集合；critic仍6292维 |
| 集合/上下文 | 5→64→64，masked mean＋log1p人数/log11；170→128→128 |
| 扩散 | 8步，余弦0.008，β≤0.999，反向std≥0.05，末步随机 |
| 示范 | 最新baseline best671744步；128拟合＋32选模场景，64000条联合转移 |
| BC | 50轮、batch1024、lr3e-4、两分支按各自维度平均后等权 |
| 在线PPO | actor1e-4、critic5e-4；clip0.1、value clip0.2；10 epochs×4 minibatches |
| 价值 | baseline时间因子、PopArt、Huber10及Value Clipping；γ0.995、GAE0.95 |
| 熵/在线BC | 0/0 |
| 预算 | 种子0、1M，489轮，末轮576条，每个优化器19560次更新 |
| WSL/服务器 | CPU8/Numba4/微批32；单3090、CPU4/Numba8/微批8192（完整rollout实测选择，仅改变梯度累积） |

命令、恢复与服务器准备见[扩散DPPO实验说明](diffusion_dppo_experiment.md)。已上传至 `/home/dym/uav_sac/`，CUDA短测与性能记录见[服务器验收](diffusion_dppo_server_deployment_20261105.md)；正式训练由用户启动。采集和预训练成本另计；新旧关联机制及观测均变化，不能把结果差异只归因于actor架构。

## 容量10 baseline的稳定性奖励对照（2026-11-08）

以`mappo_coverage_throughput_capacity10_1m.yaml`为来源，新增三组从零训练1M的服务器配置：`mappo_capacity10_stability_disconnection_1m_server.yaml`、`mappo_capacity10_stability_robustness_1m_server.yaml`、`mappo_capacity10_stability_combined_1m_server.yaml`。覆盖权重1、吞吐权重0.1；断联权重分别0.1/0/0.1，鲁棒权重分别0/0.5/0.5。

保留共享高斯actor、时间因子critic及原全部学习和物理参数，仅运行设备改为单GPU。三组可分别占用GPU0/1/2、CPU0–7/8–15/16–23同时训练；各组CPU8、Numba4线程。独立输出与tmux会话，不是多卡联合更新一个模型。

一键启动、预检查、单组启动及覆盖骤降报告命令见[稳定性奖励实验说明](mappo_capacity10_stability_rewards_1m.md)。主要比较相同1M预算last，best为补充；同时报告覆盖、吞吐和共同回报，避免用不同训练目标的奖励总值直接比较优劣。正式训练由用户启动。

## 容量10 baseline的邻居观测对照（2026-11-08）

新增`mappo_capacity10_neighbors4_1m_server.yaml`与`mappo_capacity10_neighbors5_1m_server.yaml`：可观察直连邻居槽位分别由3增加到4／5，其余学习与物理参数保持容量10、覆盖＋0.1归一化吞吐的1M baseline，鲁棒性和断联权重均为0。固定观测由105变为129／153，actor输入162／186，critic保持6292维。新增输入列零初始化、正常训练，保留原baseline初始策略函数、critic初值和随机流。

两组各从零训练种子0、1M，默认分别使用GPU0／2，每组CPU8、Numba4线程。一键tmux启动、单组启动、验证与回退见[邻居观测实验说明](mappo_capacity10_neighbor_observation_1m.md)。正式训练由用户启动。

进一步新增6／7／8邻居各1M配置：`mappo_capacity10_neighbors6_1m_server.yaml`、`mappo_capacity10_neighbors7_1m_server.yaml`、`mappo_capacity10_neighbors8_1m_server.yaml`。固定观测177／201／225维，actor输入210／234／258维，critic保持6292维；其余沿用容量10、覆盖＋0.1吞吐baseline，鲁棒性和断联权重为0。服务器三组独立tmux启动命令见[6／7／8邻居实验说明](mappo_capacity10_neighbors678_1m.md)。

## 3邻居断联惩罚强度对照（2026-11-08）

新增容量10 baseline的三组1M从零训练配置，断联权重分别0.5、2、5；覆盖权重1、吞吐权重0.1，鲁棒性和边界权重0，邻居数保持3。其余场景、策略、时间因子critic与学习参数不变。三卡独立tmux启动：

```bash
cd /home/dym/uav_sac
bash scripts/launch_mappo_disconnection_sweep_server.sh
```

默认GPU0／1／2、各PyTorch8线程／Numba4线程；正式训练由用户启动。说明、单组命令、输出目录及比较口径见 [断联权重对照](mappo_disconnection_weight_sweep_1m.md)。

## 2026-11-08：跳数缩放、联合服务评分的10M双组实验

新增Diffusion DPPO与Gaussian服务评分MAPPO，容量10、三邻居、中心UE、相同Top-K；奖励rho+.1tau。两组alpha_min=.25、k=.4，动作前跳数缩放运动，历史使用反射后实际位移/30。均采用actor/critic lr1e-4/5e-4、actor/value clip .1/.2、熵0；其他时间因子critic、GAE与PPO设置保持。每组10M、4883轮PPO、195320次各优化器更新，16×128。扩散重新采集64000条示范并预训练50轮，Gaussian从零；额外成本单列。旧配置维持原行为。详见[新实验及命令](hop_scaling_joint_service_comparison.md)。
