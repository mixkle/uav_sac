# UE 运动与奖励目标：六组可回退对照

继续以原 C 组为基准的 [解析时间因子／平均覆盖率 critic 校准](critic_time_factor_experiment.md)，冻结 actor、复用原 MC 数据，对比直接价值与两种时间因子损失；独立于在线 MAPPO 和此处 A–F 训练。

另设 [中央 2500×2500 m UE 初始化实验](central_ue_initialization_experiment.md)，仅修改 C 组的 UE 初始分布，从头训练 250k；不并入此处 A–F 控制变量矩阵。

C 组同时作为 [critic Value Clipping 单变量对照](critic_value_clipping_ablation.md)的基准。该实验只关闭价值裁剪，不改变此处六组配置，也不加入原 A–F 汇总脚本。

后续 [冻结 actor、MC Return 拟合 critic](critic_mc_coverage_experiment.md)同样以原 C 组 last 为起点，检查现有 MLP 的拟合与泛化；属于独立离线诊断，不并入六组在线训练结果。

## 实验问题与配置

本实验分别检查UE运动、断联惩罚和服务资格对学习的影响。保持现有MAPPO、九动作、观察、关联与无线模型，通过配置选择任务。`env.reward_primary`缺失时使用`coverage`。

定义实际服务覆盖率为 \(\rho\)、断联UAV比例为 \(D\)、瞬时去重感知率为 \(P\)。所有组均设置吞吐权重w=0、鲁棒性权重lambda=0。

| 组别 | 配置文件（位于configs/） | UE速度标准差 | reward_primary | mu | 每步奖励 |
|---|---|---:|---|---:|---|
| A | discrete_mappo_ablation_a_mobile_connected.yaml | 1 m/s | coverage | 0.5 | rho−0.5D |
| B | discrete_mappo_ablation_b_static_connected.yaml | 0 | coverage | 0.5 | rho−0.5D |
| C | discrete_mappo_ablation_c_mobile_coverage.yaml | 1 m/s | coverage | 0 | rho |
| D | discrete_mappo_ablation_d_mobile_sensing.yaml | 1 m/s | sensing | 0 | P |
| E | discrete_mappo_ablation_e_mobile_sensing_connected.yaml | 1 m/s | sensing | 0.5 | P−0.5D |
| F | discrete_mappo_ablation_f_static_coverage.yaml | 0 | coverage | 0 | rho |

B−A只改变UE运动；C−A只取消断联惩罚；D−C只替换奖励第一项。E−D检查感知目标下的断联惩罚，E−A在相同断联惩罚下比较感知与覆盖；F−C检查纯覆盖下的UE运动，F−B检查静止UE下的断联惩罚。不要把D−A称为单变量对照。本轮不增加静止UE×感知奖励交叉组。

六组均为3500×3500 m、18 UAV、150 UE、3个角落BS，UAV与UE独立均匀初始化；高度100 m、每步1 s、九动作最大位移30 m、边界反射、400步真正终止。UU=650 m、UG三维=1500 m。保留满负载下行SINR及独立上行信标。完整参数显式保存在六份配置中，仅reward_primary、user_speed_std、disconnection_weight三项不同。

所有组从头训练，种子0、250000条联合转移、CPU、16环境、PyTorch8线程、Numba4线程；每轮16×128=2048条转移、PPO 10 epochs/4 minibatches，最后9个向量步采集剩余144条，共123轮更新。actor/critic学习率5e−4、gamma=0.995、GAE lambda=0.95、PPO clip=0.2、熵系数0.01、梯度裁剪0.5。每约10k转移在完整更新边界验证。网络和价值损失保持原设置。

## 奖励与运动的准确含义

在动作执行、UE移动和拓扑更新后的状态计算

\[
r_t=R_{\rm primary}(s_{t+1})+w\tau_{t+1}+\lambda J_{t+1}-\mu D_{t+1},
\quad
R_{\rm primary}=\begin{cases}\rho,&\texttt{coverage},\\P,&\texttt{sensing}.\end{cases}
\]

\[
P=\frac1M\sum_m\mathbf1\{\exists u:\mathrm{SNR}^{\rm beacon}_{u,m}\ge\Gamma_{\rm sense}\}.
\]

P是当前所有UAV感知集合的并集大小除以M。同一UE被多架UAV发现只计一次；UAV断网也可以贡献P；下行SINR与容量不参与感知判定。累计发现仅作诊断，不作为奖励。D/E组仍计算真实服务关联和回传，以便评估实际覆盖。E组在P基础上扣除0.5D，断联惩罚是负贡献。

移动UE延用每步独立高斯速度幅度及均匀方向，幅度截断±3 m/s，越界反射。B/F组设置`user_speed_std=0`，仍按原流程消费运动随机数。相同场景种子生成相同初始位置；A/C/D/E的UE移动序列相同，不随UAV动作改变。记录UE相对reset位置的位移，以检查运动实际有多大。

## 启动与回退

在项目根目录依次运行，六组输出目录独立。以下为正式命令，本次验收不自动执行250k训练；已有A–D结果保留，新增E/F单独从头训练。

```bash
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_a_mobile_connected.yaml --output runs/mappo_ablation_a_mobile_connected_250k_seed0
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_b_static_connected.yaml --output runs/mappo_ablation_b_static_connected_250k_seed0
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_c_mobile_coverage.yaml --output runs/mappo_ablation_c_mobile_coverage_250k_seed0
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_d_mobile_sensing.yaml --output runs/mappo_ablation_d_mobile_sensing_250k_seed0
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_e_mobile_sensing_connected.yaml --output runs/mappo_ablation_e_mobile_sensing_connected_250k_seed0
.venv/bin/python -m uav_research train --config configs/discrete_mappo_ablation_f_static_coverage.yaml --output runs/mappo_ablation_f_static_coverage_250k_seed0
```

正式训练后，验证并生成六组对照报告：

```bash
.venv/bin/python scripts/report_ue_reward_ablations.py --runs \
  runs/mappo_ablation_a_mobile_connected_250k_seed0 \
  runs/mappo_ablation_b_static_connected_250k_seed0 \
  runs/mappo_ablation_c_mobile_coverage_250k_seed0 \
  runs/mappo_ablation_d_mobile_sensing_250k_seed0 \
  runs/mappo_ablation_e_mobile_sensing_connected_250k_seed0 \
  runs/mappo_ablation_f_static_coverage_250k_seed0 \
  --output analysis/ue_reward_ablations_250k_seed0
```

可用`report_mappo_training.py --run ... --output ...`分析单组。汇总工具接受按A–F顺序的六个目录，也继续兼容原A–D四目录命令；参数数量为其他值时明确报错。工具检查实验参数、其他场景/训练设置、实际完成预算和配对场景；混入不同配置时拒绝比较。它加载best/last进行验证，不启动训练。

回退到原配置即可恢复原任务；`reward_primary: coverage`恢复覆盖项，`user_speed_std: 1.0`恢复运动，`disconnection_weight: 0.5`恢复A组奖励。禁止改动后跨组续训，应新建目录从头训练。完整检查点保存配置与环境/优化器/RNG。旧配置和权重仍可读取；旧CSV缺少新增诊断列，旧完整检查点继续同一任务时应恢复到新目录，避免向旧CSV追加新schema。

## 日志、评估和解释

- `coverage`始终是实际联网服务比例，`discovery_fraction`是瞬时感知率；配置中的reward_primary是字符串，数值日志中的同名字段记录当前选中的奖励值。
- 训练日志记录目标、覆盖/感知/断联、UE位移、UAV最近邻距离及100 m内有邻居的UAV比例，并保留损失、分类熵、KL、clipping和梯度。训练指标为日志时刻最后一个向量步的均值，不是rollout均值。
- 验证历史和`episodes.csv`包含全回合均值、终态、最后100步覆盖/感知/断联；同时记录UE运动与聚集诊断。
- `reward_curves.npz`的`raw`第一列随目标切换，由`raw_names`标识；`contributions`是实际有符号奖励贡献，行和等于训练奖励。另存逐步实际覆盖/感知/断联和位移等诊断。
- 轨迹的`reward_terms`保持历史物理含义`[rho,tau,J,D]`；`reward_contributions`第一项是实际选中的奖励，由`reward_contribution_names`标识。物理项权重`reward_weights`在感知模式下第一项为0，额外`sensing_reward_weight=1`。总奖励可用`reward_terms@reward_weights+sensing_reward_weight*discovery_fraction`复核。
- 图输出PNG/PDF/SVG，曲线另有CSV。拓扑固定选择前两个验证场景，展示初始/中间/终态；绿色UE已服务、紫色仅感知、灰色未感知，不按表现挑选。

正式验证沿用10000–10031共32个场景，保留20000–20099共100个测试场景。统一采样评估、动作种子880001、场景顺序、批量与CPU配置，隔离训练随机流。

主要比较相同250k预算的last；best按自身奖励选择并单列，跨奖励定义不按总回报排名。汇总CSV包含均值/终态/最后100步指标及last按场景的B−A、C−A、D−C、E−D、E−A、F−C、F−B差值；单训练种子不展示种子置信区间。

B优于A支持运动增加难度；C优于A支持当前惩罚限制覆盖优化；若D感知提升而C覆盖不提升，继续拆分回传、下行干扰和容量条件；若D也学不好，继续检查策略优化、观察和协作。单种子结果不能确定唯一根因。

## 本地验收

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_ue_reward_ablations.py
```

第二条命令仅运行六组小场景流程及主场景每组2048条联合转移。小场景检查训练、精确恢复、评估、导出和绘图；主场景保留16环境、256宽网络、400步回合和全部PPO参数，仅覆盖预算、日志/验证间隔、验证种子30030/30031与TensorBoard开关。覆盖项写入产物，不使用测试集。

六组验收产物默认位于`analysis/ue_reward_ablation_acceptance_six_groups_20261019/`，原四组产物保留在`analysis/ue_reward_ablation_acceptance_20261019/`。目录包含`acceptance.json`、实际配置、检查点和对照图。2048条转移仅一轮PPO，这些结果用于实现检查，不能视为正式实验成绩。

2026-10-19验收：139项pytest全部通过（26.93秒）；四组主场景各2048条转移、一轮PPO，单组训练及内部验证约6.3–6.7秒，整个进程峰值RSS约1193 MiB。小场景四组的连续训练与完整恢复后权重、Adam/PopArt状态精确一致，导出及绘图通过。B组UE位移严格为0；其余组配对场景UE位移一致。加载已有mu=0.5覆盖奖励best模型，全部32个验证场景的回报、覆盖、感知和断联率与历史记录完全相同。未启动正式250k、服务器或多种子训练。

E/F增补验收：扩展后的针对性测试28项全部通过（15.19秒），覆盖六组配置、奖励、静止UE、环境执行路径、精确续训及四/六目录校验。E和F主场景各运行2048条联合转移、一轮PPO，保存/验证/绘图通过；六组汇总及原四组命令兼容检查通过。新产物在`analysis/ue_reward_ablation_extension_ef_20261019/`，A–D复用此前验收结果，未重新训练；这些短流程结果不用于判断正式效果。

## B best：冻结策略的动作温度诊断

检查“收紧同一个策略的动作分布，能否维持已形成的部署”。加载B的best（200704条联合转移），冻结actor，保持环境、奖励和网络权重，采样分布改为`Categorical(logits=logits/T)`。比较T=1、0.7、0.4、0.2及argmax；argmax平局取最小编号。训练时的熵系数没有变化。

```bash
.venv/bin/python scripts/evaluate_policy_temperature.py
```

默认输出`analysis/ue_reward_ablations_250k_seed0/b_best_temperature/`。使用原32个验证场景10000–10031，四种采样温度各使用动作种子880001–880005；argmax没有采样随机性，每个场景只执行一次，共672个完整回合，不使用保留测试集。不同温度配对场景、动作种子、环境批量与执行顺序。T=1、880001复核原验证的逐场景覆盖、整体回报及前两个场景的完整轨迹；检查点执行前后SHA256保持一致。

输出逐场景/动作种子的指标、逐时刻曲线、拓扑、分类熵/最大概率/悬停概率和二维指令位移方差。`fixed_reference_states.csv`额外在T=1访问的相同状态上计算各温度的概率指标，用于区分分布变尖与后续状态分布改变。位移方差为边界反射前的指令位移协方差迹，单位m²。

配对差值先对五个动作种子求场景均值，再按32个场景bootstrap，区间仅表示这些动作重复下的场景差异，不是训练种子置信区间；argmax只计算一次，不把重复执行当作独立样本。同时报告采样种子之间的结果波动。重点看最后100步覆盖、终态覆盖、联网感知范围、贴边与多跳节点，不仅看全回合奖励；两个拓扑场景固定为10000/10001，使用动作种子880001，不按结果挑选。

降低温度后若维持明显改善，支持采样随机性参与退化；若更快贴边或覆盖下降，说明更偏好的动作也可能有问题。该实验改变评估策略及后续访问状态，不能单独证明训练中的熵系数过大，也不会自动修改正式训练配置。

本次672回合已完成：[完整报告与图表](../analysis/ue_reward_ablations_250k_seed0/b_best_temperature/report.md)。T=1的五动作种子平均覆盖38.64%、最后100步35.88%；T=0.7/0.4/0.2的最后100步覆盖分别32.08%/29.01%/28.30%，argmax为33.15%。终态联网UAV贴边比例由T=1的15.48%升至29.36%/43.62%/44.85%，argmax为34.58%。温度确实降低了动作熵与指令方差，但未维持较好的服务布局。原动作种子880001仍准确复现此前39.64%的覆盖率，五种子均值与旧单次数字不同来自采样重复。

## D纯感知：critic准确性诊断

2026-10-20完成：[报告与图表](../analysis/ue_reward_ablations_250k_seed0/d_critic_audit/report.md)。冻结第0步best、180224步、250000步的actor/critic，分别使用32个验证场景×3动作种子采集完整回合，按G_t=r_t+0.995G_(t+1)、G_400=0计算实际回报。V使用PopArt反归一化后的原始尺度。另在last的48个固定状态上各独立续采样16次，随机化未来动作和UE运动；本次没有校准网络或更新策略。

```bash
.venv/bin/python scripts/diagnose_sensing_critic.py
```

主要发现：last最后一个动作前V均值91.18、实际最后一步回报0.544，96条完整验证轨迹的V全部超出单步奖励上界1。前100步V/回报均值100.80/102.10，最后100步为92.28/23.83，表现为有限回合后期严重高估。相同状态重复采样确认该偏差远大于随机误差。固定物理状态，仅改变时间输入，V从t=0的100.42降至t=399的97.28，时间敏感性很弱。

已核对奖励确为感知率、状态时间输入存在、动作前V与动作后奖励对齐、真正终止不bootstrap、原验证重放一致。诊断GAE不等于历史训练rollout，critic大误差尚不能单独证明actor所有更新方向错误。下一步应先解决价值预测/剩余时长校准，再用受控更新验证其对策略的影响；本次未改动训练参数。
