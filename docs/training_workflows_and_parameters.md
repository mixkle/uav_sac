# SAC 与离散 MAPPO：训练流程及完整参数表

本文对应当前代码和解析后的YAML。SAC保持原场景；离散MAPPO直接改为参考论文规模，鲁棒性奖励权重为0。参数来源、补充假设与本地校准记录在下文区分。本次未启动正式训练、服务器或多种子训练。

## 1. 配置与样本计数

| 版本 | 主配置 | WSL配置（SAC 50k / MAPPO 250k） | 场景与动作 |
|---|---|---|---|
| SAC | `configs/main.yaml` | `configs/wsl_50k.yaml` | 1600² m、16 UAV、180 UE、2 BS；连续圆盘 |
| 离散MAPPO | `configs/discrete_mappo.yaml` | `configs/discrete_mappo_wsl_50k.yaml` | 3500² m、18 UAV、150 UE、3 BS；九动作 |
| 连续MAPPO兼容路径 | 连续配置指定 `--algo mappo` | 同左 | 使用SAC场景，不是新版离散任务的同场景对照 |

配置优先级为dataclass默认值、YAML显式值、适用CLI覆盖；以运行目录展开`config.yaml`为准。全部字段共90个：EnvConfig 43、AlgoConfig 27、RunConfig 20。共享离策略字段不在MAPPO路径生效。

一条联合转移是一个环境中全部UAV同时动作后的一次转移，不再乘UAV数量。16环境一次向量step计16条。SAC 50k使各环境运行3125步，共80个完整600步回合＋16个125步尾段。历史离散MAPPO 250k使各环境运行15625步，共624个完整400步回合＋16个25步尾段；当前WSL 100k为每环境6250步，共240个完整回合＋16个250步尾段。

## 2. 场景与物理参数（全部EnvConfig字段）

| 字段 | 含义 | SAC | 离散MAPPO | 单位/语义 |
|---|---|---|---|---|
| `width` | 区域宽度 | 1600.0 | 3500.0 | m |
| `height` | 区域高度 | 1600.0 | 3500.0 | m |
| `num_uavs` | UAV数量 | 16 | 18 | 架 |
| `num_users` | UE数量 | 180 | 150 | 个 |
| `capacity` | 每架最大服务人数/子带数 | 20 | 20 | 个 |
| `altitude` | 固定UAV高度 | 100.0 | 100.0 | m |
| `bs_positions` | BS坐标 | [[0.0, 0.0, 0.0], [0.0, 1600.0, 0.0]] | [[0.0, 0.0, 0.0], [0.0, 3500.0, 0.0], [3500.0, 0.0, 0.0]] | m；地面网关，不参与BS–BS转发 |
| `horizon` | 真实任务长度 | 600 | 400 | 步；到期terminated=true，不bootstrap |
| `dt` | 单步时长 | 1.0 | 1.0 | s |
| `max_speed` | 最大速度 | 10.0 | 30.0 | m/s |
| `user_speed_std` | UE高斯移动幅度标准差 | 1.0 | 1.0 | m/s；均值固定0 |
| `user_speed_clip` | UE幅度截断绝对值 | 3.0 | 3.0 | m/s；截至[-3,3] |
| `user_initialization` | UE初始化模式 | "gaussian_mixture" | "uniform" | uniform全区域均匀；gaussian_mixture截断混合 |
| `user_initial_margin_fraction` | UE初始边界留白比例 | 0.1 | 不生效（默认0.1） | 每轴区域尺寸的比例 |
| `user_cluster_std_fraction` | UE分量每轴标准差比例 | 0.08 | 不生效（默认0.08） | 每轴区域尺寸的比例 |
| `user_cluster_centers` | 等权混合分量中心 | [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]] | 不生效（默认[[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]） | 按各轴区域尺寸归一化 |
| `uav_boundary` | UAV边界动力学 | clip | reflect | 超出边界的位移逐轴反向；连续版保持裁剪 |
| `uu_range` | UAV–UAV回传距离 | 400.0 | 650.0 | m；同高度水平距离 |
| `ug_range` | UAV–BS三维回传距离 | 350.0 | 1500.0 | m；包含100 m高度 |
| `frequency_hz` | 传播计算载频 | 2000000000.0 | 2400000000.0 | Hz；上下行共用路径损耗 |
| `bandwidth_hz` | 下行总带宽 | 20000000.0 | 20000000.0 | Hz；每机均分20子带 |
| `total_power_dbm` | 每架下行总发射功率 | 30.0 | 30.0 | dBm |
| `noise_density_dbm_hz` | 噪声谱密度 | -174.0 | -174.0 | dBm/Hz |
| `noise_figure_db` | 下行接收噪声系数 | 9.0 | 15.0 | dB |
| `los_a` | LoS概率参数a | 9.61 | 9.61 | 无量纲；仰角以度输入 |
| `los_b` | LoS概率参数b | 0.16 | 0.16 | 仰角以度输入 |
| `los_extra_db` | LoS额外损耗 | 1.0 | 1.0 | dB |
| `nlos_extra_db` | NLoS额外损耗 | 20.0 | 20.0 | dB |
| `sinr_threshold_db` | 下行关联门槛 | -3.0 | -3.0 | dB |
| `rate_reference_bps` | 吞吐归一参考速率 | 5000000.0 | 5000000.0 | bit/s，每UE参考值 |
| `throughput_weight` | 吞吐奖励权重w | 0.1 | 0.1 | 无量纲 |
| `robustness_weight` | 鲁棒性奖励权重lambda | 0.5 | 0.0 | 无量纲 |
| `disconnection_weight` | 断联惩罚权重mu | 0.1 | 0.1 | 当前YAML显式起点；缺失字段的历史配置为0 |
| `reward_primary` | 奖励第一项 | coverage | coverage | 可选sensing；缺失仍为coverage，实际服务指标不变 |
| `neighbor_slots` | 最近直连邻居槽数 | 3 | 3 | 固定为3 |
| `sinr_scale_db` | 观察SNR归一化尺度 | 30.0 | 30.0 | dB；上行SNR除30，不裁到[0,1] |
| `action_space` | 动作类型 | "continuous" | "discrete9" | continuous连续圆盘；discrete9九个离散指令 |
| `uav_initialization` | UAV初始化模式 | "uniform" | "uniform" | 全区域独立均匀，不筛选连通/覆盖 |
| `launch_radius` | 历史BS出发半径 | 不生效（默认100 m） | 不生效（默认100 m） | m；uniform模式不生效，仅bs_launch使用 |
| `beacon_power_dbm` | UE上行信标功率 | 0.0 | 26.020599913279625 | dBm |
| `beacon_bandwidth_hz` | 独立信标带宽 | 180000.0 | 180000.0 | Hz |
| `beacon_noise_figure_db` | UAV信标接收噪声系数 | 9.0 | 15.0 | dB |
| `sensing_threshold_db` | 发现SNR门槛 | 5.0 | 20.0 | dB；无硬编码感知距离 |
| `backend` | 环境数值实现 | "numba" | "numba" | Numba；numpy作为参考实现 |

### 论文参数与补充假设

参考 [Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models](../Scalable_UAV_Multi-Hop_Networking_via_Multi-Agent_Reinforcement_Learning_with_Large_Language_Models.pdf) 第V-A节（PDF第11页）、表I（第12页）、图8（第16页）。

- 采用3500×3500 m、18 UAV、150 UE、3 BS、30 m/步、400步任务。图8三个BS为左下、左上、右下。
- UE采用论文允许的全区域均匀分布；UAV也独立全区域均匀采样，这是用户确认的补充假设，论文未给出UAV reset采样公式。高度100 m、每步1 s为补充假设。
- 不按初始连通、覆盖、服务成功率重采样；UE没有边界留白或聚类。`user_initialization=uniform`时高斯中心、标准差、留白均不生效。`uav_initialization=uniform`时launch_radius不生效。
- UE运动保留每秒独立有符号高斯幅度（均值0、σ=1、截断±3 m/s）及均匀方向、边界反射；没有改为论文恒速布朗运动。论文未给出可直接采用的UE速度数值。
- 载频2.4 GHz、接收噪声系数15 dB、UE信标0.4 W参考表I；LoS 9.61/0.16、额外损耗1/20 dB保持。
- 按用户确认，保留满负载同频干扰的下行SINR和独立上行发现信标。论文无干扰SNR≥25 dB不用于下行关联；没有改为统一SNR通信模型。
- 每机下行总功率仍1 W，总带宽20 MHz均分20子带：每UE链路1 MHz、50 mW。不同于论文的1 W UAV发射功率直接参与单链路预算。独立信标180 kHz也是本实验保留设置。
- UU=650 m、UG三维=1500 m为本实验的距离近似与校准选择；论文通过链路预算判断回传，没有直接给出这些硬距离。UU参考表I参数推得的约628 m尺度，UG为工程近似；BS上下行功率不同，不能称其为原文精确通信半径。

### 无线、服务与归一化

上下行共用平均dB空地路径损耗。下行干扰来自全部其余UAV，即使断网或未服务；不存在统一UAV–UE通信半径。关联同时要求当前发现、下行SINR达标、任一BS可达及容量；每UE最多一个关联、每机容量20。UE按候选数、编号排序，候选机按SINR、跳数、编号排序。

| 派生量 | SAC | 离散MAPPO |
|---|---:|---:|
| 子带带宽/功率 | 1 MHz / 50 mW | 1 MHz / 50 mW |
| 子带接收噪声 | −105 dBm | −99 dBm |
| 独立信标噪声 | −112.45 dBm | −106.45 dBm |
| UG水平直接接入半径 | 335.41 m | 994.99 m |
| 发现水平范围 | 由5 dB链路预算决定 | 约523 m，20 dB门槛 |
| 全局状态 | 6643 | 6292 |
| 自身/单邻居特征 | 21 / 22 | 23 / 24 |
| 固定观察 | 91 | 101 |
| 编码后actor输入 | 124 | 134 |
| 集中critic输入 | 双Q：6675（含32维动作） | 单V：6292 |

MAPPO无其他UAV干扰的单链路下，−3 dB达标水平距离约989 m；这只是预算上界诊断，实际满负载SINR随布局变化，不能把989 m视为服务圆半径。信标独立正交、忽略信标间干扰；断网UAV仍能发现。

局部观察为自身、全部BS先验xy、三个最近直连邻居及全部当前发现UE。UE行是相对xy/区域尺寸、上行SNR/30、本机服务标记，不按容量截断。编码器4→32→32两层ReLU、分组均值、空组0，再拼接人数/20；人数可超过1。actor不读全局状态、时间或历史累计发现，无RNN。

全局状态含节点xyz、A/B/C/Y、t/T，不添加信标矩阵或编码结果。跳数除U+1（SAC17/MAPPO19），邻居ID=(ID+1)/U，坐标按区域归一，时间分别t/600、t/400。B分别16×2、18×3；BS不作为隐藏转发节点。多个子网可分别接入任一BS。

### 奖励与故障

\[
\rho=\frac{\sum_u n_u}{M},\quad
\tau=\frac{\sum_{u,m}C_{um}R_{um}}{M v_{\rm ref}},\quad v_{\rm ref}=5\times10^6.
\]

SAC主配置：\(r=\rho+0.1\tau+0.5J-0.1D\)；离散MAPPO主配置：\(r=\rho+0.1\tau-0.1D\)，鲁棒性权重为0。WSL MAPPO当前覆写throughput_weight=0、total_steps=100000，其奖励为\(r=\rho-0.1D\)；文中250k为历史实验示例，实际预算以当前展开配置为准。

\(D=U^{-1}\sum_u\mathbf1[\min_g b_{u,g}>U]\) 是动作及拓扑更新后的断联比例。直接或多跳到任一BS都视为可达，多个子网分别接入BS不受惩罚。mu=0.1为原文未给出数值时的可调整起点，可用env.disconnection_weight或训练CLI --mu-disconnect覆盖。旧配置缺少字段时按mu=0复现，旧检查点及保存回报不改写。

奖励允许为负，不裁零；无发现奖励、运动惩罚或额外奖励归一化。日志/评估新增断联比例和实际扣分，轨迹保存[ρ,τ,J,D]及带权贡献[ρ,wτ,lambda J,−mu D]和实际reward。评估分别统计负奖励步数、连续初始零奖励和首次正奖励，不能把负奖励误称为零奖励。

J仍作诊断记录，不影响新版MAPPO目标。逐一删除UAV后重算到任一BS的可达性，固定关联和名义SINR，不重新分配用户；失败机用户丢失。\(J=U^{-1}\sum_f\rho^{(-f)}\)，\(0\le J\le(1-1/U)\rho\)。故障不会真的移除下一步环境节点。保留比 \(q=\sum_tJ_t/\sum_t\rho_t\)，零覆盖单独报告。

SAC仍保留四中心截断高斯UE，中心(480,480)/(1120,480)/(480,1120)/(1120,1120) m、标准差128 m、初始范围[160,1440]²；分量等权独立随机，越界在所属分量重采样。留白只作用于reset。两版本不增加碰撞、能耗、加速度、有限回传容量或BS故障。

### 本地物理校准（2026-10-13；不训练策略）

先用30000–30031检查32个初态，再用30000/30001各400步检查14组参数×四种布局，共112个诊断回合。扫描发现15/20/25 dB×下行−6/−3/0 dB及UU、UG分别±20%；与验证/测试隔离，不按策略成绩筛选场景。

| 下行−3 dB；32个初态 | 发现15 dB | 20 dB（默认） | 25 dB |
|---|---:|---:|---:|
| 发现水平范围近似 | 844 m | 523 m | 359 m |
| 均匀随机布局发现率 | 91.50% | 66.00% | 41.62% |
| 均匀随机布局覆盖率 | 17.92% | 17.33% | 12.79% |
| 固定连通诊断布局覆盖率 | 62.04% | 61.31% | 51.50% |

20 dB保留接近15 dB的服务能力、减少远距离不可服务发现；25 dB明显限制有效候选。默认32初态平均18架中12.16架断网，回传仍是主要瓶颈，不能保证策略容易学成。

| 默认参数；2场景×400步平均 | 静止随机初态 | 随机移动 | 追踪固定诊断布局 | 固定诊断布局静止 |
|---|---:|---:|---:|---:|
| 覆盖率 | 20.61% | 26.71% | 49.27% | 58.96% |
| 当前发现率 | 59.89% | 62.80% | 61.28% | 61.85% |

固定布局为三个角落附近六节点网格、邻机间距600 m，不读UE位置；不是reset规则、最优部署或局部baseline。UU降到520 m后该固定布局覆盖为7.90%，反映拓扑与距离门槛的可行性变化，不是算法收益。默认作为实现起点，不以2场景诊断声称论文效果。原始CSV/轨迹在 `/tmp/uav_mappo_paper_scene/calibration/`；未恢复旧训练结果。

## 3. 算法参数（所有AlgoConfig字段）

| 字段 | 含义 | SAC生效值 | 离散MAPPO生效值 | 说明 |
|---|---|---|---|---|
| `name` | 算法 | sac | mappo | SAC / 新版离散MAPPO |
| `hidden` | actor/critic主干宽度 | 256 | 256 | 两层ReLU |
| `ue_hidden` | UE集合编码器宽度 | 32 | 32 | 4→32→32，ReLU |
| `actor_lr` | actor学习率 | 0.0001 | 0.0005 | 包含UE编码器 |
| `critic_lr` | critic/value学习率 | 0.0003 | 0.0005 | SAC双Q共用优化器；MAPPO单V |
| `discount` | 折扣gamma | 0.995 | 0.995 | 均为0.995 |
| `temperature_lr` | 温度学习率 | 0.0001 | — | 仅SAC |
| `initial_alpha` | 初始温度alpha | 0.01 | — | 仅SAC；优化log(alpha) |
| `target_entropy` | 目标联合熵 | -32（自动-2U） | — | SAC未显式指定时自动-2U=-32 |
| `polyak` | 目标critic新参数权重 | 0.005 | — | SAC软更新tau；不是旧参数权重 |
| `log_std_min` | 高斯log标准差下限 | -5.0 | — | 仅SAC及连续MAPPO |
| `log_std_max` | 高斯log标准差上限 | 2.0 | — | 仅SAC及连续MAPPO |
| `batch_size` | 离策略采样batch | 256 | — | 仅SAC；MAPPO按rollout分mini batch |
| `replay_capacity` | 联合回放容量 | 50000 | — | 仅SAC，不按UAV数扩增 |
| `warmup_steps` | 学习前随机探索预算 | 10000 | — | 仅SAC，联合转移计数 |
| `transitions_per_update` | 每次更新所需新增转移 | 4 | — | 仅SAC |
| `rollout_steps` | 每环境每轮采集长度 | — | 128 | 仅MAPPO，向量步128 |
| `ppo_epochs` | 每轮PPO重复遍历次数 | — | 10 | 仅MAPPO |
| `ppo_minibatches` | 每epoch分块数 | — | 4 | 仅MAPPO，按联合转移切分 |
| `gae_lambda` | GAE lambda | — | 0.95 | 仅MAPPO |
| `ppo_clip` | 策略及价值clip尺度 | — | 0.2 | 仅MAPPO；价值在归一化尺度clip |
| `entropy_coef` | 平均单机熵系数 | — | 0.01 | 仅MAPPO，固定值，无温度学习 |
| `max_grad_norm` | 梯度范数上限 | — | 0.5 | 仅MAPPO，actor和V分别裁剪；SAC未启用裁剪 |
| `td3_policy_delay` | MATD3保留字段 | — | — | 仅MATD3；SAC/MAPPO不使用；共享字段默认2 |
| `td3_target_noise` | MATD3保留字段 | — | — | 仅MATD3；SAC/MAPPO不使用；共享字段默认0.2 |
| `td3_noise_clip` | MATD3保留字段 | — | — | 仅MATD3；SAC/MAPPO不使用；共享字段默认0.5 |
| `td3_exploration_noise` | MATD3保留字段 | — | — | 仅MATD3；SAC/MAPPO不使用；共享字段默认0.1 |

### 网络、优化器与代码固定常量

| 设置 | SAC | 离散MAPPO |
|---|---|---|
| actor主干 | 124→256→256→4：二维均值＋二维log std | 134→256→256→9 logits |
| 集中critic | 两个独立6675→256→256→1 Q；另有冻结目标双Q | 一个6292→256→256→1 V |
| 参数共享 | 所有UAV共用actor及UE编码器；critic不共享编码器 | 相同 |
| 主干激活 | 两层ReLU；输出层无激活 | 相同 |
| 主干初始化 | orthogonal；隐藏gain=√2，actor输出gain=.01，critic输出gain=1；bias=0 | 相同 |
| 编码器初始化 | PyTorch Linear默认初始化 | 相同 |
| Adam beta / weight decay | (0.9,0.999) / 0 | 相同 |
| Adam eps | actor/critic/温度均1e-8 | actor/value均1e-5 |
| SAC Q损失 | 两个Q的MSE相加，无1/2系数 | — |
| 价值损失系数 | — | 1；actor/V各自优化，无额外总损失加权 |
| 价值Huber delta | — | 10.0 |
| 优势归一化epsilon | — | 1e-5；std使用unbiased=false |
| PopArt初始统计 | — | mean=0、variance=1、count=0 |
| PopArt方差下限 | — | 1e-4，std至少.01；同步重标定输出层和Adam矩 |
| 回报/价值归一化 | 不启用 | 运行回报统计＋PopArt；全局团队V |
| 奖励归一化 | 不启用 | 不启用；归一化的是优势/价值目标 |
| actor/critic梯度裁剪 | 不启用，记录梯度范数 | 各自clip_grad_norm=.5 |
| KL提前停止 / 学习率衰减 | 不启用 / 不启用 | 不启用 / 不启用 |
| BatchNorm / LayerNorm / dropout / RNN | 均不使用 | 均不使用 |
| 精度 | 物理float64；网络、原始观察与回放float32；偏移整数 | 相同；整数动作int64 |
| AMP / torch.compile | 不启用 | 不启用 |
| 优化器foreach | 未显式设置；交由PyTorch选择，未强制开启 | 相同 |

## 4. SAC训练流程

1. 创建16环境、共享actor、集中双Q、冻结目标双Q与log温度；前10000条用圆盘面积均匀随机探索。
2. 采二维高斯z，log std截[-5,2]；径向映射 \(a=z/\sqrt{1+\|z\|^2}\)，\(\log|\det J|=-2\log(1+\|z\|^2)\)。每步指令10a m，边界裁剪；确定性评估映射均值。
3. 原始当前/下一观察、状态、动作、奖励、真终止标志复制入联合回放；下一观察在reset前保存。容量50000、batch256、有放回采样，不缓存旧编码。
4. warmup后每新增4条联合转移更新一次；16环境一步通常4次更新。
5. 下一观察重新编码、停止梯度计算目标，更新双Q：
   \[
   y=r+\gamma(1-d)\left[\min_jQ'_j(s',\mathbf a')-\alpha\sum_u\log\pi_u(a'_u|o'_u)\right].
   \]
6. 冻结Q参数、保留动作梯度，重新采样更新actor与编码器：
   \[
   L_\pi=\mathbb E[\alpha\sum_u\log\pi_u(a_u|o_u)-\min_jQ_j(s,\mathbf a)].
   \]
7. 用停止梯度的联合log概率更新温度：
   \[
   L_\alpha=-\mathbb E[\log\alpha(\sum_u\log\pi_u+\mathcal H_{\rm tar})],
   \quad\mathcal H_{\rm tar}=-32.
   \]
   随后软更新 \(\phi'\leftarrow0.995\phi'+0.005\phi\)。无目标actor或编码器。
8. 真终止不bootstrap；按日程验证、日志和受控内存保存。

WSL 50k：10000条探索＋40000条学习采集，10000次更新；每次分别更新双Q优化器、actor、温度及目标Q。联合熵求和，不除以U。

## 5. 离散MAPPO训练流程

世界坐标前=+y、右=+x，\(c=1/\sqrt2\)。

| ID | 动作 | 归一指令(dx,dy) |
|---:|---|---|
| 0 | 前 | (0,1) |
| 1 | 后 | (0,-1) |
| 2 | 左 | (-1,0) |
| 3 | 右 | (1,0) |
| 4 | 左前 | (-c,c) |
| 5 | 左后 | (-c,-c) |
| 6 | 右前 | (c,c) |
| 7 | 右后 | (c,-c) |
| 8 | 悬停 | (0,0) |

八个移动动作均30 m/步，斜向每轴30/√2 m；无航向或边界动作掩码。悬停时UE和时间仍推进。接受整数[U]/[E,U]，浮点/非法ID/形状拒绝；边界概率对应原编号。训练和默认验证均Categorical采样，对环境输出整数动作。验证动作随机种子固定为880001，并隔离训练随机流。显式确定性诊断仍支持argmax，平局最小ID。UAV越界位移逐轴反射，不再裁剪停在边界。

1. 创建共享分类actor、集中单V、独立优化器，无回放、warmup、SAC温度或目标网络。
2. 冻结当前策略采128步×16环境＝2048条，存原始集合、状态、整数动作、旧log概率、原始价值、reset前下一状态价值、奖励及结束标志。
3. 反向团队GAE：
   \[
   \delta_t=r_t+\gamma(1-d_t)V_{\rm old}(s_{t+1})-V_{\rm old}(s_t),\quad
   A_t=\delta_t+\gamma\lambda_{\rm GAE}c_tA_{t+1},\quad R_t=A_t+V_{\rm old}(s_t).
   \]
   真终止无bootstrap；外部截断可bootstrap但切断跨回合GAE；片段边界用下一价值、尾carry为0。联合优势归一化，所有agent共享。
4. 更新回报统计与PopArt，重标定V输出层及优化器矩；旧值/目标用同一新统计归一化。
5. 10 epochs，每epoch随机分4份；正常minibatch512联合转移含18架动作，共9216个agent概率项。重新编码集合，逐agent计算：
   \[
   r_{t,u}=\exp[\log\pi_\theta(a_{t,u}|o_{t,u})-\log\pi_{\rm old}(a_{t,u}|o_{t,u})].
   \]
   不乘全部agent概率比。actor损失：
   \[
   L_\pi=-\mathbb E_{t,u}\min[r_{t,u}\tilde A_t,
   \operatorname{clip}(r_{t,u},0.8,1.2)\tilde A_t]-0.01\,\mathbb E_{t,u}\mathcal H(\pi_\theta).
   \]
   精确熵 \(-\sum_{k=0}^8p_k\log p_k\)，单机最大log9≈2.1972 nats；对样本/agent取均值。
6. V损失取当前值和clip值的Huber(delta10)较大者；归一化价值相对旧值clip±.2。actor/V分别更新、梯度裁剪.5。编码器只由actor更新。
7. 完成后丢弃rollout、换新策略；完整保存只在更新安全点。停止信号先完成已收集片段更新，失败不发布新恢复点。

250k=122×2048＋144；末轮9向量步、minibatch36条，共123轮PPO、4920次actor和4920次V优化；不补足251904条。

连续MAPPO兼容路径仍使用高斯actor：二维均值与两个状态无关共享log std（初始0），径向映射及Jacobian同SAC。PPO保存latent z，连续熵由高斯熵加采样log Jacobian估计。PPO/GAE相同，物理场景由配置决定。SAC与新版离散MAPPO场景已不同，不能直接把训练差异归因于算法/动作；应另外建立同场景连续对照。

## 6. 运行与评估参数（所有RunConfig字段）

两种算法的评估策略不同，WSL训练预算也不同；环境字段已分开。下表区分主配置与WSL，避免把dataclass默认总步数200万误认为当前YAML主预算100万。

| 字段 | 主配置实际值 | WSL实际值（SAC / MAPPO） | 语义 |
|---|---|---|---|
| `seed` | 0 | 0 | 唯一训练seed；本次不启动多种子 |
| `device` | auto | cpu | auto优先可用CUDA；WSL指定CPU |
| `num_envs` | 16 | 16 | 并行环境数 |
| `vector_backend` | batch | batch | batch单进程批量物理计算；process可选 |
| `cpu_threads` | 1 | 8 | PyTorch intra-op线程数，不是环境进程数 |
| `numba_threads` | 4 | 4 | Numba线程上限；进程worker各1线程 |
| `total_steps` | 1000000 | 50000 / 250000 | 完整联合转移预算，需整除num_envs |
| `eval_interval` | 50000 | 10000 | 评估阈值；MAPPO于越过阈值的更新边界执行 |
| `validation_seeds` | 10000–10031（32个） | 10000–10031（32个） | 32个固定验证场景 |
| `test_seeds` | 20000–20099（100个） | 20000–20099（100个） | 100个保留测试场景 |
| `log_interval` | 1000 | 1000 | 日志阈值；更新边界记录实际步数 |
| `checkpoint_minutes` | 30.0 | 10.0 | 完整检查点墙钟间隔 |
| `max_wall_seconds` | 未设置 | 未设置 | 运行时间上限；未设则不触发 |
| `hourly_price` | 未设置 | 未设置 | 可选每小时租价；不代表当前服务器价格 |
| `budget` | 未设置 | 未设置 | 可选成本上限；要求同时提供hourly_price |
| `tensorboard` | true | true | 启用TensorBoard |
| `memory_limit_gib` | 12.0 | 12.0 | 进程RSS上限，超过明确停止 |
| `evaluation_policy` | SAC: deterministic / MAPPO: sample | SAC: deterministic / MAPPO: sample | 验证动作选择规则 |
| `evaluation_action_seed` | 880001 | 880001 | 采样验证固定动作种子，与训练流隔离 |
| `deterministic` | true | true | 启用PyTorch确定性算子；不是策略argmax开关 |

固定实现：Torch inter-op线程1；deterministic=true时启用确定性算子、关闭CuDNN benchmark，并在未设置时指定CUBLAS_WORKSPACE_CONFIG=:4096:8；TF32未在代码中显式指定，按PyTorch后端设置。物理Numba线程受实际NUMBA_NUM_THREADS上限限制。模型仍随机采样训练；`run.deterministic=true`只限制计算可复现性。

训练各环境初始化seed由SeedSequence([训练seed,101])分别派生；每场景UE移动随机流由SeedSequence([场景seed,201])独立派生。验证与训练Torch/NumPy/Python随机流隔离，不改变后续训练采样。校准使用30000起的独立种子；不是默认测试集。

### 验证协议和保存

- SAC默认确定性验证；离散MAPPO默认采样验证，动作种子880001。评估配置保存模式和种子；相同场景顺序、批量大小及设备下可复现，验证前后恢复训练随机状态。正式测试使用独立100场景。旧argmax曲线不能与新采样曲线混为同一协议。
- 初始模型在step0验证；SAC WSL在10000/20000/30000/40000/50000验证。MAPPO WSL每跨过10000条的阈值，在PPO更新边界验证，如10240/20480/30720/40960/51200；持续至250000，最终模型也验证，记录实际转移数。
- `best.pt`按32场景平均团队回报选择，`last.pt`在验证及完整保存前写入。保留最佳及最后模型，不能只报告最有利的指标。
- 主配置30分钟、WSL10分钟墙钟触发完整保存，另在正常结束/可处理停止时保存；保留最近2个完整目录。MAPPO在更新安全点触发。
- SAC完整检查点每块最多256条联合转移，CPU Tensor分块、SHA256校验、临时目录写完原子发布、逐块恢复；包含回放原始集合。MAPPO无回放，index为空，仅完整元数据与模型状态。
- 完整状态包括模型/优化器、SAC温度或MAPPO PopArt统计、环境、原始当前观察、随机状态、采样/更新游标、历史最佳模型和选择记录。`last.pt`/`best.pt`单独不能完整续训。
- 模型版本beacon-set-v2；离散映射world-nine-v1/categorical，连续映射unit-disk-v1/gaussian。动作空间、映射或旧观察格式不兼容时拒绝恢复。
- 相对输出在WSL位于项目目录；AutoDL自动落到 `/root/autodl-tmp/uav_sac/`，或用 `UAV_DATA_ROOT`指定。显式绝对路径优先，不连接训练服务器。

### Baseline、消融和小配置

静止保持随机初态；连续随机为圆盘面积均匀，离散随机九动作等概率含悬停。局部启发式只读相同局部观察，离散版本将建议向量投影到最近合法指令，平局最小编号。

`configs/discrete_mappo_no_robustness.yaml` 当前为CPU 250k入口，w=0、lambda=0、mu=0.5，与主配置的奖励权重不同；不应当作仅改变鲁棒性权重的对照。SAC消融用CLI `--lambda-rob 0`，不改观察或通信。

### UE运动与奖励目标对照（2026-10-19）

新增六份`configs/discrete_mappo_ablation_*.yaml`，均为16环境、种子0、250000联合转移，完整主场景3500×3500 m、18 UAV、150 UE、3 BS；全部w=lambda=0。实验仅改变以下三项，其余场景与MAPPO更新参数完全一致。

| 组别/文件后缀 | `user_speed_std` | `reward_primary` | `disconnection_weight` | 奖励 |
|---|---:|---|---:|---|
| a_mobile_connected | 1.0 | coverage | 0.5 | rho−0.5D |
| b_static_connected | 0.0 | coverage | 0.5 | rho−0.5D |
| c_mobile_coverage | 1.0 | coverage | 0.0 | rho |
| d_mobile_sensing | 1.0 | sensing | 0.0 | P |
| e_mobile_sensing_connected | 1.0 | sensing | 0.5 | P−0.5D |
| f_static_coverage | 0.0 | coverage | 0.0 | rho |

`user_speed_clip=3.0`显式冻结。P为当前被任意UAV感知的去重UE比例，不要求联网，不受下行SINR或容量约束，不是累计发现率。D组只用P；coverage日志始终保留真实联网服务比例。未指定reward_primary时所有历史配置仍使用覆盖项。

六组从头训练，独立输出目录，禁止跨奖励/运动设置续训。主要比较同预算last，另报各自奖励选出的best。固定32个验证场景和采样协议；首轮仅单种子。本次实现仅进行小配置与每组2048条主场景转移的本地验收，不自动开启250k。

完整启动、回退、奖励分解数据格式和结果解释见[六组实验说明](ue_motion_reward_ablations.md)。单组绘图工具已支持覆盖/感知目标；汇总入口`scripts/report_ue_reward_ablations.py`接受A–F六个目录，并兼容旧A–D四目录命令。

`configs/smoke.yaml` / `configs/discrete_mappo_smoke.yaml`：400×400 m，4 UAV、24 UE、容量6、UU200 m、UG180 m、horizon32，BS为(0,0,0)/(0,400,0)。小SAC仍为高斯UE，小离散MAPPO改为均匀UE且鲁棒性权重0；二者UAV均匀。小配置10 m步长、双BS与原无线参数为显式流程验证覆盖，不是新版同参缩小实验。主干32，batch32、回放512、warmup64（仅SAC生效）、rollout8、PPO epochs2/minibatches2（仅MAPPO）；2环境，CPU1线程、Numba2线程，总256、验证128、日志64，验证10000/10001、测试20000/20001，TensorBoard关闭。

`configs/local_validation.yaml`：保持主物理场景，CPU、2环境、Torch1/Numba2线程，SAC batch64、replay4096、warmup256，总4096，验证2048、日志256，验证10000–10003、TensorBoard关闭。它们是流程验证设置，不是主实验超参数。

## 7. 从头训练命令

离散MAPPO的WSL配置当前预算为100000、吞吐权重为0；此前250000为历史实验预算。文件名`discrete_mappo_wsl_50k.yaml`保留以兼容旧命令；SAC仍为50000，评估间隔和其余训练参数保持原值。

```bash
# 原SAC，连续动作；原场景
.venv/bin/python -m uav_research train --config configs/wsl_50k.yaml --output runs/wsl_sac_disconnect_50k_seed0

# 新版论文规模离散MAPPO；250k，从头训练，使用新目录
.venv/bin/python -m uav_research train --config configs/discrete_mappo_wsl_50k.yaml --output runs/wsl_discrete_mappo_disconnect_100k_seed0

# 新场景检查和参数扫描（不学习）
.venv/bin/python -m uav_research check-env --config configs/discrete_mappo.yaml --output runs/mappo_paper_scene/check.json
.venv/bin/python -m uav_research calibrate --config configs/discrete_mappo.yaml --output runs/mappo_paper_scene/calibration
```

改变区域、节点数和BS数量后旧MAPPO检查点不能用于新版续训，须从头训练。actor导出仍包含编码器、主干与缩放，不依赖全局UE数量。上述仅是使用命令，本次只做临时目录中的正确性及短流程验收。

本地验收：89项pytest通过；主场景CPU单线程、2环境，先256条（1轮标准128步PPO）后完整恢复到512条（2轮），保留256宽网络/10 epochs/4 minibatches，回合仍400步。临时验证仅1个校准场景30020，另30021完成评估、导出和绘图。过程峰值RSS约836 MiB；这是小批次流程结果，不代表16环境正式训练速度或内存。临时产物在 `/tmp/uav_mappo_paper_scene/pipeline/`，无正式训练成绩。
