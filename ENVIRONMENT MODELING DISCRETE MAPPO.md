# 九动作离散 MAPPO：纯覆盖奖励基准

本文件以当前 **C 组：MAPPO＋九动作离散空间＋移动 UE＋仅联网服务覆盖奖励**的代码为基准，独立定义物理环境、观察、奖励与学习规则。基准配置为 [discrete_mappo_ablation_c_mobile_coverage.yaml](configs/discrete_mappo_ablation_c_mobile_coverage.yaml)，对应历史运行目录 `runs/mappo_ablation_c_mobile_coverage_250k_seed0/`。核对日期：2026-10-20。

本文所称“基准”明确使用 \(r_t=\rho^{t+1}\)，吞吐、鲁棒性、断联惩罚权重均为零。UE 仍然移动；静止 UE 的纯覆盖版本为 F 组，属于对照实验。通用代码保留其他配置，但文件名为 `discrete_mappo.yaml` 或 `discrete_mappo_no_robustness.yaml` 的配置不代表本文基准。

物理与观察建模源自连续版本 [ENVIRONMENT MODELING.md](ENVIRONMENT%20MODELING.md)，本文以当前离散实现为准。本次基准整理不将独立 critic 校准实验或评估策略温度实验中的改动并入训练方法。


## 一、场景

无人机（Unmanned Aerial Vehicles, UAVs），通常也称为 drones，正越来越广泛地被部署为**移动感知平台和通信平台**。

其中一个重要的应用场景，是在地面通信基础设施因自然灾害而受损，或者在大型人员密集活动中因负载过高而拥塞时，利用自主 UAV 作为能够**快速部署的空中中继节点**。

在森林火灾监测和战场监视等场景中，UAV 还可能工作在超出操作人员直接控制范围的区域，因此需要具备**机载自主决策能力（onboard autonomy）**以及 **UAV 之间的点对点协同能力（peer-to-peer coordination）**。

在这些场景中，UAV 集群必须自主决定各自应部署在什么位置，从而最大化对**动态目标区域**的感知覆盖或通信覆盖。

许多多 UAV 覆盖与部署任务都可以从最大覆盖选址问题**（Maximum Coverage Location Problem, MCLP）**的角度进行描述；然而，该问题属于 **NP-hard 问题**。

此外，真实部署环境还具有两个重要特征。

一方面是**部分可观性**，即 UAV 只能获取通信范围内的邻机信息和信标可感知的地面用户信息；全部 BS 坐标作为预先已知信息提供。

另一方面是**通信约束**，即UAV–UE需要满足发现SNR及下行SINR门槛，UAV–UAV和UAV–BS回传使用距离门槛近似；UE还需经服务UAV接入任一BS。

为了解决该问题，多智能体强化学习**（Multi-Agent Reinforcement Learning, MARL）**提供了一套良好的方法



## 二、CTDE

本方案使用 **集中式训练、分布执行（CTDE）**架构。

在训练阶段，集中式 critic 可以访问全局信息，以提高训练过程的稳定性；在执行阶段，每架 UAV 运行一个**共享策略（参数共享的策略网络）**，其输入仅包括自身的局部观察（其中 BS 坐标为预先已知信息）。



## 三、系统模型

本文考虑应急通信场景，地面接入设施受损或覆盖不足，需要多架无人机在空中形成多跳网络。模型中的三个 BS 是仍可用的核心网接入点；可移动地面用户通过 UAV 获得服务，再经直接或多跳回传接入任一 BS。

### A. 网络模型

**实验实例约定：** 主实验区域为 \(3500\times3500\,\mathrm m\)，\(U=18,M=150,G=3\)。BS位于 \((0,0,0),(0,3500,0),(3500,0,0)\)，对应参考论文图8的左下、左上、右下。全部BS可接入核心网且不失效；可有多个互不连接的UAV子网，只要各自可达任一BS。BS仅为路由终点，不增加BS–BS隐藏转发边。参数以本文末尾的基准参数表及 C 组配置为准。

场景规模、数量、UE均匀分布、30 m/步及400步任务参考 *Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models* 第V-A节、表I和图8。UE和UAV在整个[0,3500]²内独立均匀采样，不留边界空白、不按覆盖或连通性筛选。论文允许UE均匀或多中心高斯，本版选择均匀；UAV均匀reset是用户确认的补充假设，论文未给出采样公式。高度100 m、时间步1 s亦为补充假设。静止baseline保持同一随机初态。当前模式均为uniform；高斯参数与launch_radius不生效。

任务400步、每步1 s，八个移动动作均30 m。UE保留每秒独立有符号高斯幅度（σ=1、截断±3 m/s）、均匀方向与物理边界反射；不是原论文的恒速布朗模型。UU距离650 m，UG三维距离1500 m（水平约1496.7 m）为本实验回传距离近似，没有照搬统一SNR回传计算。

通信保留原下行SINR及独立发现信标，未改为论文无干扰SNR模型。上下行载频2.4 GHz，接收噪声系数15 dB；LoS参数9.61/0.16、额外损耗1/20 dB。每UAV总功率1 W/30 dBm，20 MHz均分20子带，每UE链路1 MHz/50 mW，满负载同频干扰；这不同于原论文1 W直接参与单链路预算。信标功率0.4 W/26.02 dBm、带宽180 kHz、门槛20 dB（发现水平范围约523 m）；信标与下行正交、信标间干扰忽略。下行门槛−3 dB，真实服务范围受布局和干扰影响。无线参数选择与独立联合扫描见参数文档。

本基准奖励仅为 \(r_t=\rho^{t+1}\)。只有 UE 实际关联到可接入核心网的 UAV 才计入覆盖。单纯发现 UE 或 UAV 恢复联网均不单独产生奖励。吞吐、断联率和固定关联单 UAV 故障保留覆盖仍可用于诊断；不增加感知奖励、势函数奖励、运动惩罚或能耗约束。MAPPO 学习参数保持现有 C 组设置。

#### 1. 网络节点定义

令 \[ \mathcal{G}=\{1,2,\ldots,G\} \] ，表示所有可用基站构成的集合，各基站的坐标位置表示为：
$$
p_{g}=(x_{g},y_{g},0), \qquad \forall g=1,2,\ldots,G.
$$
令 \[ \mathcal{U}=\{1,2,\ldots,U\} \] ，表示所有 UAV 构成的集合，各 UAV 的坐标位置表示为：
$$
p_{u}=(x_{u},y_{u},h_{u}), \qquad \forall u=1,2,\ldots,U.
$$
由于已有大量研究对 UAV-BS 的最优飞行高度进行了研究，并且一旦最优高度确定，其变化相对于 UAV-BS 的最优二维水平位置来说通常较小。

因此本文假设所有 UAV 均在相同高度飞行，即 \[ h_{u}=h. \]

令 \[ \mathcal{M}=\{1,2,\ldots,M\} \] ，表示所有 地面用户UE 构成的集合，各 UE 的坐标位置表示为：
$$
p_{m}=(x_{m},y_{m},0), \qquad \forall m=1,2,\ldots,M.
$$

#### 2. 实体距离 

任意两实体之间的距离使用 **欧式距离**。

实体 \[e_1\] 与 \[e_2\] 之间的三维距离可以表示为：
$$
d_{{e_1},{e_2}} = \sqrt{(x_{e_1}-x_{e_2})^2 + (y_{e_1}-y_{e_2})^2+(h_{e_1}-h_{e_2})^2}.
$$

#### 3. 连接变量

令 \[a_{{u_i},{u_j}}\] 表示 UAV \[u_i\] 与 UAV \[u_j\] 的连接关系，因此，
$$
a_{{u_i},{u_j}} = \begin{cases} 1,&u_i\text{ 连接 }u_j\\ 0,&u_i\text{ 不连接 }u_j \end{cases}
$$
所有的 UAV 连接关系变量组合为 **UAV 连接矩阵**： \[\bold A\] ，其中 \[\bold A\]为 U * U 矩阵，每个元素代表具体两架 UAV 的连接关系，对角线元素为1.

令 \[b_{u,g}\] 表示 UAV \[u\] 与 基站 \[g\] 的最短连接路径跳数关系（直接或多跳间接），因此，
$$
b_{u,g} = \begin{cases} 
1, &u\text{ 可直接接入基站 }g\\
k, &u\text{ 可通过最短 }k\text{ 跳接入}g \\
U+1, &u\text{ 不可接入基站 }g 
\end{cases}
$$
所有 UAV-BS 连接关系变量组合为 **UAV 接入矩阵**：\[ \bold B \] ，其中 \[ \bold B \] 为 U * G 矩阵。

令 \[c_{m}^{u}\] 表示 UAV \[u\] 与 地面用户 \[m\] 的连接关系。因此，
$$
c_{u,m} = \begin{cases} 1,&m\text{ 接入 }u\\ 0,&m\text{ 不接入 }u \end{cases}
$$
所有的 UAV-UE 连接关系组成 **服务关联矩阵**：\[ \bold C \] ，其中 \[ \bold C \] 为 U * M 矩阵。

### B. 通信模型

#### 1. LoS 和 NLoS 

无线信号传播被划分为两类链路：一类是**视距（LoS）链路**，另一类是**非视距（NLoS）链路**。

LoS 和 NLoS 链路出现的概率可以分别表示为：
$$
P_{e_1,e_2}^{\mathrm{LoS}}
=
\frac{1}{
1+a\exp[-b(\theta_{e_1,e_2}-a)]
}
$$

$$
P_{e_1,e_2}^{\mathrm{NLoS}} = 1 - P_{e_1,e_2}^{\mathrm{LoS}}
$$

\[a\] 和 \[b\] 是与具体环境传播有关的参数，而 \[\theta\] 为 \[e_1\] 和 \[e_2\] 之间的仰角，代入此公式时单位为度。代码使用 LoS/NLoS 概率加权的平均路径损耗，不逐步随机采样遮挡状态。

LoS 和 NLoS 情况下的路径损耗(dB)分别表示为：
$$
\zeta_{e_1,e_2}^{\mathrm{LoS}}
=
20\log d_{e_1,e_2}
+
20\log f
+
20\log\left(\frac{4\pi}{c}\right)
+
\eta_1,
$$

$$
\zeta_{e_1,e_2}^{\mathrm{NLoS}}
=
20\log d_{e_1,e_2}
+
20\log f
+
20\log\left(\frac{4\pi}{c}\right)
+
\eta_2.
\tag{4}
$$

其中，\[f\] 表示载波频率，\[c\] 表示光速，\[\eta_1\] 和 \[\eta_2\] 分别表示 LoS 和 NLoS 传播路径所产生的额外损耗。以上路径损耗公式的 \(\log\) 均为以10为底，距离使用米、频率使用 Hz。

#### 2. UAV-UE 通信

无人机 UAV 与 地面用户 UE 之间的**平均路径损耗**(dB)可以表示为：
$$
\zeta_{u,m} = 
P_{u,m}^{\mathrm{LoS}}\zeta_{u,m}^{\mathrm{LoS}} +
P_{u,m}^{\mathrm{NLoS}}\zeta_{u,m}^{\mathrm{NLoS}}
$$

#### 3. UAV-UAV 与 UAV-BS 通信

UAV–UAV 和 UAV–BS 回传在代码中采用固定距离门槛。作为门槛近似的物理解释，可考虑自由空间传播损耗：
$$
\zeta_{u_1,u_2}
=
20\log d_{u_1,u_2}
+
20\log f
+
20\log\left(\frac{4\pi}{c}\right)
$$

#### 4. 带宽分配

假设系统的总带宽为 \[B\] ，并采用**频分复用(FDM)**。

若一台 UAV 最多可以同时服务 \[N\] 个 UE，则 UAV 分配给每个 UE 的带宽为：
$$
B_m = \frac{B}{N},
\qquad
\forall m=1,2,\ldots,M.
$$

#### 5. 下行 SINR 与上行发现 SNR

**(1) UAV-UE 下行 SINR**

各 UAV 内部采用正交频分接入，不同 UAV 复用同一接入频段。采用固定子带功率和满负载干扰近似，候选链路的干扰不随实际关联人数及子带占用改变。即使某架 UAV 未联网或没有服务用户，也仍作为干扰源参与计算。

假设多台 UAV 使用相同频段为 UE 提供服务，因此不同 UAV 之间会产生**同频干扰**。

于是，UAV 与 UE 之间的 **SINR** 可以表示为：
$$
\gamma_{u,m} =
p_u - \zeta_{u,m} - 
10\log_{10}\left(\sigma^2 + \sum_{i \in \mathbf{U} \setminus u} I_{m,i}\right)
$$
其中，\[p_u\] 表示 UAV \[u\] 的在一个用户带宽 \[ B/N \] 内的发射功率，单位 dBm；\[\sigma^2\] 表示该用户带宽内的噪声功率，单位 mW。

\[I_{m,i}=10^{(p_i-\zeta_{i,m})/10}\] 表示 UAV \[i\] 对 UE \[m\] 造成的干扰。 

所有 \[\gamma_{i,j}\] 组合成 SINR 矩阵 \[ \bold Y \] ，其中 \[\gamma_{i,j}\] 代表具体 UAV \[i\] 与 UE \[j\] 之间的 SINR。

**(2) UE-UAV 上行发现 SNR**

记 UE \(m\) 到 UAV \(i\) 的上行功率增益为：
$$
g_{m,i}^{\mathrm{ul}} = 10^{-\zeta_{i,m}/10}
$$
信标与下行正交，信标之间的资源重叠干扰忽略，因此使用上行 SNR：
$$
\gamma_{i,m}^{ul}
=
10\log_{10}
\left(
\frac{P_m 10^{-\zeta_{i,m}/10}}
{\sigma_b^2}
\right)
$$
其中，\[P_m\] 是 UE 的发射功率，单位 mW，\[\sigma_b^2\] 是上行噪声功率，单位 mW。

#### 6. 连接条件

**(1) UAV-UE 连接**

对于 UAV-UE 连接，下行 SINR 达标是必要条件：

也就是说，只有满足：
$$
\gamma_{u,m}\geq \gamma_T
$$
同时还必须满足上行发现门槛、UAV 可达任一 BS、该 UAV 尚有服务容量，才允许：
$$
c_{u,m} = 1.
$$

**(2) UAV-UAV 连接**

假设所有 UAV 具有相同的最大机间通信距离 \(D_{\mathrm{UU}}\)，当两架 UAV 的距离不超过该范围时，认为二者可以建立直接通信链路。

设机间链路允许的最大路径损耗为 \(\zeta_{\mathrm{UU}}^{\max}\)，则由
$$
\zeta_{i,j}
=
20\log_{10}\!\left(\frac{4\pi f d_{i,j}}{c}\right)
\leq\zeta_{\mathrm{UU}}^{\max}
$$
可得：
$$
D_{\mathrm{UU}}
=
\frac{c}{4\pi f}\,
10^{\zeta_{\mathrm{UU}}^{\max}/20}.
$$
**(3) UAV-BS 连接**

假设所有 UAV 具有相同的最大 UAV–BS 三维通信距离 \(D_{\mathrm{UG}}\)，当 UAV与BS 的距离不超过该范围时，认为二者可以建立直接通信链路。

设 UAV–BS 链路允许的最大路径损耗为 \(\zeta_{\mathrm{UG}}^{\max}\)，可得：
$$
D_{\mathrm{UG}}
=
\frac{c}{4\pi f}\,
10^{\zeta_{\mathrm{UG}}^{\max}/20}.
$$


上述自由空间公式只解释距离门槛的含义。当前代码直接使用配置中的 \(D_{\mathrm{UU}}=650\) m 和 \(D_{\mathrm{UG}}=1500\) m，不根据发现/下行门槛动态推导回传距离，也不计算回传 SINR 或容量瓶颈。

#### 7. 下行数据率

候选 UAV–UE 链路的数据率为：
$$
r_{u,m}=
\frac{B}{N} \log_2 \left(
1+10^{\gamma_{u,m}/10}
\right).
$$
代码计算全部候选链路速率，再通过关联矩阵得到实际服务速率。UE 每时刻最多关联一架 UAV：
$$
R_{u,m}=c_{u,m}r_{u,m},\qquad v_m=\sum_u R_{u,m}.
$$

所有数据率组成数据率矩阵 \[ \bold R \]。

#### 8. 感知条件

所有需要服务的 UE 能广播发现信标，信标携带用户标识和位置；当 UAV \[u\] 接收到的 UE \[m\] 上行信号 SNR 不低于感知阈值，则可感知到该 UE：
$$
\gamma_{u,m}^{ul} \geq \gamma_{obs}
$$


## 四、离散动作 MAPPO 建模

### A. POMDP

POMDP 通常定义为如下元组：
$$
\mathcal M=
\langle
\mathcal N,\mathcal S,\mathcal A,\mathcal P,\Omega,\mathcal O,\mathcal R
\rangle
$$
其中：

- \[\mathcal N\] : 智能体集合
- \[\mathcal S\] : 状态空间
- \[\mathcal A\] : 动作空间
- \[\mathcal P\] : 状态转移概率
- \[\Omega\] : 观测空间
- \[\mathcal O\] : 观测概率函数
- \[\mathcal R\] : 奖励函数

在任意时间步 \[t \in T\] ，每个智能体 \[ i\in\mathcal N \] 根据当前观测 \[ o_i^t\in\Omega \] 选择动作 \[ a_i^t\in\mathcal A \] ，其中观测 \[ o_i^t \] 通常是系统真实状态 \[ s^t \in \mathcal S \] 的某种函数。

随后，整个系统获得一个联合奖励：
$$
r^t=R(s^t,a^t,s^{t+1})=\rho^{t+1},
$$
并根据状态转移概率：
$$
s^{t+1}
\sim
P(\cdot|s^t,a^t)
$$
转移到新的状态。

每个智能体 \[i\] 根据观测概率函数：
$$
O_i(o_i^{t+1}\mid s^{t+1})
$$
获得下一时刻的原始局部观测。当前模型采用无历史、确定性的观测规则，可写为 \(o_i^{t+1}=g_i(s^{t+1})\)。用户集合的神经网络编码属于 actor 内部计算，不属于环境观测函数。

#### 1. 状态空间 \[\mathcal S\]

状态空间包含全局环境信息，建模为：
$$
\mathcal S^t = [
p_{\mathcal U},p_{\mathcal G}, p_{\mathcal M},
\bold A, \bold B, \bold C ,\bold Y,
\frac{t}{T}
]
$$

\[ (p_{\mathcal U}, p_{\mathcal G}, p_{\mathcal M},) \] 分别是 所有UAV，基站，地面用户的坐标集合。

 \[\bold A\] 是 UAV 连接拓扑矩阵，维度是 \[ U \times U \]；\[ \bold B \] 是 UAV 接入跳数矩阵；\[ \bold C \] 是服务关联矩阵 ；\[ \bold Y \] 是 SINR 矩阵，维度是 \[ U \times M \]。

\[ t, T \] 分别是当前时间步和总时间步。

实现中坐标按 \((L_x,L_y,h)\) 归一化，\(\bold B\) 除以 \(U+1\)，\(\bold Y\) 除以 30 dB；\(\bold A,\bold C\) 保持二值。状态维度为
$$
3(U+G+M)+U^2+UG+2UM+1=6292.
$$
上行发现矩阵和 UE 集合的神经编码结果不额外加入全局状态。

#### 2. 局部观察

根据实际可部署环境下 UAV 可获得的信息，局部观察可建模为：
$$
\mathbf o_i^t=[\mathbf o_i^{self}, \mathbf o_i^{BS}, \mathbf o_i^{neighbor}, \mathbf o_i^{ue}]
$$
**(1) 自身状态 \[o_i^{self} \]** 

自身状态 \[o_i^{self} \] 为 UAV 自身的位置信息与连接信息，可表示为：
$$
\mathbf o_i^{self} = [x_i, y_i, h_i, c_i, n_i, \bold a_i]
$$
其中，\[(x_i, y_i, h_i)\] 是 UAV 位置信息；\[c_i\] 为该 UAV \[i\] 能接入核心网的最短跳数的归一化：
$$
c_i=
\frac {\min_{g\in\mathcal G}b_{i,g}} {U+1}
$$
\[n_i\] 为 UAV \[i\] 当前服务的 UE 数量；\[\bold a_i\] 为 U 维矢量，记录 UAV \[i\] 与各 UAV 的连接情况，代码取 \(\bold A\) 的第 \(i\) 行（此无向图矩阵对称）。

上式为物理量表示，实际输入为 \([x_i/L_x,y_i/L_y,h_i/h,c_i,n_i/N,\bold a_i]\)，共 \(5+U=23\) 维。当前固定高度下 \(h_i/h=1\)；断联时 \(c_i=1\)。

**(2) 基站状态 \[ o_i^{BS} \]**

基站状态 \[ o_i^{BS} \] 为基站位置的先验信息：
$$
\mathbf o_i^{BS} = \{x_j, y_j\}_{j=1}^G
$$
**(3) 邻居状态 \[o_i^{neighbor}\]** 

邻居状态 \[o_i^{neighbor}\] 为 UAV \[i\] 可直接连接的3架**最近邻** UAV 邻居的信息的聚合，可表示为：
$$
\mathbf o_i^{neighbor} = \{
u_j,x_j,y_j,h_j,c_j,n_j, \bold a_j
\}_{j=1}^3
$$
其中，排除自身后按距离、编号升序选取邻居；不足三个时整槽补零，不另加掩码维度。邻居槽包含编号与其完整自身状态，实际编号使用从 1 开始的编号除以 \(U\)，其余字段按自身状态规则归一化，每槽 \(6+U=24\) 维。直接相邻不必然意味着可达 BS：断联子网内的邻居也会被观察到。

BS 坐标输入为 \((x_g/L_x,y_g/L_y)\)。因此固定部分共 \(23+2G+3\times24=101\) 维。actor 不接收远处 UAV 的坐标、全局 UE 列表或历史轨迹。

**(4) 用户状态 \[o_i^{ue}\]** 

定义本机感知集合：
$$
\mathcal D_i = \{m:\gamma_{i,m}^{ul,t}\geq\gamma_{\mathrm{obs}}\}, \qquad n_i^{obs}=|\mathcal D_i|
$$
UAV \[i\] 感知到的 UE \[j \in \mathcal D_i^t\] 信息为：
$$
e_{i,j} = 
\left[
\frac{x_j-x_i}{L_x}, \frac{y_j-y_i}{L_y},
\frac{\gamma_{i,j}^{ul}}{30}, c_{i,j} 
\right]
$$
其中，\[ (x_j, y_j) \] 表示 UE 坐标；\[\gamma_{i,j}^{ul}\] 为以 dB 表示的上行 SNR，除以 `sinr_scale_db=30`；\[c_{i,j}\] 表示该 UE \[j\] 是否正由 UAV \[i\] 服务。断联 UAV 仍可感知 UE，发现集合不按服务容量截断。

环境提供的原始 UE 观察是可变长度集合：
$$
\mathbf o_i^{ue}
=\{\mathbf e_{i,j}:j\in\mathcal D_i\}.
$$
原始局部观察 \( o_i^t \) 由自身、基站、邻居信息和上述用户集合组成，是结构化数据，而非已经编码的固定长度向量。下文 MAPPO 部分定义可训练的集合编码器，将其转换为 actor 内部的固定长度特征。

#### 3. 动作空间

每架 UAV 的动作是离散编号 \(a_i^t\in\{0,1,\ldots,8\}\)。前为世界坐标 +y，右为 +x，不定义机体航向。设 \(c=1/\sqrt2\)：

| 编号 | 动作 | 归一化方向 \(\mathbf d(a)\) |
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

非悬停动作位移长度均为 \(v_{\max}\Delta t=30\) m；斜向分量为 \(30/\sqrt2\) m。每个动作在每步重新选择。边界采用逐轴反射，无动作掩码；超出边界的位移量反向应用，PPO保存并计算原始指令编号的概率。

#### 4. 状态转移

**(1) UAV 位置更新**

$$
\mathbf p_i^{t+1}=\operatorname{reflect}_{W,H}\left(\mathbf p_i^t+v_{\max}\Delta t\,\mathbf d(a_i^t)\right).
$$

反射逐轴定义为 $r_L(x)=\min(x\bmod 2L,2L-(x\bmod 2L))$，适用于角落和单步多次越界。

UAV保持高度100 m。保存动作编号、二维指令与实际位移，区分指令及边界反射效果。悬停不移动UAV，但时间、UE移动、关联与奖励仍更新。

**(2) UE 位置更新** 

地面用户的位置更新使用 **随机游走模型(Random Walk)** 。

每次更新时，随机选择一个速度与方向：
$$
v_m(t)\sim \mathcal N(0,\sigma_v^2), \qquad \sigma_v=1 m/s \\
\theta_m(t)\sim U(0,2\pi).
$$
\(v_m(t)\)：移动速度；\(\theta_m(t)\)：移动方向；\(G(a,b)\)：高斯分布； \(U(a,b)\)：区间 \([a,b]\) 上的均匀分布。

位置更新：
$$
\begin{aligned}
x_m^{t+1}&=x_m^t+v_m(t)\cos\theta_m(t)\Delta t\\
y_m^{t+1}&=y_m^t+v_m(t)\sin\theta_m(t)\Delta t
\end{aligned}
$$
其中，\[ \Delta t \] 是时间步长。

实现时先将有符号移动幅度 \(v_m(t)\) 截断到 \([-3,3]\) m/s，再计算上述候选位置，最后对两个坐标分别使用与 UAV 相同的边界反射函数。移动幅度和方向每步独立重采样，没有速度惯性。UE 运动使用独立随机流。

**(3) UAV 连接矩阵更新**

在完成位置更新后，根据更新后的 UAV 间距离重新确定直接连接关系。

直接连接矩阵按距离门槛更新为：
$$
a_{i,j}^{t+1}
=
\begin{cases}
1, & i=j,\\
1, & i\neq j,\ 
d_{i,j}^{t+1} \leq D_{UU},\\
0, & \text{else},
\end{cases}
$$
**(4) UAV-BS 连接更新**

在完成 UAV 位置及 UAV 间连接矩阵更新后，根据 UAV 与基站之间的距离，更新直接连接关系，并判断每架 UAV 是否能够通过直接或多跳链路接入核心网。

**① UAV–BS 直接连接更新**

由于基站位于地面，UAV \(i\) 与基站 \(g\) 在时间步 \(t+1\) 的三维距离为：
$$
d_{i,g}^{t+1}
=
\sqrt{
(x_i^{t+1}-x_g)^2+
(y_i^{t+1}-y_g)^2+
h^2
}.
$$
当 UAV-BS 之间的三维距离不超过 \[ D_{UG} \] 时，UAV 可直接接入核心网。
$$
b_{i,g}^{t+1,(0)}
=
\begin{cases}
1, & d_{i,g}^{t+1} \leq D_{UG},\\
U+1, & \text{else},
\end{cases}
$$
**② 多跳可达状态传播**

当 UAV-BS 之间的距离大于 \[ D_{UG} \] 时，若其可连接的邻接 UAV 可接入核心网时，该 UAV 也可接入。

根据当前 UAV 连接矩阵 \(\boldsymbol A^{t+1}\)​ ，定义 UAV \(i\) 当前的直接邻居集合：
$$
\mathcal N_i^{t+1} = \{j\in\mathcal U\setminus\{i\}:a_{i,j}^{t+1}=1\}. 
$$
随后同步迭代：
$$
b_{i,g}^{t+1,(k+1)}
=
\min\left\{
b_{i,g}^{t+1,(k)},
\;
1+\min_{j\in\mathcal N_i^{t+1}}
b_{j,g}^{t+1,(k)}
\right\}.
$$
当 \(\mathcal N_i=\varnothing\) 时，保持该 UAV 的当前跳数估计不变。

当矩阵不再变化时停止迭代（最多迭代 U-1 轮）：
$$
\boldsymbol B^{t+1,(k+1)}
=
\boldsymbol B^{t+1,(k)}.
$$
**(5) SINR 矩阵更新**

完成 UAV 与 UE 位置更新后，根据新的三维距离和仰角，按照通信模型重新计算所有 UAV–UE 链路的平均路径损耗 \(\zeta_{u,m}^{t+1}\)。

在得到更新后的平均路径损耗 \(\zeta_{u,m}^{t+1}\) 后，重新计算各 UAV 在 UE 处产生的同频干扰功率：
$$
I_{m,i}^{t+1} = 10^{(p_i-\zeta_{i,m}^{t+1})/10}
$$
因此，UAV \(u\) 与 UE \(m\) 之间的 SINR 更新为：
$$
\gamma_{u,m}^{t+1}
=
p_u-\zeta_{u,m}^{t+1}
-
10\log_{10}\left(
\sigma^2+
\sum_{i\in\mathcal U\setminus\{u\}}
I_{m,i}^{t+1}
\right).
$$
**(6) 服务关联矩阵更新**

在得到 UAV 接入矩阵 \(\boldsymbol B^{t+1}\) 和 SINR 矩阵 \(\boldsymbol Y^{t+1}\) 后，重新确定用户服务关联。

这是一个多用户争抢有限服务名额资源的问题，可以使用以下贪心算法解决，即：**候选 UAV 较少的用户优先，在剩余容量中选择 SINR 最大的 UAV**。

**① 确定候选 UAV**

对于 UE \[m\]，定义候选 UAV 集合，即只有同时满足可感知，接入 SINR 阈值和核心网可达条件的 UAV，才能为用户提供服务：
$$
\mathcal F_m^{t+1}
=
\left\{
u\in\mathcal U:
\gamma_{u,m}^{ul,t+1} \geq \gamma_{obs},\quad
\gamma_{u,m}^{t+1}\geq\gamma_T,\quad
\min_{g\in\mathcal G}b_{u,g}^{t+1}\leq U
\right\}
$$
**② 初始化与排序**

将服务关联矩阵及各 UAV 的已分配用户数初始化为零：
$$
\boldsymbol C^{t+1}=\boldsymbol 0,
\qquad n_u=0,\quad\forall u\in\mathcal U.
$$
按候选集合大小 \(|\mathcal F_m^{t+1}|\) 从小到大处理用户，并列时按用户编号升序处理。候选集合为空的用户保持未接入。

**③ 按剩余容量分配**

处理 UE \(m\) 时，确定仍有服务容量的候选集合：
$$
\mathcal F_{m,\mathrm{free}}^{t+1}
=
\left\{
u\in\mathcal F_m^{t+1}:n_u<N
\right\}.
$$
若该集合非空，选择：
$$
u_m^\star = \underset{u\in\mathcal F_{m,\mathrm{free}}^{t+1}} {\arg\max}\; \gamma_{u,m}^{t+1}
$$
SINR 相同时，优先选择接入核心网最短跳数较少的 UAV；仍相同时，选择编号较小者。随后更新：
$$
c_{u_m^\star,m}^{t+1}=1,
\qquad
n_{u_m^\star}\leftarrow n_{u_m^\star}+1.
$$
若没有剩余容量可用，则该用户对应的关联矩阵列保持为零。

#### 5. 奖励函数：仅实际联网服务覆盖率

本基准所有 UAV 共用同一个团队奖励。动作执行、UE 移动、拓扑和服务关联更新完成后，计算：
$$
\boxed{r_t=\rho^{t+1}=\frac{1}{M}\sum_{m\in\mathcal M}\sum_{u\in\mathcal U}c_{u,m}^{t+1}}.
$$
每个 UE 最多关联一架 UAV。服务关联要求同时满足发现门槛、下行 SINR 门槛、核心网可达和服务容量约束，因此 \(\rho\) 是实际联网服务覆盖率，既不是几何覆盖面积，也不是瞬时感知率。

基准配置固定为：

```yaml
env:
  reward_primary: coverage
  throughput_weight: 0.0
  robustness_weight: 0.0
  disconnection_weight: 0.0
```

因此 \(0\le r_t\le1\)。奖励使用当前覆盖水平，不使用相邻时刻覆盖增量；维持相同覆盖可以在每一步继续获得相同奖励。UAV 自身联网不直接加分，断联不单独扣分；联网只有在使 UE 获得服务时才带来覆盖收益。仅发现 UE、累计发现新用户、减少跳数、增加中继数量或扩大网络面积均没有额外奖励。没有软服务势差项。

训练目标是最大化有限回合的期望折扣覆盖累积：
$$
\max_\theta\ \mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{T-1}\gamma^t\rho^{t+1}\right],
\qquad T=400,\quad\gamma=0.995.
$$
对应原始尺度的真实策略价值为：
$$
V^\pi(s_t)=\mathbb E_\pi\!\left[\sum_{k=t}^{T-1}\gamma^{k-t}\rho^{k+1}\mid s_t\right],
\qquad V^\pi(s_T)=0.
$$
所以 \(0\le V^\pi(s_t)\le(1-\gamma^{T-t})/(1-\gamma)\)，最后一次动作之前的价值上界为 1。这里是价值函数的定义与校验边界；当前网络输出没有强制截断或乘以剩余时间，不能据此假定其预测已经校准。集中状态包含 \(t/T\)，actor 局部观察不包含时间。

#### 6. 保留的诊断指标与非基准实验

**(1) 吞吐指标。** 假设回传容量足以承载全部接入业务；回传与接入资源正交，忽略回传拥塞、调度开销及路由收敛时延。
$$
v_m=\sum_u c_{u,m}\frac BN\log_2(1+10^{\gamma_{u,m}/10}),
\qquad \tau=\frac{\sum_m v_m}{M v_{\rm ref}},\quad v_{\rm ref}=5\ \mathrm{Mbps}.
$$
吞吐仍可报告，但其权重为零，速率大小不直接进入基准奖励。

**(2) 固定关联故障保留覆盖。** 对每个 UAV \(f\) 的假想失效，移除该节点并重新计算其余 UAV 到任一 BS 的可达性，记为 \(z_u^{(-f)}\)。失败节点的可达性为零；允许切换网关，但保持原来的 UE 关联，不重新分配用户。
$$
\rho^{(-f)}=\frac1M\sum_{u,m}c_{u,m}z_u^{(-f)},\qquad
J_{\rm rob}=\frac1U\sum_f\rho^{(-f)},\qquad
0\le J_{\rm rob}\le(1-1/U)\rho.
$$
故障评估为当前状态的反事实诊断，不在正常训练轨迹中实际移除 UAV；\(J_{\rm rob}\) 的奖励权重为零。

**(3) 断联率。**
$$
z_u=\mathbf1\{\min_g b_{u,g}\le U\},\qquad
D=\frac1U\sum_u(1-z_u).
$$
直接或多跳到任一 BS 均视为联网。多个子网分别接入不同 BS 不算断联；机间互联但整体无法到达 BS 仍算断联。\(D\) 是当前断联比例，不是新增断联数或故障后的比例；本基准不把它作为惩罚。

**(4) 瞬时感知率。**
$$
P=\frac1M\sum_m\mathbf1\{\exists u:\gamma^{\rm ul}_{u,m}\ge\gamma_{\rm obs}\}.
$$
同一 UE 去重计数，不要求感知它的 UAV 联网，也不要求下行达标或有剩余容量。\(P\) 与历史累计发现率分开记录，均不进入本基准奖励。

**(5) 通用代码中的对照开关。** 共用环境实现仍支持
$$
r_t=Q^{t+1}+w\tau^{t+1}+\lambda J_{\rm rob}^{t+1}-\mu D^{t+1},
\qquad
Q=\begin{cases}\rho,&\texttt{reward\_primary=coverage},\\P,&\texttt{reward\_primary=sensing}.\end{cases}
$$
这只是可配置实现的统一表达式，本文基准固定 \(Q=\rho,w=\lambda=\mu=0\)。六组对照的定位如下，全部 \(w=\lambda=0\)：

| 组别 | UE 运动 | 奖励 | 与本文的关系 |
|---|---|---|---|
| A | 移动 | \(\rho-0.5D\) | 增加断联惩罚 |
| B | 静止 | \(\rho-0.5D\) | 静止且有断联惩罚 |
| **C** | **移动** | **\(\rho\)** | **本文基准** |
| D | 移动 | \(P\) | 改为纯感知目标 |
| E | 移动 | \(P-0.5D\) | 感知加断联惩罚 |
| F | 静止 | \(\rho\) | 仅关闭 UE 运动 |

静止组设置 `user_speed_std=0`，保持同一初始化分布。各组独立训练、保存完整配置，不能跨奖励或运动设置直接续训。实验入口、指标和回退方法见 [UE 运动与奖励对照说明](docs/ue_motion_reward_ablations.md)。其他奖励配置不能与 C 组混为同一基准。

### B. MAPPO

#### 1. 集中训练、分布执行

所有 UAV 共享局部 actor \(\pi_\theta\)，只读取前文定义的结构化局部观察。集中式价值网络 \(V_\phi(s)\) 在训练时读取全局状态，并输出单一团队价值；执行时不需要全局状态。关联继续由固定贪心规则决定，不是策略输出。

参数 \(\theta=(\psi,\omega)\) 包括 UE 编码器与分类策略主干。编码器不与价值网络共享参数，无预训练或监督标签，通过 PPO actor 损失端到端更新。

#### 2. 集合编码与分类策略

对每架 UAV 当前发现集合中的每行 \(\mathbf f_{im}\in\mathbb R^4\)，使用共享两层 ReLU编码器 \(h_\psi:4\to32\to32\)：
$$
\bar{\mathbf h}_i=\begin{cases}|\mathcal D_i|^{-1}\sum_{m\in\mathcal D_i}h_\psi(\mathbf f_{im}),&|\mathcal D_i|>0,\\\mathbf0,&|\mathcal D_i|=0.\end{cases}
$$
拼接固定101维特征、32维集合均值和人数 \(|\mathcal D_i|/20\)，得到134维内部特征。人数可超过1。全部当前已发现UE参与编码，不按容量截断，不输入历史累计发现或全局UE数量。

两层256单元ReLU主干输出9个logits \(l_{i,k}\)：
$$
\pi_\theta(a_i=k\mid o_i)=\frac{\exp l_{i,k}}{\sum_{j=0}^8\exp l_{i,j}}.
$$
采集和默认验证时各机均按分类分布采样；验证动作种子固定为880001，与训练随机流隔离。确定性argmax仅用于显式诊断，不用于默认模型选择。条件联合策略可写为各agent分布的乘积，但PPO更新不使用联合概率比。

#### 3. On-policy采集与GAE

冻结本轮采集策略 \(\theta_{\rm old}\)，16环境各采集128步，即2048条联合转移。保存当前原始观察、状态、动作编号、旧log概率、原始尺度价值、下一状态价值、奖励及终止标志。每轮更新重新编码当前原始集合，禁止将旧集合编码作为训练输入。

令 \(d_t\) 为真正终止标志，\(c_t\) 为是否继续向下一条轨迹累积优势（真正终止或截断时为0）：
$$
\delta_t=r_t+\gamma(1-d_t)V_{\rm old}(s_{t+1})-V_{\rm old}(s_t),\quad
\hat A_t=\delta_t+\gamma\lambda_{\rm GAE}c_t\hat A_{t+1},\quad
\hat R_t=\hat A_t+V_{\rm old}(s_t).
$$
400步到期为真正终止，不能bootstrap。未结束环境的采集片段末尾使用下一状态价值，尾端GAE累积从0开始。下一状态必须是reset前的真实下一状态。团队优势在本轮联合转移上标准化，同一团队优势用于各机策略项。

#### 4. PPO actor与熵正则

每个agent分别计算概率比：
$$
r_{t,i}=\exp\big[\log\pi_\theta(a_{t,i}\mid o_{t,i})-\log\pi_{\rm old}(a_{t,i}\mid o_{t,i})\big].
$$
$$
L_{\rm actor}=-\mathbb E_{t,i}\!\left[\min\{r_{t,i}\tilde A_t,\operatorname{clip}(r_{t,i},1-\epsilon,1+\epsilon)\tilde A_t\}\right]-\beta\mathbb E_{t,i}[\mathcal H(\pi_\theta(\cdot\mid o_{t,i}))].
$$
分类熵精确计算为 \(-\sum_k p_k\log p_k\)，单机最大值为 \(\log9\)。熵系数固定为0.01，对样本和agent求平均；联合熵仅作为诊断。无SAC温度、自适应目标熵、双Q网络或目标actor。

#### 5. 集中价值更新

价值网络为 \(6292\to256\to256\to1\)，两层ReLU。沿用运行回报统计与PopArt：更新归一化统计时同步重标定输出层，保持原始尺度价值。对本轮回报 \(\hat R_t\) 与采集旧价值使用同一更新后的均值/标准差归一化，记为 \(\tilde R_t,\tilde V_{{\rm old},t}\)。当前网络输出为 \(\tilde V_\phi(s_t)\)，使用clipping与Huber损失：
$$
\tilde V_{\rm clip}=\tilde V_{\rm old}+\operatorname{clip}(\tilde V_\phi-\tilde V_{\rm old},-\epsilon,\epsilon),
$$
$$
L_V=\mathbb E_t\max\{\operatorname{Huber}_{10}(\tilde V_\phi-\tilde R_t),\operatorname{Huber}_{10}(\tilde V_{\rm clip}-\tilde R_t)\}.
$$
actor与价值优化器独立。价值更新不改变UE编码器；actor通过log概率与熵梯度更新编码器，无需对离散动作求环境梯度。

基准的 `algo.value_clipping` 缺省为 `true`。独立对照配置 `configs/discrete_mappo_critic_no_value_clip.yaml` 将其设为 `false`，仅把上述价值损失替换为未裁剪的 \(\mathbb E_t[\operatorname{Huber}_{10}(\tilde V_\phi-\tilde R_t)]\)，保留 actor 的 PPO clipping、PopArt 和全部其他设置。该对照不替换本文基准；说明与结果见 [Value Clipping 对照实验](docs/critic_value_clipping_ablation.md)。

#### 6. 参数、预算与恢复

actor/critic学习率均5e-4，Adam eps=1e-5；折扣0.995，GAE 0.95，clip 0.2，熵系数0.01，10 epochs、4 minibatches，梯度范数裁剪0.5。minibatch按联合转移切分，每条保留全部18架UAV。每次更新后丢弃已用rollout，不设随机warmup或离策略回放池。

WSL预算为250000条联合转移：122个完整2048条rollout后剩144条，最后采集9个向量步再更新，共123轮PPO，不补足预算。评估在跨过10k阈值的完整更新边界执行，保存实际步数；最后模型也评估。

完整检查点在完成rollout更新的安全点发布，包含模型、优化器、PopArt统计、环境与随机状态、步数、历史最佳模型，不包含回放池。接收停止信号后完成当前已收集片段的更新再保存。检查点按当前原子目录机制发布、保留最近两份；last.pt为轻量模型文件，不能独立完整续训。

## 五、实现约定与实验协议（beacon-set-v2 / discrete9）

本版本与连续动作版本并存，共用传播、发现、关联、UE运动和固定关联单机故障实现。本文基准固定为 C 组的 \(r_t=\rho^{t+1}\)，不因通用配置类或其他 YAML 的默认权重不同而改变。状态6292维，固定局部特征101维，集合编码后actor输入134维；空槽、排序和固定物理尺度归一化沿用前文。

配置env.action_space=discrete9、algo.name=mappo；动作映射版本world-nine-v1。标量接口接受整数[U]、向量接口接受整数[E,U]。浮点编号、越界编号、错误形状拒绝执行。检查点与actor导出标记动作空间、映射与categorical分布，不迁移连续权重。

静止baseline永远编号8；随机baseline九动作等概率；局部启发式只读取同一局部观察，将连续建议向量投影到欧氏距离最近的合法指令，平局最小编号。所有方法共享随机初始化与评估场景，保留零鲁棒性配置作为兼容入口；主版同样为0，因此当前不构成有效消融对照。固定目标部署只用于物理诊断；离散追踪目标会量化并可能悬停于附近，不能当成可部署局部策略。

记录九动作频率、悬停比例、精确熵、最大动作概率、PPO clipping与KL、价值损失及编码器梯度；轨迹保存编号、指令、实际位移、位置、图与逐故障结果。动作频率训练曲线明确标为日志所在向量步的统计，评估频率为整回合统计。

验证32个固定场景、测试100个保留场景，随机流隔离。按验证集的**全回合未折扣总奖励的场景平均值**选择 `best`，同时报告 `last`。本基准每回合均为400步且只有覆盖奖励，所以该选择等价于按全回合平均覆盖率选择，**不是按终态覆盖率**。训练仍使用前述折扣和GAE；选择模型的统计量与训练中的折扣价值需要区分。

报告全回合平均、最后100步与终态的覆盖、感知、断联及拓扑指标；不同奖励定义的总回报不能直接比较高低。跨动作空间比较应配对场景、训练预算，并提供连续MAPPO来分离算法和动作空间的影响。单种子结果用于诊断，不能据此宣称论文优势。熵鼓励探索，但不保证形成协调回传或提高覆盖。

### 基准参数索引

以下表格对应 [C 组完整配置](configs/discrete_mappo_ablation_c_mobile_coverage.yaml)。配置中其他算法共用的 `warmup_steps`、`replay_capacity`、`batch_size`、温度、Polyak、TD3 噪声等字段不参与本离散 MAPPO 路径；高斯 UE 初始化参数和 `launch_radius` 在当前均匀初始化下也不生效。

| 场景/环境参数 | 基准值 |
|---|---|
| 区域；UAV / UE / BS 数量 | 3500×3500 m；18 / 150 / 3 |
| BS 位置 | (0,0,0)、(0,3500,0)、(3500,0,0) m |
| UAV / UE 初始化 | 全区域独立均匀；不筛选初始覆盖或连通性 |
| 高度；服务容量 | 100 m；每 UAV 20 UE |
| 回合；时间步；UAV 速度 | 400 步；1 s；30 m/s |
| 动作；UAV/UE 边界 | `discrete9`；逐轴反射 |
| UE 运动 | 每步独立高斯幅度，标准差1 m/s，截断±3 m/s；均匀方向 |
| UAV–UAV / UAV–BS 距离门槛 | 650 m / 1500 m（三维） |
| 载频；总带宽；总功率 | 2.4 GHz；20 MHz；每 UAV 30 dBm |
| 每子带带宽 / 功率 | 1 MHz / 约16.99 dBm（50 mW） |
| 噪声密度；下行/信标噪声系数 | −174 dBm/Hz；均15 dB |
| LoS 参数；LoS/NLoS 额外损耗 | 9.61、0.16；1 / 20 dB |
| 下行 SINR 门槛 | −3 dB；满负载同频干扰 |
| 信标功率；带宽；发现 SNR 门槛 | 26.0205999133 dBm（0.4 W）；180 kHz；20 dB |
| 奖励主项；吞吐/鲁棒性/断联权重 | `coverage`；0 / 0 / 0 |
| 速率参考值（仅诊断） | 5 Mbps |
| 邻居槽；SNR/SINR 缩放 | 3；除以30 dB |
| 全局状态 / 局部固定特征 / 编码后特征 | 6292 / 101 / 134 维 |

| 网络/优化参数 | 基准值 |
|---|---|
| UE 编码器 | 4→32→32，ReLU，集合均值＋人数/20 |
| actor | 134→256→256→9，隐藏层ReLU，分类分布 |
| critic | 6292→256→256→1，隐藏层ReLU，团队价值 |
| actor / critic 学习率 | 均 \(5\times10^{-4}\)，固定 |
| 优化器 | 独立Adam，eps=\(10^{-5}\) |
| 折扣 / GAE λ | 0.995 / 0.95 |
| PPO 比率裁剪 / 归一化价值裁剪 | 均0.2 |
| 价值损失 | PopArt＋Huber（δ=10）＋value clipping |
| 优势标准化 | 当前rollout的联合转移维度上标准化 |
| 分类熵系数 | 0.01，固定 |
| 每轮采集 | 16环境×128步＝2048条联合转移 |
| 更新 | 10 epochs，每epoch 4 minibatches |
| actor / critic 梯度范数上限 | 均0.5 |
| 训练/默认验证动作 | 原分类策略采样；不收紧分布，argmax仅诊断 |

| 本地运行/评估参数 | 基准值 |
|---|---|
| 训练种子；联合转移预算 | 0；250000 |
| 设备；向量后端；环境后端 | CPU；`batch`；`numba` |
| PyTorch / Numba 线程 | 8 / 4 |
| 验证场景 | 10000–10031，共32个 |
| 保留测试场景 | 20000–20099，共100个 |
| 验证动作种子 | 880001，与训练随机流隔离 |
| 验证间隔 | 每10000条附近的完整更新边界，记录实际步数 |
| 日志间隔；TensorBoard | 1000条；开启 |
| 完整检查点间隔；保留数 | 10分钟，在安全点保存；最近两份 |
| 内存预算；确定性运行开关 | 12 GiB；`deterministic: true` |

`deterministic` 是复现设置，不将采样策略改为 argmax。运行结果保存完整配置，历史实验保持独立。本次更新仅明确基准建模，不启动新训练，也不改变已有检查点。
