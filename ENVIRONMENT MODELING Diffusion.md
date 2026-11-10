# ENVIRONMENT MODELING

## 一、场景

无人机（Unmanned Aerial Vehicles, UAVs），通常也称为 drones，正越来越广泛地被部署为**移动感知平台和通信平台**。

其中一个重要的应用场景，是在地面通信基础设施因自然灾害而受损，或者在大型人员密集活动中因负载过高而拥塞时，利用自主 UAV 作为能够**快速部署的空中中继节点**。

在森林火灾监测和战场监视等场景中，UAV 还可能工作在超出操作人员直接控制范围的区域，因此需要具备**机载自主决策能力（onboard autonomy）**以及 **UAV 之间的点对点协同能力（peer-to-peer coordination）**。

在这些场景中，UAV 集群必须自主决定各自应部署在什么位置，从而最大化对**动态目标区域**的感知覆盖或通信覆盖。

许多多 UAV 覆盖与部署任务都可以从最大覆盖选址问题**（Maximum Coverage Location Problem, MCLP）**的角度进行描述；然而，该问题属于 **NP-hard 问题**。

此外，真实部署环境还具有两个重要特征。

一方面是**部分可观性**，即 UAV 只能感知到其附近的实体（其余UAV，基站BS，地面用户UE）。

另一方面是**通信约束**，即两个实体（UAV-UAV, UAV-UE, UAV-BS）之间的通信链路质量SINR超过某个阈值时，它们才能通信交换信息。

为了解决该问题，多智能体强化学习**（Multi-Agent Reinforcement Learning, MARL）**提供了一套良好的方法



## 二、CTDE

本方案使用 **集中式训练、分布执行（CTDE）**架构。

在训练阶段，集中式 critic 可以访问全局信息，以提高训练过程的稳定性；在执行阶段，每架 UAV 运行一个**共享策略（参数共享的策略网络）**，其输入仅包括自身的局部观察（其中 BS 坐标为预先已知信息）。



## 三、系统模型

本文考虑应急通信场景，现有通信基站设施已经失效，此时需要使用多架无人机部署于空中形成多跳的无人机网络；可移动地面用户可连接无人机，并通过多跳无人机网络连接至基站。

### A. 网络模型

**实验实例约定：** 主实验区域为 \(1600\times1600\,\mathrm m\)，可用应急网关 BS 数量为 \(G=2\)，位置分别为 \((0,0,0)\) 与 \((0,1600,0)\)。两个 BS 均可接入核心网且不失效；区域原有接入设施失效。UAV 运行中允许形成多个机间子网，只要各自可直接或多跳到达任一 BS。BS 仅作为路由终点，不增加 BS–BS 转发边。完整数值约定与默认参数见 `docs/model_contract.md` 和 `configs/main.yaml`。

主场景采用 16 架 UAV 和 180 个按四中心二维高斯混合初始化的 UE。UAV 高度固定为 100 m，初始水平坐标在整个 [0,1600]×[0,1600] m 区域内独立均匀采样，不按覆盖率、回传可达性或连通性筛选，不要求任何 UAV 初始接入 BS。静止部署 baseline 保持同一随机初始位置不动，所有算法使用相同初始化分布。配置 uav_initialization=uniform；bs_launch 仅保留用于历史实验复现和初始化对照，不能将两种分布的结果混合比较。

UE 初始化参考 *Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models* 第 V-A 节提出的多中心二维高斯混合分布，但论文未明确指定边界留白、混合中心或方差。本文采用四个等权中心 (480,480)、(1120,480)、(480,1120)、(1120,1120) m，各轴标准差 128 m。每个 UE 独立抽取所属分量，再从该分量截断高斯采样，使初始坐标位于 [160,1440]×[160,1440] m；不裁剪到边界，不按服务效果筛选。160 m 留白及中心、标准差是本实验自己的参数。留白仅用于 reset；后续 UE 可进入留白区，仍在完整 1600×1600 m 区域内反射运动。

任务持续 600 步，每步 1 s，最大速度 10 m/s；UU 距离 400 m，UG 三维距离 350 m。发现信标功率 0 dBm、独立带宽 180 kHz、UAV 接收噪声系数 9 dB、感知门槛 5 dB；下行门槛 −3 dB。上行与下行资源正交，信标间干扰忽略，不通过硬编码距离确定发现集合。两种门槛需按各自链路预算联合校准，而不是直接比较数值大小。

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

\[a\] 和 \[b\] 是与具体环境传播有关的参数，而 \[\theta\] 为 \[e_1\] 和 \[e_2\] 之间的仰角。

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

其中，\[f\] 表示载波频率，\[c\] 表示光速，\[\eta_1\] 和 \[\eta_2\] 分别表示 LoS 和 NLoS 传播路径所产生的额外损耗。

#### 2. UAV-UE 通信

无人机 UAV 与 地面用户 UE 之间的**平均路径损耗**(dB)可以表示为：
$$
\zeta_{u,m} = 
P_{u,m}^{\mathrm{LoS}}\zeta_{u,m}^{\mathrm{LoS}} +
P_{u,m}^{\mathrm{NLoS}}\zeta_{u,m}^{\mathrm{NLoS}}
$$

#### 3. UAV-UAV 与 UAV-BS 通信

对于 UAV 与 UAV 或 UAV 与 BS 之间的信道，由于通常不存在障碍物遮挡，因此直接采用**自由空间传播损耗模型**：
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

#### 5. SINR

**(1) UAV-UE 下行 SINR**

各 UAV 内部采用正交频分接入，不同 UAV 复用同一接入频段。采用固定子带功率和满负载干扰近似，候选链路的干扰不随实际关联人数及子带占用改变。

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

**(2) UAV-UE 上行 SINR**

记 UE \(m\) 到 UAV \(i\) 的上行功率增益为：
$$
g_{m,i}^{\mathrm{ul}} = 10^{-\zeta_{i,m}/10}
$$
假设信标足够稀疏，资源重叠干扰可忽略，那么上行 SINR ：
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

对于 UAV-UE 连接，只有当 UAV 与 UE 之间的 SINR 超过某个给定阈值 \[\gamma_T\] 时，UAV 才能为 UE 提供服务。

也就是说，只有满足：
$$
\gamma_{u,m}\geq \gamma_T
$$
才允许：
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

假设所有 UAV 具有相同的最大机间通信距离 \(D_{\mathrm{UG}}\)，当 UAV与BS 的距离不超过该范围时，认为二者可以建立直接通信链路。

设机间链路允许的最大路径损耗为 \(\zeta_{\mathrm{UG}}^{\max}\)，可得：
$$
D_{\mathrm{UG}}
=
\frac{c}{4\pi f}\,
10^{\zeta_{\mathrm{UG}}^{\max}/20}.
$$


#### 7. 下行数据率

如果 UE \[m\] 与 UAV \[u\] 建立连接，则二者之间的下行数据率可以表示为：
$$
r_{u,m}=
B_n \log_2 \left(
1+10^{\gamma_{u,m}/10}
\right).
$$
UE \[m\] 每时刻只能与一架 UAV \[u\]，若 UE \[m\] 与 UAV \[u\] 已建立连接，则 \[m\] 与其余 UAV 的数据率为0：
$$
r_{i,m} = 0,
\qquad
\forall i \in \mathbf{U} \setminus u.
$$

所有数据率组成数据率矩阵 \[ \bold R \]。

#### 8. 感知条件

所有需要服务的 UE 能广播发现信标，信标携带用户标识和位置；当 UAV \[u\] 接收到的 UE \[m\] 上行信号 SINR 大于某一感知阈值，则可感知到该 UE：
$$
\gamma_{u,m}^{ul} \geq \gamma_{obs}
$$


## 四、RL 建模

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
r^t=R(s^t,a^t),
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
获得下一时刻的原始局部观测。无历史基准采用确定性规则 \(o_i^{t+1}=g_i(s^{t+1})\)；新增一步历史实验采用 \(o_i^{t+1}=g_i(s^{t+1},a_i^t,c_i^t,|\mathcal D_i^t|)\)，历史只由本机已获得的信息更新。用户集合的神经网络编码属于 actor 内部计算，不属于环境观测函数。

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

#### 2. 局部观察

根据实际可部署环境下 UAV 可获得的信息，局部观察可建模为：
$$
\mathbf o_i^t=[\mathbf o_i^{self}, \mathbf o_i^{BS}, \mathbf o_i^{neighbor}, \mathbf o_i^{ue}, \mathbf s_i^{t-1}]
$$
**(1) 自身状态 \[o_i^{self} \]** 

自身状态 \[o_i^{self} \] 为 UAV 自身的位置信息与连接信息，可表示为：
$$
\mathbf o_i^{self} = [
\frac{x_i}{L_x}, \frac{y_i}{L_y}, c_i, n_i, \bold a_i
]
$$
其中，\[(x_i, y_i, h_i)\] 是 UAV 位置信息，为归一化坐标，\[ L_x, L_y \] 是场景大小；\[c_i\] 为该 UAV \[i\] 能接入核心网的最短跳数的归一化：
$$
c_i=
\frac {\min_{g\in\mathcal G}b_{i,g}} {U+1}
$$
\[n_i\] 为 UAV \[i\] 当前服务的 UE 数量；\[\bold a_i\] 为 U 维矢量，记录 UAV \[i\] 与各 UAV 的连接情况，可从全局UAV连接矩阵 \[\bold A\] 中 取出 第 \[i\] 列获得。

**(2) 基站状态 \[ o_i^{BS} \]**

基站状态 \[ o_i^{BS} \] 为基站位置的先验信息，坐标转化为以自身UAV为原点的相对归一化坐标：
$$
\mathbf o_i^{BS} = \{
\frac{x_j-x_i}{L_x}, \frac{y_j-y_i}{L_y}
\}_{j=1}^G
$$
**(3) 邻居状态 \[o_i^{neighbor}\]** 

邻居状态 \[o_i^{neighbor}\] 为 UAV \[i\] 可直接连接的3架**最近邻** UAV 邻居的信息的聚合，可表示为：
$$
\mathbf o_i^{neighbor} = \{
u_j,\frac{x_j-x_i}{L_x},\frac{y_j-y_i}{L_y},c_j,n_j, \bold a_j
\}_{j=1}^3
$$
 其中，\[u_j\] 为该 UAV 的编号，若连接 UAV 不足 3，则多余的向量该位置0作掩码；\[(x_j,y_j,h_j,c_j,\bold a_j)\]  为 UAV \[j\] 的自身状态观察。

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
\gamma_{i,j}^{ul}, \gamma_{i,j}^{DL}, c_{i,j} 
\right]
$$
其中，\[ (x_j, y_j) \] 表示 UE 坐标；\[\gamma_{i,j}^{ul}\] 表示 UAV 与 UE 之间的上行 SINR；\[c_{i,j}\] 表示该 UE \[j\] 是否正由 UAV \[i\] 服务。

环境提供的原始 UE 观察是可变长度集合：
$$
\mathbf o_i^{ue}
=\{\mathbf e_{i,j}:j\in\mathcal D_i\}.
$$
原始局部观察 \(o_i^t\) 由自身、基站、邻居信息、上述用户集合以及可选的一步历史组成，是结构化数据，而非已经编码的固定长度向量。下文 SAC 部分定义可训练的集合编码器，将其转换为 actor 内部的固定长度特征。

**(5) 历史状态 \[ s_i^{t-1}\]** 

历史状态 \[ s_i^{t-1} \] 为 UAV \[i\] 在 \[ t \] 时刻 的上一时刻 \[ t-1 \] 时刻的归一化二维指令动作、联网跳数与感知人数历史信息：
$$
\mathbf s_i^{t-1} = \{
\Delta x_i^{t-1}, \Delta y_i^{t-1}, c_i^{t-1}, |\mathcal D_i^{t-1}|/K
\}
$$
\(c_i^{t-1}=\min_g b_{i,g}^{t-1}/(U+1)\) 为归一化最短联网跳数；\(K\) 是每机容量，M1-C中为20；\[ |\mathcal D_i^{t-1}| \]  为上一时刻的感知人数。

并且，在 \[ t=0 \] 时刻，上一刻动作为悬停，上一时刻UAV最短联网跳数与感知人数与目前 \[ t=0 \] 相同。 

#### 3. 动作空间

动作空间为**二维连续运动矢量**与**服务连接评分**：
$$
a_i^t = \{ \Delta x_i^t, \Delta y_i^t, \bold q_i^t \}, \qquad \forall i=1,2,\ldots,U.
$$
其中，\(\Delta x_i^t\)、\(\Delta y_i^t\) 分别表示 UAV \(i\) 在 X、Y 轴方向上的指令位移，取值可以为正或负。

为满足 UAV 最大飞行速度约束：
$$
\sqrt{(\Delta x_i^t)^2+(\Delta y_i^t)^2}
\le v_{\max}\Delta t
$$
定义 UAV \[i\] 可服务 UE 的集合：
$$
\mathcal F_i^t=
\left\{
m\in\mathcal D_i^t
\;\middle|\;
\gamma_{i,m}^{\mathrm{DL},t}\ge\gamma_T,\quad
\min_{g\in\mathcal G}b_{i,g}^t\le U
\right\}
$$
\[ \bold q_i \] 是 UAV 与 可服务用户集合的资源分配评分：
$$
\mathbf q_i^t=
\{q_{i,m}^t\mid m\in\mathcal F_i^t\}
$$
因此完整动作输出的维度为 \[ \dim(a_i^t)=2+|\mathcal F_i^t| \].

#### 4. 状态转移

**(1) UAV 位置更新**

UAV \[i\] 作出动作 \[a_i^t\] 后，需要更新新的位置：
$$
(x_i^{t+1}, y_i^{t+1}) = (x_i^t + \Delta x_i^t, y_i^t + \Delta y_i^t)
$$
若更新后位置会超出区域边界，则裁剪超出增量。

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

注意，速度采样绝对值大于 3m/s 时需要截断。

**(3) UAV 连接矩阵更新**

在完成位置更新后，根据更新后的 UAV 间距离重新确定直接连接关系。

令 \(\gamma_{i\rightarrow j}^{t+1}\) 表示位置更新后，UAV \(i\) 向 UAV \(j\) 发送信号时，在接收端 \(j\) 测得的 SINR，则直接连接矩阵更新为：
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
当 UAV-BS 之间的三维距离小于 \[ D_{UG} \] 时，UAV 可直接接入核心网。
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

在完成 UAV 与 UE 的位置更新、UAV 回传网络拓扑更新以及下行 SINR 计算后，根据各 UAV 的 Actor 输出的服务评分，采用分布式关联机制确定 UAV 与 UE 的最终服务关系。

每架 UAV 独立根据自身局部观察，由共享 Diffusion Actor 为可服务 UE 集合中的全部 UE 生成服务评分。为满足 UAV 的最大服务容量限制，采用 **Top-K 服务提议与 UE 自主选择** 的关联方式。

**① UAV 生成服务提议集合**

设 UAV \[i\] 的最大服务容量为 \[N\]，在时刻 \[ t \] 的可服务 UE 集合为 \[\mathcal F_i^t\]，Actor 输出对应的服务评分：
$$
\mathbf q_i^t=
\{q_{i,m}^t\mid m\in\mathcal F_i^t\}
$$
为限制 UAV 的服务数量，首先计算：
$$
K_i^t=\min(N,|\mathcal F_i^t|)
$$
随后按照 Actor 输出评分从高到低排序，选择评分最高的 \[K_i^t\]个 UE（评分相同时优先选择 UE 编号较小者），形成服务提议集合：
$$
\mathcal P_i^t=
\operatorname{Top}{K_i^t}
\left(
{q_{i,m}^t\mid m\in\mathcal F_i^t}
\right)
$$
**② 更新服务提议的有效性**

由于 UAV 和 UE 在执行动作后均可能发生位置变化，时刻 (t) 满足通信条件的 UE 在时刻 \[t+1\] 不一定仍然可以被服务。

因此，在更新 UAV 回传拓扑及下行 SINR 后，重新计算时刻 \[t+1\] 的可服务 UE 集合：
$$
\mathcal F_i^{t+1}=
\left\{
m\in\mathcal D_i^{t+1}
\;\middle|\;
\gamma_{i,m}^{\mathrm{DL},t+1}\ge\gamma_T,\quad
\min_{g\in\mathcal G}b_{i,g}^{t+1}\le U
\right\}
$$
保留原服务提议中仍满足通信条件的 UE：
$$
\widetilde{\mathcal P}_i^{t+1}
=
\mathcal P_i^t\cap\mathcal F_i^{t+1}
$$
**③ UE 选择服务 UAV**

对于 UE \[m\]，定义其收到有效服务提议的 UAV 集合：
$$
\mathcal O_m^{t+1}
=
\left\{
i\in\mathcal U
\;\middle|\;
m\in\widetilde{\mathcal P}_i^{t+1}
\right\}
$$
若 \(\mathcal O_m^{t+1}\neq\varnothing\)，UE \(m\) 选择服务评分最高的 UAV：
$$
i_m^\star
=
\underset{i\in\mathcal O_m^{t+1}}{\arg\max}
\;q_{i,m}^{t}
$$
当多个 UAV 的评分相同时，选择编号最小的 UAV。

若 \(\mathcal O_m^{t+1}=\varnothing\)，则 UE \(m\) 在该时刻不建立服务连接。

**④ 更新服务关联矩阵**

首先初始化：
$$
\boldsymbol C^{t+1}=\boldsymbol 0
$$
根据每个 UE 的选择结果，更新关联变量：
$$
c_{i,m}^{t+1} = 
\begin{cases} 
1, & \mathcal O_m^{t+1}\neq\varnothing \;\land\; i=i_m^\star,\\ 
0, & \text{otherwise}. 
\end{cases}
$$

#### 5. 奖励函数

奖励函数应同时包含**覆盖率、数据率、鲁棒性**的考虑，同时还有**断联惩罚**：
$$
r^t = \rho^{t+1} + w \tau^{t+1} + \lambda J^{t+1} - \mu D^{t+1}
$$
其中，\[ w, \lambda, \mu \] 为权重参数；\(\mu\) 对应 `env.disconnection_weight`。当前配置以0.1作为可调整起点；原文未指定其数值，缺失该字段的历史配置按0复现。

**(1) 覆盖率 \[ \rho \]**
$$
\rho^{t+1}
=
\frac1M
\sum_{m\in\mathcal M}
\sum_{u\in\mathcal U}
c_{u,m}^{t+1}
$$
因为每个用户最多关联一架 UAV，且关联时已经检查核心网可达性，所以不会重复计数，也不会把无法接入核心网的用户算作已覆盖。

**(2) 数据率 \[ \tau \]**

假设回传容量足以承载全部接入业务；回传与接入使用正交资源，忽略回传拥塞、调度开销及路由收敛时延。

用户 \[m\] 的实际服务速率为：
$$
v_m^{t+1}
=
\sum_{u\in\mathcal U}
c_{u,m}^{t+1}
\frac BN
\log_2\left(1+10^{\gamma_{u,m}^{t+1}/10}\right).
$$
数据率项 \[ \tau \] :
$$
\tau^{t+1}
=
\frac{1}{M v_{\mathrm{ref}}}
\sum_{m\in\mathcal M}v_m^{t+1}
$$
其中，\[ v_{ref} \] 为速率归一参数。

**(3) 鲁棒性 \[ J \]**

依次假设 UAV \(f\in\mathcal U\) 失效，计算剩余 UAV 的可接入性：
$$
z_u^{(-f)}
=
\begin{cases}
1, & u\neq f\text{，且移除 }f\text{ 后仍能接入核心网},\\
0, & \text{else}.
\end{cases}
$$
使用当前的服务关联矩阵计算故障后的保留覆盖率：
$$
\rho^{(-f)}
=
\frac1M
\sum_{m\in\mathcal M}
\sum_{u\in\mathcal U}
c_{u,m}z_u^{(-f)}
$$
计算所有 UAV 的故障平均表现作为奖励项：
$$
J_{\mathrm{rob}}
=
\frac1U\sum_{f\in\mathcal U}\rho^{(-f)}
$$

**(4) 断联惩罚**

定义在时刻 \[ t \] 的 UAV \[u\] 网络可达指标：
$$
z_u^t=
\begin{cases}
1,&\displaystyle\min_{g\in\mathcal G}b_{u,g}^t\leq U\\
0,&\displaystyle\min_{g\in\mathcal G}b_{u,g}^t=U+1
\end{cases}
$$
断联惩罚项定义为**断联UAV与总UAV数的比例**：
$$
D^{t+1} =
\frac1U\sum_{u\in\mathcal U}
(1-z_u^{t+1})
$$
其中，\(D^{t+1}\in[0,1]\)。当所有 UAV 均能接入至少一个 BS 时，\(D^{t+1}=0\)；当所有 UAV 均断联时，\(D^{t+1}=1\)。

