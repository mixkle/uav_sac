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


## 四、SAC 建模

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
\[n_i\] 为 UAV \[i\] 当前服务的 UE 数量；\[\bold a_i\] 为 U 维矢量，记录 UAV \[i\] 与各 UAV 的连接情况，可从全局UAV连接矩阵 \[\bold A\] 中 取出 第 \[i\] 列获得。

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
\gamma_{i,j}^{ul}, c_{i,j} 
\right]
$$
其中，\[ (x_j, y_j) \] 表示 UE 坐标；\[\gamma_{i,j}^{ul}\] 表示 UAV 与 UE 之间的上行 SINR；\[c_{i,j}\] 表示该 UE \[j\] 是否正由 UAV \[i\] 服务。

环境提供的原始 UE 观察是可变长度集合：
$$
\mathbf o_i^{ue}
=\{\mathbf e_{i,j}:j\in\mathcal D_i\}.
$$
原始局部观察 \( o_i^t \) 由自身、基站、邻居信息和上述用户集合组成，是结构化数据，而非已经编码的固定长度向量。下文 SAC 部分定义可训练的集合编码器，将其转换为 actor 内部的固定长度特征。

#### 3. 动作空间

动作空间为**二维连续矢量**，解耦 UAV x轴 与 y轴 的运动：
$$
a_i^t = \{ \Delta x_i, \Delta y_i \}, \qquad \forall i=1,2,\ldots,U.
$$
\[ \Delta x, \Delta y \] 分别是在 x轴方向 和 y轴方向的位移标量。

为了满足 UAV 的运动速度限制，有以下约束：
$$
\sqrt{\Delta x_i^2 + \Delta y_i^2} \leq v_{max}\Delta t
$$
其中，\[v_{max}\] 为最大速度。

#### 4. 状态转移

**(1) UAV 位置更新**

UAV \[i\] 作出动作 \[a_i^t\] 后，需要更新新的位置：
$$
(x_i^{t+1}, y_i^{t+1}) = (x_i^t + \Delta x_i, y_i^t + \Delta y_i)
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

在得到 UAV 接入矩阵 \(\boldsymbol B^{t+1}\) 和 SINR 矩阵 \(\boldsymbol Y^{t+1}\) 后，重新确定用户服务关联。

这是一个多用户争抢有限服务名额资源的问题，可以使用以下贪心算法解决，即：**候选 UAV 较少的用户优先，在剩余容量中选择 SINR 最大的 UAV**。

**① 确定候选 UAV**

对于 UE \[m\]，定义候选 UAV 集合，即只有同时满足可感知，接入 SINR 阈值和核心网可达条件的 UAV，才能为用户提供服务：
$$
\mathcal F_m^{t+1}
=
\left\{
u\in\mathcal U:
\gamma_{u,m}^{ul} \geq \gamma_{obs},\quad
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

### B. SAC

#### 1. 算法总体框架

本文采用集中式训练、分布执行框架下的参数共享多智能体 SAC 方法，学习 UAV 的连续移动策略。

所有 UAV 共享同一个 actor 网络，包括 UE 集合编码器和策略主干，并分别根据自身原始局部观察生成移动动作。记完整 actor 参数为
$$
\theta=(\psi,\omega),
$$
其中 \(\psi\) 为 UE 编码器参数，\(\omega\) 为策略主干及动作分布输出层参数。

UE 编码器不与集中式 critic 共享参数，不设置单独的监督标签或预训练目标；它通过 actor 损失端到端训练。

训练阶段采用两个集中式 critic，利用全局状态与全部 UAV 的联合动作评估系统长期回报。

服务关联矩阵由前文定义的贪心规则更新，不作为 actor 的输出。UAV 移动会改变通信拓扑、链路质量与服务关联，进而影响覆盖率、吞吐量及鲁棒性奖励。因此，actor 通过环境反馈学习有利于整体服务性能的部署策略。

定义联合观察和联合动作：
$$
\boldsymbol o^t=(o_1^t,\ldots,o_U^t),
\qquad
\boldsymbol a^t=(a_1^t,\ldots,a_U^t).
$$

#### 2. 最大熵优化目标

给定局部观察，各 UAV 使用共享参数、独立噪声采样动作，联合策略写为：
$$
\pi_\theta(\boldsymbol a^t\mid\boldsymbol o^t)
=
\prod_{i=1}^{U}\pi_\theta(a_i^t\mid o_i^t).
$$
因此：
$$
\log\pi_\theta(\boldsymbol a^t\mid\boldsymbol o^t)
=
\sum_{i=1}^{U}\log\pi_\theta(a_i^t\mid o_i^t).
$$
最大熵目标可写为：
$$
\mathcal J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\beta^t
\left(
r^t
-
\alpha
\sum_{i=1}^{U}
\log\pi_\theta(a_i^t\mid o_i^t)
\right)
\right]
$$
其中，\[ \beta \] 为折扣因子，\[ \alpha \] 为熵温度参数。

#### 3. 参数共享的局部 actor

actor 以 UAV 自身状态、基站先验、邻居状态及可变长度 UE 集合组成的原始局部观察 \(o_i^t\) 为输入，先编码用户集合，再输出二维高斯分布的均值和标准差。

每个被感知用户的信息经过共享编码器 \(\phi_\psi:\mathbb R^4\to\mathbb R^{d_h}\)，再进行平均聚合：
$$
\mathbf h_i^t=
\begin{cases}
\displaystyle\frac{1}{n_i^{obs,t}}
\sum_{j\in\mathcal D_i^t}\phi_\psi(\mathbf e_{i,j}^t),
&n_i^{obs,t}>0,\\[6pt]
\mathbf0,&n_i^{obs,t}=0.
\end{cases}
$$
加入本机感知人数，得到固定长度特征：
$$
\mathbf z_i^{ue,t}=[\mathbf h_i^t,n_i^{obs,t}],
\qquad
\mathbf x_{i,\psi}^t=
[\mathbf o_i^{self,t},\mathbf o_i^{BS,t},
\mathbf o_i^{neighbor,t},\mathbf z_i^{ue,t}].
$$
\(\mathbf z_i^{ue,t}\) 与 \(\mathbf x_{i,\psi}^t\) 是依赖当前参数的网络内部特征，不写入回放池来替代原始观察。

策略主干 \(\operatorname{Head}_\omega\) 接收 \(\mathbf x_{i,\psi}^t\)。完整 actor 定义为
$$
\operatorname{Actor}_\theta(o_i^t)
=\operatorname{Head}_\omega(\mathbf x_{i,\psi}^t).
$$
例如，用户编码器可采用 \(4\to32\to32\) 的 MLP，此时 UE 集合特征为 33 维。

所有用户及所有 UAV 共用同一套编码器参数。

所有 UAV 共享网络参数，但由于输入的局部观察不同，可以生成不同的移动动作。
$$
\left(
\boldsymbol\mu_\theta(o_i^t),
\log\boldsymbol\sigma_\theta(o_i^t)
\right)
=
\operatorname{Actor}_\theta(o_i^t).
$$
通过重参数化采样：
$$
\boldsymbol\epsilon_i^t\sim\mathcal N(\boldsymbol0,\boldsymbol I_2),\\
\boldsymbol z_i^t
=
\boldsymbol\mu_\theta(o_i^t)
+
\boldsymbol\sigma_\theta(o_i^t)\odot\boldsymbol\epsilon_i^t.
$$
再将样本映射为符合动作约束的移动动作：
$$
a_i^t
=
\mathcal T(\boldsymbol z_i^t)
=
f_\theta(o_i^t,\boldsymbol\epsilon_i^t).
$$
其中，\[ \mathcal T \] 是动作变换。

#### 4. 集中式双 critic 更新

两个 critic 定义为：
$$
Q_{\phi_1}(s,\boldsymbol a),
\qquad
Q_{\phi_2}(s,\boldsymbol a).
$$
它们评估的是**联合移动动作对应的团队软价值**，而不是某架 UAV 自己的即时奖励。

经验回放池存储同一环境时间步的联合转移：
$$
\left(
s^t,\boldsymbol o^t,\boldsymbol a^t,r^t,
s^{t+1},\boldsymbol o^{t+1},d^t
\right).
$$
其中，\(d^t\) 为任务终止标记。有限时域任务到达真正的任务终点时停止 bootstrap；若仅因采样时间限制截断持续任务，则保留 bootstrap，不能把两者混为一谈。

回放池中的 \(\boldsymbol o^t,\boldsymbol o^{t+1}\) 保存固定部分的原始观察，以及每架 UAV 当时实际发现的 UE 特征列表。保存数据副本，避免后续环境更新覆盖历史记录；不保存旧编码向量作为训练输入。采样单位仍是同一时刻的联合转移，不能独立抽取各 UAV 的记录后拼成联合样本。

**无填充的集合批处理：** 对包含 \(K\) 个联合转移的 minibatch，将当前观察中所有 UAV 的 UE 列表拼接成 \(X\in\mathbb R^{L\times4}\)，其中 \(L=\sum_{b=1}^K\sum_{i=1}^U n_{b,i}^{obs}\)。为每行记录所属的“样本—UAV”组索引，共有 \(KU\) 组。对 \(X\) 运行共享编码器后，按组使用求和聚合，再除以各组人数，恢复为 \([K,U,d_h]\) 特征；空组显式输出零。即使整个 minibatch 没有 UE，也保留全部 \(KU\) 个空组，不对空张量求平均。组索引和人数是批处理元数据，不是用户特征或掩码。下一时刻观察单独打包，它的总用户行数和分组可以与当前时刻不同。

从回放池采样后，使用**当前编码器和当前策略主干**重新编码下一时刻的原始局部观察，再生成：
$$
a_i'\sim\pi_\theta(\cdot\mid o_i'),
\qquad
\boldsymbol a'=(a_1',\ldots,a_U').
$$
构造目标：
$$
y
=
r+
\beta(1-d)
\left[
\min_{k=1,2}Q_{\bar\phi_k}(s',\boldsymbol a')
-
\alpha\sum_{i=1}^{U}
\log\pi_\theta(a_i'\mid o_i')
\right]
$$
critic 损失为：
$$
L_Q(\phi_k)
=
\mathbb E_{\mathcal D}
\left[
\left(Q_{\phi_k}(s,\boldsymbol a)-y\right)^2
\right],
\quad k=1,2.
$$
使用两个目标 critic 的较小值构造学习目标，以缓解价值过估计。优化 critic 时，将整个目标 \(y\) 视为常量：下一时刻 UE 编码、动作采样、log 概率与目标 critic 计算均停止梯度。因此 critic 更新只改变 \(\phi_1,\phi_2\)，不会改变 \(\psi,\omega\)。本方案不需要额外的目标 actor 或目标 UE 编码器。

目标网络进行软更新：
$$
\bar\phi_k
\leftarrow
(1-\eta)\bar\phi_k+\eta\phi_k,
\quad k=1,2.
$$
其中，\[ \eta \] 为目标网络软更新系数。

#### 5. actor 与温度参数更新

**(1) actor 更新**

从回放池取出当前状态和原始局部观察，在梯度开启的条件下使用当前编码器重新计算 UE 集合特征，再由共享 actor 重新采样所有 UAV 的动作：
$$
\tilde a_i=f_\theta(o_i,\epsilon_i),
\qquad
\tilde{\boldsymbol a}=(\tilde a_1,\ldots,\tilde a_U).
$$
最小化：

$$
L_\pi(\theta)
=
\mathbb E_{\mathcal D,\boldsymbol\epsilon}
\left[
\alpha\sum_{i=1}^{U}
\log\pi_\theta(\tilde a_i\mid o_i)
-
\min_{k=1,2}
Q_{\phi_k}(s,\tilde{\boldsymbol a})
\right].
$$
更新 actor 时固定 critic 参数，并将温度 \(\alpha\) 视为常量，但保留 critic 对动作的梯度，不能对整个 critic 前向计算使用停止梯度。梯度通过联合动作、策略主干、集合聚合传回 UE 编码器；全部 UAV 的动作分支共同参与共享参数更新：
$$
(\psi,\omega)\leftarrow(\psi,\omega)
-\eta_\pi\nabla_{(\psi,\omega)}L_\pi.
$$
actor 优化器必须同时包含 \(\psi\) 和 \(\omega\)，且不对集合特征执行 detach。用户列表的发现、筛选与组索引不需要可微；反向传播只经过编码器、聚合及后续策略计算。空用户组不向 UE 编码器贡献梯度，但仍通过自身、基站和邻居信息训练策略主干。

**(2) 温度更新**

全体 UAV 共用的温度参数 \( \alpha \)，定义：
$$
\xi=\log\alpha,\qquad \alpha=e^\xi.
$$
使用以下损失函数更新：
$$
L_{\mathrm{temp}}(\xi)
=
-\mathbb E
\left[
\xi\,
\operatorname{sg}
\left(
\sum_{i=1}^{U}
\log\pi_\theta(\tilde a_i\mid o_i)
+
\mathcal H_{\mathrm{tar}}
\right)
\right]
$$
其中，\(\operatorname{sg}\) 表示停止梯度，\(\mathcal H_{\mathrm{tar}}\) 是**联合动作策略的目标熵**。温度更新只改变 \(\xi\)，不更新 UE 编码器或策略主干。加入集合编码器并未改变每架 UAV 的二维动作空间，因此目标熵不随感知用户数变化，也不按集合特征维度设置。

#### 6. 训练流程与分布执行

**Input:** episode count E, horizon T, batch size K, replay threshold L_start ≥ K, discount β, soft-update coefficient η, and joint target entropy H_tar.

**Output:** shared actor parameters θ = (ψ, ω), including the UE encoder and policy head.

```text
 1: Initialize shared UE encoder φ_ψ and policy head Head_ω; set θ = (ψ, ω).
 2: Initialize twin critics Q_φ₁, Q_φ₂, target critics φ̄ₖ ← φₖ,
    log temperature ξ, α ← exp(ξ), and joint replay buffer D.
 3: for episode = 1, …, E do
 4:     Reset environment; obtain global state s and raw local observations o,
        including each UAV's variable-length detected-UE list.
 5:     for t = 0, …, T − 1 do
 6:         Without gradients, encode each local UE set with current ψ,
            aggregate by UAV, and sample aᵢ from the full policy π_θ(· | oᵢ).
 7:         Execute joint action a; update positions, A, B, downlink SINR Y,
            uplink discovery measurements, and service association C.
 8:         Compute reward r; construct s′ and raw observations o′ from the
            updated discovery sets and association; obtain terminal flag d.
 9:         Store raw joint transition (s, o, a, r, s′, o′, d) in D.
10:         if |D| ≥ L_start then
11:             Sample K aligned joint transitions.
12:             Pack current and next UE lists separately; retain group indices
                and counts for all K × U groups, including empty groups.
13:             Without gradients, re-encode next UE lists with current ψ,
                sample next joint actions with current ω, and construct target y
                using target critics and the joint log probability.
14:             Update only φ₁ and φ₂ using critic losses and stored joint actions.
15:             Freeze critic weights; enable gradients through their action inputs.
16:             Re-encode current UE lists with ψ; sample new joint actions;
                minimize actor loss jointly over ψ and ω with α held constant.
17:             Restore critic weights' trainability.
18:             Update ξ using detached joint log probabilities from step 16;
                set α ← exp(ξ).
19:             Soft-update target critic parameters φ̄ₖ.
20:         end if
21:         Set s ← s′ and o ← o′; break if the episode ends.
22:     end for
23: end for
24: return (ψ, ω).
```

执行阶段，每架 UAV 部署共享 UE 编码器和策略主干，只处理本机原始局部观察；训练时的 batch 分组不是 UAV 之间的信息交换。部署包应同时保存编码器、策略主干及所用特征缩放配置，不需要 critic 或全局用户数。当前集中式 critic 仍可使用仿真器提供的全局状态；actor 对可变用户数的支持不意味着现有固定维度 critic 自动支持跨全局用户规模训练。


## 五、新版实验实现约定（beacon-set-v2）

实现中的 UE 行为相对坐标、上行 SNR/30 dB、本机服务标记；保留全部当前发现 UE，不按服务容量截断。共享编码器为 4→32→32 ReLU，按 UAV 求均值；空组输出零，再拼接感知人数/20。感知人数可以大于 1，累计发现量仅作历史诊断，不进入观察或奖励。主场景每架 UAV 固定部分 91 维，编码后 124 维，全局状态 6643 维；原始观察是固定数组、连续 UE 行及偏移三部分。

每步关联候选同时满足本机发现、下行 SINR 达标、任一 BS 可达。发现不依赖核心网可达。通信计算使用 float64，模型与回放 float32。600 步为真正终止，不 bootstrap；UE 越界反射。

SAC 折扣 0.995，actor（含编码器）学习率 1e-4，critic 3e-4，温度 1e-4，初始温度 0.01，联合目标熵 −32。经验池保存 50,000 条原始联合转移；当前与下一观察分别打包，禁止保存编码后的历史特征作为训练输入。完整检查点按最多 256 条转移分块写入、原子发布，按块在 CPU 恢复。旧观察模型明确拒绝加载。MAPPO 和 MATD3 采用相同原始观察与编码器。

完整默认值见 `configs/main.yaml`，数值与故障语义见 `docs/model_contract.md`。本地验证设置见 `configs/local_validation.yaml`，仅用于正确性和短流程检查；正式训练与多种子实验等待通知。熵正则有助于动作探索，但不保证形成协调展开、更高覆盖或超过 MAPPO。
