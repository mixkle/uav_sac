# UAV 多跳覆盖：完整奖励连续 MAPPO baseline 建模

> 核对日期：2026-10-29。本文对应 [mappo_full_reward_baseline_1m.yaml](configs/mappo_full_reward_baseline_1m.yaml) 及当前实际代码。文件名沿用历史名称 `ENVIRONMENT MODELING DISCRETE MAPPO.md`；**当前 baseline 使用连续动作、共享 actor 和解析时间因子 critic**。本文不再定义九动作离散、纯覆盖 C 组或旧短预算实验。

## 一、任务定义与基准范围

本研究考虑地面通信覆盖不足时的 UAV 应急接入与多跳回传部署。UAV 在固定高度自主调整水平位置，发现并服务移动地面用户（UE），通过直接或 UAV 中继路径连接任一可用基站（BS），共同优化联网服务覆盖、总吞吐、单 UAV 故障保留能力和正常运行时的联网程度。

使用集中式训练、分布执行（CTDE）：18 架 UAV 共享一个局部 actor；集中式 critic 读取全局物理状态，输出一个团队价值。执行时每架 UAV 仅使用其局部观察及一步历史。UE 关联由固定的集中贪心规则计算，不是 actor 的动作，也不是一个额外学习阶段。

这是有限时域的合作部分可观测控制任务。各机条件独立地从共享策略采样，联合策略为

\[
\pi_\theta(\mathbf a_t\mid\mathbf o_t)
=\prod_{i=1}^{U}\pi_\theta(a_{t,i}\mid o_{t,i}),\qquad
\max_\theta\ \mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

期望包含随机初始布局、UE运动及策略采样。第八节的PPO和熵正则是实际优化方法；这里的任务回报本身只含环境奖励。

本 baseline 固定如下条件：

- 3500×3500 m 区域，18 UAV、150 UE、3 BS；UAV 全场均匀初始化，UE 中心2500×2500 m均匀初始化后持续移动。
- 连续二维高斯经圆盘变换生成动作，最大30 m/步，边界反射，400步真正终止。
- 相对坐标局部观察、全部当前发现 UE 的集合编码、4维一步历史。
- 完整团队奖励为覆盖率、归一化吞吐、故障保留覆盖及断联惩罚四项；边界惩罚权重为0。
- 共享连续 MAPPO，解析时间因子团队 critic，从头训练种子0、1,000,000条联合转移。

这份配置是后续正式训练基准，其模型与优化器均重新初始化；已保留的纯覆盖 PPO/SAC 运行只作历史对照，不能视作本 baseline 的结果。本文描述已实现的方法，不预设正式训练会提高覆盖或维持部署。

## 二、场景、初始化与运动

### 2.1 节点与几何

记 UAV、UE、BS 集合分别为

\[
\mathcal U=\{1,\ldots,U\},\quad
\mathcal M=\{1,\ldots,M\},\quad
\mathcal G=\{1,\ldots,G\}.
\]

本基准 \(U=18,M=150,G=3\)，水平区域 \(\mathcal X=[0,W]\times[0,L]\)，\(W=L=3500\) m。节点位置为

\[
p_i^t=(x_i^t,y_i^t,h),\quad h=100\ \mathrm m;
\qquad p_m^t=(x_m^t,y_m^t,0),
\]

\[
p_1^{\rm BS}=(0,0,0),\quad
p_2^{\rm BS}=(0,3500,0),\quad
p_3^{\rm BS}=(3500,0,0).
\]

距离采用三维欧式距离；UAV 等高，因此机间三维距离等于水平距离。所有 BS 始终可用并可接入核心网，研究的故障对象为单 UAV，而非 BS。

### 2.2 初始化分布

每架 UAV 的水平位置在整个 \(\mathcal X\) 内独立均匀采样。每个 UE 在

\[
[500,3000]\times[500,3000]\ \mathrm m
\]

内独立均匀采样，对应 `user_initialization: uniform_interior`、`user_initial_margin_fraction: 1/7`。

不按初始覆盖、连通性、BS接入或算法表现筛选场景，不强制 UAV 分组、两端出发或形成初始多跳网络。UE 的中心范围只约束初始化；之后 UE 可以在整个3500 m区域运动。配置中的高斯混合参数和 `launch_radius` 在当前初始化模式下不生效。

中心初始化的初始 UE 密度为24人/km²；全场均匀150人时约12.24人/km²。因此中心初始化同时改变初始密度及 UE 与 BS 的相对位置，不能仅解释为“避免边缘用户”。场景规模等参考背景见 [UE初始化参考说明](docs/ue_initialization_reference.md)；中心留白、UAV随机初始化及当前无线参数应作为本研究的具体假设说明。

### 2.3 时间、UE运动与边界

时刻 \(t=0,\ldots,T\)，\(T=400\)，动作发生于 \(t=0,\ldots,399\)，\(\Delta t=1\) s。执行动作 \(a^t\) 后得到时刻 \(t+1\) 的拓扑、关联与奖励。UAV 运动见第六节。

UE 每步独立采样

\[
v_m^t=\operatorname{clip}(\xi_m^t,-3,3),\quad
\xi_m^t\sim\mathcal N(0,1^2)\ \mathrm{m/s},
\qquad \varphi_m^t\sim\operatorname{Unif}[0,2\pi),
\]

\[
\mathbf q_m^{t+1}
=\mathcal R_{W,L}\!\left(
\mathbf q_m^t+\Delta t\,v_m^t[\cos\varphi_m^t,\sin\varphi_m^t]
\right).
\]

这里是每步重新采样的有符号幅度与方向，不保留上一速度。UE 运动使用独立随机流，初始化或验证采样不会消费训练的动作随机流。

UAV 与 UE 均使用逐轴反射。对长度为 \(D\) 的一维区间，定义

\[
\mathcal R_D(q)=\min\{q\bmod 2D,\ 2D-(q\bmod 2D)\}.
\]

二维反射分别作用于 x、y，可处理越界位移。模型没有航向状态、惯性、加速度、碰撞或能耗约束。边界反射是动力学；当前没有边界奖励成本。

## 三、无线、发现、回传及服务关联

### 3.1 UAV–UE 平均路径损耗

水平距离为 \(r_{im}\)，三维距离和仰角为

\[
d_{im}=\sqrt{r_{im}^2+h^2},\qquad
\vartheta_{im}=\frac{180}{\pi}\operatorname{atan2}(h,r_{im}).
\]

\[
p_{im}^{\rm LoS}
=\frac{1}{1+a\exp[-b(\vartheta_{im}-a)]},
\]

\[
\ell_{im}
=20\log_{10}\!\left(\frac{4\pi f_c d_{im}}{c_0}\right)
+p_{im}^{\rm LoS}\eta_{\rm LoS}
+(1-p_{im}^{\rm LoS})\eta_{\rm NLoS}.
\]

本基准 \(f_c=2.4\) GHz，\(c_0=299792458\) m/s，\(a=9.61,b=0.16\)，附加损耗为1/20 dB。仰角按度代入概率公式；距离使用米、频率使用Hz。代码计算概率加权的平均dB损耗，不每步随机抽取 LoS/NLoS，也没有小尺度快衰落。

上下行采用同一损耗；该模型及参数是传播近似，不代表实测场景校准结果。

### 3.2 下行SINR与速率

每架 UAV 总带宽 \(B_{\rm tot}=20\) MHz、总功率 \(P_{\rm tot}=30\) dBm、最大服务容量 \(K=20\)。每用户子带固定为

\[
B_{\rm sub}=B_{\rm tot}/K=1\ \mathrm{MHz},\qquad
P_{\rm sub}=P_{\rm tot}-10\log_{10}K\approx16.9897\ \mathrm{dBm}.
\]

每子带功率为50 mW；不会因当前服务用户较少而重新增大每用户带宽或功率。

噪声谱密度为−174 dBm/Hz，下行噪声系数为15 dB：

\[
N_{\rm dl,dBm}=-174+10\log_{10}B_{\rm sub}+15=-99\ \mathrm{dBm},
\quad N_{\rm dl}=10^{N_{\rm dl,dBm}/10}\ \mathrm{mW}.
\]

令 \(S_{im}=10^{(P_{\rm sub}-\ell_{im})/10}\) mW，则

\[
Y_{im}=10\log_{10}\frac{S_{im}}
{N_{\rm dl}+\sum_{j\in\mathcal U\setminus\{i\}}S_{jm}},
\qquad
R^{\rm cand}_{im}=B_{\rm sub}\log_2(1+10^{Y_{im}/10}).
\]

UAV 内部子带正交，不同 UAV 复用同一频段。干扰采用满负载近似：所有其他 UAV 均产生干扰，不取决于是否联网、是否服务用户或真实子带占用。BS 不加入该接入频段干扰。

### 3.3 上行信标发现

UE 发射独立发现信标，功率 \(P_{\rm beacon}=26.0205999133\) dBm（0.4 W）、带宽180 kHz、UAV接收噪声系数15 dB。信标与下行正交，忽略信标间干扰：

\[
N_{\rm beacon,dBm}=-174+10\log_{10}(180000)+15
\approx-106.4473\ \mathrm{dBm},
\]

\[
S^{\rm ul}_{im}=P_{\rm beacon}-\ell_{im}-N_{\rm beacon,dBm},\qquad
\mathcal D_i^t=\{m:S^{\rm ul,t}_{im}\ge20\ \mathrm{dB}\}.
\]

信标提供用户标识与位置。断网 UAV 仍可发现 UE；发现不等于可服务。发现范围由链路预算导出，不硬编码圆形半径；下行服务范围还受干扰与回传影响。累计发现只作诊断，不进入观察或奖励。

### 3.4 UAV回传图与最短跳数

定义机间邻接矩阵

\[
A_{ij}^t=\mathbf1\{\|p_i^t-p_j^t\|\le650\ \mathrm m\}.
\]

矩阵对称，代码保留 \(A_{ii}=1\)。计算真实邻居时排除自身。

当 \(\|p_i^t-p_g^{\rm BS}\|\le1500\) m 时 UAV 可直接接入该 BS。此处使用三维距离，高度100 m对应水平直接接入半径约1496.66 m。

对每个 BS 单独执行最短路搜索，形成 \(B^t\in\mathbb N^{U\times G}\)：直接接入跳数为1，经 \(k-1\) 个 UAV 中继为 \(k\)，不可达编码为 \(U+1=19\)。令

\[
b_i^t=\min_g B_{ig}^t,\qquad z_i^t=\mathbf1\{b_i^t\le U\}.
\]

BS 是路径终点，不是 UAV 之间的中继节点；不设置 BS–BS 边或经核心网隐藏转发的机间路径。允许多个互不连通的 UAV 子网分别接入任一 BS。回传采用距离门槛，不计算回传SINR、拥塞、路由时延或容量限制；每步视为路由即时收敛。

### 3.5 唯一关联与容量分配

UE \(m\) 的候选 UAV 集合为

\[
\mathcal K_m^t=\{i:m\in\mathcal D_i^t,\ Y_{im}^t\ge-3\ \mathrm{dB},\ z_i^t=1\}.
\]

关联矩阵 \(C_{im}^t\in\{0,1\}\) 满足

\[
\sum_i C_{im}^t\le1,\qquad
n_i^t=\sum_m C_{im}^t\le K.
\]

关联采用固定规则，不保证全局最优匹配：

1. 先按候选数 \(|\mathcal K_m^t|\) 升序处理 UE，同数时按UE编号升序；候选数是分配容量之前计算的。
2. 对当前 UE，从仍有容量的候选 UAV 中选择下行SINR最高者。
3. SINR相同时选择最短联网跳数更小者，再平局取UAV编号更小者。
4. 无可用容量则该UE本步不服务。

实际用户速率为 \(v_m^t=\sum_i C_{im}^tR^{\rm cand,t}_{im}\)。每步重新计算关联；观测仍保留所有已发现UE，不能只保留最终关联用户或按20人容量截断。

## 四、完整团队奖励与故障评估

### 4.1 四个物理量

**联网服务覆盖率：**

\[
\rho^t=\frac1M\sum_{i,m}C_{im}^t=\frac1M\sum_i n_i^t.
\]

只计算实际服务用户，发现但未关联或不能接入核心网的UE不计入覆盖。

**归一化总吞吐：**

\[
\tau^t=\frac{\sum_m v_m^t}{M v_{\rm ref}},
\qquad v_{\rm ref}=5\times10^6\ \mathrm{bit/s}.
\]

吞吐奖励不直接使用bps，不除以实际服务人数，不裁剪到 \([0,1]\)。

**单 UAV 故障保留覆盖：** 对每个失效节点 \(f\)，从回传图中删除该节点并重新计算剩余 UAV 到所有 BS 的可达性 \(z_i^{t,(-f)}\)。保留名义关联及名义SINR，不重新关联UE、不重新分配容量，也不因干扰源失效重新计算链路速率：

\[
\rho^{t,(-f)}=\frac1M\sum_{i\ne f}n_i^t z_i^{t,(-f)},\qquad
J_{\rm rob}^t=\frac1U\sum_f\rho^{t,(-f)}.
\]

失效节点自身服务的UE全部丢失；其他节点可切换到另一BS路径以保留其原关联用户。该评估是固定关联的单故障诊断，并非向正常训练轨迹实际注入故障。它满足

\[
0\le J_{\rm rob}^t\le\left(1-\frac1U\right)\rho^t.
\]

“同时可达多个BS”不保证路径节点独立，不等同于抗单节点故障保证。

**正常运行断联率：**

\[
D^t=\frac1U\sum_i(1-z_i^t).
\]

它是当前正常状态下断网UAV比例，不是新增断网人数，也不是故障评估中的失联比例。

### 4.2 动作后的团队奖励

训练收到的标量奖励为

\[
\boxed{
r_t=\rho^{t+1}+0.1\tau^{t+1}
+0.5J_{\rm rob}^{t+1}-0.1D^{t+1}.
}
\]

权重对应 `throughput_weight=0.1`、`robustness_weight=0.5`、`disconnection_weight=0.1`。`reward_primary=coverage`；感知率不是主奖励。`boundary_penalty_weight=0`，因此当前训练奖励与原四项 `task_reward` 相等。

所有 actor 使用同一团队奖励与团队优势。虽然 \(\rho\) 可以分解为各机服务人数贡献，当前并不按本机覆盖单独计算GAE，也不对中继节点另设局部奖励。奖励允许负值；完整奖励及其未来平均值不限制在 \([0,1]\)。

本基准未启用势函数塑形、发现奖励、贴边成本、运动成本、能耗或碰撞惩罚。MAPPO熵正则只出现在actor优化目标中，不加入环境奖励或critic的真实回报标签。

## 五、状态、局部观察与集合编码

### 5.1 全局物理状态

集中式critic输入按如下顺序展平、拼接：

\[
s_t=[\widetilde p_{\mathcal U}^t,
\widetilde p_{\mathcal G},\widetilde p_{\mathcal M}^t,
\operatorname{vec}A^t,\operatorname{vec}(B^t/(U+1)),
\operatorname{vec}C^t,\operatorname{vec}(Y^t/30),t/T].
\]

位置按 \((W,L,h)\) 归一化，高度槽位UAV为1、地面节点为0；这里使用绝对归一化位置，不使用各actor的相对坐标。A、C保留二值，Y是未按候选条件过滤的全部下行SINR。状态不额外加入发现矩阵、用户集合编码、速率矩阵或actor历史。

\[
d_s=3(U+G+M)+U^2+UG+2UM+1=6292.
\]

`t/T` 为最后一个字段，解析时间因子直接从该字段获得。

### 5.2 当前固定局部特征

令 \(c_i^t=b_i^t/(U+1)\)，联网直连为 \(1/19\)，断网为1。自身特征为

\[
f_i^{\rm self,t}
=[x_i^t/W,y_i^t/L,1,c_i^t,n_i^t/K,A_{i,:}^t],
\]

共 \(5+U=23\) 维。自身坐标为绝对归一化位置；“相对坐标模式”并不将自身位置设为零。

BS位置为预先已知信息，按配置顺序提供相对本机偏移：

\[
f_i^{\rm BS,t}
=\left[\frac{x_g^{\rm BS}-x_i^t}{W},
\frac{y_g^{\rm BS}-y_i^t}{L}\right]_{g=1}^G,
\]

共6维，不加入BS的额外高度槽位。

从所有直接相连且非自身的UAV中，按距离、编号排序取最近三架。每个有效邻居 \(j\) 的槽位为

\[
f_{ij}^{\rm nbr,t}
=[j/U,(x_j^t-x_i^t)/W,(y_j^t-y_i^t)/L,
1,c_j^t,n_j^t/K,A_{j,:}^t],
\]

这里数学编号为1到U，对应代码 `(index+1)/U`。每槽 \(6+U=24\) 维，不足三架则整槽补零。有效编号为正，零槽不需额外mask维度。邻居可能属于断网子网；直接相连不意味着已联网。自身邻接行仍包含全部直接邻居的连接标记，三槽限制的是邻居详细状态。

当前固定特征共

\[
d_{\rm current}=23+6+3\times24=101.
\]

### 5.3 一步历史及时间对齐

在101维当前固定特征末尾追加

\[
h_i^t=
[a_{i,x}^{t-1},a_{i,y}^{t-1},c_i^{t-1},|\mathcal D_i^{t-1}|/K].
\]

历史动作是上一条归一化圆盘指令，不是高斯latent，也不是反射后的实际位移。执行 \(a_t\) 前保存当前跳数及感知人数；返回 \(t+1\) 观察时，历史为 \([a_t,c_t,|\mathcal D_t|/K]\)。

初始化时 \(h_i^0=[0,0,c_i^0,|\mathcal D_i^0|/K]\)。读取观察、刷新无线或恢复快照不额外推进历史；新回合重置，避免跨回合泄漏。感知人数除20，允许超过1。

当前原始固定观察维度为 \(101+4=105\)。不增加上一服务人数、历史UE集合、累计发现量、显式差分或RNN。

### 5.4 当前UE集合与actor编码

每个已发现UE的原始特征为

\[
e_{im}^t=
[(x_m^t-x_i^t)/W,(y_m^t-y_i^t)/L,
S_{im}^{\rm ul,t}/30,C_{im}^t]\in\mathbb R^4.
\]

集合按UE编号打包，但神经聚合对集合排列不敏感。全部已发现UE均进入集合，服务标记可为0，断网UAV的集合也可非空。固定物理尺度归一化不改变空槽零值，不对SNR或人数额外做范围裁剪。

UE编码器属于actor，所有UAV共用 \(h_\psi:4\to32\to32\)，两层ReLU：

\[
\bar e_i^t=\begin{cases}
|\mathcal D_i^t|^{-1}\sum_{m\in\mathcal D_i^t}h_\psi(e_{im}^t),&|\mathcal D_i^t|>0,\\
\mathbf0\in\mathbb R^{32},&|\mathcal D_i^t|=0.
\end{cases}
\]

actor内部输入为

\[
\chi_i^t=[f_i^{\rm current,t},h_i^t,\bar e_i^t,|\mathcal D_i^t|/K],
\qquad d_\chi=101+4+32+1=138.
\]

环境提供固定数组、连续UE行及分组偏移，编码在actor内实时计算。rollout保存原始观察副本，PPO每次更新重新编码，不缓存旧网络编码作为输入。

### 5.5 部分可观性与critic输入限制

各机无法读取全局UE列表、远处UAV坐标或全局状态，也不直接读取全局时间。设完整信息状态为 \(x_t=(s_t,h_{1:U}^t)\)，局部观察可写为 \(o_i^t=g_i(x_t)\)。一步历史是环境递推生成的策略输入，不是完整历史记忆。

当前critic仍只计算 \(V_\phi(s_t)\)，没有读取 \(h_{1:U}^t\)。因此相同物理状态可能因历史不同而产生不同的后续动作，critic存在信息遗漏；它是受限输入的团队价值近似，不能宣称 \(s_t\) 对当前历史策略一定充分。该限制与已保留的共享PPO实验一致，本baseline不同时扩展critic输入，后续通过MC误差及动作价值诊断检查其影响。

## 六、连续策略、动作与位置更新

共享策略主干为 \(138\to256\to256\to4\)，隐藏层ReLU，输出二维均值和二维状态相关log标准差：

\[
(\mu_i^t,\ell_i^t)=f_\theta(\chi_i^t),\qquad
\sigma_i^t=\exp(\operatorname{clip}(\ell_i^t,-5,2)).
\]

采样与径向圆盘变换为

\[
z_i^t=\mu_i^t+\sigma_i^t\odot\epsilon_i^t,
\quad\epsilon_i^t\sim\mathcal N(0,I_2),\qquad
\boxed{a_i^t=\frac{z_i^t}{\sqrt{1+\|z_i^t\|^2}}}.
\]

它将 \(\mathbb R^2\) 可逆映射到开单位圆盘，\(\|a_i^t\|<1\)。这里没有九动作编号、Categorical分布或最大概率动作选择。固定世界坐标下，+x为右、+y为前，不引入机头朝向。

\[
\mathbf p_{i,xy}^{t+1}
=\mathcal R_{W,L}\left(\mathbf p_{i,xy}^t+30\Delta t\,a_i^t\right).
\]

圆盘动作同时控制方向和幅度。指令为零时UAV悬停，UE及时间仍推进；连续分布不为精确悬停设置离散概率质量。

变换的Jacobian为

\[
\log|\det J(z)|=-2\log(1+\|z\|^2),
\]

\[
\log\pi_\theta(a_i|o_i)
=\log\mathcal N(z_i;\mu_i,\operatorname{diag}\sigma_i^2)
+2\log(1+\|z_i\|^2).
\]

log概率对应归一化指令，不将反射后实际位移当作策略随机变量；乘30 m的固定单位换算不加入当前实现的概率密度。标量环境输入为 `[U,2]`，向量环境为 `[E,U,2]`，拒绝非法形状、非有限值及超出圆盘的指令。

采集和默认验证均从原策略采样。显式确定性诊断使用高斯均值的圆盘映射，不属于模型选择协议。`run.deterministic=true` 控制运行复现设置，不将随机策略变成确定性策略。

## 七、解析时间因子团队critic

### 7.1 价值定义与参数化

critic预测完整环境奖励的有限回合累计价值，折扣 \(\gamma=0.995\)。对已给定完整状态与策略，理论目标为未来奖励的条件期望；实际网络使用前述受限物理状态输入近似它。

剩余动作次数 \(n_t=T-t\)，定义

\[
H_t=\sum_{k=0}^{n_t-1}\gamma^k
=\frac{1-\gamma^{T-t}}{1-\gamma},\qquad
H_T=0.
\]

\(\gamma=1\) 时使用 \(H_t=T-t\)。本基准 \(H_0\approx173.0684\)、\(H_{399}=1\)、\(H_{400}=0\)。时间取自全局状态末尾已有的 \(t/T\)，无新增状态维度。

保持 \(6292\to256\to256\to1\) 两层ReLU网络。令标量网络输出为 \(f_\phi(s_t)\)，平均奖励归一化统计为 \((\mu_m,\sigma_m)\)：

\[
\widehat m_t=\sigma_m f_\phi(s_t)+\mu_m,\qquad
\boxed{\widehat V_t=H_t\widehat m_t}.
\]

\(\widehat m\) 表示未来奖励的**折扣加权平均**，不是普通算术平均，更不是只预测覆盖率。对MC标签 \(G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}\)，相应平均标签为 \(G_t/H_t\)。只有终止状态H为0，不能相除；训练rollout当前状态均满足H>0。

不使用sigmoid，不把平均预测、GAE标签或累计价值裁到 \([0,1]\) 或 \([0,H_t]\)。完整任务奖励允许负数及超过1，相关越界统计只描述数值分布，不是合法性错误。

### 7.2 初始化与梯度

actor和critic从种子0按当前代码顺序构建，无预训练或旧权重。策略主干与critic由代码的 `mlp()` 构造，其线性层采用正交初始化、零偏置；隐藏层增益为 \(\sqrt2\)，actor最终层增益0.01，critic最终层增益1。UE编码器的两层线性层沿用PyTorch默认初始化，不套用上述正交／零偏置规则。一步历史新增输入列初始化为零，之后参与正常训练。

时间因子模式进一步将critic最终层权重及偏置除以 \(H_0\)，归一化器初始均值0、方差1、计数0。因此初始价值为原MLP输出乘 \(H_t/H_0\)，避免将随机初值直接放大H倍。该初始化不是继承离线校准模型。

actor和critic参数、优化器彼此独立，UE编码器仅由actor优化器更新。actor使用PPO概率比梯度，不通过 \(\nabla_a V\) 或环境运动求导；critic梯度经过 \(H_t\) 回传网络。

## 八、在线MAPPO采集与更新

### 8.1 Rollout与GAE

固定本轮旧策略，16个环境各采集最多128步，即通常2048条联合转移。每条包含全部18架UAV；预算不再乘UAV数量。保存原始观察、全局状态、latent \(z\)、逐机旧log概率、旧累计价值、下一累计价值、完整奖励及终止标志。

在真正下一状态上计算下一价值，再对已结束环境reset。令 \(d_t\) 为真正终止，\(q_t\) 为截断标志，GAE从片段末尾以0开始反向递推：

\[
\delta_t=r_t+\gamma(1-d_t)\widehat V_{\rm old}(s_{t+1})
-\widehat V_{\rm old}(s_t),
\]

\[
A_t=\delta_t+\gamma\lambda_{\rm GAE}(1-d_t)(1-q_t)A_{t+1},
\qquad \lambda_{\rm GAE}=0.95,
\]

\[
\widehat R_t^{\rm GAE}=A_t+\widehat V_{\rm old}(s_t).
\]

第400步为真正终止，环境返回 `terminated=true, truncated=false`，不bootstrap。非终止rollout末端仍使用下一状态累计价值。GAE的所有价值接口均使用 \(\widehat V\)，不能把平均输出 \(\widehat m\) 直接代入。

优势在本轮所有联合转移（时间×环境）上统一标准化：

\[
\widetilde A_t=\frac{A_t-\operatorname{mean}A}
{\operatorname{std}_{\rm population}A+10^{-5}}.
\]

所有UAV使用相同团队优势。注意 \(\widehat R^{\rm GAE}\) 使用标准化之前的A，而非 \(\widetilde A\)。

### 8.2 两套归一化与PopArt

每轮更新前，仅使用该训练rollout的标签更新以下统计：

- 平均任务奖励归一化器：输入 \(\widehat R_t^{\rm GAE}/H_t\)，对critic输出层及其Adam状态做PopArt重标定。
- 累计价值归一化器：输入 \(\widehat R_t^{\rm GAE}\)，只更新统计，用于损失及裁剪，不再次修改输出层。

统计为所有历次输入按样本数累计的总体均值和方差，不是每个minibatch独立标准化，也不是仅使用最新rollout。方差下限为 \(10^{-4}\)，对应标准差至少0.01；不是离线校准实验的固定标准差。

记平均归一化器更新前后统计为 \((\mu_o,\sigma_o)\)、\((\mu_n,\sigma_n)\)，最终层为 \(w,b\)，PopArt使用

\[
w'=(\sigma_o/\sigma_n)w,\qquad
b'=(\sigma_o b+\mu_o-\mu_n)/\sigma_n.
\]

Adam一阶矩乘 \(\sigma_o/\sigma_n\)，二阶矩乘其平方，保持原始尺度 \(\widehat m\) 及 \(\widehat V\) 在统计更新前后不变。GAE目标可能暂时超出真实回报范围，不强行裁剪。验证及测试数据不更新归一化统计。

### 8.3 逐UAV PPO目标与连续熵

每次minibatch重新编码原始UE集合，在采集时保存的同一个latent \(z_{t,i}\) 上重算新log概率：

\[
\eta_{t,i}=\exp\bigl[
\log\pi_\theta(a_{t,i}|o_{t,i})-
\log\pi_{\rm old}(a_{t,i}|o_{t,i})\bigr].
\]

条件联合策略可写为各UAV分布的乘积，但更新使用逐UAV比率，不把18个比率相乘。固定同一latent时，圆盘Jacobian在新旧log概率之差中抵消。

\[
L_{\rm actor}=
-\mathbb E_{t,i}\min\left\{
\eta_{t,i}\widetilde A_t,
\operatorname{clip}(\eta_{t,i},1-\epsilon,1+\epsilon)\widetilde A_t
\right\}
-\beta\mathbb E_{t,i}\widehat{\mathcal H}_i,
\]

其中 \(\epsilon=0.2,\beta=0.01\)，对联合转移和UAV取均值。连续熵估计为

\[
\widehat{\mathcal H}_i
=\mathcal H[\mathcal N(\mu_i,\operatorname{diag}\sigma_i^2)]
-2\log(1+\|\widetilde z_i\|^2),
\quad \widetilde z_i\sim\mathcal N(\mu_i,\operatorname{diag}\sigma_i^2).
\]

高斯熵解析计算，Jacobian期望每次使用新的重参数化样本估计，不能用旧rollout latent代替该新样本。熵梯度可进入均值、标准差和UE编码器。微分熵允许负数，不具有分类策略的 \(\log9\) 上限。

本MAPPO路径不使用SAC自动温度、目标熵、双Q、目标网络、离策略回放或随机warmup。

### 8.4 累计价值空间的Huber与Value Clipping

记本轮更新后的累计价值归一化为

\[
N_V(x)=(x-\mu_V)/\sigma_V.
\]

对当前预测、旧累计价值及GAE目标使用同一统计：

\[
v_t=N_V(\widehat V_\phi(s_t)),\quad
v_t^{\rm old}=N_V(\widehat V_{\rm old}(s_t)),\quad
y_t=N_V(\widehat R_t^{\rm GAE}),
\]

\[
v_t^{\rm clip}=v_t^{\rm old}
+\operatorname{clip}(v_t-v_t^{\rm old},-0.2,0.2).
\]

定义与PyTorch一致的Huber残差损失

\[
\ell_\delta(e)=\begin{cases}
\frac12e^2,&|e|\le\delta,\\
\delta(|e|-\frac12\delta),&|e|>\delta,
\end{cases}\qquad\delta=10.
\]

\[
L_V=\mathbb E_t\max\{\ell_{10}(v_t-y_t),
\ell_{10}(v_t^{\rm clip}-y_t)\}.
\]

这是累计价值空间的损失，不是直接拟合平均任务奖励的损失。保持Value Clipping开启；actor的PPO clipping与价值裁剪是两项独立操作。critic每个minibatch更新一次，不能因18个UAV而重复18次。

### 8.5 优化、预算和保存

actor与critic各自使用Adam，学习率均 \(5\times10^{-4}\)，eps=\(10^{-5}\)，默认betas=(0.9,0.999)，无权重衰减；固定学习率，无衰减调度。每个epoch对联合转移重新随机排列，切为4个minibatch，保留每条转移的全部UAV；每轮10个epoch。

actor和critic分别按全局梯度范数上限0.5裁剪。共享actor只有一套参数与优化器，编码器与策略主干一起更新；critic不共享编码器。当前没有基于KL的提前停止，KL和clip比例用于诊断。

预算1,000,000条联合转移：488个完整2048条rollout后剩576条，最后采集36个向量步再更新，共489轮PPO、各19560次actor/critic优化，不补足到额外完整rollout。

rollout使用后丢弃。每约10分钟在更新完成的安全点发布完整检查点，包含模型、优化器、两套归一化、环境与历史数组、随机状态、进度及历史最佳，保留最近两份。验证和完整保存前先写轻量 `last.pt`。`best.pt/last.pt`可用于推理，但不能替代完整恢复目录。跨奖励、观察、动作或共享模式的直接续训应被拒绝。

## 九、评估、诊断及结果口径

训练开始前评估step0，之后每跨过约10000条转移的完整更新边界评估，并记录实际步数；训练结束评估last。验证使用32场景10000–10031、采样策略、动作种子880001，batch16，与训练随机流隔离。保留测试场景为20000–20099，未用于训练选模。

`best_metric=task_return`，按各验证场景**400步未折扣完整奖励总和的均值**选择best，包含step0。当前无边界成本，所以task_return与训练reward之和相同。它不等价于纯覆盖排序，也不按终态选模；训练的折扣价值与选模的未折扣回报应分别说明。

主要比较同预算last，best作为补充。至少报告：

- 全回合、最后100步及终态的覆盖、感知、吞吐、故障保留、断联和四项实际奖励贡献；总回报提高不能代替覆盖改善。
- 故障保留比例 \(\sum_tJ_{\rm rob}^t/\sum_t\rho^t\)，零覆盖回合单独标记。
- 覆盖峰值及部署维持：例如 \(\overline\rho_{301:400}-\overline\rho_{51:150}\)，两段均按动作后的1–400步编号，同时报告绝对覆盖。
- 指令与实际位移、条件动作方差、近悬停、靠边、反射、UAV间距、聚集与联网链路保持；悬停增加本身不是有效部署证据。
- 感知、下行、回传、容量四阶段淘汰人数；它们与已服务UE构成互斥分解。
- actor/critic loss、连续熵、KL、PPO clipping、价值裁剪触发比例、梯度、归一化统计、耗时和内存。

冻结actor/critic后可采集完整验证回合，计算

\[
G_t=r_t+\gamma G_{t+1},\qquad G_T=0,
\]

比较 \(\widehat V_t\) 与 \(G_t\)、\(\widehat V_t/H_t\) 与 \(G_t/H_t\)，检查全程与末端误差、相同时刻的场景区分能力。标签使用完整四项环境奖励，不加MAPPO熵。平均曲线正确或loss较小，均不足以证明动作优劣判断正确。

固定展示场景10000、10001的初始、中间及终态拓扑，不按效果挑选。区分已服务、仅发现、未发现UE及UAV核心网可达性。训练日志中的瞬时采集均值不直接等同于整回合验证均值。

保存展开配置、源码及依赖信息、种子、日志、模型、逐回合指标和必要轨迹。图表从保存数据生成PDF/SVG/高分辨率PNG。单训练种子的场景bootstrap只能解释为场景差异区间，不能标为训练种子置信区间。历史纯覆盖及不同预算结果不作为同奖励严格对照。

## 十、完整参数索引

### 10.1 场景与环境

|参数／配置字段|baseline值|
|---|---|
|区域 `width/height`|3500 / 3500 m|
|节点 `num_uavs/num_users`；BS数量|18 / 150；3|
|BS位置（m）|(0,0,0)、(0,3500,0)、(3500,0,0)|
|高度 `altitude`|100 m|
|UAV初始化 `uav_initialization`|`uniform`，全区域|
|UE初始化 `user_initialization`|`uniform_interior`，中心[500,3000]²|
|UE留白 `user_initial_margin_fraction`|1/7|
|UE运动 `user_speed_std/user_speed_clip`|1 / 3 m/s|
|任务 `horizon/dt/max_speed`|400步 / 1 s / 30 m/s|
|动作／边界 `action_space/uav_boundary`|`continuous` / `reflect`|
|通信距离 `uu_range/ug_range`|650 m / 1500 m（三维）|
|容量 `capacity`|20 UE/UAV|
|载频／总带宽／总功率|2.4 GHz / 20 MHz / 30 dBm|
|噪声谱密度／下行NF／信标NF|−174 dBm/Hz / 15 dB / 15 dB|
|LoS `los_a/los_b/los_extra_db/nlos_extra_db`|9.61 / 0.16 / 1 / 20 dB|
|下行 `sinr_threshold_db`|−3 dB|
|信标功率／带宽／SNR门槛|26.0205999133 dBm / 180 kHz / 20 dB|
|奖励 `reward_primary`|`coverage`|
|吞吐／鲁棒性／断联权重|0.1 / 0.5 / 0.1|
|速率参考 `rate_reference_bps`|5,000,000 bit/s|
|边界惩罚权重／缓冲距离|0 / 150 m；权重0时不进入奖励|
|观察坐标／历史|`relative` / `one_step`|
|邻居槽／SNR-SINR缩放|3 / 除以30 dB|
|状态／当前固定／含历史固定／actor输入维度|6292 / 101 / 105 / 138|
|物理后端／精度|Numba＋NumPy / float64；网络及观察float32|

### 10.2 网络与MAPPO

|参数／配置字段|baseline值|
|---|---|
|算法／共享 `name/actor_sharing`|`mappo` / `shared`|
|UE编码器 `ue_hidden`|4→32→32，ReLU，均值＋人数|
|策略 `hidden/mappo_state_dependent_std`|138→256→256→4 / `true`|
|log std范围|[−5,2]|
|critic `critic_parameterization`|`time_factor`；6292→256→256→1|
|actor／critic学习率|均5e-4|
|Adam eps／betas／weight decay|1e-5 / (0.9,0.999) / 0|
|折扣 `discount`／GAE `gae_lambda`|0.995 / 0.95|
|PPO `ppo_clip`／`value_clipping`|0.2 / `true`|
|价值损失|累计归一化价值Huber，delta=10；Value Clipping|
|归一化|联合转移优势标准化；平均奖励PopArt＋累计价值统计|
|熵 `entropy_coef`|0.01，固定；变换后连续微分熵|
|rollout／epochs／minibatches|128 / 10 / 4|
|梯度范数 `max_grad_norm`|actor、critic分别0.5|
|目标网络／回放／warmup|MAPPO路径均不使用|

### 10.3 运行、评估及保存

|参数／配置字段|baseline值|
|---|---|
|种子／预算 `seed/total_steps`|0 / 1,000,000条联合转移|
|设备 `device`／并行环境 `num_envs`|CPU / 16|
|向量／环境后端|`batch` / `numba`|
|PyTorch／Numba线程|8 / 4|
|PPO轮数／每网络优化次数|489 / 19560|
|验证场景／保留测试场景|10000–10031 / 20000–20099|
|验证策略／动作种子|`sample` / 880001|
|选模 `best_metric`|`task_return`，完整奖励全回合总和的场景均值|
|验证间隔／日志间隔|约10000条的更新边界 / 1000条|
|TensorBoard／复现开关|`true` / `deterministic: true`|
|完整保存间隔／保留数|10分钟的更新安全点 / 最近两份|
|进程RSS上限 `memory_limit_gib`|12 GiB|
|时间／价格／预算上限|当前均未设置；总步数限定为1M|

## 十一、代码对应与手动运行

- [配置与维度](src/uav_research/config.py)、[baseline配置](configs/mappo_full_reward_baseline_1m.yaml)。
- [物理、初始化、历史和奖励](src/uav_research/environment.py)；[路由、关联、故障及原始观察](src/uav_research/kernels.py)。
- [集合编码、圆盘策略、归一化和时间因子](src/uav_research/models.py)；[GAE、PPO和价值更新](src/uav_research/algorithms.py)。
- [训练与检查点流程](src/uav_research/train.py)、[验证与选模指标](src/uav_research/evaluate.py)。

手动从头启动：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python -u -m uav_research train \
  --config configs/mappo_full_reward_baseline_1m.yaml \
  --output runs/mappo_full_reward_baseline_1m_seed0
```

建议在tmux中运行。恢复、验证、绘图和服务器数据盘路径见 [训练说明](docs/training_workflows_and_parameters.md) 与 [服务器执行说明](docs/server_execution.md)。原C组、九动作及短预算实验不再作为本文当前定义；旧结果清理范围见 [结果索引](docs/results_inventory.md)。本次文档更新不启动训练。
