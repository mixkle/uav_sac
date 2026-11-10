# Set-Conditioned Diffusion Policies for Decentralized Joint Mobility and User Association in Multi-UAV Networks

**中文参考题名：面向多无人机网络分散式运动与用户关联联合优化的集合条件化扩散策略**

目标期刊：*IEEE Transactions on Vehicular Technology*（TVT）。

文稿性质：Introduction 与 Related Work 双语初稿。英文与中文按相同段落逻辑组织，共用参考文献编号。本次修订依据[跳数运动缩放建模](../ENVIRONMENT_MODELING_Diffusion_DPPO_Hop_Scaling.md)，采用容量10、覆盖率与数据率奖励及真实位移历史。将“面向部署维持的跳数感知运动缩放”列为第三项方法贡献；时间因子critic保留为训练实现，不再作为核心贡献。不把尚未完成的公平对照、多种子或独立测试写成既有结论。文末“作者修订说明”不属于投稿正文。

---

## English manuscript

### I. Introduction

Unmanned aerial vehicles (UAVs) offer a flexible means of extending wireless coverage when terrestrial infrastructure is damaged, unavailable, or unable to meet local demand. By acting as aerial access points and relays, multiple UAVs can connect ground users to operational base stations beyond their direct communication range. This capability is particularly valuable for emergency communications, where users may be dispersed and move over time. Realizing this potential requires UAVs to coordinate their deployment so that improved access to users is accompanied by reliable connectivity to the terrestrial network. [1, Xu et al.](https://doi.org/10.1109/TMC.2026.3669346)

UAV deployment and user association are closely coupled in such networks. Moving a UAV changes its access links, its potential service users, and its connections to other aerial relays. Association decisions, in turn, determine how limited service capacity is distributed among users in overlapping coverage regions. A favorable deployment therefore does not necessarily translate into effective communication service unless suitable associations and gateway paths are also available. Recent research has addressed this coupling through joint deployment and association optimization, including the iterative multi-agent soft actor–critic approach of Cao et al. [10](https://doi.org/10.1109/JIOT.2023.3329346), while work on multi-hop networking has emphasized coordination among UAVs and the maintenance of network connectivity. [1](https://doi.org/10.1109/TMC.2026.3669346)

Multi-agent reinforcement learning provides a promising way to adapt deployment decisions to changing network conditions. Centralized training with decentralized execution allows agents to learn cooperative behavior while making operational decisions from local observations. [2, Yu et al.](https://arxiv.org/abs/2103.01955) For joint mobility and association control, however, choosing a learning algorithm is only part of the problem. The policy must also represent locally observed users, coordinate competing service preferences, and account for the effect of motion on the network that supports those services. These requirements motivate the following challenges.

First, each UAV has only a partial view of the communication environment. Neighboring UAVs and locally detected users provide information about nearby service opportunities, but do not reveal the complete topology or user distribution. Moreover, the number of observed users changes over time. A local policy must extract useful context from these changing observations without requiring a fixed list of all users. Second, mobility and association involve interdependent decisions of different forms: continuous movement determines future service opportunities, while user-specific preferences determine which opportunities are used. Representing these decisions together requires a policy that can accommodate a variable number of service candidates. Third, reaching a useful deployment is insufficient if it cannot be maintained. Continued movement may extend coverage, but it can also disrupt relay paths and cause several UAVs to lose access to the core network. Balancing network expansion with deployment maintenance is especially difficult when the downstream effects of a local action are not directly observable.

To address these challenges, this paper develops a set-conditioned diffusion policy for decentralized joint mobility and user association in multi-UAV networks. The policy combines neighboring-UAV information with a set representation of locally observed users, and jointly generates continuous motion commands and user-specific service scores. A capacity-constrained association mechanism converts these scores into service connections. The framework builds on set representation learning and diffusion policies, using demonstration-based pretraining followed by online refinement inspired by Diffusion Policy Policy Optimization (DPPO). [3, Zaheer et al.](https://arxiv.org/abs/1703.06114) [4, Chi et al.](https://arxiv.org/abs/2303.04137) [5, Ren et al.](https://arxiv.org/abs/2409.00588)

We further introduce hop-aware motion scaling to support deployment maintenance. The mechanism uses locally available routing information to moderate the movement of UAVs with short gateway paths while retaining greater mobility for UAVs farther along the relay network or disconnected from it. This provides a way to balance continued exploration with more conservative adjustment near gateway access points. The mechanism is integrated into policy learning and execution, with the communication objective remaining service coverage and aggregate throughput.

The main contributions of this paper are summarized as follows:

1. **A decentralized joint mobility and association framework.** We formulate service coverage and throughput optimization in a multi-hop UAV network under local observations and limited service capacity. The framework connects learned user preferences with feasible association decisions while accounting for access-link quality and gateway reachability.
2. **A set-conditioned diffusion policy for coupled control.** We design a local policy that combines neighboring-UAV information with variable-size user observations to jointly generate movement and service scores. A two-stage learning procedure combines denoising pretraining with DPPO-style fine-tuning to optimize these coupled decisions.
3. **A hop-aware motion mechanism for deployment maintenance.** We introduce routing-dependent motion scaling to balance coverage expansion and the persistence of established service. The mechanism uses local gateway-hop information and is incorporated into both training and execution without requiring the complete network topology.

The remainder of this paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Sections IV and V describe the joint control framework and learning method, respectively. Section VI evaluates communication performance and deployment maintenance. Section VII concludes the paper.

### II. Related Work

#### A. UAV Mobility, User Association, and Multi-Hop Networking

Learning-based UAV communication research has examined the interaction between deployment and resource decisions. Feng et al. propose a graph-attention multi-agent trust-region framework for trajectory design and resource assignment, using graph-based temporal representations to support value estimation. [6, Feng et al.](https://arxiv.org/abs/2401.17880) This work illustrates the relevance of network structure to cooperative aerial control. In a complementary direction, Xu et al. investigate multi-hop UAV networking through a framework combining agent grouping, reward decomposition, behavioral constraints, and knowledge distillation from large language models. [1, Xu et al.](https://arxiv.org/abs/2505.08448) These studies motivate treating access service and inter-UAV coordination as related networking decisions.

Our focus is the local policy–association interface within a network whose users must reach a gateway through their serving UAVs. We retain physical eligibility checks and service-capacity constraints, but expose association preferences as learned outputs alongside motion. In this formulation, the policy can influence which users receive the available service slots, while the protocol resolves competing proposals. The contribution is thus a particular combination of local information, variable-size joint decisions, and executable association rules, rather than the general idea of jointly optimizing UAV deployment and communication resources.

#### B. Multi-Agent Policy Learning and Set Representations

MAPPO demonstrates that on-policy optimization can provide a strong baseline for cooperative multi-agent tasks when its implementation and optimization settings are appropriately chosen. [2, Yu et al.](https://arxiv.org/abs/2103.01955) We adopt the CTDE separation between local action generation and centralized value learning, but use a diffusion actor to represent the joint control outputs. Global information available to the training critic is not made available to the deployed actor. Local execution nevertheless relies on specified control information, including gateway locations, neighbor messages, and user link measurements; it does not imply communication-free operation.

Deep Sets provides a basis for constructing permutation-invariant and permutation-equivariant neural representations. [3, Zaheer et al.](https://arxiv.org/abs/1703.06114) We apply this principle to two distinct user sets: detected users provide contextual information, whereas service-eligible users receive action scores. Pooling forms a local context without discarding the individual embeddings needed for scoring. This distinguishes the requirements of observation aggregation from those of action generation. Neighboring UAVs, in contrast, are represented through a fixed number of ordered local slots in the present implementation. Accordingly, accommodation of variable user-set cardinality should not be interpreted as a claim of unrestricted generalization to arbitrary UAV fleet sizes.

#### C. Diffusion Policies and Multi-Agent Policy Optimization

Diffusion Policy formulates action generation as conditional denoising for robot control. [4, Chi et al.](https://arxiv.org/abs/2303.04137) DPPO subsequently develops a policy-gradient framework for fine-tuning diffusion policies using their denoising process. [5, Ren et al.](https://arxiv.org/abs/2409.00588) Diffusion policies have also been studied directly in multi-agent learning. Li et al. introduce OMAD, an online off-policy framework that combines decentralized diffusion policies with a joint distributional value function and an entropy-based objective. [7, Li et al.](https://arxiv.org/abs/2602.18291)

These methods establish that diffusion-based action generation and multi-agent coordination are existing research directions. Our formulation instead addresses a communication-specific action structure: one bounded motion command and a changing set of per-user preferences, followed by capacity-limited proposals and user-side conflict resolution. The fine-tuning procedure uses an on-policy probability ratio for each UAV and each denoising transition. Log densities are summed over valid action coordinates, and the clipped objective is averaged across environment samples, UAVs, and denoising steps. It does not use the marginal likelihood of the final physical action or multiply all UAV ratios into a team-level ratio. The method is a DPPO-style adaptation to the specified joint-control problem, rather than a reproduction of every engineering choice in the original DPPO implementation.

The centralized value learner retains adaptive target normalization based on PopArt, which preserves predictions on their original scale as normalization statistics change. [8, van Hasselt et al.](https://arxiv.org/abs/1602.07714) The analytic remaining-horizon factor and the value-training settings are shared with the Gaussian comparison. They are supporting implementation choices rather than the novelty emphasized in this work.

#### D. Connectivity Control and Deployment Maintenance

Connectivity control in mobile robot networks studies how motion can maintain or modify a communication graph. Zavlanos et al. review centralized and distributed approaches based on graph spectra, potential fields, and hybrid control, including methods for maintaining links and controlling network connectivity. [9, Zavlanos et al.](https://doi.org/10.1109/JPROC.2011.2157884) This literature establishes that communication-aware motion and connectivity maintenance are existing control objectives.

Our setting requires each serving UAV to reach at least one gateway; it does not require all UAVs to belong to a single connected inter-UAV component. We use the locally available gateway hop count as a coarse cue for motion amplitude, combined with learned motion directions and user scores. The scaling rule does not compute a spectral connectivity measure, identify a critical relay, or project an action onto constraints that preserve selected links. In particular, a low hop count does not necessarily imply a large downstream relay load, and a smaller displacement can still break a short-margin link when neighboring UAVs move simultaneously.

The intended contribution is the integration of this routing-conditioned execution prior with decentralized joint mobility and association learning. Its effectiveness must be assessed through coverage retention, abrupt service drops, and disconnection recovery, alongside average coverage and throughput. A controlled comparison with scaling disabled and with uniform speed reduction is needed to distinguish the value of hop-dependent adaptation from the effect of simply moving more slowly.

---

## 中文对应稿

### I. 引言

当现有地面通信基础设施受损、缺失或无法满足局部需求时，无人机能够以灵活的方式扩展无线覆盖。多架无人机可以作为空中接入点和中继，将地面用户连接到直接通信范围之外的可用基站。这一能力对用户分布分散且位置不断变化的应急通信场景尤为重要。要实现这一目标，无人机必须协同调整部署，使地面接入能力的提升与通往地面网络的可靠连接相协调。[1，Xu等](https://doi.org/10.1109/TMC.2026.3669346)

在此类网络中，无人机部署与用户关联紧密耦合。无人机移动会改变其接入链路、潜在服务用户及与其他空中中继的连接；关联决策则决定有限的服务容量如何分配给重叠覆盖区域内的用户。因此，有利的部署位置并不必然转化为有效通信服务，还需要合适的用户关联和可用的网关路径。已有研究通过联合部署与关联优化处理这种耦合，例如Cao等提出的迭代式多智能体软演员—评论家方法。[10](https://doi.org/10.1109/JIOT.2023.3329346) 多跳组网研究则进一步强调了无人机间协作与网络连通性维持的重要性。[1](https://doi.org/10.1109/TMC.2026.3669346)

多智能体强化学习为适应动态网络条件的部署决策提供了一条可行路径。集中式训练、分散式执行允许智能体在训练中学习协作行为，并在实际运行时依据局部观测作出决策。[2，Yu等](https://arxiv.org/abs/2103.01955) 然而，对于运动与关联的联合控制，选择学习算法只是问题的一部分。策略还需要表示本地观察到的用户、协调相互竞争的服务偏好，并考虑运动对支撑这些服务的网络所产生的影响。由此产生以下几个关键挑战。

首先，每架无人机只能获得通信环境的部分信息。邻居无人机和本地发现用户能够提供附近服务机会的信息，却不能揭示完整的网络拓扑与用户分布；同时，所观察到的用户数量还会随时间变化。因此，局部策略必须从动态变化的观测中提取有效信息，而不能依赖包含全场用户的固定列表。其次，运动与关联包含形式不同、相互影响的决策：连续运动决定未来的服务机会，面向具体用户的偏好则决定如何利用这些机会。将二者统一表示，需要策略能够适应数量变化的服务候选用户。最后，仅形成良好部署仍不充分，还需要维持该部署所提供的服务。持续运动虽然可能扩大覆盖，却也可能破坏中继路径，导致多架无人机失去核心网接入。当局部动作对下游网络的影响无法被直接观察时，兼顾网络展开与部署维持尤其困难。

针对上述挑战，本文提出面向多无人机网络分散式运动与用户关联联合优化的集合条件化扩散策略。策略结合邻居无人机信息与本地用户集合表示，联合生成连续运动指令和用户服务评分，再通过容量受限的关联机制将评分转化为服务连接。在集合表示学习与扩散策略的基础上，框架采用基于示范的预训练，并借鉴扩散策略策略优化（Diffusion Policy Policy Optimization，DPPO）进行在线微调。[3，Zaheer等](https://arxiv.org/abs/1703.06114) [4，Chi等](https://arxiv.org/abs/2303.04137) [5，Ren等](https://arxiv.org/abs/2409.00588)

为支持部署维持，本文进一步引入跳数感知运动缩放机制。该机制利用本地可获得的路由信息，适当降低通往网关路径较短的无人机的运动幅度，同时为中继链上跳数较大或已经断联的无人机保留更大的移动能力，以兼顾持续探索与网关接入端附近更谨慎的位置调整。该机制融入策略学习与执行过程，通信优化目标仍为服务覆盖率和系统总吞吐量。

本文的主要贡献概括如下：

1. **分散式运动与关联联合优化框架。** 在局部观测和有限服务容量条件下，建立多跳无人机网络的服务覆盖率与吞吐量优化问题。框架综合考虑接入链路质量与网关可达性，将学习到的用户偏好转化为可行关联决策。
2. **面向联合控制的集合条件化扩散策略。** 设计融合邻居无人机信息与变长用户观测的局部策略，联合生成运动指令与服务评分。采用去噪预训练与DPPO风格微调相结合的两阶段学习过程，优化这两类耦合决策。
3. **面向部署维持的跳数感知运动机制。** 引入依赖路由信息的运动缩放，兼顾覆盖展开与已有服务的维持。该机制仅使用本地网关跳数信息，并同时融入训练和执行，无需获取完整网络拓扑。

本文其余部分安排如下：第二节回顾相关工作；第三节介绍系统模型与问题建模；第四节和第五节分别介绍联合控制框架与学习方法；第六节评估通信性能与部署维持能力；第七节总结全文。

### II. 相关工作

#### A. 无人机运动、用户关联与多跳组网

基于学习的无人机通信研究已经关注部署与资源决策之间的相互作用。Feng等提出基于图注意力的多智能体信赖域框架，用于轨迹设计与资源分配，并通过图结构的时序表示支持价值估计。[6，Feng等](https://arxiv.org/abs/2401.17880) 该研究说明了网络结构信息对于协同空中控制的意义。另一方面，Xu等研究无人机多跳组网，将智能体分组、奖励分解、行为约束与大语言模型知识蒸馏相结合。[1，Xu等](https://arxiv.org/abs/2505.08448) 这些研究启发我们将接入服务与无人机间协作作为相互关联的网络决策进行处理。

本文关注的是用户必须通过服务无人机到达网关时，局部策略与关联过程之间的接口。我们保留物理可行性检查和服务容量约束，同时将关联偏好与运动一起作为策略的可学习输出。在这一建模中，策略能够决定可用服务名额优先分配给哪些用户，而协议负责处理相互竞争的提议。因此，本文的贡献在于局部信息、变长联合决策与可执行关联规则的具体结合，而不是一般意义上的无人机部署与通信资源联合优化概念。

#### B. 多智能体策略学习与集合表示

MAPPO表明，在合理选择实现和优化设置时，同策略优化能够成为协作式多智能体任务的有力基准。[2，Yu等](https://arxiv.org/abs/2103.01955) 本文采用CTDE中局部动作生成与集中式价值学习相分离的方式，并以扩散actor表示联合控制输出。训练critic使用的全局信息不会提供给部署actor。不过，局部执行仍依赖明确规定的控制信息，包括网关位置、邻居消息和用户链路测量；它并不意味着无需通信即可运行。

Deep Sets为构造具有置换不变性和置换等变性的神经表示提供了基础。[3，Zaheer等](https://arxiv.org/abs/1703.06114) 本文将这一原则用于两类不同用户集合：发现用户提供上下文信息，可服务用户获得动作评分。池化形成局部上下文，同时保留评分所需的独立用户嵌入，从而区分观测聚合与动作生成的不同需求。相比之下，当前实现中的邻居无人机采用固定数量、按规则排序的局部槽位表示。因此，对可变用户集合规模的适应性，不应被解释为能够不受限制地泛化到任意无人机数量。

#### C. 扩散策略与多智能体策略优化

Diffusion Policy将机器人控制中的动作生成建模为条件去噪过程。[4，Chi等](https://arxiv.org/abs/2303.04137) DPPO进一步利用去噪过程构建扩散策略微调的策略梯度框架。[5，Ren等](https://arxiv.org/abs/2409.00588) 扩散策略也已被用于多智能体学习。Li等提出在线离策略框架OMAD，将分散式扩散策略、联合分布式价值函数与基于熵的目标相结合。[7，Li等](https://arxiv.org/abs/2602.18291)

这些方法说明，扩散动作生成与多智能体协作本身已经是已有研究方向。本文面向的是一种具有通信含义的动作结构：一个有界运动指令，以及数量随时间变化的用户偏好，随后通过容量受限提议和用户侧冲突消解得到实际关联。微调过程为每架无人机的每次去噪转移计算同策略概率比。对数密度在有效动作坐标上求和，裁剪目标在环境样本、无人机与去噪步之间取平均。该方法不使用最终物理动作的边缘似然，也不将全部无人机的概率比相乘为团队概率比。因此，本文方法是面向所述联合控制问题的DPPO风格适配，而不是对原始DPPO实现全部工程选择的复现。

集中式价值学习保留基于PopArt的自适应目标归一化，在归一化统计变化时保持原始尺度上的预测不变。[8，van Hasselt等](https://arxiv.org/abs/1602.07714) 解析剩余时域因子及价值训练设置与Gaussian对照共用，属于支持训练的实现选择，而非本文强调的创新点。

#### D. 连通性控制与部署维持

移动机器人网络中的连通性控制研究如何通过运动维持或改变通信图。Zavlanos等综述了基于图谱、势场和混合控制的集中式与分散式方法，包括链路维持与网络连通性控制。[9，Zavlanos等](https://doi.org/10.1109/JPROC.2011.2157884) 这一研究脉络表明，通信感知运动和连通性维持本身已经是已有控制目标。

本文要求每个提供服务的无人机能够接入至少一个网关，并不要求全部无人机构成单一的机间连通分量。我们将本机可获得的网关跳数作为调节运动幅度的粗粒度线索，与学习到的运动方向和用户评分结合。缩放规则不计算图谱连通性指标，不识别关键中继，也不将动作投影到维持指定链路的约束集合。特别是，较低跳数并不必然意味着更大的下游中继负载；当邻居同时移动时，较小的位移仍可能使距离裕量不足的链路失效。

本文拟贡献的是将这一路由条件化的执行先验与分散式运动、用户关联联合学习相结合。其有效性需要通过覆盖保持、服务骤降和断联恢复，以及平均覆盖与吞吐等指标共同评估。还需要在关闭缩放和统一降低速度的条件下进行受控比较，以区分按跳数自适应调整的价值与单纯减速带来的效果。

---

## References / 参考文献

文献编号与两种语言正文一致。预印本按预印本标识，不推测其未核实的期刊卷期或DOI。

1. Y. Xu et al., “Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models,” *IEEE Transactions on Mobile Computing*, accepted author version, 2026, doi: 10.1109/TMC.2026.3669346. [出版标识](https://doi.org/10.1109/TMC.2026.3669346) · [作者预印本](https://arxiv.org/abs/2505.08448). 本地PDF已标明接收及DOI，此处不补写未核对的卷期页码。
2. C. Yu et al., “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” *Advances in Neural Information Processing Systems*, vol. 35, Datasets and Benchmarks Track, 2022. [会议出版页面](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html).
3. M. Zaheer et al., “Deep Sets,” *Advances in Neural Information Processing Systems*, 2017. [作者预印本](https://arxiv.org/abs/1703.06114).
4. C. Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” *Robotics: Science and Systems*, 2023. [作者项目页面](https://diffusion-policy.cs.columbia.edu/) · [作者预印本](https://arxiv.org/abs/2303.04137).
5. A. Z. Ren et al., “Diffusion Policy Policy Optimization,” *International Conference on Learning Representations*, 2025. [会议页面](https://openreview.net/forum?id=mEpqHvbD2h) · [作者预印本](https://arxiv.org/abs/2409.00588).
6. Z. Feng, D. Wu, M. Huang, and C. Yuen, “Graph Attention-based Reinforcement Learning for Trajectory Design and Resource Assignment in Multi-UAV Assisted Communication,” *arXiv preprint*, arXiv:2401.17880, 2024. [作者预印本](https://arxiv.org/abs/2401.17880).
7. Z. Li, H. Zhong, X. Wang, Q. Xia, L. Zhang, and L. Huang, “Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies,” *Proceedings of the 43rd International Conference on Machine Learning*, PMLR, vol. 306, pp. 72744–72767, 2026. [会议出版页面](https://proceedings.mlr.press/v306/li26jv.html) · [作者预印本](https://arxiv.org/abs/2602.18291).
8. H. van Hasselt, A. Guez, M. Hessel, V. Mnih, and D. Silver, “Learning values across many orders of magnitude,” *Advances in Neural Information Processing Systems*, 2016. [作者预印本](https://arxiv.org/abs/1602.07714).
9. M. M. Zavlanos, M. B. Egerstedt, and G. J. Pappas, “Graph-Theoretic Connectivity Control of Mobile Robot Networks,” *Proceedings of the IEEE*, vol. 99, no. 9, pp. 1525–1540, Sep. 2011, doi: 10.1109/JPROC.2011.2157884. [出版记录](https://doi.org/10.1109/JPROC.2011.2157884) · [作者出版列表](https://www.michaelmzavlanos.org/publications).

10. Y. Cao, Y. Luo, H. Yang, and C. Luo, “UAV-Based Emergency Communications: An Iterative Two-Stage Multiagent Soft Actor–Critic Approach for Optimal Association and Dynamic Deployment,” *IEEE Internet of Things Journal*, vol. 11, no. 16, pp. 26610–26622, Aug. 2024, doi: 10.1109/JIOT.2023.3329346. [出版标识](https://doi.org/10.1109/JIOT.2023.3329346).

---

## 作者修订说明（不属于投稿正文）

### A. TVT定位与稿件边界

本次重写将Introduction限定为研究动机与贡献概述：不列公式、仿真参数、网络维度、Top-K逐步时序、历史字段或策略概率计算。缩放公式、动作映射、协议流程及训练实现应在系统模型和方法章节展开；Related Work保留文献比较功能。中英文引言的段落与贡献一一对应。

本稿按TVT的无线通信方向组织：先讲多跳接入、容量和干扰造成的网络问题，再讲算法。没有为了迎合刊名而虚构车联网、道路交通或车辆用户模型。TVT官方范围可见[期刊说明](https://vtsociety.org/publication/ieee-transactions-vehicular-technology/information-editors)。

当前初稿的三项贡献是方法设计贡献，不等于已经证明每一模块都提升性能。尤其需要避免以下表述：首次将扩散用于MARL、保证全局最优关联、保证稳定部署、保证任意UAV规模泛化、完全无需控制通信、已解决回传容量或实际路由时延问题。

本文的核心逻辑更新为：**局部信息下的运动与关联耦合 → 集合条件化联合动作与可执行评分协议 → 通过跳数感知运动执行兼顾展开与维持**。去噪预训练与DPPO微调支撑联合策略学习；时间因子critic、PopArt和价值裁剪作为共用训练实现，不再独立主张创新。

“部署稳定维持”在本稿中指服务在回合后段的保持及骤降程度，不指已经证明的李雅普诺夫稳定性、必然收敛或链路安全。当前缩放是有设计动机的确定性运动先验，正文未宣称它已经解决断链或贴边。本文的“分散执行”对应局部actor、评分提议与UE选择的信息需求；当前仿真实现并不是已部署的分布式通信系统。

| 拟贡献 | 对应问题 | 需要支撑的证据 |
|---|---|---|
| 局部联合建模与评分关联接口 | 本地运动和有限服务名额相互耦合 | 评分作用、Top-K损失、服务覆盖与吞吐 |
| 集合条件化扩散策略及两阶段训练 | 变长候选集合与运动/评分联合分布的表达和学习 | 同协议Gaussian评分策略、预训练与在线增益、推理成本 |
| 跳数感知运动缩放 | 展开中的持续运动可能扰动网关路径和已形成服务 | 开关缩放、统一减速对照、覆盖保持、骤降、断联恢复及展开速度 |

当前实现取 \(U=18\)、\(\alpha_{\min}=0.25\)、\(k_h=0.4\)，断联编码为19；直连BS的运动比例为0.25，断联时的运动比例为1。比例由动作前跳数决定，先缩放、再镜面反射；历史保存最终真实位移，评分不缩放。奖励仍为 \(\rho+0.1\tau\)，不加入鲁棒性、断联或边界惩罚。跳数低不等于关键中继，也不代表已经识别其下游负载；这应是方法解释的明确边界。

### B. 当前结果与后续摘要写作

此前[5M报告](../analysis/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/report.md)属于未采用本次跳数缩放的新旧算法比较，不能直接用其中的优势数值支撑新版的部署维持贡献。

当前新实验为Diffusion DPPO与Gaussian服务评分MAPPO，两组均启用跳数缩放、真实位移历史及相同Top-K协议，均已完成在线训练10M步。[完整训练验证](../analysis/hop_scaling_joint_service_10m_seed0/report.md)记录两组step0、best与last，主要比较同预算last；该对照仍不能隔离缩放本身的效果。Gaussian从零初始化，扩散另有示范与预训练成本，因此即使在线预算相同，也需要明确初始化和额外数据差异。训练过程中的临时验证报告已由完整训练报告替代。

上述结果均来自单训练种子及参与选模的验证场景。摘要和Introduction暂不写“所提方法显著优于所有基线”或“已解决部署稳定性”；独立测试、多种子及关键消融补齐后，再加入中英文一致的结果概述段。

### C. 正式投稿前的文献与实验补充

- 当前10篇文献构成初稿的代表性脉络，不是完整文献综述。新增[9]将运动缩放放在已有连通性控制研究背景中；引用内容依据作者摘要、出版信息及公开方法说明，未将未核验的缺陷归于具体论文。
- 本次已阅读本地Xu等论文和Cao等论文的引言，参考其“应用背景—已有进展—关键挑战—方法与贡献”的行文层次，未照搬原文。Cao等的联合部署与关联研究已作为[10]纳入引言；关于具体信息条件、回传与关联时序的深入比较仍须结合其方法章节核实，不能据引言概述推断其局限。Xu等本地PDF已标明TMC接收及DOI，因此同步修正[1]的出版状态。
- 正式文献表应从出版方导出BibTeX，统一IEEE格式；对预印本核实是否已有最终出版版本。不得为增加TVT引用数量而加入不相关文献。
- 扩散表达的关键对照是同观测、同Top-K协议、同缩放规则、同时间因子critic和同预算的Gaussian联合运动/评分策略；另外需要对齐或单独拆分示范使用机会。当前“扩散预训练、Gaussian从零”比较只能评价两套完整方法，不能单独归因于策略分布。
- 新增贡献最直接的消融是**同一类actor开启/关闭跳数缩放**：两组均采用真实位移历史、相同奖励和关联协议，并在各自执行规则下重新训练；扩散组应重新采集对应示范。不能拿旧版指令历史模型与新版实际位移历史模型的差异全部归因于缩放。
- 补充统一速度缩放的对照，按独立开发场景预先确定常数比例或平均执行位移匹配规则，检查收益是否仅来自整体减速。对 \(\alpha_{\min}\) 与 \(k_h\) 做独立敏感性分析，不能事后按正式验证最优点包装为普遍规律。
- 部署维持同时报告全回合/后100步覆盖、51–150步到301–400步的维持变化、覆盖5%分位、最大单步及5步跌幅、≥10个百分点骤降事件、联网节点突降及恢复时间；并保留展开速度、吞吐和实际位移，避免把运动变慢本身视作成功。
- 用固定SINR评分、去除邻居信息、仅预训练策略等对照支撑评分、局部协作和在线微调。direct/time-factor critic可作为训练实现诊断保留，不再承担核心贡献的证据角色。当前actor使用一步历史而critic仍只读取当前物理状态和时间的信息差异，也应在方法局限中保留说明。
- 系统模型应明确满负载干扰、充分回传容量、理想路由收敛、固定飞行高度、信标与下行正交，以及移动前评分/移动后过滤的假设。未来针对这些假设的敏感性结果可加强TVT稿件的通信解释力。

### D. 术语对齐

| English | 本稿中文用语 | 说明 |
|---|---|---|
| decentralized execution | 分散式执行 | 不表示训练阶段无全局信息 |
| service coverage | 服务覆盖率 | 实际获得可达网关服务的UE比例，不是发现率 |
| aggregate throughput | 系统总吞吐量 | 奖励中采用固定参考尺度归一化 |
| detected user set | 发现用户集合 | 独立上行信标SNR达标 |
| service-eligible candidate set | 可服务候选集合 | 同时满足发现、下行与回传条件 |
| association score | 关联评分/服务评分 | 排序偏好，不是连接成功概率；不同于扩散的概率密度score function |
| set-conditioned diffusion policy | 集合条件化扩散策略 | 条件输入为局部集合，输出包含变长候选评分 |
| hop-aware motion scaling | 跳数感知运动缩放 | 依据动作前最短网关跳数缩放指令幅度，不改变评分 |
| deployment maintenance | 部署维持 | 服务覆盖与网关接入随时间的保持，不等同于形式化稳定性保证 |
| actual-displacement history | 真实位移历史 | 缩放、反射后的新旧坐标差，按最大指令位移归一化 |
| time-factor critic | 时间因子critic | 共用训练实现：预测平均任务奖励并还原累计价值，不列为核心贡献 |
