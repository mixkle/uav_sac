# 奖励建模更新与实现验收（2026-10-14）

以最新ENVIRONMENT MODELING.md的奖励节为准：

\[
r^t=\rho^{t+1}+w\tau^{t+1}+\lambda J^{t+1}-\mu D^{t+1},\qquad
D^{t+1}=\frac{1}{U}\sum_u\mathbf1[\min_gb_{u,g}^{t+1}>U].
\]

在动作之后，更新位置、链路、最短跳数和关联，再计算断联比例。可通过直接或多跳路径到达任一BS即为可达；允许多个子网各自接入BS。惩罚所有当前断联UAV，包括未服务UE的断联节点。不能把D解释为UE断网比例、新增断联数、缺乏双BS路径或假想故障下的断网比例。

文档未指定mu数值，询问尚无回复时采用0.1作为可调整起点，当前9份YAML均显式设置。EnvConfig字段disconnection_weight默认0，保证缺失字段的历史配置和检查点继续使用原奖励。CLI训练--mu-disconnect可覆盖。当前生效权重：

| 配置 | w | lambda | mu |
|---|---:|---:|---:|
| main.yaml / wsl_50k.yaml（SAC） | 0.1 | 0.5 | 0.1 |
| discrete_mappo.yaml（主MAPPO） | 0.1 | 0 | 0.1 |
| discrete_mappo_wsl_50k.yaml（WSL MAPPO） | 0 | 0 | 0.1 |

WSL MAPPO此前已被设置为100000步和吞吐权重0，保留用户已有改动。SAC和MAPPO读取同一UAVEnvironment返回的团队奖励，更新算法中不重复扣除。允许负奖励，不裁零。不修改场景、动作、观测维度、网络、学习率或训练预算。

新增日志disconnection_ratio及disconnection_penalty（mu D）；评估报告均值、终态及最后100步。轨迹reward_terms=[rho,tau,J,D]，reward_contributions=[rho,w tau,lambda J,−mu D]，reward为实际标量；绘图直接读取实际奖励，避免硬编码w=0.1遗漏新惩罚。零奖励、负奖励、首次正奖励与首次服务分别记录。

验收：121项pytest全部通过，包括参考/Numba、连续/九动作、批量/进程环境，动作后断联、多跳/分网接入、负奖励进入SAC回放及MAPPO rollout并完成更新、评估和轨迹。主配置单步检查（独立校准种子30000，不训练）：SAC断联6/16，扣分0.0375；离散MAPPO断联7/18，扣分约0.038889。历史best.pt加载后mu=0、来源训练步数221184，奖励保持旧定义。

历史250k训练及critic校准结果没有改写。当前更新后训练需使用新配置和新结果目录；旧检查点按自身旧配置复现，现有完整恢复检查会拒绝跨奖励权重续训。本次未启动正式训练或连接服务器。
