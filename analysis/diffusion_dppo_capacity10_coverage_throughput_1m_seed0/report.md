# Diffusion DPPO 容量10对照报告

本次last实际在线步数：**1000000条联合转移**。示范和预训练成本另计；短流程不能作为正式1M性能结论。

奖励为覆盖率＋0.1×归一化吞吐；全部模型采用采样评估。

| 模型 | 全程覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐 Mbps | 回合回报 |
|---|---:|---:|---:|---:|---:|
| step0 | 51.999% | 54.151% | 53.396% | 452.702 | 232.1385 |
| best | 54.887% | 57.513% | 53.917% | 482.829 | 245.2974 |
| last | 49.973% | 45.976% | 41.271% | 440.231 | 223.3727 |
| teacher_topk | 52.890% | 55.918% | 48.708% | 465.385 | 236.3801 |
| baseline_best | 56.438% | 58.107% | 58.562% | 466.249 | 250.6204 |
| baseline_last | 56.708% | 59.933% | 58.167% | 471.218 | 251.9654 |

主要比较同预算last及相对预训练后step0的增益。教师在旧集中式协议训练，新协议含Top-K竞争与一步评分时序；新旧差异不能只归因于扩散结构。

Critic在各自on-policy轨迹上与真实终止的两项奖励MC比较，不包含熵或去噪概率。

场景区间见 paired_scene_differences.csv；不是训练种子置信区间。示范采集和预训练成本另计，保留测试集未参与本报告。

图表：

- [覆盖、吞吐和奖励](figures/task_curves.png)
- [部署维持和边界行为](figures/deployment.png)
- [提议与竞争](figures/protocol.png)
- [奖励分量](figures/reward_components.png)
- [条件动作随机性](figures/action_randomness.png)
- [评分与近悬停](figures/score_and_motion_distribution.png)
- [Critic与MC](figures/critic_mc.png)
