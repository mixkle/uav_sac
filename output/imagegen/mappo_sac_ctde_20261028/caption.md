# 论文图注建议

**English:** Centralized training and decentralized execution for continuous-action MAPPO and SAC in UAV coverage control. Each UAV encodes its local observation and discovered-user set, then samples a Gaussian latent and maps it to a bounded disk action. MAPPO uses on-policy rollouts, a centralized time-factor team value function, and per-agent clipped PPO updates with a shared team advantage. SAC uses joint experience replay, time-factor twin soft-Q critics, reparameterized policy gradients, and automatic temperature tuning. The depicted experiment uses connected-service coverage as its sole environment reward; centralized critics are used only for training.

**中文：** UAV覆盖控制中连续动作MAPPO与SAC的集中训练、分散执行框架。各UAV编码本地观测和已发现用户集合，采样高斯latent并映射为有界圆盘动作。MAPPO使用当前策略rollout、集中式时间因子团队价值函数，以及基于团队优势的逐智能体PPO更新；SAC使用联合经验回放、时间因子双soft-Q、重参数化策略梯度与自动温度调整。图示实验的环境奖励仅包含联网服务覆盖率，集中式critic仅参与训练。

## 文件及使用范围

- 最终图片：`mappo_sac_ctde.png`，1774×887像素。
- 便于插入文档的PDF：`mappo_sac_ctde.pdf`，内容为位图封装，不是可编辑矢量。
- 代码核对：`code_review.md`。
- 最终构图提示词：`final_prompt.txt`；最后公式修正：`typography_correction_prompt.txt`。
- 来源及校验：`provenance.json`。

前两版草稿及`mappo_sac_ctde_layout.png`用于生成过程追溯，包含已被修正的连线或公式问题，不作为论文最终图。论文排版时采用本文件中的完整图注；图片底部的小字号生成图注不替代正式图注。
