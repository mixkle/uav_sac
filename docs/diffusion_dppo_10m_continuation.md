# 从5M完整状态续训到累计10M

配置：`configs/diffusion_dppo_capacity10_coverage_throughput_10m_server.yaml`。
脚本：`scripts/resume_diffusion_dppo_10m_server.sh`。

只将累计在线训练预算从5,000,000改为10,000,000，追加5,000,000条联合转移。
其他场景、奖励、学习率、去噪步数、PPO参数、单GPU与线程设置全部继承5M配置。

默认恢复来源：
`/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/checkpoints/full_005000000`。

新结果目录：
`/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0/`。

恢复actor、critic、Adam、归一化统计、实际关联、历史观测、环境与随机状态，以及历史best和验证游标。
不重新采集示范或执行预训练。初次启动时复制日志历史并记录续训来源；保留5M原始运行。
之后重复启动脚本会优先选取10M目录下最新的完整检查点。脚本有输出锁，防止重复启动。

登录服务器后启动（物理GPU0）：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock new -s dppo-10m -c /home/dym/uav_sac 'CUDA_VISIBLE_DEVICES=0 bash scripts/resume_diffusion_dppo_10m_server.sh'
```

按Ctrl+B，再按D离开会话。重新进入：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t dppo-10m
```

只做完整检查点、配置兼容、CUDA和空间预检查：

```bash
cd /home/dym/uav_sac
CUDA_VISIBLE_DEVICES=0 bash scripts/resume_diffusion_dppo_10m_server.sh --check
```

预检查不启动训练、不创建正式运行目录。全部服务器写入均限制在`/home/dym/`。
