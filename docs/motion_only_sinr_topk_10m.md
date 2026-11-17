# M1/M2 从1M完整状态续训至累计10M

两组原始1M已完成。此次每组**额外9M**联合转移，目标累计10,000,000；不从best重新初始化，不重置Adam、critic或随机流。正式续训由用户启动。

| 组 | 网络 | 物理GPU | CPU集合 | 原始完整状态 |
|---|---|---|---|---|
| M1 | Gaussian仅运动PPO | 2 | 12–17 | `runs/motion_only_sinr_topk_1m/M1_seed0/checkpoints/full_001000000` |
| M2 | Diffusion仅运动DPPO | 3 | 18–23 | `runs/motion_only_sinr_topk_1m/M2_seed0/checkpoints/full_001000000` |

## 续训设置

`configs/motion_only_sinr_topk_10m/{M1,M2}.yaml`与各自1M配置**只有 `run.total_steps` 不同**。场景、容量10、SINR Top-K、跳数缩放、历史观测、M2参考残差参数化、覆盖＋0.1数据率、学习率和所有优化/验证规则保持不变；两份实验契约逐项一致。

恢复完整actor/critic、两套Adam、双归一化、PopArt、环境位置与实际关联、历史、CPU/CUDA/UE随机状态、累计步数、历史最佳分数和下次评估时点。每约50k在完整更新边界评估；best继续在累计0–10M范围按后100步覆盖优先、相等时按吞吐选择。继续保留每10分钟的安全点完整检查点，最新两份。

**保留原1M结果，使用新目录：**

- 数据根 `/mnt/disk2/dym/uav_sac`；代码 `/home/dym/uav_sac`。
- 新运行 `runs/motion_only_sinr_topk_10m/M1_seed0`、`M2_seed0`。
- 第一次实际启动时原子准备续训目录，复制原始step0与截至1M的CSV前缀，记录 `continuation.json` 和来源哈希；原始结果不会被覆盖。
- 此后中断重启时，自动使用**10M目录中最新完整检查点**；缺失来源、不兼容契约、日志比检查点更新、已达10M等情况明确报错，不静默重头训练或回滚。
- 1M阶段最后一轮为部分rollout，续训按余量继续，因此PPO更新轮次不应直接用一次从零10M的轮次数套算。

## 一条命令同时启动

```bash
cd /home/dym/uav_sac && bash scripts/launch_motion_only_10m_server.sh
```

创建独立tmux会话 `motion-10m-M1`、`motion-10m-M2`，分别使用GPU2、3。所选卡必须空闲；先对两组做完整预检，再启动。关闭SSH不会中断tmux中的进程。默认每进程PyTorch4线程、Numba4线程。

只检查、不开始训练：

```bash
cd /home/dym/uav_sac && bash scripts/launch_motion_only_10m_server.sh --check-only
```

查看日志和会话：

```bash
tail -f /mnt/disk2/dym/uav_sac/logs/motion_only_sinr_topk_10m/M1.log
tail -f /mnt/disk2/dym/uav_sac/logs/motion_only_sinr_topk_10m/M2.log
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t motion-10m-M2
# Ctrl-b，然后d：脱离会话，保持训练。
nvidia-smi -i 2,3
```

两组都退出后，可以重用同一启动命令自动从最新安全点恢复。若只需恢复一组，确认该组旧进程已经退出，然后在tmux中运行对应指令：

```bash
CUDA_VISIBLE_DEVICES=2 UAV_CPU_AFFINITY=12-17 bash scripts/resume_motion_only_10m_server.sh train --group M1
CUDA_VISIBLE_DEVICES=3 UAV_CPU_AFFINITY=18-23 bash scripts/resume_motion_only_10m_server.sh train --group M2
```

## 完成后的冻结验证

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=2 .venv-dppo/bin/python -u scripts/motion_only_comparison.py report \
  --total-steps 10000000 --device cuda:0
# 仅从保存数据重建：
.venv-dppo/bin/python scripts/motion_only_comparison.py report --total-steps 10000000 --rebuild
```

输出 `analysis/motion_only_sinr_topk_10m/`；原1M分析目录保留。验证step0/best/last，32场景、三个动作种子，主要比较10M last；报告原始step0与整个训练历史的best。仍使用绿色/蓝色和不同线型，后100/50步和终态覆盖、吞吐、稳定性等原指标不变。保留测试集不参与。

## 已完成的10M验证（2026-11-17）

完整报告：[M1/M2 10M冻结验证](../analysis/motion_only_sinr_topk_10m/report.md)，数值汇总：[unified_best_last.csv](../analysis/motion_only_sinr_topk_10m/metrics/unified_best_last.csv)。已验证step0/best/last、32场景、三个动作种子，模型与结果已下载本地，来源哈希一致。

M1/M2 best后100步覆盖分别为62.397%/79.016%，best实际来自4,100,672/6,500,928步；10M last分别为62.161%/65.022%。M2 best能形成并在平均曲线上维持较高覆盖，但后期训练退化明显；last后100步覆盖标准差7.859个百分点，≥10个百分点骤降平均5.115次。不能把历史best与最终last混为一个模型，也不据此自动增加训练预算。

补充图表和报告可仅从保存数据重建：运行上述`--rebuild`命令后，再运行 `analysis/motion_only_sinr_topk_10m/rebuild_comparison.py`。无需重新训练或采样。

实现与部署检查见 `output/motion_only_10m_continuation_acceptance/`；预检不会实际执行正式梯度更新。未调整学习率，也不因观察到曲线还在变化而预先断言10M一定改善。
