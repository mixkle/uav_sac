# 扩散 DPPO：从 1M 续训至累计 3M

本轮只延长在线训练预算：从完整 `full_001000000` 恢复，再训练2,000,000条联合转移。场景、容量10、覆盖＋0.1数据率奖励、学习率、PPO/GAE、时间因子critic和验证口径保持原设置。不会重新采集示范、预训练或重置Adam。

1M验证曲线在约530k时覆盖35.12%，到1M回升到49.97%，尚未稳定；原best为40,960步、覆盖54.89%。延长训练用于检验回升能否持续，不预设3M一定更好。

## 目录与恢复内容

- 源运行：`/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/`，保留不变。
- 源完整检查点：上述目录的 `checkpoints/full_001000000/`。
- 新运行：`/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0/`。
- 完整恢复actor、critic、两套归一化、Adam、环境实际关联、历史、随机状态、历史best及验证游标。
- 原step0和截至1M的CSV日志复制到新目录，使曲线可连续重建；`continuation.json`记录来源及检查点哈希。新TensorBoard文件只记录续训阶段，原TensorBoard留在1M目录。
- 在1M处曾有一次不满128步的PPO，因此本次新增977轮PPO，最后一轮72个向量步＝1152条转移；累计1466轮、58,640次actor/critic优化。不能套用从零训练3M的轮数。

## 单卡与双卡

配置：

- 单卡：`configs/diffusion_dppo_capacity10_coverage_throughput_3m_server.yaml`。
- 双卡：`configs/diffusion_dppo_capacity10_coverage_throughput_3m_server_dual.yaml`。

两份配置的学习语义一致。`diffusion.actor_update_devices: [cuda:0, cuda:1]` 开启actor更新的数据并行：保存的UAV去噪链按样本分片，完整候选集合随样本路由，梯度归并至主actor后只进行一次全局梯度裁剪和Adam更新。团队critic每个minibatch仍只更新一次，采集及正式验证仍在主卡。没有增加环境样本、去噪步数或PPO更新次数。

单卡和双卡都使用微批上限16384，表示一次梯度累积波次的**总UAV样本数**，不是新增PPO minibatch。原1M服务器配置的8192保留。不同分片的浮点求和顺序会略有差异，不承诺跨单/双卡逐位相同。

同一真实1M检查点、同一2048条转移、同一Adam和随机排列的短测：

| 模式 | 40次PPO更新耗时 | 峰值已分配显存 |
|---|---:|---:|
| 单卡，微批8192 | 4.97秒 | 约2.69GiB |
| **单卡，微批16384** | **4.35秒** | 约3.01GiB |
| 双卡，微批16384 | 6.07秒 | 主卡约1.60GiB，副卡约1.49GiB |

本次双卡慢约40%，不推荐以它加速当前小型MLP。原因包括每个minibatch的复制、线程调度和跨卡梯度归并；CPU环境与完整验证仍占较大时间。保留双卡开关供使用，但默认单卡。此为单个rollout短测，不是整个3M耗时保证。原1M在线耗时约3.46小时，按相近吞吐估计额外2M约6–8小时，实际随候选规模和系统负载变化。

GPU0/1处于同一NUMA节点；如使用GPU2/3，CPU绑定改为12–23。`CUDA_VISIBLE_DEVICES`中的物理卡会在进程内重新编号为cuda:0、cuda:1。

## 正式启动（由用户执行）

推荐单卡：

```bash
ssh dym@10.16.63.110
cd /home/dym/uav_sac
tmux -S /home/dym/uav_sac/.cache/tmux.sock new -s dppo-3m
CUDA_VISIBLE_DEVICES=0 bash scripts/resume_diffusion_dppo_3m_server.sh single
```

若希望使用双卡，在同一tmux会话内改为下面这条，二选一，不同时运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/resume_diffusion_dppo_3m_server.sh dual
```

使用物理GPU2/3：

```bash
CUDA_VISIBLE_DEVICES=2,3 UAV_CPU_AFFINITY=12-23 \
  bash scripts/resume_diffusion_dppo_3m_server.sh dual
```

按Ctrl+B再按D分离。重新连接：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t dppo-3m
```

脚本执行的是 `dppo train --resume`，**不要再次执行原来的 `dppo all`**。所有显式写入仍在 `/home/dym/uav_sac/` 内，包含数据、临时文件与缓存。

## 再次中断与报告

Ctrl+C会请求在完整PPO更新结束后保存；待保存提示完成再关闭会话。重新运行同一启动命令会选择3M目录中最新完整检查点。若突然退出导致CSV包含比最近完整检查点更新的步数，程序会拒绝混写，需要恢复到另一新目录，不能用轻量last.pt替代完整检查点。

训练完成后：

```bash
cd /home/dym/uav_sac
export CUDA_VISIBLE_DEVICES=0
source scripts/dppo_server_env.sh
.venv-dppo/bin/python -u -m uav_research dppo report \
  --run runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0 \
  --output analysis/diffusion_dppo_capacity10_coverage_throughput_3m_seed0 \
  --device cuda:0
```

## 已执行验收

- 本地DPPO测试覆盖旧路径、续训配置契约、来源保护和CSV历史复制。
- 双卡与单卡保存链log概率最大差异2.15e-5，梯度相对L2差异1.78e-6。
- 双卡连续执行与完整恢复继续执行的参数/Adam最大差异为0。
- 从真实1M检查点独立试续训2048条转移，达到1,002,048步、490轮PPO，40次优化，损失有限并成功保存。历史best仍为40960步。
- 短测存放在服务器 `deployment/continuation_3m_*`，不作为正式3M初值；正式续训依然从原始1M检查点开始。

2026-11-05部署验收时没有启动正式3M训练。用户随后完成正式运行，2026-11-06已下载并验证：best为2,792,000步，last为3,000,000步。详见[3M验证与收敛分析](../analysis/diffusion_dppo_capacity10_coverage_throughput_3m_seed0/diagnosis.md)。本次分析没有追加训练预算。

## 继续到5M（2026-11-06）

服务器已准备 `configs/diffusion_dppo_capacity10_coverage_throughput_5m_server.yaml` 和 `scripts/resume_diffusion_dppo_5m_server.sh`。仅将累计预算改为5M，从 `full_003000000` 再训练2M，保留原3M运行；模型、Adam、两套归一化、环境、随机状态和2,792,000步的历史best一并恢复。新数据写入 `runs/diffusion_dppo_capacity10_coverage_throughput_5m_seed0/`。

已完成脚本语法、配置控制变量、实际3M完整检查点兼容性、CUDA和空间预检查；没有启动正式续训。在服务器终端执行：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock new -s dppo-5m -c /home/dym/uav_sac 'CUDA_VISIBLE_DEVICES=0 bash scripts/resume_diffusion_dppo_5m_server.sh'
```

脚本只运行在线训练，不重复示范与预训练。后续再次运行时优先恢复5M目录中最新已完整发布的检查点；存在比检查点更新的日志时仍拒绝混写。仅做预检查可执行 `bash scripts/resume_diffusion_dppo_5m_server.sh --check`。
