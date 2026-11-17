# M1/M2：仅运动 Actor＋SINR Top-K，1M 从零训练

1M完成后续训至累计10M：见[10M续训说明](motion_only_sinr_topk_10m.md)。原1M配置、模型和日志保留。

本轮移除学习评分这个动作分支，只比较两类运动策略。旧 A–F 配置、模型和结果保留；新组使用独立命名空间。

| 组别 | Actor | 优化方式 | 初始化 | 默认物理GPU |
|---|---|---|---|---|
| M1 | 集合条件化 Gaussian，仅二维运动 latent | PPO，每UAV一个概率比 | 种子0随机初始化 | 2 |
| M2 | 集合条件化 Diffusion，仅二维运动 latent | DPPO，每UAV每去噪步一个概率比 | 种子0随机初始化，无BC | 3 |

两组均从0在线训练1,000,000条联合转移，不读取任何教师、示范、预训练或旧优化器状态。M1与旧D4的算法和设置相同，属于独立重跑；M2与旧F2a的重要差别是取消示范与BC。这里的M1/M2是本轮局部命名，不覆盖历史桥接实验中的同名组。

## 固定设置

配置来自 `configs/system_ablation_1m/D4.yaml`，新文件位于 `configs/motion_only_sinr_topk_1m/`；可用 `scripts/build_motion_only_configs.py` 重建。两份配置仅算法名称、运行编号、未使用的BC目标标签以及扩散初始化参数化不同。

- 场地3500×3500 m，18 UAV、150 UE、3个角落BS；每UAV容量10，可见邻居3。
- UAV全场均匀初始化；UE中心2500×2500 m均匀初始化后持续移动。
- 400步真正终止；最大指令速度30 m/s；镜面反射，指数跳数速度缩放，真实位移一步历史。
- UU 650 m、UG三维1500 m；上下行、满负载干扰和候选资格不变。
- 奖励只有 `coverage + 0.1 * normalized_throughput`；鲁棒性、断联、边界权重全部为0。
- 105维固定观察与5维变长UE特征保留；移除评分输出不删除这些可用于运动决策的信息。
- 6292维时间因子团队critic；原GAE、双归一化、PopArt、Huber10、Value Clipping保留。
- actor/critic学习率1e-4/5e-4，折扣0.995，GAE0.95，actor/value clip0.1/0.2，10epochs×4minibatches，梯度裁剪0.5，熵系数0。
- 16环境×128步；共489轮PPO，最后576条转移，actor/critic各19,560次优化。每组PyTorch4线程、Numba4线程，单卡FP32。
- M2保留既有8步余弦DDPM、预测噪声参数化、反向标准差下限0.05，采用此前A1b的 `normal_reference_residual` 初始化并仅适配运动分支：噪声预测为 `sqrt(1-alpha_bar[k])*x_k + residual_theta`，残差输出层为零初值且正常训练。它不使用示范或已训练权重。采样和概率重算使用相同转移。原epsilon初始化在本次审计中latent平均模长约189、指令平均模长0.9998；不把这种先天饱和当作正常的分布对照。旧A1/F2a及联合A1b行为不变，记录M2 step0分布，不能假定与Gaussian完全一致。

## 固定关联与概率

动作前，用当前可服务候选的SINR生成全局一致的单调评分 `sigmoid((SINR-threshold)/10)`，本机取Top10；执行一次运动后仅过滤失效提议，UE选择最高评分UAV。不补位，不在移动后重新评分，不接纳本步新发现的候选，平局按既有编号规则。

该评分与S2的 `Q=log(1+SNR)/log(11)` 均为相同SINR的全局严格单调变换，按排序决策时等价；不引入保持项或负载项。

Actor没有评分头参数，也不采样评分latent；rollout的评分链为空、评分log概率为0。公共ServiceAction消息中仍携带确定性规则评分以复用现有环境接口，环境会依据动作前SINR自行核验/生成执行评分，不能把消息字段误称为网络评分输出。PPO只使用运动latent或运动去噪链，不用执行位移重算概率。

## 服务器启动与进度

代码在 `/home/dym/uav_sac`，全部结果、缓存及日志在 `/mnt/disk2/dym/uav_sac`。正式训练由用户执行：

```bash
cd /home/dym/uav_sac && bash scripts/launch_motion_only_server.sh
```

仅检查、不启动：`bash scripts/launch_motion_only_server.sh --check-only`。

默认M1使用物理GPU2、CPU12–17，M2使用物理GPU3、CPU18–23；两组CPU集合互不重叠，均邻近两张卡所在NUMA节点。进程内部为cuda:0。脚本先检查两张卡空闲、空间、配置、旧输出和tmux会话，再分别创建 `motion-M1`、`motion-M2`。发现同名运行时拒绝覆盖，不自动续训。

```bash
# 两组各自日志
tail -f /mnt/disk2/dym/uav_sac/logs/motion_only_sinr_topk_1m/M1.log
tail -f /mnt/disk2/dym/uav_sac/logs/motion_only_sinr_topk_1m/M2.log
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t motion-M1
# Ctrl-b 后按 d 脱离；关闭SSH不影响tmux训练。
nvidia-smi -i 2,3
```

独立输出为 `runs/motion_only_sinr_topk_1m/M1_seed0`、`M2_seed0`（服务器均位于数据根目录）。每约50k完整更新边界验证32场景；包含step0，best按后100步覆盖优先、数值相等时后100步吞吐选取。完整检查点每10分钟在更新安全点保存，保留最近两份；另存step0/best/last。

中断后应先确认旧进程已退出，再显式恢复；完整检查点缺失时拒绝重启为零：

```bash
CUDA_VISIBLE_DEVICES=2 UAV_CPU_AFFINITY=12-17 bash scripts/train_motion_only_server.sh train --group M1 --auto-resume
CUDA_VISIBLE_DEVICES=3 UAV_CPU_AFFINITY=18-23 bash scripts/train_motion_only_server.sh train --group M2 --auto-resume
```

这些恢复指令本身在前台运行；长期续训可置于新的tmux会话。轻量last.pt不能替代完整恢复检查点。

## 训练后验证

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=2 .venv-dppo/bin/python -u scripts/motion_only_comparison.py report --device cuda:0
# 不重新采样，只重建已保存的数据：
.venv-dppo/bin/python scripts/motion_only_comparison.py report --rebuild
```

默认评估step0/best/last，验证场景10000–10031、三个动作种子880001–880003，不接触保留测试集。输出 `analysis/motion_only_sinr_topk_1m/`：

- `metrics/unified_best_last.csv`：后100/50步及终态覆盖/吞吐、覆盖标准差、回报、联网比例、骤降、最大跌幅、P05和形成时间。
- `metrics/paired_differences.csv`：M2−M1的2000次配对场景bootstrap；先在场景内平均动作重复，不是训练种子区间。
- 训练/验证/loss曲线、后期保持、固定场景10000/10001拓扑，以及各自on-policy的critic MC审计。绿色为M1，蓝色为M2，配不同线型。

主要比较同预算last，best和相对step0的增益作补充。两类策略的动作分布、初始分布和PPO比率定义不同；性能差异不能仅归因于参数量。若M2失败，应结合step0动作尺度和优化诊断判断，不能直接归因于扩散表达能力。

## 验收记录

见 `output/motion_only_acceptance_20261116/` 的本地测试、服务器源码哈希和独立短测记录。短测输出不进入正式运行目录，正式训练重新初始化。
