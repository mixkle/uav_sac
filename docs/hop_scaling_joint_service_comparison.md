# 跳数速度缩放：Diffusion DPPO 与 Gaussian MAPPO，单卡各10M

两组均输出运动与变长UE评分，共用Top-K协议、容量10、三邻居、相对坐标和真实位移历史。场景以容量10 baseline为来源。奖励严格为 `coverage + 0.1*tau`，其余权重为0；`tau=throughput/(150*5e6)`。扩散使用重新采集的示范和新预训练，Gaussian从零训练。两组在线各10M联合转移，种子0；正式运行由用户启动。

## 执行和策略

按动作前最短联网跳数计算
`alpha=.25+.75*(1-exp(-.4*(h-1)))/(1-exp(-.4*18))`。
直连h=1时alpha=.25，断联h=19时alpha=1。先缩放二维圆盘指令，再逐坐标反射；不缩放UE评分，不添加链路安全约束。下一步4维历史为反射后实际坐标差/30、动作前h/19、动作前感知人数/10。原命令、缩放位移、实际位移分别保存。旧实验默认关闭缩放、保留原指令历史，不能跨语义续训。

两组固定观察105维、5维UE集合、6292维时间因子critic；每机只对动作前可服务集合评分并提议Top10，移动后过滤，UE选最高分；不补位。Reset保留原初始贪婪关联，第一步起Top-K。

扩散保持8步DDPM与MLP集合编码；Gaussian使用同样5→64→64编码、170→128→128上下文，候选embedding均值64维；运动192→256→256→4，评分256→256→256→2，输出状态相关高斯参数，log std限制[-5,2]。运动圆盘映射、评分sigmoid。Gaussian PPO按每UAV全部运动和有效评分latent的联合log概率求和；扩散按每UAV/每去噪步求和有效坐标，再对样本/UAV/去噪步平均。两者都保存原始采样，不对Top-K反向传播。

两组actor/critic学习率1e-4/5e-4，Adam eps1e-5，gamma .995、GAE .95，actor/value clip .1/.2，10epochs/4minibatches，grad clip .5，熵和在线BC均0。时间因子critic保留PopArt、双归一化及累计价值空间Huber10和clipping。新组critic使用独立的固定初始化流（seed+714031）保持两组初值相同；actor及后续采样不受该初始化流扰动。16环境×128步；10M共4883轮，末轮1664条，每优化器195320次更新。

## 先准备扩散示范与预训练

服务器 `/home/dym/uav_sac`，依赖 `.venv-dppo`。缓存、临时文件、数据均在 `/home/dym`。准备默认GPU0、CPU0–11：

```bash
cd /home/dym/uav_sac
mkdir -p .cache
tmux -S /home/dym/uav_sac/.cache/tmux.sock new-session -s hop-prepare 'cd /home/dym/uav_sac && bash scripts/prepare_hop_scaling_diffusion_server.sh all'
```

示范300000–300127拟合、301000–301031选模，共64000联合转移；50轮预训练。运动教师为旧容量10 best、671744步，冻结哈希；专用适配器保留旧教师的原指令历史，新扩散示范观察保存实际位移历史，监督目标仍为缩放前高斯latent。评分教师logit `(SINR-threshold)/10`。不复用旧示范或预训练。

准备脚本也支持 `collect`、`pretrain`、`resume-pretrain`；采集按已完成场景安全继续，预训练中断需显式用 `resume-pretrain`。

## 同时启动两组在线10M

准备完成后：

```bash
cd /home/dym/uav_sac
bash scripts/launch_hop_scaling_comparison_server.sh --check
bash scripts/launch_hop_scaling_comparison_server.sh
```

默认扩散GPU0/CPU0–11、Gaussian GPU2/CPU12–23，两个独立tmux会话 `hop-diffusion`、`hop-gaussian`。每组PyTorch4线程、Numba4线程、单GPU；服务器先测128/256/512，再根据低利用率与显存余量测1024/2048/4096，选4096，仅改变梯度累积粒度；内部设备均cuda:0。自定义例：

```bash
UAV_HOP_GPUS=1,3 UAV_HOP_CPUSETS='0-11|12-23' bash scripts/launch_hop_scaling_comparison_server.sh
```

在线数据分别为：
- `runs/diffusion_dppo_hop_scaling_capacity10_10m_seed0/`
- `runs/gaussian_mappo_topk_hop_scaling_capacity10_10m_seed0/`

日志：`runs/hop_scaling_launch_logs/{diffusion,gaussian}.log`。查看 `tail -f`；也可 `tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t hop-diffusion`。完整检查点每10分钟及结束时保存，保留最近两份；另保留step0/best/last。开启前同时检查空间余量、预训练契约、输出目录、GPU及tmux会话。

## 恢复

使用最近完整目录，不能使用轻量last.pt。替换以下占位路径：

```bash
CUDA_VISIBLE_DEVICES=0 UAV_CPU_AFFINITY=0-11 \
UAV_RESUME_CHECKPOINT=/home/dym/uav_sac/runs/diffusion_dppo_hop_scaling_capacity10_10m_seed0/checkpoints/full_XXXXXXXXX \
bash scripts/train_hop_scaling_group_server.sh diffusion

CUDA_VISIBLE_DEVICES=2 UAV_CPU_AFFINITY=12-23 \
UAV_RESUME_CHECKPOINT=/home/dym/uav_sac/runs/gaussian_mappo_topk_hop_scaling_capacity10_10m_seed0/checkpoints/full_XXXXXXXXX \
bash scripts/train_hop_scaling_group_server.sh gaussian
```

在tmux内执行恢复命令。SIGINT/SIGTERM/SIGHUP会请求完整PPO更新后保存退出；kill -9不能触发保存。

## 验证和报告

服务器执行前先source运行环境；GPU编号可换：

```bash
cd /home/dym/uav_sac
export CUDA_VISIBLE_DEVICES=0
source scripts/dppo_server_env.sh
.venv-dppo/bin/python -u scripts/report_hop_scaling_comparison.py \
  --runs runs/diffusion_dppo_hop_scaling_capacity10_10m_seed0 \
         runs/gaussian_mappo_topk_hop_scaling_capacity10_10m_seed0 \
  --output analysis/hop_scaling_joint_service_10m_seed0 --device cuda:0
```

本地用 `.venv-dppo-compat/bin/python`、`--device cpu`。报告采样32场景10000–10031、动作种子880001、批次16；step0/best/last，主要比较同预算last。best按覆盖+.1tau全回合回报选取。扩散的step0已预训练，Gaussian未训练；不能将差异全部归因于分布类型。

报告保存任务/部署/协议曲线、MC价值误差、动作随机性、固定10000/10001拓扑、训练/验证曲线、骤降与维持指标、2000次场景bootstrap。绿色扩散、蓝色Gaussian，主曲线配合不同线型；场景区间不是训练种子置信区间。原始数据可重建：

```bash
.venv-dppo-compat/bin/python scripts/report_hop_scaling_comparison.py \
  --output analysis/hop_scaling_joint_service_10m_seed0 --rebuild
```

正式训练之前只执行独立小配置与≤2048步主场景短验收，不继承短训权重。旧结果和旧生产配置保留。

## 2026-11-10 完整10M验证

两组均已完成10M联合转移。最终结果保存在本地与服务器相同的相对目录
`analysis/hop_scaling_joint_service_10m_seed0/`，详见[完整验证报告](../analysis/hop_scaling_joint_service_10m_seed0/report.md)。

| 组别 | best保存步数 | best平均覆盖 | last平均覆盖 | last后100步覆盖 | last吞吐 Mbps |
|---|---:|---:|---:|---:|---:|
| Diffusion | 7,501,824 | 60.587% | 57.662% | 59.612% | 505.563 |
| Gaussian | 970,752 | 69.211% | 59.322% | 66.553% | 557.489 |

本次冻结六个模型，使用32场景和64次独立条件动作采样复验。两组均比各自step0改善，但last相对best存在回退；不能仅从未严格收敛推断继续延长预算一定有效。最终模型、原始数据和训练文件已核对哈希。

本次完整中文报告使用以下入口从保存数据重建，无需重采样：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv-dppo-compat/bin/python analysis/hop_scaling_joint_service_10m_seed0/build_report.py
```

服务器加载运行环境后将Python路径换为`.venv-dppo/bin/python`。本地和服务器的两次`hop_scaling_interim_20261109*`中途评估目录已删除，完整训练目录和最终验证保留；清单见最终目录的`cleanup_manifest.json`。
