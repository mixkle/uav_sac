# 扩散 DPPO：服务器部署与 GPU 验收（2026-11-05）

代码已上传到 `dym@10.16.63.110:/home/dym/uav_sac/`，隔离环境、缓存、示范和训练结果也全部位于该目录内。仅执行部署验收与短基准，**没有启动正式1M训练**。旧MAPPO/SAC实现和教师检查点保留。

## 安装与路径

| 项目 | 实际配置 |
|---|---|
| GPU | 4×RTX3090 24GB；本实验只用GPU0 |
| CPU | 双路E5-2678 v3，24物理核／48线程 |
| 内存 | 约64GB |
| 系统／驱动 | Ubuntu20.04／535.230.02 |
| 隔离环境 | `/home/dym/uav_sac/.venv-dppo` |
| Python／PyTorch | 3.11.16／2.7.1+cu118，CUDA可用 |
| 正式输出 | `/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/` |
| 短测与监控 | `/home/dym/uav_sac/deployment/` |

依赖按 `requirements/dppo-server-cu118.lock.txt` 安装，没有修改系统Python、驱动或系统配置。服务器Python补丁版本与本地兼容环境的3.11.18不同，GPU验收单独记录。

`scripts/dppo_server_env.sh` 将临时文件、uv/pip、Python字节码、Numba、Matplotlib、CUDA和PyTorch缓存重定向至项目内。手动运行命令时先source该脚本。显式写入均限制在 `/home/dym/`，未使用 `/root/` 或系统临时目录。

## 优化与测量

保留FP32、禁用TF32，保持16环境×128步、10 epochs、4 minibatches、8步去噪及所有学习参数。服务器使用PyTorch4线程、Numba8线程，CPU绑定0–11，与GPU0相邻。

实现优化包括：观察编码时使用CPU生成的整数候选索引，避免CUDA布尔索引的输出大小同步；每次编码缓存候选归属，供去噪和概率计算复用；CPU验证分段长度后跳过CUDA重复校验。原始观察与检查点格式不变。固定权重、观察及随机状态下，优化前后动作、评分、完整去噪链和log概率最大差异均为0。

单次采样＋概率重算的Profiler中，`repeat_interleave`调用从47降至15，CUDA kernel launch从1204降至954。这些小算子优化本身没有带来明显的整轮收益：512条转移、微批256的更新从9.71秒变为10.14秒，属于短测波动。主要可复现加速来自扩大梯度累积微批。

下表使用**同一真实规模的2048条联合转移**，每档重新载入相同模型和Adam初值，完成40次actor及critic更新；各档采集数据相同：

| 每微批UAV样本数 | PPO更新时间 | PyTorch峰值显存 |
|---:|---:|---:|
| 512 | 19.59秒 | 277MiB |
| 2048 | 6.76秒 | 547MiB |
| **8192（服务器默认）** | **4.66秒** | **1587MiB** |

8192相对512的更新时间减少76.2%，约快4.2倍。它只改变完整minibatch内部的梯度累积，未改变PPO样本量、40次优化或算法公式；浮点求和顺序可能改变，不承诺不同微批之间的训练轨迹逐位相同。本地微批32保持不变。

这一轮128向量步的推理约3.09秒、物理环境约5.33秒。混合三档的监控均值GPU利用率13.5%、峰值36%，`nvidia-smi`峰值显存4556MiB，RSS峰值1375MiB。该监控包含初始化、采集和三档更新，不能当成8192档的稳态平均利用率；PyTorch显存统计为已分配内存，和驱动统计口径不同。小网络、CPU环境和频繁的短GPU任务仍限制利用率，没有据此开启多卡或改变学习预算。

按这一完整rollout基准线性外推，纯在线采集＋更新约1.8小时；**不含示范、50轮预训练、正式32场景验证、保存及训练后数据分布变化**，不是正式全流程耗时承诺。

## 验收结果

- 本地Python3.11／PyTorch2.7.1 CPU：18项扩散DPPO测试通过，包括新整数路由、聚合值与梯度检查；此前旧路径回归结果保留在本地验收报告。
- RTX3090：采样与保存链概率重算最大误差 `2.43187e-5`，PPO比率最大误差同量级；UE编码器、运动头、评分头均获得有限非零梯度。
- CUDA中断恢复：完整模型、Adam、随机状态及环境状态恢复后，继续采集更新与连续执行的最大参数差异为 **0**；末步时间因子1、终止状态0均通过。
- 真实场景两阶段短流程：拟合2场景＋选模1场景，共1200条示范；预训练1轮15个batch；在线2048条联合转移、1轮PPO、40次优化。容量10、18 UAV、150 UE、400步回合和原无线参数保持正式设置。
- 示范约6.49秒，预训练及完整选模回合检查约12.37秒，在线及两次2场景验证约36.72秒。整个监控进程约60.62秒，RSS峰值1427MiB、驱动显存峰值3426MiB。
- 在线短流程中，PPO约4.35秒、推理3.11秒、环境5.62秒；验证约22.84秒。实际短训已显示验证开销不可忽略。
- 检查点、best/last/step0、导出均生成成功；导出actor和训练actor在相同随机状态下的运动与评分最大差异均为0。

短流程预训练仅1轮，且只使用3个示范场景，**不能用于判断正式算法效果，也不作为正式训练初始权重**。正式输出目录尚未创建，启动时从头执行正式示范、预训练和1M在线训练。

详细JSON、CSV、日志和配置已下载至 [本地服务器验收数据](../analysis/diffusion_dppo_server_deployment_20261105/)。服务器原始数据位于 `deployment/`，最终上传校验清单另存为 `deployment/final_upload_manifest.json`。

## 正式启动

登录后，用项目内的tmux socket保留终端会话：

```bash
ssh dym@10.16.63.110
cd /home/dym/uav_sac
tmux -S /home/dym/uav_sac/.cache/tmux.sock new -s diffusion-dppo
bash scripts/train_diffusion_dppo_server.sh
```

按 `Ctrl+B`，再按 `D` 分离会话。重新连接：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t diffusion-dppo
```

启动脚本自动设置缓存位置和CPU绑定，依次执行正式示范采集、50轮预训练、1M在线训练。可在启动前指定已分配的GPU及邻近CPU，例如GPU2：

```bash
CUDA_VISIBLE_DEVICES=2 UAV_CPU_AFFINITY=12-23 bash scripts/train_diffusion_dppo_server.sh
```

## 中断恢复

先加载同一环境：

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
```

示范采集可用相同输出重跑 `dppo collect`，已校验完成的场景会跳过。预训练从本阶段 `last.pt` 恢复：

```bash
taskset -c 0-11 .venv-dppo/bin/python -u -m uav_research dppo pretrain \
  --config configs/diffusion_dppo_capacity10_coverage_throughput_1m_server.yaml \
  --demonstrations runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/demonstrations \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain \
  --resume runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain/last.pt
```

在线训练必须从完整目录恢复，`full_XXXXXXXXX`替换为实际最近一份目录：

```bash
taskset -c 0-11 .venv-dppo/bin/python -u -m uav_research dppo train \
  --config configs/diffusion_dppo_capacity10_coverage_throughput_1m_server.yaml \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 \
  --resume runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/checkpoints/full_XXXXXXXXX
```

若原日志已有比该完整检查点更新的步数，程序会拒绝混写；此时将 `--output` 指向新的恢复目录，仍使用原完整检查点。轻量 `last.pt`不含完整环境与随机状态，不能代替完整恢复目录。
