# 集合条件化扩散Actor＋DPPO：容量10、覆盖与数据率

代码：`src/uav_research/diffusion_dppo/`。流程为冻结教师采集示范、联合去噪预训练、逐UAV/逐去噪转移PPO微调。原Gaussian MAPPO和SAC保留。1M服务器运行已完成；后续累计3M续训与单/双卡命令见 [续训说明](diffusion_dppo_3m_continuation.md)，由用户启动。

## 场景和协议

主配置 `configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml` 的所有env字段与 `mappo_coverage_throughput_capacity10_1m.yaml` 一致：3500² m、18 UAV、150 UE、3角落BS、容量10、UE中心2500² m初始化后移动、UAV全场均匀初始化、400步、30 m/步、镜面反射、UU650 m、UG三维1500 m。保留满负载SINR和独立信标，每UE链路2 MHz、20 dBm。

奖励只有 `coverage + 0.1*tau`，其中 `tau=throughput/(150*5e6)`；其他奖励权重为0。资源分配指服务名额和关联，不表示学习带宽、功率或RB。

Reset使用baseline初始贪婪关联。第一步起，在动作前可服务集合上生成全部评分，每机提议Top10；移动后过滤失效提议，UE选最高评分，平局按UE/UAV编号。不补位、不驻留、不在移动后第二次推理；新发现但没评分的UE等待一步。刷新或恢复不重新匹配。

观察固定105维＋全部感知UE的5维集合：相对xy、上行SNR/30、下行SINR/30、本机服务标记。只有可服务集合生成评分；UE编号只作路由元数据。服务及历史人数除以10；集合人数用 `log1p(count)/log(11)`，允许超过1。下行SINR反馈沿用模型假设，不计测量误差、延时和信令成本。

## 网络和参数

| 模块 | 设置 |
|---|---|
| UE编码 | 5→64→64，ReLU，masked mean |
| 上下文 | (105＋64＋1)→128→128 |
| noisy-score配对 | 65→128→64，再按候选集合均值聚合 |
| 去噪步编码 | 32维固定正弦 |
| 运动／评分头 | 各两层256 ReLU，输出2维／每候选1维噪声 |
| 扩散 | 8步DDPM，余弦偏移0.008、最大β0.999、后验std下限0.05，末步也随机 |
| 输出 | 运动圆盘映射、评分sigmoid；不截断latent和采样噪声 |
| Critic | 6292→256→256→1，时间因子、平均奖励PopArt和累计价值归一化 |
| 在线学习率 | actor 1e-4，critic 5e-4；Adam eps1e-5 |
| 折扣／GAE | 0.995／0.95 |
| Actor／Value clip | 0.1／0.2，独立配置 |
| Value loss | 归一化累计价值空间Huber δ10、Value Clipping |
| PPO | 10 epochs、4 minibatches、梯度裁剪0.5 |
| 熵／在线BC系数 | 0／0 |
| 在线采集 | 16×128；1M；489轮，末轮576条，19,560次actor和critic优化 |

log密度对有效动作坐标求和；PPO按单UAV、单去噪转移计算ratio和clip，然后对样本/UAV/去噪步求平均。团队GAE在联合转移上标准化。不使用全链联合ratio、不平均坐标log密度、不裁剪log概率。圆盘、sigmoid和Top-K是其后的环境映射，内部转移概率不加它们的Jacobian。critic每个minibatch只更新一次。

actor有历史，critic仍读取当前物理状态，此限制保留并用MC审计观察。固定反向高斯条件熵不是最终动作熵，不作为可学习熵奖励。

## 示范与预训练

运动教师 `runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt`，实际671,744步。只使用actor，在新Top-K环境真实执行，通过5维观察投影得到原4维教师观察。评分教师原始logit为 `(SINR_DL-threshold)/10`，无跳数和关联历史偏好，与固定布局诊断一致。

- 拟合集300000–300127：128×400＝51,200条联合转移。
- 选模集301000–301031：32×400＝12,800条；总示范64,000条，与在线验证及测试隔离。
- 教师动作主种子980001，按场景派生流；保存运动高斯latent，不反解接近圆盘边界的动作。
- 50轮、batch1024、Adam 3e-4/eps1e-5、梯度裁剪0.5。
- 运动噪声MSE按2维平均，评分MSE按本机候选数平均，空集为0，两项等权。
- 每轮洗牌场景块和块内样本，全部样本遍历一次，批次可跨块；内存映射、最多缓存4个场景。
- 选模按独立集固定噪声、样本序号模8指定扩散步的总去噪损失，含epoch0。正式验证不参与预训练选模。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python -u -m uav_research dppo collect \
  --config configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/demonstrations

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python -u -m uav_research dppo pretrain \
  --config configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml \
  --demonstrations runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/demonstrations \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain
```

预训练恢复在第二条命令后追加 `--resume .../pretrain/last.pt`。保存模型、Adam、排列、游标、未完成轮次统计及CPU/CUDA随机状态。结束后best在选模场景执行完整回合；表现不及教师只记录，不自动扩充预算。

## 在线训练和恢复

加载预训练best，重新初始化critic和在线Adam；不继承教师critic或预训练Adam。rollout保存当前候选映射、完整去噪链和逐步旧概率；不能用新集合、新采样替换。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python -u -m uav_research dppo train \
  --config configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml \
  --pretrained runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain/best.pt \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0
```

这是完整1M命令，不能作为短测。首次运行也可用 `dppo all --config ... --output ...` 依次执行两阶段；恢复时分别执行阶段命令。

`last.pt`在验证前和每轮PPO后保存。完整目录在更新安全点、每10分钟及结束时原子发布，保留最近两份。恢复必须使用完整目录：

```bash
.venv-dppo-compat/bin/python -u -m uav_research dppo train \
  --config runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/config.yaml \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 \
  --resume runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/checkpoints/full_XXXXXXXXX
```

`--steps`是累计目标。跨实验、观察、协议或学习设置续训被拒绝；可换设备、线程和微批，不承诺跨设备逐位一致。SIGINT/SIGTERM/SIGHUP请求在当前完整更新后保存退出，终端仍建议用tmux。非有限数值停止并保留failure诊断，中间PPO状态不能续训。

## 评估和重建

验证10000–10031、采样种子880001、批次16，约每10k更新边界评估。best按两项奖励全回合回报，包含预训练后step0；保留测试集不参与开发。

```bash
.venv-dppo-compat/bin/python -u -m uav_research dppo report \
  --run runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 \
  --output analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 --device cpu

.venv-dppo-compat/bin/python -m uav_research dppo report \
  --run runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 \
  --output analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 --rebuild

.venv-dppo-compat/bin/python -m uav_research dppo export-actor \
  --checkpoint runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/best.pt \
  --output runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/actor.pt
```

报告包含原baseline best/last、同一教师＋新Top-K、扩散step0/best/last，记录实际步数和哈希。主要比较同预算last及相对step0的增益。示范、预训练另计，关联、UE输入和actor均有变化，不能将差异只归因于扩散结构。

完整MC使用两项环境奖励和真实终止，不含熵或去噪概率。条件动作方差每20步独立采样64次，不改变执行流；可用 `--moment-repeats 0` 跳过并在报告记录。图表从保存数据重建，输出PDF/SVG/PNG。配对场景bootstrap 2000次、种子950001，不是训练种子置信区间。

## 本地验收和服务器配置

`diffusion_dppo_local_smoke.yaml`：4 UAV、24 UE、小回合；用于端到端测试。`diffusion_dppo_local_probe.yaml`：真实物理场景，拟合/选模2/1场景、预训练1轮、验证2场景、在线2048条，仅完整性验收。输出独立，不能作为正式1M初值。

隔离依赖，不覆盖现有 `.venv`：

```bash
uv venv --python 3.11 .venv-dppo-compat
uv pip install --python .venv-dppo-compat/bin/python \
  --index-strategy unsafe-best-match -r requirements/dppo-cpu.txt -e .
```

这里合并PyPI和PyTorch官方索引候选，避免uv优先索引规则使common.lock的固定版本找不到。服务器对应 `requirements/dppo-server-cu118.txt`：Python3.11/PyTorch2.7.1 CUDA11.8，不套用旧CUDA12.8环境或升级驱动。

已部署并复核：4×RTX3090 24GB、双路E5-2678 v3（24物理核/48线程）、约64GB RAM、驱动535.230.02、Ubuntu20.04。代码、Python环境、缓存、示范和结果均位于 `/home/dym/uav_sac/`，不使用 `/root/autodl-tmp`。部署和性能记录见 [服务器验收报告](diffusion_dppo_server_deployment_20261105.md)。

服务器配置 `diffusion_dppo_capacity10_coverage_throughput_1m_server.yaml`：单GPU、PyTorch4线程、Numba8线程、UAV微批8192；本地8/4线程、微批32。微批仅控制完整PPO minibatch内的梯度累积，不改变16×128的采集量、4个minibatch或Adam更新次数。由 `CUDA_VISIBLE_DEVICES` 指定卡，程序使用cuda:0。启动脚本默认绑定CPU0–11，对应GPU0/1；GPU2/3可设置 `UAV_CPU_AFFINITY=12-23`。

采用批量物理计算、变长数组、一次观察编码复用8步、固定页传输和完整minibatch加权梯度累积。默认FP32，不默认多卡、AMP或编译。RSS目标12GiB、显存20GiB、磁盘保留15GiB；不足时停止，不丢UE、不缩预算。进度记录采集、编码、去噪、传输、更新、验证和保存。

已完成完整128向量步基准：512、2048、8192个UAV样本的微批分别需19.59、6.76、4.66秒完成同一轮40次PPO更新。选择8192，峰值PyTorch显存约1.55GiB。线性短估算不含完整验证、示范与预训练，不能当作精确1M时间；只优化设备/线程/微批，不改16×128或学习参数。

已上传并安装环境。以下为正式两阶段启动命令，由用户执行，部署期间没有启动正式1M训练：

```bash
cd /home/dym/uav_sac
bash scripts/train_diffusion_dppo_server.sh
```

启动脚本首先加载 `scripts/dppo_server_env.sh`，使临时文件及Numba、CUDA、Matplotlib等缓存也保存在项目内。手动执行分阶段命令或恢复命令时，先 `source scripts/dppo_server_env.sh`。冻结教师best已上传，路径和SHA256由采集入口校验。GPU编号应按届时分配修改；建议用tmux保持会话。CUDA概率重算、梯度和精确续训已通过短测。
