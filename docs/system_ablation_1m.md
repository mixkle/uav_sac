# Diffusion / Gaussian 1M 系统消融：实施与服务器命令

2026-11-17新增：[M2-NoHop / M2-N5 / M2-N7 / M2-N9四组10M对照](m2_motion_neighbor_controls_10m.md)。仅运动扩散DPPO，从零训练；自动沿用1M→10M阶段，GPU0–3各一组，用户启动。原M2三邻居＋跳数缩放结果保留为参考。

M1/M2两组仅运动实验已完成1M；[追加至累计10M的续训命令](motion_only_sinr_topk_10m.md)使用GPU2/3，完整状态恢复，原1M结果独立保留。

2026-11-16新增：[M1/M2仅运动策略＋SINR Top-K，从零训练1M](motion_only_sinr_topk_1m.md)。两组不学习评分、不使用BC，独立目录保存；默认GPU2/3，由用户启动。与历史桥接实验同名编号无关。

2026-11-14追加：[T1–T3冻结关联时序诊断](association_timing_t1_t3.md)。冻结A3的best/last，
比较学习评分、运动前SINR和运动后SINR一轮Top-K；576个回合已全部执行并验收，未增加训练。

2026-11-13结果：19组（含可选C4）均完成1M；step0/best/last各用32个验证场景与3个动作种子冻结评估完毕。[完整分析](../analysis/diffusion_gaussian_ablation_1m/analysis.md) · [A–F数值表](../analysis/diffusion_gaussian_ablation_1m/comparison_analysis.md) · [last完整指标CSV](../analysis/diffusion_gaussian_ablation_1m/metrics/last_late_metrics.csv)。正式训练没有追加，保留测试集未使用。

依据根目录《UAV_Diffusion_Gaussian_1M_系统消融实验方案.md》（2026-11-12）。本轮交付代码、配置与验证工具，不自动启动正式训练。

代码：`/home/dym/uav_sac`。全部输出：`/mnt/disk2/dym/uav_sac`。旧配置、模型与结果保留。

## 冻结约定

- 18个必做独立运行，额外提供可选C4。A2/B1/C1/D1/E1复用唯一G0；F1b/F2b复用A3。
- 每组种子0、各1M在线联合转移（A4为在教师之后额外1M），16环境×128步，489轮PPO，最后576条；actor/critic学习率1e-4/5e-4，PPO clip0.1、value clip0.2、10epochs×4minibatches。8步DDPM，其余继承联合10M配置。
- 覆盖＋0.1数据率，容量10、中心UE、真实位移历史、相对坐标、时间因子critic。B3特意改为指令历史。
- 所有组约每50k在完整更新边界验证32场景10000–10031，记录实际步数。best主指标是后100步覆盖；仅数值相等（容差1e-12）时按后100步吞吐决胜。未采用示例0.5pp容差，以免允许吞吐补偿覆盖降低。step0参与选模；历史10M的旧选模不变。
- A4只加载新G0的best actor，critic、归一化、Adam、在线随机流和计数从新阶段初始化。不继承教师critic/优化器；记录教师已训练步数。这是额外1M，不是合计1M。
- B5固定c=0.5，属于预先指定的常数减速，不是拟合G0平均速度；应结合实际速度解释。
- A3/A5读取完全相同的G0联合latent示范：128拟合＋32选模完整回合，共64k联合转移；A3去噪MSE，A5联合高斯latent NLL。均50轮、batch1024、学习率3e-4；分支损失的绝对数值不能跨分布比较。F2a使用同一示范，只监督运动。
- F1a直接复用A3的BC best，在线关闭Hop；不重新采样或重训BC。应报告其初始化分布失配。
- D3/E3刻意保留无执行作用评分概率（E3与E2保持同一网络），不作为推荐算法。D4/F2a完全移除评分随机变量及概率，规则评分不参与PPO。

## 实验映射

| ID | 改动 |
|---|---|
| G0 | 共享Gaussian联合运动＋评分；指数Hop；完整集合；局部Top-K |
| A1 | 随机Diffusion，无BC |
| A3 | 新G0 best示范→Diffusion BC→新1M在线阶段 |
| A4 | 直接加载G0 best actor→新1M在线阶段 |
| A5 | 同一份示范→Gaussian NLL BC→新1M在线阶段 |
| B2 | 线性Hop，alpha(1)=0.25，alpha(U+1)=1 |
| B3 | 指令历史代替真实位移历史 |
| B4 | 不缩放，真实位移历史 |
| B5 | 全队恒定0.5缩放 |
| C2 | 上行信标SNR降序、UE ID裁决的前10UE，5维特征＋mask展平编码 |
| C3 | 同一前10UE，集合均值编码 |
| C4（可选） | 全部感知UE的均值、二阶矩、最大值统计，经线性投影；无逐UE学习聚合 |
| D2 | 仅运动＋移动后集中式贪婪参考 |
| D3 | 联合运动＋评分训练，但执行用动作前SINR规则Top-K |
| D4 | 仅运动训练＋同一动作前SINR规则Top-K |
| E2 | 全局信息＋同一局部候选资格/Top-K |
| E3 | 与E2相同的全局联合actor，但评分执行改为移动后集中式贪婪参考 |
| F1a | A3相同BC初始化，在线关闭Hop |
| F2a | Diffusion仅运动＋SINR Top-K；同一示范仅运动BC |

C2/C3的运动上下文均不读取前10之外的逐UE特征；评分分支仍保留全部合法候选，固定观测中的人数历史不删除。C2/C3参数量不同，日志记录实际数量，不能宣称完全匹配网络容量。

E组是全局信息oracle：额外输入每架UAV相对位置、跳数、服务人数，以及全部UE的相对位置和当前是否服务，共522维，`522→128→64`编码后加入本机上下文。原局部资格不扩展、UE编号仅对齐。它增加观察权限和参数量，不是可部署协议。E3保留与E2相同的联合actor与评分概率，但集中式贪婪执行端不使用这些评分；该无效分支与D3一样会带来优化噪声，应报告。E2/E3的物理权限和关联时序均不同，不能把差距解释为某一种集中求解器的纯收益。

集中式参考复用旧`kernels.associate`：移动后依据全局SINR候选排序、唯一关联、容量限制，不受动作前提议限制；是贪婪，不是最大匹配或理论上界。其与Top-K的权限/时序差异在报告中标明。

## 推荐启动顺序

SSH登录后：

```bash
cd /home/dym/uav_sac
# 第一批：4张单卡，各自独立tmux。正式训练由你执行这一命令启动。
UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_system_ablation_server.sh G0 A1 D3 D4
```

全部组可独立启动；不要在同一张忙碌GPU上叠加作业。默认会话`abl-G0`等；tmux socket不变。

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock ls
tail -f /mnt/disk2/dym/uav_sac/logs/system_ablation_1m/G0.log
```

G0完成1M以后，先采集共享示范并预训练三个学生（这一步也需要用户启动）：

```bash
cd /home/dym/uav_sac
tmux -S /home/dym/uav_sac/.cache/tmux.sock new-session -d -s abl-prepare \
  'cd /home/dym/uav_sac && CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_system_ablation_server.sh > /mnt/disk2/dym/uav_sac/logs/system_ablation_1m/prepare.log 2>&1'
```

该脚本顺序执行collect A3→pretrain A3→A5→F2a。中断恢复可分阶段显式执行；已完成示范会校验并跳过，不重新生成。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_system_ablation_server.sh pretrain --group A3 \
  --resume /mnt/disk2/dym/uav_sac/runs/system_ablation_1m/pretraining/A3/last.pt
```

准备完成并释放GPU后，按批次启动；每批完成后再启动下一批：

```bash
UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_system_ablation_server.sh A3 A4 A5 F2a
UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_system_ablation_server.sh B2 B3 B4 B5
UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_system_ablation_server.sh C2 C3 D2 F1a
UAV_ABLATION_GPUS=0,1 bash scripts/launch_system_ablation_server.sh E2 E3
# C4可选
CUDA_VISIBLE_DEVICES=0 bash scripts/train_system_ablation_server.sh train --group C4
```

### 前两批完成后：一条命令排完剩余实验

G0、A1、D3、D4、A3、A4、A5、F2a都完成1M后，可以替代上面的手动分批方式：

```bash
cd /home/dym/uav_sac && UAV_ABLATION_GPUS=0,1,2,3 bash scripts/launch_remaining_system_ablation_server.sh --include-optional
```

该命令在独立tmux会话`abl-remaining`中调度B2、B3、B4、B5、C2、C3、D2、F1a、E2、E3及可选C4。每张卡同时只运行一组；哪张卡先完成，就自动接下一组，不等待同批其他卡。去掉`--include-optional`则只运行剩余10个必做实验。不会自动开始验证报告。

已完成1M的组自动跳过；已有未完成结果默认报错，确认需要续训时追加`--resume-incomplete`，仅从完整检查点恢复。如果一组报错，队列停止派发新组，其他正在训练的组继续完成；不会自动改参数或重启失败组。关闭SSH终端不会关闭tmux中的队列。运行前所选4张卡应无其他计算任务。

```bash
# 总队列：START / DONE / FAILED及对应GPU
tail -f /mnt/disk2/dym/uav_sac/logs/system_ablation_1m/queue.log
# 每组仍使用独立日志，例如B2
tail -f /mnt/disk2/dym/uav_sac/logs/system_ablation_1m/B2.log
```

启动器默认为四组分配互不重叠的CPU集合0–5、6–11、12–17、18–23，可用`UAV_ABLATION_CPUSETS`（竖线分隔）覆盖。默认每个进程PyTorch4、Numba4线程，支持`UAV_TORCH_THREADS`、`UAV_NUMBA_THREADS`覆盖。显卡由`UAV_ABLATION_GPUS`或`CUDA_VISIBLE_DEVICES`指定，进程内总是cuda:0；不使用多卡actor并行。

完整恢复示例（只接受full目录，不能用轻量last.pt恢复）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_system_ablation_server.sh train --group G0 --auto-resume
# --auto-resume没有完整检查点时明确报错，不会重新初始化。
```

启动器拒绝已有输出覆盖；从错误共享模式、运动规则、关联或初始化语义的检查点恢复会报错。A4不是`--resume`，它由配置显式执行新阶段actor初始化。

## 验证与报告

以下仅验证，不更新权重。统一验证场景，默认动作种子880001；可追加880002、880003。保留测试集20000–20063仅在最终检查时通过`--split test`显式启用，不用于选模。

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=0 .venv-dppo/bin/python -u scripts/report_system_ablation.py \
  --groups G0 A1 A3 A4 A5 B2 B3 B4 B5 C2 C3 D2 D3 D4 E2 E3 F1a F2a \
  --device cuda:0 --action-seeds 880001 880002 880003
# 首批完成可只传 --groups G0 A1 D3 D4，无需等待全部训练。
# 仅读取保存的轨迹重画，无需模型与服务器GPU：
.venv-dppo/bin/python scripts/report_system_ablation.py --rebuild
```

输出：`/mnt/disk2/dym/uav_sac/analysis/diffusion_gaussian_ablation_1m/`。

- `metrics/unified_best_last.csv`：step0/best/last，后100/50覆盖、吞吐、逐场景覆盖标准差、终态、回报、联网比例、骤降、P05、最大跌幅、首次达标。
- `metrics/per_scene_action.csv`：逐场景、逐动作种子数据。时间t=301–400；骤降包含300→301，阈值10pp。
- `metrics/paired_differences.csv`：配对差值及Hop/评分交互；每场景先平均动作重复，再2000次场景bootstrap（950001）。不是训练种子置信区间。
- 每组保存原始`step_metrics.npz`、两条代表轨迹、critic真实MC标签与预测；MC不含actor熵。
- `figures/A_*`至`F_*`：学习与任务曲线；固定场景10000/10001拓扑。绿色/蓝色为首两条，额外曲线用橙、灰、浅蓝等并配线型。
- `sources.json`：checkpoint SHA256、真实步数、初始化、配置、参数量。不同方法的示范/BC/教师成本另计。

主要比较last与其step0增益；best仅补充。尚未完成组留空，不从历史10M替代G0或教师。C4为可选；不自动追加训练种子、训练步数或扫描超参数。

补充诊断：日志将原始高斯/DDPM转移的运动与评分log概率分开记录，提供movement/score conditional KL及各自比率的clip比例。优化始终使用原联合比率；拆分统计只用于检查评分维度是否主导裁剪，不把DDPM单分支条件比率冒充最终动作边缘概率。`--moment-repeats 64`可开启独立随机流的条件动作方差审计，计算成本另计。报告还绘制冻结best/last的累计价值及平均奖励预测与真实MC曲线。

## 评分诊断 S1／S2

T1/T2冻结对照之后，新增[UE评分排序、随机性与规则消融诊断](score_rule_s1_s2.md)。该实验不训练，固定A3 best/last；在独立场景锁定保持与负载权重，再做旧32场景及新32场景的三动作种子确认。输出 `analysis/score_rule_s1_s2_20261116/`，完整主表为 `s2/comparison.csv`，不能混作新的1M训练组。
