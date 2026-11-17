# F2b、F2c、A1b追加定位实验：运行说明

本轮1M训练与32场景×3动作种子复评已完成。结果解读见[验证分析](../analysis/diffusion_followup_f2b_f2c_a1b_20261113/analysis.md)，全部step0/best/last指标见[统一CSV](../analysis/diffusion_followup_f2b_f2c_a1b_20261113/metrics/unified_step0_best_last.csv)。F2b last后100步覆盖76.23%、吞吐850.86 Mbps；F2c为71.81%、710.65 Mbps；A1b为50.84%、423.61 Mbps。当前均为单训练种子的诊断结果。

依据根目录`UAV_Diffusion_三项追加定位实验_F2b_F2c_A1b_执行方案.md`。新实现沿用1M系统消融的物理场景、容量10、指数Hop、真实位移历史、覆盖＋0.1吞吐、8步DDPM及DPPO参数。三组均为种子0、各1M在线联合转移，正式运行由用户启动。

**命名隔离：** 旧`configs/system_ablation_1m/manifest.json`中的`F2b=A3`是旧2×2表的别名，继续保留。本轮真正的新F2b必须使用`diffusion_followup.py`及下面的新配置，不能使用旧`system_ablation.py --group F2b`。

|组|新配置|初始化与唯一指定改动|
|---|---|---|
|F2b|`configs/diffusion_followup_1m/F2b.yaml`|复制原A3在线`step0.pt`的全部actor参数及去噪buffer；critic、归一化、Adam从新阶段初始化。保留联合评分采样、8步链和联合log概率，只有执行评分改为已有D3的动作前本地SINR规则。|
|F2c|`configs/diffusion_followup_1m/F2c.yaml`|同G0教师、同64k示范及原始联合latent标签，重新执行50轮BC。仅将送入运动去噪头的noisy-score pair聚合置零；输入宽度、参数、UE上下文与评分头保留。Gate同时用于BC、在线、概率重算和导出加载。|
|A1b|`configs/diffusion_followup_1m/A1b.yaml`|无BC、无教师权重。噪声预测为`√(1−ᾱ)z＋Δθ`，运动与评分残差输出层零初始化且可训练。反向日程、方差下限与概率公式保留。|

A1b的反向均值使用与原DDPM后验**代数等价**的写法，避免高噪声时浮点运算`z−b*b*z`相消。采样和log概率调用同一均值函数；没有裁剪或隐藏缩放latent。该分布不保证严格标准正态。

当前pair聚合同时包含UE embedding与评分latent。F2c将这条混合分支送入运动头的值整体置零，但主UE集合上下文继续保留。因此应将结论限定为“该混合条件分支”的作用，不能进一步唯一归因于评分噪声分量。

## 存储

- 服务器代码：`/home/dym/uav_sac/`。
- 新训练：`/mnt/disk2/dym/uav_sac/runs/diffusion_followup_1m/{F2b,F2c,A1b}_seed0/`。
- F2c新BC：同目录下`pretraining/F2c/`。
- 前置诊断：同目录下`audits/`。
- 日志：`/mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/`。
- 最终对照：`/mnt/disk2/dym/uav_sac/analysis/diffusion_followup_f2b_f2c_a1b_20261113/`。

原A1/A3/G0/F2a/A5模型、旧BC和64k示范只读复用；不覆盖旧配置或结果，不重新采集示范。

## 一条命令：准备完成后三卡同时在线训练

```bash
cd /home/dym/uav_sac && UAV_FOLLOWUP_GPUS=0,1,2 bash scripts/launch_diffusion_followup_server.sh
```

脚本先检查GPU空闲、空间、参考运行和输出目录，创建tmux会话`followup-prepare`，依次完成：

1. A1b完整初始化审计、F2c耦合敏感性审计。
2. F2b冻结A3的step0/best/last评分替换验证，与F2c的50轮BC并行运行。
3. 两项都成功后，创建`follow-F2b`、`follow-F2c`、`follow-A1b`三个独立tmux会话，**同时开始各自1M在线训练**。

因此不是命令执行后立即三组在线训练；F2c必须先获得自己结构下的BC。准备阶段只需等待，脚本随后自动启动三组。上传代码时不会执行这个启动命令。

默认物理GPU0/1/2分别给F2b/F2c/A1b，每组内部使用`cuda:0`，每组PyTorch4、Numba4线程，CPU集合0–5/6–11/12–17互不重叠。可用`UAV_FOLLOWUP_GPUS`及`UAV_FOLLOWUP_CPUSETS='0-5|6-11|12-17'`覆盖。GPU3不占用。

查看准备与训练进度：

```bash
tail -f /mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/prepare.log
tail -f /mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/F2c_pretrain.log
tail -f /mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/F2b.log
tail -f /mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/F2c.log
tail -f /mnt/disk2/dym/uav_sac/logs/diffusion_followup_1m/A1b.log
tmux -S /home/dym/uav_sac/.cache/tmux.sock ls
```

关闭SSH不会停止tmux。准备失败时三组在线训练都不启动，查看`audit_init.log`、`audit_coupling.log`、`frozen.log`、`F2c_pretrain.log`定位；不要通过修改审计门槛强行放行。

## 初始化与冻结审计

A1b固定池包含原16个训练worker初始场景，以及128个独立场景932000–932127；三个初始策略A1/G0/A1b各对每个本地观察采样64次，不读取验证回报调参。门槛在配置manifest中提前冻结：

- 运动latent模长均值0.1–5，P99≤15；圆盘指令模长均值0.1–0.95、P95≤0.99。
- 有效评分latent标准差≥0.05；sigmoid小于0.01或大于0.99的比例≤5%。另报0.05/0.95饱和比例。
- 所有概率有限；同链重算容差为`atol=0.002, rtol=0.0002`；有效坐标概率求和、动作/ID/Mask检查通过。
- 完整审计`gate.json`记录配置契约和初始actor SHA256，缺失、不通过或契约/初值不符时，A1b训练入口明确拒绝。

`--smoke`只用于少量样本验收，**不会产生可用于正式启动的通过标志**。

F2c的敏感性审计将原A3 BC权重分别装入Gate开/关网络，仅用于验证机制；正式F2c仍必须重新BC。固定观察和运动latent、扰动评分噪声时，Gate=0的运动预测差应严格为0，但UE上下文仍可影响运动、评分输出仍正常。

F2b冻结验证使用32场景和3个动作种子，保留A3原actor、critic与归一化，不更新权重。已有A3 learned验证数据只在SHA与场景匹配后复用；SINR分支重新执行完整回合。另在原策略轨迹t=0/150/200/399处，用动作种子880001对所有32场景做完整快照的单步干预，保存快照、同一动作及去噪链、两套提议/关联和差值。该随机流下每一步还用同一物理转移解析两种评分的提议、Jaccard和关联归属，保存`step_protocol_diagnostics.csv`，不额外移动或再调用actor。两条长期分支状态会分化，不能声称之后每步仍是相同运动动作。

## 单阶段命令及恢复

```bash
# 只读诊断，可独立重建/续做完整缓存
CUDA_VISIBLE_DEVICES=2 bash scripts/train_diffusion_followup_server.sh audit-init --group A1b
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_followup_server.sh audit-coupling --group F2c
CUDA_VISIBLE_DEVICES=0 bash scripts/train_diffusion_followup_server.sh frozen --group F2b

# 新BC；--auto-resume已有last时精确恢复，已完整50轮时验证并跳过
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_followup_server.sh pretrain --group F2c --auto-resume

# 中断后的在线恢复：需要full检查点，找不到时不会从零重启
CUDA_VISIBLE_DEVICES=0 bash scripts/train_diffusion_followup_server.sh train --group F2b --auto-resume
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_followup_server.sh train --group F2c --auto-resume
CUDA_VISIBLE_DEVICES=2 bash scripts/train_diffusion_followup_server.sh train --group A1b --auto-resume
```

每个在线配置仍为16环境×128步，1M共489轮PPO，最后576条联合转移，actor/critic共19,560次优化；约每50k完整更新边界验证。best仍按880001下后100步覆盖优先、吞吐仅精确平局裁决。

完整检查点每10分钟在更新安全点保存，保留最近两份。A1b另在第一次达到100k的安全更新点（正常为100,352）写轻量`audit_100k.pt`，用于动作尺度复核，不改变训练随机流或更新次数。

## 训练后统一报告

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=0 .venv-dppo/bin/python -u scripts/report_diffusion_followup.py --device cuda:0
```

将A1/G0/A3/A5/F2a的已冻结验证经SHA核对后复用，与新F2b/F2c/A1b的step0/best/last统一比较。新组仍为32场景×3动作种子；不按复评重新选择best，不使用保留测试集。

- `metrics/unified_step0_best_last.csv`、`per_scene_action.csv`、`paired_differences.csv`、`online_gains.csv`、`internal_diagnostics.csv`。
- `F2b`冻结干预结果与其重新训练结果分开保存，不合并称为在线收益。
- `A1b_scale_progress`：固定审计观察池上的step0／100k／best／last动作尺度与两个显式sigmoid饱和阈值；与on-policy分布区分。
- `figures/F2_followup`、`A1_initialization`、`A_BC_references`：覆盖、吞吐、奖励、训练loss/KL/clip；另有各模型固定10000/10001拓扑及MC价值误差。

服务事件`last100_direct_switches`等字段是该窗口**每步事件数的平均**，乘100才是每回合窗口总次数。历史`score_saturation_fraction`是`|latent|>10`，不可和本次显式0.01/0.99、0.05/0.95阈值混用。

只读取保存数据重画，不重新采样：

```bash
.venv-dppo/bin/python scripts/report_diffusion_followup.py --rebuild
```

本轮不自动追加训练种子、调整超参数、延长预算或启动F2b＋F2c组合。是否改善需要分别检查step0能力、在线净收益、last、稳定性和配对场景区间。
