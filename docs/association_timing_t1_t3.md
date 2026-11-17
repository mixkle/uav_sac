# T1–T3 冻结关联时序诊断

2026-11-14已在服务器执行并验收：576个完整回合，18组冻结评估；T1/T2共12组历史复现误差均为0。
源A3 best为251,904步，last为1,000,000步，未重新训练或重选模型。
[完整分析](../analysis/t1_t2_t3_post_motion_association/analysis.md) ·
[主指标CSV](../analysis/t1_t2_t3_post_motion_association/summary/performance.csv) ·
[验收记录](../analysis/t1_t2_t3_post_motion_association/tests/acceptance.json)。

T2较T1提高后100步覆盖约0.76/0.85个百分点（best/last），吞吐约58.65/72.01 Mbps，显著减少中断。
T3较T2的覆盖增益只有约0.28/0.30个百分点，区间均跨0，last中断反而增加。
同布局时序增益约99.6%来自接纳移动后新候选；旧分数更新和旧候选重排的即时作用很小。

执行方案见根目录 `T1-T3_关联时序消融实验执行方案.md`。
本次不训练网络，不改变奖励和模型，不占用保留测试集。

新增 `association_timing.py` 评估适配层：生产 `prepare` 仍验证动作并且只移动一次，
生产 `finish` 只提交选中的一个关联方案，独立影子求解不写入环境。
T1 保留学习评分，T2 使用运动前 SINR，T3 使用运动后 SINR 与完整可服务集合。
P1 固定有效旧提议、只更新 UE 选择；P2 在新旧候选交集中重新执行一轮 Top-K。
评分映射、平局、容量和故障规则沿用现有代码。不补位、不使用最大匹配。

冻结 A3 best、last，各自执行 32 场景、3 动作种子、400 步、批次16；
PyTorch8线程、Numba4线程，每个协议单 GPU，三个协议并行。
模型源为 `runs/system_ablation_1m/A3_seed0/{best,last}.pt`，记录其实际训练步数和哈希。
T1 同物理状态影子池保存五种方案，不混入其他策略轨迹。可选 F2b 状态池本轮未执行。

服务器启动：

```bash
cd /home/dym/uav_sac
bash scripts/launch_association_timing_server.sh
```

结果存储于 `/mnt/disk2/dym/uav_sac/analysis/t1_t2_t3_post_motion_association/`。
三个 `.exit` 均为0后，读取已保存结果生成报告：

```bash
source scripts/dppo_server_env.sh
.venv-dppo/bin/python scripts/diagnose_association_timing.py --rebuild
.venv-dppo/bin/python scripts/summarize_association_timing.py
```

本地重建：

```bash
.venv/bin/python scripts/diagnose_association_timing.py --rebuild
```

原始数据：`rollout/{best,last}/{T1,T2,T3}/{seed}/`，包含每步指标、
每场景指标、闭环坐标/动作/关联块和固定场景轨迹。
T1 另外保存完整扩散链、候选映射、动作前后物理量及五种关联结果，
每50步一块；初始快照包含随机流，可以核查轨迹。
`one_step/` 是同状态影子对照；`summary/` 是场景级汇总与配对区间。
报告及图可直接从文件重建，不重新采样。

覆盖比例存0–1，吞吐原始数据存bps，汇总吞吐为Mbps。
后100步指301–400；标准差先逐轨迹计算，再平均动作重复和场景。
区间由场景级2000次bootstrap产生，不是训练种子区间。
服务中断同时记录每步均值和窗口累计；已结束服务段长度与仍在服务的年龄分开报告，
后者有右删失，不能当成完整服务时长。

验收包括 T1 复现已有 A3，T2 复现此前冻结 SINR 换协议结果，
三种闭环 UE 路径哈希相同，源权重哈希前后不变，影子效应严格望远镜分解。
T3 具有运动后即时测量和交换假设，尚未计入信令时延。
