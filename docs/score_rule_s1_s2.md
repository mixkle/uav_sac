# S1／S2：冻结评分诊断与规则消融

执行依据根目录的 `S1_S2_UE评分机制诊断与规则消融_实验计划.md`。新增独立诊断模块，不修改训练算法、生产关联协议或历史结果。所有模型固定为系统消融A3的best（251,904步）和last（1,000,000步），无梯度、无critic调用、无优化器。

## 数据与控制变量

- 原始T1/T2轨迹：`analysis/t1_t2_t3_post_motion_association/rollout/`，重放全部32×400×3×2×2转移，逐步核对关联、位置、覆盖、吞吐、回报、中断和外生UE路径哈希。
- 固定时点：动作编号50/100/150/200/250/300/325/350/375/400。例如第50步取动作执行前t=49观察，影子使用已记录的第50次物理转移。
- 每场景每动作种子按候选11–15、>15两个层各补抽一个非固定时点；保存纳入概率及逆概率权重。额外拥挤样本不混入固定时点主估计。
- 同观察16次完整联合扩散采样，独立诊断Generator，不消耗执行随机流；仅把新评分投影到相同原运动后的物理状态。保存运动、最终评分和原始logit，不把这解释为给定运动后的条件分布。
- S2独立校准场景740000–740015、动作种子890001；Q及第10/11名间距先测量，网格{0,.05,.15,.30}。在非零β/λ组合中最大化两检查点平均即时任务奖励，精确相等则选更小参数。
- 锁定β=.05、λ=.05后才执行开发场景10000–10031和确认场景750000–750031。两组checkpoint共用同一参数，不按闭环结果修改。
- 开发集三个新规则R1/R2/R3合计230,400条转移；新确认集R0/R1/R2/R3合计307,200条；独立校准12,800条。全部为冻结评估，不算训练步数。
- 不补位、不加入驻留、不做动作后新评分；局部前状态Q、当前关联和负载是唯一规则输入。负载项不改变本机Top-K。

## 命令

服务器进入 `/home/dym/uav_sac`，先 `source scripts/dppo_server_env.sh`。数据统一写 `/mnt/disk2/dym/uav_sac/analysis/score_rule_s1_s2_20261116/`。

```bash
.venv-dppo/bin/python -u scripts/diagnose_score_rules.py calibrate --device cuda:0

# 可将R1/R2/R3/R0分配给不同空闲卡；R0工作进程还执行全部S1重放和16次采样。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_score_rule_worker_server.sh R1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_score_rule_worker_server.sh R2
CUDA_VISIBLE_DEVICES=2 bash scripts/run_score_rule_worker_server.sh R3
CUDA_VISIBLE_DEVICES=3 bash scripts/run_score_rule_worker_server.sh R0

.venv-dppo/bin/python scripts/diagnose_score_rules.py verify
.venv-dppo/bin/python scripts/diagnose_score_rules.py report
```

每个独立case有完整来源与完成标记；重复执行复用已完成且契约一致的case，不覆盖T1/T2历史。失败case可重跑，参数锁定后不会重新校准。冒烟必须使用独立输出目录并加 `--smoke`。

下载结果后本地重建：

```bash
.venv/bin/python scripts/diagnose_score_rules.py report
```

服务器无需安装pandas、scipy或Parquet库，连续数组以NPZ存储。可选本地依赖 `requirements/score_diagnostics.txt` 用于额外导出逐UE、逐次采样的Parquet；NPZ是完整可重建源，Parquet并非必需运行依赖。

## 文件与口径

- `report.md`：完整中文结果；`s1/report.md`、`s2/report.md`分开说明。
- `configs/preregistered_design.json`：校准/确认划分及事前选择准则；`locked_rules.json`：权重、来源、Q分布和完整性哈希。
- `s1/{best,last}/{动作种子}/score_samples/`：变长原始观察、候选映射、16次输出、原评分、前SINR与固定后状态速率。
- `s1/ranking_summary.csv`：固定样本与补抽拥挤层分开；未定义Spearman留空。
- `s1/score_effect_paired.csv`：单次评分、16次平均评分、SINR在同运动下的配对差值。
- `s1/event_attribution.csv`：每次中断的互斥原因与重叠标签；直接换UAV另计。
- `s1/service_spells.csv`、`survival.csv`：已结束与右删失服务段、Kaplan–Meier生存。
- `s2/comparison.csv`：各集合、checkpoint、规则的一行完整汇总，含后100/50步、终态、稳定性、连续性和资源利用。
- `s2/paired_differences.csv`：R1−R0、R2−R0、R3−R0及2×2交互的场景配对区间。
- `s2/shadow_paired.csv`：T1/T2两种参考布局分开的固定转移影子结果。
- `s2/per_episode.csv`：逐场景、逐动作种子；`per_scene.csv`先平均三个动作种子；`per_step.csv`为整合逐步数值。
- `audit/baseline_regression.json`：冻结来源、旧轨迹逐步回归、UE随机路径一致性；`figures/`同时输出PNG/PDF/SVG。

后100步是动作301–400，后50步是351–400。覆盖σ先逐条400步轨迹取后期时间标准差，再平均动作重复及场景。事件同时保存每步均值与窗口累计次数，不混用。区间先平均同场景三次执行，再做2000次场景级bootstrap，不是训练种子区间。

初始关联来自baseline贪婪分配，仅在reset发生。只要UE收到至少一个有效提议，本协议就会为它选择一架UAV，因此纯跨UAV归属竞争本身不能让该UE完全中断；中断归因不把直接切换当成丢服务。

## 本轮已执行结果

已完成全部S1与S2、独立新场景确认、R0重新调用actor的回归；源权重未改。共检查2,296个固定观察快照（其中1,920个为预定固定时点），每个16次采样。最终整合表有18行组别汇总、1,728行逐回合指标、691,200行逐步指标，Parquet保存4,506,960条候选UE评分样本；图表9组，均导出三种格式。

- 拥挤集学习评分与SINR Spearman：best 0.0014，last −0.0036；重复采样Top-K Jaccard约0.65/0.64，第10/11名翻转约48%。
- 后期学习评分服务中断中，89.58%/91.08%主要归于Top-K未提议。该指标是UE级丢服务，不等于所有整体覆盖骤降。
- 固定运动下16次评分均值对吞吐的改善远小于SINR；仅收紧采样未恢复物理质量排序。
- 保持项在best上减少一些中断，但存在吞吐取舍；负载项及组合未显示稳定的覆盖、吞吐联合收益。12项覆盖配对对照均无区间下界大于0的结果。

完整数值与区间见 [实验报告](../analysis/score_rule_s1_s2_20261116/report.md)；主表 [comparison.csv](../analysis/score_rule_s1_s2_20261116/s2/comparison.csv)。本轮未启动训练，未修改生产协议，未使用保留测试集。
