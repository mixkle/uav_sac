# Diffusion / Gaussian 10M 定位实验

执行依据：[定位计划](../Diffusion_vs_Gaussian_10M_定位实验计划.md)。结果入口：[诊断报告](../analysis/diffusion_gaussian_diagnosis_20261110/report.md)。

本轮为独立诊断，不替换生产策略、物理环境或原10M模型。固定32个开发验证场景10000–10031、执行动作种子880001。原四条完整轨迹均精确复现覆盖序列后再分叉。

## 实际执行

1. 同状态交换运动/评分：两种源策略、best/last、固定t25/100/200/300/375，另收集每场景最严重骤降前状态；每状态64个原始条件样本，其中16个用于单步配对干预。
2. 多步分叉：两种源策略各32场景的t100/t200，共128个起点/权重类型；四种组合运行50步，汇总10/25/50步。每组合一条随机轨迹，UE移动流相同。
3. 全事件归因：32×400步中所有≥10个百分点相邻步骤降，不计reset→首动作；每场景最严重事件另外做18个“只保持某架UAV”的干预。
4. 条件分布：D step0/best/last和G best/last，每状态64次，区分原始指令、跳数缩放和反射后位移；保存真实UE编号、变长评分、原始latent。额外执行0.8标准差干预的完整回合。
5. 历史更新审计：按原日志实际字段检查分步KL/clip、候选数、梯度和概率求和。缺失历史明细明确列出，不造数据。
6. 门控通过后执行A/B：同D best actor、critic、两套归一化，Adam均重置、相同新环境/RNG。各200k额外转移。原设置1e-4/10epochs/clip0.1；保守3e-5/4/0.05。不自动增加预算。

输入保存在原`analysis/hop_scaling_joint_service_10m_seed0/`，只读哈希核验。所有新输出在`analysis/diffusion_gaussian_diagnosis_20261110/`。服务器仅在`/home/dym/`下写入；单GPU运行，每个进程独立绑定CPU，未打断任何原训练。

## 重新运行与重建

使用兼容环境`.venv-dppo-compat`（本地CPU）或`.venv-dppo`（服务器）。完整冻结回放需CUDA与原配置配对，以复现原GPU采样轨迹。下面是数据重建命令，不重新采样、不训练：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv-dppo-compat/bin/python scripts/report_diffusion_gaussian_diagnosis.py audit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv-dppo-compat/bin/python scripts/report_diffusion_gaussian_diagnosis.py report
# report会生成基础统计，先保留为detailed_statistics.md，再运行最终解释入口。
cp analysis/diffusion_gaussian_diagnosis_20261110/report.md \
   analysis/diffusion_gaussian_diagnosis_20261110/detailed_statistics.md

for stage in fork risk cases noise; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-dppo-compat/bin/python scripts/extend_diffusion_gaussian_diagnosis.py "$stage"
done

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv-dppo-compat/bin/python scripts/finalize_diffusion_gaussian_diagnosis.py
```

采集入口`diagnose_diffusion_gaussian.py`提供collect/probe/fork/cases/noise阶段，禁止覆盖既有状态采集；微调`finetune_diffusion_diagnosis.py`必须先读取`gate_decision.json`，已有初始化必须显式指定恢复路径。最终冻结微调模型由`evaluate_diagnosis_finetuning.py`评估。不要把微调新增计数0误称为从零训练网络。

## 结论边界

2000次bootstrap按场景配对，先平均动作重复与状态，不将时间步当独立样本。它不是多训练种子置信区间。相同状态混合运动与评分存在分布外局限；事件案例是有选择的反事实诊断。微调同时改动三个优化参数，不用于判定单参数因果。未使用保留测试集。

## 仓库发布

完整诊断数据单独发布为[diagnosis-2026-11-10](https://github.com/mixkle/uav_sac/releases/tag/diagnosis-2026-11-10)，下载及逐文件校验见[快照说明](../artifacts/20261110-diagnosis/README.md)。原10M来源模型在此前的results-2026-11-10 Release中，未被本次诊断覆盖。
