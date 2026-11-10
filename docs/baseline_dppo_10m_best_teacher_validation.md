# Baseline与DPPO的10M best、以及新teacher验证

结果与图表：[三组验证报告](../analysis/baseline_dppo_10m_best_teacher_seed0/report.md)。

本次只比较三组，颜色依次为红、蓝、橙：

1. `runs/mappo_coverage_throughput_capacity10_10m_seed0/best.pt`：原集中式关联，10M训练中的best实际来自3,821,568步。
2. `runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0/best.pt`：学习运动和评分、Top-K关联，best来自9,550,656步。
3. 第一组同一个冻结运动actor＋固定局部SINR评分＋Top-K，作为新baseline的teacher闭环对照，不单独训练。不是第二组示范预训练时使用的旧1M教师。

32场景10000–10031、400步、动作种子880001、采样执行、批次16、CUDA FP32；每20步使用独立随机流抽样64次计算条件运动方差。teacher不使用原集中式critic作为其价值预测。

baseline原始结果与完整恢复文件、日志、TensorBoard及验证历史已从服务器下载；新baseline与新teacher在服务器复验，DPPO复用相同协议且已核对模型SHA256的冻结best评估。未使用保留测试集，未开始新训练。

本地只读保存数据重建：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv-dppo-compat/bin/python scripts/report_baseline_dppo_10m.py \
  --baseline-run runs/mappo_coverage_throughput_capacity10_10m_seed0 \
  --dppo-run runs/diffusion_dppo_capacity10_coverage_throughput_10m_seed0 \
  --output analysis/baseline_dppo_10m_best_teacher_seed0
```

服务器若需要重新采样两组新评估，先在`/home/dym/uav_sac`加载`scripts/dppo_server_env.sh`，使用`.venv-dppo/bin/python`执行同一脚本，分别传`--evaluate baseline_best`或`--evaluate teacher_topk`。物理GPU由`CUDA_VISIBLE_DEVICES`选择，脚本内部设备为`cuda:0`。DPPO可继续复用来源一致的缓存。

来源、下载校验、逐场景配对区间、完整逐步指标、MC回报及拓扑见报告目录中的JSON、CSV与NPZ。图表输出PDF/SVG/PNG。
