# SAC纯覆盖1M对照（已完成）

共享actor、连续圆盘动作、中心UE初始化、相对坐标、一步历史、时间因子双Q。与共享连续PPO纯覆盖1M使用同一物理场景。

配置：`configs/sac_coverage_time_factor_relative_central_history_1m.yaml`（CPU）与 `configs/sac_coverage_time_factor_relative_central_history_server_1m.yaml`（CUDA）。CPU8线程、Numba4线程。

SAC actor/critic/温度学习率1e-4/3e-4/1e-4，初始alpha=.01、目标联合熵−36、gamma=.995、Polyak=.005、batch256、回放50k、warmup10k、每4条转移更新一次，共247500次更新。时间缩放为 `Q=(H/H0)*f(s,a)`，不使用MAPPO的PopArt。

原服务器CUDA口径：初始覆盖29.29%；best460k覆盖30.87%；last1M覆盖20.79%。统一CPU三动作种子：best27.57%，last23.30%，共享PPO last64.94%。SAC后期明显退化，不能仅凭TD loss或平均Q接近目标就判断动作价值排序正确。

完整数据与说明见[报告](../analysis/sac_coverage_central_history_1m_seed0/report.md)和[解读](../analysis/sac_coverage_central_history_1m_seed0/interpretation.md)。重建已保存图表：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
.venv/bin/python scripts/report_sac_coverage.py --rebuild
```

本地运行目录保留服务器轻量best/last及日志，完整回放检查点在原服务器。未达到1M的本地SAC中断运行已删除。SAC1M结果不代表完整奖励baseline的表现。
