> 新版 beacon-set-v2：服务器目前不可用，正式训练须等待通知。以下命令仅供恢复可用后参考；完整恢复文件已改为分块目录（无 .pt 后缀），旧版本检查点不能继续训练。新版参数以 configs/main.yaml 为准。

# 服务器部署与续训

代码目录 `/root/uav_sac`；实验、检查点、资源监控、虚拟环境和缓存目录 `/root/autodl-tmp/uav_sac`。

```bash
cd /root/uav_sac
source scripts/server_env.sh
python -m uav_research train --config configs/server_probe.yaml --output runs/probe_seed0
```

CLI 的相对输出路径在 AutoDL 上自动解析到数据盘，也可以通过 `UAV_DATA_ROOT` 指定其他数据根目录。绝对路径保持原义。`server_env.sh` 同时设置缓存及临时文件路径。

采集资源使用率（`psutil` 为监控脚本的额外依赖）：

```bash
python scripts/profile_run.py --output "$UAV_DATA_ROOT/profiles/probe_seed0" -- \
  python -m uav_research train --config configs/server_probe.yaml --output runs/probe_seed0
```

`resources.csv` 中 CPU 百分比以一个逻辑核为 100%；需除以容器 CPU 配额才得到相对租用资源利用率。进程树 RSS 可能重复计算共享页；容器内存以 `cgroup_memory_mib` 为参考。GPU 利用率来自 NVIDIA 采样，不能仅凭显存大小判断效率。脚本同时保存训练 stdout、执行命令及汇总。

短训默认保留主物理场景、10,000 条探索和每四条转移一次更新，只将总预算缩至 14,080 条转移、验证场景缩至四个；该配置用于效率和正确性检查，不作为论文性能证据。

学习与场景参数以 [当前训练流程及参数](training_workflows_and_parameters.md) 为准。SAC actor/critic默认学习率1e-4/3e-4，MAPPO为5e-4/5e-4，显式配置优先。

恢复示例：

```bash
python -m uav_research train --config configs/server_probe.yaml \
  --output runs/resumed_branch \
  --resume "$UAV_DATA_ROOT/runs/probe_seed0/resume_000000014080" --steps 16000
```

训练配置、并行环境数及评估日程必须兼容检查点。性能调优改变并行环境数时，应启动独立短训，不混入原训练分支。
