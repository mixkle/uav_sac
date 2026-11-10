# AutoDL执行与数据路径

项目代码可放 `/root/uav_sac`，训练数据、缓存、环境及检查点放 `/root/autodl-tmp/uav_sac`。`scripts/server_env.sh` 设置数据盘缓存及8个CPU／4个Numba线程。

已完成的SAC服务器为RTX3080Ti 12GB、10 vCPU Xeon Gold6248、30GB RAM、50GB数据盘。历史部署详情见[记录](sac_autodl_deployment_20261027.md)。

## 完整奖励MAPPO baseline

上传当前代码和配置并安装环境后，在tmux中手动启动：

```bash
tmux new -s mappo_baseline
cd /root/uav_sac
source scripts/server_env.sh
python -u -m uav_research train \
  --config configs/mappo_full_reward_baseline_1m.yaml \
  --output /root/autodl-tmp/uav_sac/runs/mappo_full_reward_baseline_1m_seed0
```

配置默认CPU8线程，适合当前MAPPO小MLP；本次没有部署或启动服务器训练。如改CUDA须另行测量，不直接套用SAC测速。

完整恢复使用训练目录config.yaml及实际的resume目录，详见[训练说明](training_workflows_and_parameters.md)。best/last是轻量模型，不足以恢复Adam、随机流和环境。

本地清理没有修改远端文件。下载服务器结果时优先轻量模型、日志和评估数据，完整SAC回放检查点较大。
