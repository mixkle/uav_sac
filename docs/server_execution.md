# 服务器执行与数据路径

## 当前服务器（2026-11-11起）

服务器为 `dym@10.16.63.110`。代码与Python环境保留在 `/home/dym/uav_sac/`，训练、检查点、示范、预训练、验证、日志及新增运行缓存统一放在 **`/mnt/disk2/dym/uav_sac/`**。

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
echo "$UAV_DATA_ROOT"  # /mnt/disk2/dym/uav_sac
```

现有启动脚本已使用数据盘路径；直接通过项目CLI传入相对 `--output runs/...` 或 `analysis/...` 也会默认使用该数据盘。本地WSL行为不变。显式绝对输出路径仍按用户参数处理。

已完成结果迁移并逐文件SHA256核验，原项目中的 `runs/analysis/output/deployment/logs` 保留为软链接，历史配置、模型引用、教师路径及恢复入口仍可使用。`/mnt/disk2` 未挂载时启动脚本会报错，避免退回系统盘保存。

迁移与验证记录见[服务器存储迁移](server_storage_20261111.md)。tmux套接字仍在 `/home/dym/uav_sac/.cache/tmux.sock`，原会话无需重建。

## 历史AutoDL部署

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
