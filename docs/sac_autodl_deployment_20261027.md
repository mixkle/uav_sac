# SAC AutoDL部署记录（2026-10-27，2026-10-29整理）

代码路径 `/root/uav_sac`；环境和数据盘根目录 `/root/autodl-tmp/uav_sac`。设备为RTX3080Ti 12GB、10 vCPU Xeon Gold6248、30GB内存、50GB数据盘。服务器Python3.12、PyTorch2.8 CUDA12.8；本地CPU PyTorch版本不同，采样随机流不要求逐位复现。

使用 `scripts/server_env.sh` 将缓存、临时文件、训练结果写到数据盘，CPU8线程、Numba4线程。手动启动脚本 `scripts/train_sac_coverage_server.sh` 使用保留的server_1m配置，不会自动开始训练。

历史部署验收包括CUDA恢复一致性、完整场景短流程及满容量回放保存恢复。短测和压力测试产物已按2026-10-29清理要求移除；当前保留完整1M运行及评估。此前单次测速显示CUDA明显快于CPU，但优化前后重复测试存在波动，不能将单次15%或25%差异当作稳定加速保证。

正式1M实际完成247500次更新，墙钟约3小时53分；进程峰值RSS约5.03 GiB，PyTorch峰值分配显存约186 MiB，未发生OOM。模型效果及CUDA/CPU分别评估见[SAC结果](sac_coverage_central_history_experiment.md)。

本地仅有轻量模型与日志；完整恢复检查点仍在原服务器运行目录。后续完整奖励MAPPO命令见[服务器说明](server_execution.md)。
