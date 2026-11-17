# 服务器结果迁移与默认数据目录

2026-11-11，服务器 `dym@10.16.63.110`。用户要求将已经完成的实验结果迁到 `/mnt/disk2/dym/uav_sac`，以后服务器结果也使用该目录；不启动训练。

按用户补充要求，本项目独占 `uav_sac/` 子目录，父目录 `/mnt/disk2/dym/` 留给其他项目。先前平铺于父目录的本项目结果、缓存和迁移记录已一并归入子目录；父目录不保留 `runs/analysis` 等本项目别名。项目代码内的旧路径软链接直接指向新子目录。

| 内容 | 实际新位置 |
|---|---|
| 训练、模型、完整恢复检查点、示范及预训练 | `/mnt/disk2/dym/uav_sac/runs/` |
| 验证、拓扑、报告及定位实验 | `/mnt/disk2/dym/uav_sac/analysis/` |
| 本地完整性/性能验收产物 | `/mnt/disk2/dym/uav_sac/output/`、`deployment/` |
| 控制台日志 | `/mnt/disk2/dym/uav_sac/logs/`、各运行目录内部日志 |
| 新运行缓存及临时文件 | `/mnt/disk2/dym/uav_sac/.cache/` |
| 历史结果压缩包 | `/mnt/disk2/dym/uav_sac/.cache/legacy_artifacts/` |
| 代码、配置、Python环境 | `/home/dym/uav_sac/`，保持原位置 |

迁移前检查没有训练/评估进程，并检查结果没有被进程打开。采用rsync复制，逐文件核对SHA256、大小、权限及修改时间；源与目标完全一致后移除系统盘副本，原位置换为软链接。共迁移6,963个文件、11,302,178,125字节（约10.53 GiB）。没有修改历史模型、验证数组和Best选择。

迁移清单及逐文件哈希保存在服务器 `/mnt/disk2/dym/uav_sac/migration_20261111/`。本地副本及后续验证结果见同目录下本项目的 `docs/server_storage_migration_20261111.json`。

## 默认路径

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
echo "$UAV_DATA_ROOT"
# /mnt/disk2/dym/uav_sac
```

服务器启动、续训和预训练脚本明确使用 `$UAV_DATA_ROOT/runs/`，日志使用数据盘；容量检查也以数据盘空余量为准。服务器配置中的教师与预训练引用改到数据盘，不改变学习参数和实验契约。

直接运行CLI时，相对输出也会自动使用数据盘。例如 `--output analysis/new_report` 指向 `/mnt/disk2/dym/uav_sac/analysis/new_report`。显式绝对路径仍尊重参数。旧绝对路径通过保留的软链接继续有效，不改写历史检查点内的路径。

环境脚本将新缓存、Numba编译、Matplotlib配置、临时目录指向数据盘。已安装Python环境及历史uv安装缓存保持原位；tmux套接字和少量锁文件保持原路径，因此现有会话不受影响。未挂载 `/mnt/disk2` 时拒绝默认保存，避免无意写回系统盘。

本地WSL的相对输出行为与旧AutoDL默认路径保持兼容；不在本地创建 `/mnt/disk2/dym/uav_sac`。
