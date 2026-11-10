# 跳数缩放双组实验验收与服务器部署，2026-11-08

2026-11-10整理说明：本页是部署前验收快照，正式两组10M现已完成，见[最终验证](../analysis/hop_scaling_joint_service_10m_seed0/report.md)。短测模型、原始轨迹和重复图表已清理，验收摘要见[acceptance_records](acceptance_records/README.md)，服务器性能与恢复核查JSON继续保留。

已实现共享执行缩放、实际位移历史、扩散两阶段新数据流程，以及独立入口Gaussian运动＋UE评分MAPPO。两组使用相同Top-K、物理场景和覆盖+.1数据率奖励。正式配置均为种子0、在线10M、单GPU；正式示范、预训练与在线训练尚未启动，由用户执行。

- 98项完整回归通过；随后新增两组大/小微批次梯度累积等价测试，7项新路径测试全部通过，合计100个不同测试通过。
- 覆盖历史时序、反射、断联、空集、置换、联合概率、梯度、局部reset、批量/进程一致性、初始化critic一致性、跨版本拒绝、保存恢复及旧SAC/MAPPO/DPPO回归。
- 本地两组各2048联合转移、40次优化，完成保存、导出及完整回合评估。峰值RSS：扩散约0.70GiB，Gaussian约0.67GiB。短测不用于性能优势判断。
- 服务器GPU0与GPU2并发执行两组各64+1984条短转移；均完成优化、保存、导出、MC评估。峰值RSS约1.4GiB。
- CUDA小配置：两种算法中断恢复与连续执行的learner/Adam张量最大绝对差均为0。
- 两组报告的step0/best/last验证、MC、拓扑及PDF/SVG/PNG已在本地跑通；只读取保存数据重建后comparison.json逐字节一致。
- 历史DPPO 10M best（9550656步）仍可按旧契约加载。旧容量10教师文件哈希保持不变。

## 性能实测

先比较128/256/512，随后因GPU利用率低且显存余量大，继续比较1024/2048/4096。最终以完整16环境×128步、40次优化复核：

| Actor | 微批次512更新耗时 | 微批次4096更新耗时 | 降幅 | 4096 PyTorch显存峰值 |
|---|---:|---:|---:|---:|
| Diffusion | 19.31s | 5.30s | 72.6% | 985.02MiB |
| Gaussian | 19.97s | 5.05s | 74.7% | 287.59MiB |

正式服务器配置选择4096；只改变一次完整minibatch中的梯度累积粒度，仍每minibatch更新一次Adam，未增加环境样本、epochs或学习率。实际候选数量会影响显存和耗时；依旧保留12GiB RSS和20GiB PyTorch显存上限。以上为短测，预训练仅使用独立小示范，不能作为正式10M总耗时的精确预测。

原始监测、微批次基准与CUDA恢复记录见 `docs/hop_scaling_server_acceptance_20261108/`；汇总见 `hop_scaling_validation_20261108.json`。

## 部署和启动

仅上传到 `/home/dym/uav_sac/`，所有显式数据、临时文件、缓存和tmux socket均在 `/home/dym/`。复用现有Python3.11/PyTorch2.7.1+cu118环境，不安装或覆盖其他环境，不删除旧结果。上传清单由SHA256逐项校验。

[准备、并行启动、恢复和报告命令](hop_scaling_joint_service_comparison.md)。先完成新扩散示范及50轮预训练，再运行并行启动脚本；启动脚本验证完整示范与预训练标记、来源哈希、GPU/CPU分配、磁盘及独立输出目录。默认扩散GPU0、Gaussian GPU2，各单卡。

原本地验收目录为 `analysis/hop_scaling_local_acceptance/`，不属于正式10M实验结果；清理后保留[短测报告摘要](acceptance_records/20261110/analysis/hop_scaling_local_acceptance/report.md)。
