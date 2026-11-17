# M2四组10M对照：实现与部署验收

日期：2026-11-17。正式训练未启动，原M1/M2及其他实验未删除或覆盖。

- 新配置：`configs/m2_motion_neighbor_controls_10m/`，四组各包含1M第一阶段及10M最终配置。除运行编号、指定的缩放/邻居设置外，逐项等同原M2；阶段之间仅预算不同。
- 原M2的1M→10M阶段被自动复现。旧完整恢复选择器仅增加可选`source_steps`参数，缺省仍为1M；原M1/M2恢复回归通过。
- 本地：31项测试通过，24.32秒；覆盖四组完整两阶段流程、第一/第二阶段中断恢复、原始step0保留、同预算参数匹配、Top-K物理一致、critic初值、概率重算、导出、报告重建和跨观察契约拒绝。
- 服务器：相同31项测试通过，80.01秒；pytest临时文件全部位于数据盘配置的TMPDIR。
- 四张3090并发执行完整主场景16环境的初始采样、概率重算和一次物理step；均为0次优化。NoHop速度系数恒1；其余三组沿用0.25/0.4指数缩放。未创建任何新组正式训练运行目录或正式训练tmux会话。
- CUDA四组`logp_recompute_abs_max`均为0；CPU合并检查中，固定采样随机流下初始指令一致，四组初始全局状态和critic预测一致。新邻居列并未另行使用零初始化，不能据此声称整个训练采样流相同。
- 稠密邻居单测确认第4–9槽真实承载额外邻居；通信图和固定动作下物理状态保持一致。无邻居时仍补零，不扩大通信半径。
- 四卡启动器检查全部配置、会话、GPU占用、CPU集合与磁盘后再启动，`--check-only`通过。四组默认依次使用GPU0/1/2/3，CPU0–5/6–11/12–17/18–23，每进程PyTorch4、Numba4线程。
- 上传18个文件逐个SHA256匹配，见`deployment.json`。生产`src/`与服务器的相关算法代码内容一致，本次未上传或改写算法实现。

代码在`/home/dym/uav_sac`，全部输出在`/mnt/disk2/dym/uav_sac`。启动、进度、恢复及报告命令见`docs/m2_motion_neighbor_controls_10m.md`。

数据文件：`local_tests.txt`、`server_tests.txt`、`local_initialization.json`、`cuda_<组>.json`、`server_preflight.log`及`deployment.json`。
