# M2运动缩放与邻居数量：四组10M对照

参照已完成的M2（仅运动Diffusion DPPO＋SINR Top-K），新增四组从零训练的实验。原M2配置、1M/10M运行、模型和分析全部保留。正式训练由用户启动。

| 组 | 可见邻居上限 | 运动缩放 | 默认物理GPU | 固定观察维度 | Actor参数 |
|---|---:|---|---:|---:|---:|
| M2，复用参考 | 3 | 指数跳数，0.25/0.4 | 已完成 | 105 | 150978 |
| M2-NoHop | 3 | 关闭，所有UAV比例1 | 0 | 105 | 150978 |
| M2-N5 | 5 | 与M2相同 | 1 | 153 | 157122 |
| M2-N7 | 7 | 与M2相同 | 2 | 201 | 163266 |
| M2-N9 | 9 | 与M2相同 | 3 | 249 | 169410 |

## 控制变量与解释

- 场景3500×3500 m，18 UAV、150 UE、3 BS，容量10；中心UE初始化后移动，UU650 m、UG1500 m、最大指令30 m/s、400步、镜面反射均不变。
- Reward仅覆盖率＋0.1归一化数据率；鲁棒性、断联和边界惩罚均为0。Gaussian、联合评分actor、教师、示范、BC均不参与这些新实验。
- 使用M2相同的8步DDPM、normal-reference残差初值、每UAV每去噪步PPO比率、actor/critic学习率1e-4/5e-4、GAE、PopArt、Huber和value clipping。
- NoHop只关闭速度缩放，仍保留反射后实际位移历史，不退回指令历史。N5/N7/N9只增加通信范围内按距离/编号排序的邻居槽，缺失补零；不扩大UU距离，不观察通信范围外UAV。
- 参考M2先到1M，再续训10M；新组自动复现相同阶段边界：先从零训练1M，再完整恢复actor、critic、Adam、归一化、环境、RNG和best状态至累计10M。共4884轮、195360次优化；不是额外10M，也不继承已训练的M2。
- 四组critic输入均6292维，参数量1677057；构造使用既有独立固定随机流，因此初始critic相同。邻居更多使actor上下文第一层增宽；不能声称网络容量完全相等。无新增全局信息或agent ID输入。
- 本次沿用原扩散网络的标准构造，未为新增邻居列另加零初始化规则。零残差初值在固定采样随机流下有相同初始动作分布；输入宽度改变可能改变构造后的训练采样随机流，不能宣称整段初始随机轨迹逐位配对。
- best沿用原规则：32场景全回合验证，按后100覆盖优先、数值相等时后100吞吐选取；约每50k完整更新边界评估，保留step0。主要比较同预算last，另报历史best，避免掩盖M2已观察到的后期退化。

配置来源：`configs/motion_only_sinr_topk_10m/M2.yaml`。新配置在 `configs/m2_motion_neighbor_controls_10m/`，每组包含主10M和自动第一阶段1M配置，manifest固定来源SHA及全部配置SHA。可用 `scripts/build_m2_control_configs.py` 重建；不要在已经启动后随意重生成变更配置。

## 一条命令，4卡并行

在服务器运行：

```bash
cd /home/dym/uav_sac && bash scripts/launch_m2_controls_server.sh
```

默认GPU0/1/2/3依次运行NoHop/N5/N7/N9，各用一张卡，进程内为cuda:0；CPU集合分别0–5、6–11、12–17、18–23，每进程PyTorch4线程、Numba4线程。每个实验独立tmux，第一阶段完成会自动转入10M阶段，不需要中途再输入命令。

启动前会检查四卡空闲、CPU集合不重叠、至少31GiB可用空间、全部配置哈希、会话和输出目录；全部通过后才派发训练。已有结果默认拒绝覆盖；不会删除任何旧数据。先检查、不训练：

```bash
cd /home/dym/uav_sac && bash scripts/launch_m2_controls_server.sh --check-only
```

自定义物理GPU顺序或CPU绑定：

```bash
UAV_M2_CONTROL_GPUS=0,1,2,3 \
UAV_M2_CONTROL_CPUSETS='0-5|6-11|12-17|18-23' \
bash scripts/launch_m2_controls_server.sh
```

全部输出位于数据盘 `/mnt/disk2/dym/uav_sac/`：

- 第一阶段：`runs/m2_motion_neighbor_controls_1m/<组>_seed0/`，完整保留。
- 最终运行：`runs/m2_motion_neighbor_controls_10m/<组>_seed0/`，CSV继承第一阶段，保留来源与初始模型。
- 日志：`logs/m2_motion_neighbor_controls_10m/<组>.log`，从第一阶段到最终阶段连续追加。
- 模型：安全点完整检查点最近两份＋step0/best/last；checkpoint约10分钟，在更新完成后原子发布。

## 进度和恢复

```bash
tail -f /mnt/disk2/dym/uav_sac/logs/m2_motion_neighbor_controls_10m/M2-NoHop.log
tail -f /mnt/disk2/dym/uav_sac/logs/m2_motion_neighbor_controls_10m/M2-N5.log
tail -f /mnt/disk2/dym/uav_sac/logs/m2_motion_neighbor_controls_10m/M2-N7.log
tail -f /mnt/disk2/dym/uav_sac/logs/m2_motion_neighbor_controls_10m/M2-N9.log
tmux -S /home/dym/uav_sac/.cache/tmux.sock ls
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t m2ctrl-M2-N9
# Ctrl-b，再按d：脱离，训练继续。
```

中断后先确认原进程已退出。四组都需要恢复且尚未到10M时：

```bash
cd /home/dym/uav_sac && bash scripts/launch_m2_controls_server.sh --auto-resume
```

只有某组需要恢复，使用新tmux会话单独运行，例如：

```bash
cd /home/dym/uav_sac
tmux -S /home/dym/uav_sac/.cache/tmux.sock new-session -d -s m2ctrl-M2-N9 \
  'cd /home/dym/uav_sac && CUDA_VISIBLE_DEVICES=3 UAV_CPU_AFFINITY=18-23 bash scripts/train_m2_controls_server.sh train --group M2-N9 --auto-resume >> /mnt/disk2/dym/uav_sac/logs/m2_motion_neighbor_controls_10m/M2-N9.log 2>&1'
```

恢复会识别第一或第二阶段；没有完整检查点、日志超前、契约不匹配或已达10M均明确报错，不会改为从零训练或静默回滚。完成的组不能再用恢复命令重跑；四组恢复命令遇到其中已完成的组也会在启动前停止，此时只恢复尚未完成的组。

## 完成后验证

```bash
cd /home/dym/uav_sac
source scripts/dppo_server_env.sh
CUDA_VISIBLE_DEVICES=0 .venv-dppo/bin/python -u scripts/m2_control_experiments.py report --device cuda:0
# 仅读保存数据重建：
.venv-dppo/bin/python scripts/m2_control_experiments.py report --rebuild
```

默认比较原M2＋四组新实验，step0/best/last，32场景10000–10031、三个动作种子880001–880003，不使用保留测试集。按组局部验证可用 `--groups M2 M2-NoHop` 等。拒绝把未完成10M的最终目录混入正式报告。

结果写入 `analysis/m2_motion_neighbor_controls_10m/`。包含后100/50/终态覆盖与吞吐、窗口内标准差、P05、骤降次数/最大跌幅、联网比例、部署形成与保持、速度/反射/靠边、critic MC误差、固定拓扑与逐场景数据。按场景先平均动作重复，再进行2000次配对bootstrap；不代表训练种子置信区间。

分别绘制运动缩放两组与邻居数量四组，采用不同颜色和线型。NoHop与原M2用于判断缩放取舍；N5/N7/N9与原M2用于判断更多邻居信息及相应输入容量的效果。不预设缩放或更多邻居必然更好，不自动延长预算。

验收记录：`output/m2_controls_acceptance_20261117/`。正式训练没有由助手启动。
