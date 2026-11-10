# 容量10、覆盖与吞吐baseline：单卡从零训练10M

配置：`configs/mappo_coverage_throughput_capacity10_10m_server.yaml`。
以原`mappo_coverage_throughput_capacity10_1m.yaml`为来源，仅将`run.total_steps`设为10,000,000、`run.device`设为`cuda:0`。场景、奖励、观察、集中式关联、共享高斯actor、时间因子critic及全部学习参数保持原值。

奖励为`coverage + 0.1 * throughput_normalized`，容量10；断联、鲁棒性、边界惩罚权重均为0。CPU8线程、Numba4线程、16环境×128步；共4883轮PPO，末轮104个向量步、1664条联合转移，共195320次actor及critic优化。

该运行从种子0重新初始化actor、critic、Adam和环境，命令不包含`--resume`，不读取旧结果或预训练权重。历史1M及DPPO结果继续保留。CUDA执行不承诺与历史CPU训练逐位一致。

服务器项目和所有数据、缓存、日志均位于`/home/dym/uav_sac/`。使用已有`.venv-dppo` CUDA11.8环境。

## 启动

登录服务器后执行，示例选择物理GPU0：

```bash
cd /home/dym/uav_sac
tmux -S /home/dym/uav_sac/.cache/tmux.sock new -s mappo-10m -c /home/dym/uav_sac 'CUDA_VISIBLE_DEVICES=0 bash scripts/train_mappo_capacity10_10m_server.sh'
```

只使用一张卡；改为`CUDA_VISIBLE_DEVICES=1`等即可更换物理GPU，程序内仍为逻辑`cuda:0`。默认CPU亲和范围0–11，若使用不同NUMA节点可通过`UAV_CPU_AFFINITY`指定分配到的CPU范围。

预检查不会开始训练，也不会创建正式运行目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_mappo_capacity10_10m_server.sh --check
```

重新进入会话：

```bash
tmux -S /home/dym/uav_sac/.cache/tmux.sock attach -t mappo-10m
```

按`Ctrl+B`再按`D`退出tmux显示，训练继续。启动脚本发现正式目录非空时会退出，防止覆盖或把旧结果误当成新训练。

## 输出与评估

- 正式结果：`/home/dym/uav_sac/runs/mappo_coverage_throughput_capacity10_10m_seed0/`。
- 终端日志：`/home/dym/uav_sac/logs/mappo_capacity10_10m_*.log`。
- 训练指标、验证、best/last、TensorBoard及完整恢复检查点沿用原baseline保存逻辑。
- 初始step0参与验证与选模，之后约每10k联合转移评估32场景；best按两项任务奖励选择。
- 验证种子10000–10031、动作采样种子880001；保留测试集不参与训练验证。

正式训练由用户手动启动。该配置用于补齐在线样本预算；DPPO另有示范和预训练成本，论文比较应另外报告。
