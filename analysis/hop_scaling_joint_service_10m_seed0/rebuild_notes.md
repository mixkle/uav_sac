
## 清理与图表重建

两端仅删除`analysis/hop_scaling_interim_20261109`和`analysis/hop_scaling_interim_20261109_2121`，本地另删除四个对应的一次性缓存脚本。共清理本地约790 MiB、服务器约712 MiB。训练目录、示范、预训练、完整恢复检查点以及本报告的六份模型副本全部保留。服务器19个源文件在清理前后哈希一致；67个最终原始数据和快照文件通过两端一致性检查。删除清单见`cleanup_manifest.json`。

从仓库根目录运行以下命令，只读取保存数据，不连接服务器、不重新采样或训练：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv-dppo-compat/bin/python analysis/hop_scaling_joint_service_10m_seed0/build_report.py
```

服务器可在加载`scripts/dppo_server_env.sh`后将Python路径改为`.venv-dppo/bin/python`。模型来源见`snapshot_manifest.json`，最终原始数据哈希见`raw_data_hashes.json`，评估数学与协议检查见`validation_audit.json`。
