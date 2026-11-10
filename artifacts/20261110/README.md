# 训练结果快照：2026-11-10

完整模型和结果通过同一仓库的[GitHub Release：results-2026-11-10](https://github.com/mixkle/uav_sac/releases/tag/results-2026-11-10)发布。仓库内直接保留源码、配置、文档、分析报告、图表及表格；Release包含本地当前保留的整个`runs/`和`analysis/`，恢复后路径与研究记录一致。

## 范围与来源

- 包含20个本地1M及以上正式运行，以及当前跳数缩放双组10M的六份冻结模型和完整评估数据。
- 包含本地现存的模型、训练日志、恢复检查点、示范、预训练、轨迹、MC预测、图表及诊断。
- 最新跳数缩放两组的完整在线恢复目录位于训练服务器；本次上传本地已有内容，没有额外下载这些远程文件。
- 已按项目清理记录删除的短测不在快照中；验收摘要保存在`docs/acceptance_records/`。
- 本地Token、虚拟环境、临时文件及参考论文原PDF不在上传范围内。

`results_manifest.json`逐文件记录路径、大小及SHA256，另记录分片大小、校验和与整个压缩包的校验和。相同内容在tar内部以硬链接去重，解压后完整恢复各原路径。每个Release分片不超过1 GiB，顺序拼接后形成一个gzip压缩tar文件。

## 下载与恢复

从克隆后的仓库根目录执行：

```bash
result_asset_dir=.cache/results-20261110
mkdir -p "$result_asset_dir"
while read -r result_sha result_name; do
  curl -fL --retry 3 \
    -o "$result_asset_dir/$result_name" \
    "https://github.com/mixkle/uav_sac/releases/download/results-2026-11-10/$result_name"
done < artifacts/20261110/SHA256SUMS

(cd "$result_asset_dir" && sha256sum -c ../../artifacts/20261110/SHA256SUMS)
cat "$result_asset_dir"/training-results-20261110.tar.gz.part* | tar -xzf - -C .
python3 scripts/verify_result_snapshot.py --detach-hardlinks
```

解压会恢复`runs/`和`analysis/`，已有同名结果会被快照内容覆盖。最后一条命令逐文件校验，并将压缩包内用于去重的硬链接转为独立文件，避免后续续训或改写日志时影响其他历史副本。只查看报告和图表无需下载完整快照。

## 研究入口

- [最新跳数缩放10M完整对照](../../analysis/hop_scaling_joint_service_10m_seed0/report.md)
- [正式结果索引](../../docs/results_inventory.md)
- [项目导航](../../docs/project_structure.md)
- [训练与恢复命令](../../docs/hop_scaling_joint_service_comparison.md)
