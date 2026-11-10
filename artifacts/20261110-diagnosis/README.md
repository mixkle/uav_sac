# Diffusion / Gaussian 10M 定位实验完整数据

[诊断报告](../../analysis/diffusion_gaussian_diagnosis_20261110/report.md) · [执行说明](../../docs/diffusion_gaussian_diagnosis.md) · [完整数据 Release](https://github.com/mixkle/uav_sac/releases/tag/diagnosis-2026-11-10)

本快照只增加2026-11-10的定位实验，不覆盖此前`results-2026-11-10`快照。仓库直接保留诊断代码、计划、报告、图表及表格；Release提供完整223文件目录（含输出校验清单），包括完整环境快照、64次条件采样原始数据、反事实分叉、两组各200k微调的模型/优化器/恢复状态及最终评估。

两组微调均由Diffusion已训练best开始、Adam重置；它们不是从零学习，也不是原训练的无缝续训。测试记录为新增诊断4通过，扩展回归32通过、1项既有终止异常优先级失败，详见原报告。

原10M输入模型与评估位于上一份Release，本次不重复打包；源文件SHA256保存在诊断manifest.json。无Token、虚拟环境、缓存或本地冒烟临时输出。

## 下载与校验

在克隆仓库根目录执行：

```bash
diagnosis_asset_dir=.cache/github-diagnosis-20261110
mkdir -p "$diagnosis_asset_dir"
while read -r diagnosis_sha diagnosis_name; do
  curl -fL --retry 3 \
    -o "$diagnosis_asset_dir/$diagnosis_name" \
    "https://github.com/mixkle/uav_sac/releases/download/diagnosis-2026-11-10/$diagnosis_name"
done < artifacts/20261110-diagnosis/SHA256SUMS

(cd "$diagnosis_asset_dir" && sha256sum -c ../../artifacts/20261110-diagnosis/SHA256SUMS)
cat "$diagnosis_asset_dir"/diagnosis-results-20261110.tar.gz.part* | tar -xzf - -C .
python3 scripts/verify_result_snapshot.py \
  --manifest artifacts/20261110-diagnosis/results_manifest.json --detach-hardlinks
```

解压恢复`analysis/diffusion_gaussian_diagnosis_20261110/`，会覆盖同路径内容。分片按文件名字典序拼接；manifest记录逐文件/分片/整个压缩包的SHA256。归档内部用硬链接去重，校验命令将其转成独立文件，避免后续修改相互影响。仅查看报告不需要下载大文件。
