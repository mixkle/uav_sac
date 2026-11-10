# 5页进展PPT素材

内容提纲：`docs/progress_presentation_diffusion_allocation_20261103.md`。

| 页码 | 文件 | 说明 |
|---|---|---|
| 1 | `fig1_baseline_framework.*` | Gaussian MAPPO部署与集中关联流程，CTDE蓝橙分区 |
| 1 | `fig1_baseline_topology_scene10000_t400.*` | 固定场景10000的真实best终态拓扑，覆盖70% |
| 2 | `fig2_baseline_validation_progress.*` | 覆盖、吞吐与回报的验证进度，来自现有日志 |
| 2 | `fig2_episode_maintenance.*` | best/last完整400步的覆盖和断联，来自32场景平均 |
| 2 | `table2_baseline_results.csv` | 初始、best、last可编辑数据表 |
| 4 | `fig4_diffusion_training_pipeline.*` | 拟实施的示范预训练＋on-policy去噪转移PPO全流程 |
| 5 | `fig5_topk_summary.*` | 覆盖上界与吞吐的诊断摘要，来自已保存审计 |
| 5 | `table5_topk_results.csv` | 三种评分、P0/P1/O1/O2可编辑数据表 |

每图包含SVG、PDF与240dpi PNG。SVG保留文本与矢量图元；PDF嵌入字体。未安装Noto Sans CJK的Windows设备可先使用PNG，以免字体替换。

第1页baseline图增加与第4页V2一致的参考风格：[PNG预览](fig1_baseline_framework_v2_reference.png)／[SVG](fig1_baseline_framework_v2_reference.svg)／[PDF](fig1_baseline_framework_v2_reference.pdf)。浅蓝/米橙分区、青蓝/杏橙模块和深色描边宽箭头；原版保留。重建入口：`scripts/build_baseline_reference_style.py`。

第4页算法图另有[三版视觉方案](fig4_style_variants/README.md)：原版、参考图风格、简洁编辑风格。[三版打包下载](fig4_style_variants/fig4_three_styles.zip)。原版文件保持不变。

风格参考论文Figure 1仅用于视觉组织，未采用其离策略算法。本项目新训练图的Replay替代为本轮Rollout Buffer；时间因子critic与团队GAE指导逐UAV、逐去噪转移PPO。新DPPO方案没有在本轮实现或训练。

全部新结果图仅从原CSV/JSON/NPZ重绘，未重新采样轨迹、未加载模型、未启动训练；来源哈希见 `assets_manifest.json`。

在完整项目内重建：

```bash
.venv/bin/python scripts/build_progress_presentation_assets.py
```
