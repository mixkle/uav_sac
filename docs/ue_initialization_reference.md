# UE 初始化参考与两个版本的选择

参考 [Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models](../Scalable_UAV_Multi-Hop_Networking_via_Multi-Agent_Reinforcement_Learning_with_Large_Language_Models.pdf)，DOI 10.1109/TMC.2026.3669346。原始PDF第11页V-A节给出约3500×3500 m、150 UE、18 UAV，UE采用均匀分布或多中心二维高斯混合；第16页图8的三个BS为左下、左上、右下。

## 当前离散MAPPO

直接采用论文规模：3500×3500 m、150 UE、18 UAV、3 BS。UE在完整区域独立均匀采样，采用论文允许的均匀选项；UAV也独立全区域均匀采样，这是用户确认的补充假设，论文未给出其reset分布。无边界留白、聚类、高斯截断、连通性或服务效果筛选。

`user_initialization=uniform`不使用高斯中心、标准差或留白字段。`uav_initialization=uniform`不使用launch_radius。UE后续保留现有高斯幅度、均匀方向与反射运动，并未复制论文恒速布朗运动；100 m高度亦为补充假设。

## 连续SAC保留设置

| 参数 | 连续版本 |
|---|---|
| UE / UAV / BS | 180 / 16 / 2 |
| 区域 | 1600×1600 m |
| UE初始范围 | [160,1440]² m |
| 等权高斯中心 | (480,480)、(1120,480)、(480,1120)、(1120,1120) m |
| 分量标准差 | 每轴128 m |
| 重采样 | 固定所属分量，仅对越界UE坐标重采样 |
| UAV | 全区域独立均匀采样 |

连续配置留白0.10、标准差0.08、中心各轴0.3/0.7，随区域轴尺寸缩放。独立随机选分量，不强制每组45；留白仅作用于reset。原论文没有给出高斯中心、方差、混合权重或留白值，以上数值为本实验设置。

读取以本地原始PDF文字和图8为准，没有将二手摘要或未给出的初始化参数当成论文事实。完整参数及通信差异见 [训练参数](training_workflows_and_parameters.md)。
