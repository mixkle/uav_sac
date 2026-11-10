# MAPPO / SAC CTDE 流程图：代码核对记录

日期：2026-10-28。仅核对实现并制作图示，没有修改训练代码或运行训练。

## 图示范围

采用当前可比较的连续动作、纯覆盖、相对坐标、一步历史、解析时间因子版本。基础配置分别为 `configs/continuous_mappo_coverage_shared_central_history_100k.yaml` 和 `configs/sac_coverage_time_factor_relative_central_history_server_1m.yaml`。图示呈现更新机制，不将不同训练预算或设备画成算法定义。

两个配置共用3500×3500 m、18 UAV、150 UE、3 BS、400步回合、中心UE初始化、移动UE、30 m最大步长和反射边界。奖励为实际联网服务覆盖率ρ，其他环境奖励权重为零。完整奖励、九动作MAPPO和独立actor属于已有可选实验，不在主图中混画。

## 核对结果

| 内容 | 实现及图示约定 | 源码 |
|---|---|---|
| 局部输入 | 105维固定特征含4维一步历史；可变UE集合4→32→32，均值与人数33维；actor输入138维 | `observations.py`, `models.py::UEEncoder`, `environment.py` |
| Gaussian actor | 状态相关二维均值和log标准差；256×256主干；圆盘变换a=z/√(1+‖z‖²)，log概率包含Jacobian | `models.py::GaussianActor`, `disk_transform` |
| MAPPO采集 | 16×128的当前策略rollout；保存原始观测、latent、旧log概率、价值、奖励和终止标记 | `train.py`, `algorithms.py::MAPPO.collect` |
| MAPPO价值 | 6292维全局物理状态；V=H·m；平均奖励归一化器PopArt重标定输出层及Adam；累计价值归一化只用于损失和裁剪 | `algorithms.py::MAPPO.value_components`, `MAPPO.update` |
| MAPPO策略更新 | 团队GAE优势，逐agent PPO比例；10 epochs×4 minibatches；actor重新编码集合；无V动作梯度 | `algorithms.py::gae`, `MAPPO.update` |
| SAC采集 | 10k圆盘均匀warmup；50k条完整联合回放，batch256，每4新增转移一次更新 | `train.py`, `replay.py`, 对应配置 |
| SAC价值 | 两个6328→256→256→1的集中式Q，输入为全局状态加36维联合动作；Q=(H/H0)f；固定尺度，无MAPPO式PopArt | `models.py::TimeFactorTwinQ`, `TwinQ` |
| SAC目标 | 当前actor采下一动作，目标双Q最小值减未来联合熵成本；目标停止梯度；只有目标critic、没有目标actor | `algorithms.py::SAC.update`, `soft_target` |
| SAC策略更新 | 冻结critic参数，保留Q对动作的梯度；联合log概率为18机log概率之和；可训练actor与UE编码器 | `algorithms.py::SAC.update` |
| SAC温度 | α初值0.01，联合目标熵−36，温度独立优化；actor及Q目标均使用α | `algorithms.py::SAC.__init__`, `SAC.update` |
| 时间因子 | H=剩余折扣和，γ=.995；从状态已有时间字段计算；t399为1、t400为0；真正终止不bootstrap | `models.py::DiscountedHorizon`, `train.py` |

## 需要保留的设计边界

1. 当前actor读取历史，但critic的6292维全局物理状态没有历史。本图准确呈现现状，不代表已解决状态信息遗漏问题。
2. 独立actor版本仍使用集中式团队V和团队优势，不能标为标准IPPO。可在另图把共享actor替换为固定编号绑定的18套actor；SAC本次对照仍是共享actor。
3. MAPPO的熵仅作为actor正则；SAC的soft-Q包含未来策略熵。二者的价值标签不同。
4. SAC圆盘均匀warmup不是actor采样，图中用回放旁注呈现，不把warmup误画为Gaussian actor。
5. 基础建模文档保留历史场景和完整奖励定义，因此主图以此次对应配置及运行代码为准，不能把旧1600 m参数混入。

## 生成方式及文件

使用内置image_gen生成工具；用户指定模型名称为gpt-image-2，内置工具不提供显式模型选择参数，不能从调用接口独立核验后端型号。完整生成提示词保存在同目录 `prompt.txt`。成图经视觉核对后保存在本目录；PNG为生成的位图，若提供PDF则为位图封装，不声称是可编辑矢量图。

最终采用简化构图的`final_prompt.txt`及单行公式修正`typography_correction_prompt.txt`。最终文件为`mappo_sac_ctde.png`和`mappo_sac_ctde.pdf`；其余图片为过程草稿。完整中英文图注见`caption.md`。图中的相同DATA标签是配对端口，表示联合数据分别进入两种算法自己的buffer；并不表示两算法同时更新同一actor。MAPPO的value loss同时依赖预测V与GAE目标，图中用loss卡片文字表达，省略部分连线。代码哈希见`provenance.json`。
