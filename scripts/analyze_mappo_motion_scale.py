"""Rebuild interpretation of saved speed interventions, with no new rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_online_time_factor import write_csv


def main(output):
    root = Path(output)
    result = json.loads((root/'summary.json').read_text())
    meta = json.loads((root/'experiment.json').read_text())
    assert meta['status'] == 'completed'
    rows, sources = [], [Path(__file__), root/'summary.json', root/'experiment.json']
    for label in ['M1-C','M1-C-H']:
        curves, paths = {}, {}
        times = np.arange(11,391)
        for speed in [15,30,45]:
            path = root/label/f'speed_{speed}/curves.npz'
            sources.append(path)
            with np.load(path) as data:
                mean = data['diagnostics'].mean((0,1))
                names = list(data['fields'])
                curves[speed] = np.convolve(mean[:,names.index('coverage')],np.ones(21)/21,'valid')
                paths[speed] = mean[10:390,names.index('cumulative_actual_distance_m')]
        # Use the common support of all three movement-scaled curves, no extrapolation.
        lower = int(np.ceil(max((times*s/30).min() for s in curves)))
        upper = int(np.floor(min((times*s/30).max() for s in curves)))
        grid = np.arange(lower,upper+1)
        reference = np.interp(grid,times,curves[30])
        path_grid = np.linspace(max(p.min() for p in paths.values()),min(p.max() for p in paths.values()),200)
        path_reference = np.interp(path_grid,paths[30],curves[30])
        for speed in [15,45]:
            raw = np.interp(grid,times,curves[speed])
            scaled = np.interp(grid,times*speed/30,curves[speed])
            path_curve = np.interp(path_grid,paths[speed],curves[speed])
            rows.append(dict(model=label,speed=speed,
                common_equivalent_step_first=lower,common_equivalent_step_last=upper,
                raw_time_curve_rmse_pp=float(np.sqrt(np.mean((raw-reference)**2))*100),
                scaled_time_curve_rmse_pp=float(np.sqrt(np.mean((scaled-reference)**2))*100),
                path_curve_rmse_pp=float(np.sqrt(np.mean((path_curve-path_reference)**2))*100)))
    write_csv(root/'curve_alignment.csv', rows)
    (root/'curve_alignment.json').write_text(json.dumps(rows,indent=2)+'\n')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (root/'interpretation_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
    lines=['# 运动尺度诊断：结果解释','',
        '完成两个固定100k模型×三档速度×64场景×3动作种子，共1152个完整400步诊断回合。没有更新actor或critic；只改变环境最大速度。30 m/s下两模型全部三次重复逐元素复现既有报告。初始布局、完整UE运动路径、动作随机流及源权重哈希核对通过。','',
        '## 峰值随速度移动','',
        '|模型|最大速度(m/s)|21步平滑峰值步|峰值覆盖|全程覆盖|后100步覆盖|终态覆盖|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f'|{r["model"]}|{r["speed"]}|{r["peak_step"]}|{r["peak_coverage"]:.2%}|{r["coverage"]:.2%}|{r["last100_coverage"]:.2%}|{r["final_coverage"]:.2%}|')
    lines+=['','15 m/s相对30 m/s的后100步覆盖增益：']
    for r in result['paired_differences']:
        if r['speed']==15 and r['metric']=='last100_coverage':
            lines.append(f'- {r["model"]}：+{r["mean"]*100:.2f}个百分点，场景区间[{r["low"]*100:.2f}, {r["high"]*100:.2f}]。')
    lines+=['','## 整条曲线的运动尺度对齐','',
        '用21步平滑后的场景平均曲线，在三档速度的共同运动归一化支持区间内插值比较。横轴为步数×速度/30，共同区间17–195；同回合时刻的对照也使用17–195，避免外推。路程对齐另外使用三档速度共同的累计路程区间。下面的RMSE衡量两条平均覆盖曲线的差异，不是critic误差，也不是显著性检验。','',
        '|模型|速度对比|原时间对齐RMSE(pp)|运动时间对齐RMSE(pp)|实际路程对齐RMSE(pp)|',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["model"]}|{r["speed"]} vs 30|{r["raw_time_curve_rmse_pp"]:.2f}|{r["scaled_time_curve_rmse_pp"]:.2f}|{r["path_curve_rmse_pp"]:.2f}|')
    lines+=['','![覆盖与运动尺度](figures/coverage_motion_scale.png)','',
        '整体曲线对齐比单个argmax更有解释力：速度改变后，策略经过的平均覆盖上升—下降过程近似按运动尺度移动。尤其历史组15 m/s存在宽平台，11步窗口峰值为179，21/41步窗口为241/238；不能将241当成精确事件，但三种窗口下均晚于30/45 m/s。',
        '', '## 空间演化与结论边界','',
        '|模型|速度(m/s)|终态感知率|终态断联率|终态距边界150 m内占比|终态最近邻距离(m)|',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f'|{r["model"]}|{r["speed"]}|{r["final_sensing"]:.2%}|{r["final_disconnection"]:.2%}|{r["final_boundary150"]:.2%}|{r["final_nearest_m"]:.0f}|')
    lines+=['','![历史组空间演化](figures/M1-C-H_deployment_diagnostics.png)','',
        '速度更大时，感知下降、近邻距离收缩和靠边增加均更快；历史组联网率反而持续提高，说明其后期覆盖损失不能主要归结为断联。数据支持“持续运动使网络空间服务范围逐渐退化”，而非固定100–150步的统一触发事件。冻结执行中没有128步PPO更新，且本实验完全不读取critic。',
        '15 m/s仍在持续移动，覆盖达到平台后仍下降。原组峰值58.07%降至终态55.07%，历史组54.49%降至49.85%；这不是策略已经学会悬停。原组降速全程覆盖仅+0.53pp，场景区间跨零，不能把后期收益当成全程无代价提升。',
        '峰值时平均净位移约534–669 m，只是这些模型和场景下的量级线索，不是已证实的临界距离；轨迹并非严格按1/v缩放。改变速度会改变策略访问的状态及一步历史变化量，UE仍按原时间移动，因此这是冻结策略的动力学诊断，不能代表重新训练的速度方案。',
        '这一步定位了退化过程对运动尺度的依赖，尚未指出策略为何持续输出有害动作。下一步优先在同一较好部署状态下比较候选动作的配对未来回报，以及一步TD优势对这些动作的排序，检验“保持部署更好”是否被当前训练信号正确表达。','',
        '峰值区间、窗口敏感性及逐场景数据见[原始统计报告](report.md)。这64场景为研究诊断集，保留测试集未使用；区间为场景bootstrap，不是训练种子区间。']
    (root/'interpretation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='analysis/mappo_central_motion_scale_100k_seed0')
    main(parser.parse_args().output)
