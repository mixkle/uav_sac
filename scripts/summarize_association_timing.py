"""Write the interpretation from the accepted, saved T1--T3 measurements."""
import csv,json
from pathlib import Path
from uav_research.runtime import resolve_output


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    root=resolve_output('analysis/t1_t2_t3_post_motion_association')
    assert json.loads((root/'tests/acceptance.json').read_text())['passed']
    perf=read(root/'summary/performance.csv');diff=read(root/'summary/paired_differences.csv')
    immediate=read(root/'summary/immediate_paired_effects.csv');mechanism=read(root/'summary/mechanism_statistics.csv')
    provenance=json.loads((root/'provenance.json').read_text())
    source={k:next(x for x in provenance['evaluations'] if x['checkpoint']==k) for k in ['best','last']}
    def n(r,k,scale=1):return float(r[k])*scale
    def interval(r,scale=1):return f"{n(r,'difference',scale):+.4f} [{n(r,'low',scale):+.4f}, {n(r,'high',scale):+.4f}]"
    def delta(c,p,k):return next(r for r in diff if r['checkpoint']==c and r['comparison']==p and r['metric']==k)
    def now(c,p,k):return next(r for r in immediate if r['checkpoint']==c and r['comparison']==p and r['metric']==k and r['window']=='last100')
    def performance(c,p):return next(r for r in perf if r['checkpoint']==c and r['protocol']==p)
    def pair_values(function,p,k,scale=1):
        return '/'.join(f"{n(function(c,p,k),'difference',scale):.4f}" for c in ['best','last'])
    score_coverage=pair_values(delta,'T2 - T1','last100_coverage',100)
    score_rate=pair_values(delta,'T2 - T1','last100_throughput_mbps')
    timing_coverage=pair_values(delta,'T3 - T2','last100_coverage',100)
    timing_rate=pair_values(delta,'T3 - T2','last100_throughput_mbps')
    interrupted=delta('last','T3 - T2','last100_service_interruptions_window_sum')
    service_change='；'.join(f"{c}从{n(performance(c,'T1'),'last100_service_interruptions_window_sum'):.2f}降至{n(performance(c,'T2'),'last100_service_interruptions_window_sum'):.2f}次/100步窗口" for c in ['best','last'])
    lines=['# T1–T3 关联时序诊断：结果与解释','',
        '**主要发现：当前主要可见损失来自学习评分相对SINR规则的质量和连续性，运动前后时序失配只有有限的额外即时收益，尚未产生明确的后期闭环性能优势。**','',
        '本轮严格为冻结策略评估，没有训练、优化器或归一化更新，也没有重选best。',
        f"源A3 best实际为{source['best']['source_online_steps']:,}步，last为{source['last']['source_online_steps']:,}步；两者均为联合扩散策略。",'',
        '## 实验与验收','',
        '- T1：运动前学习评分，旧Top-K，移动后过滤。',
        '- T2：运动前SINR评分，同样旧Top-K和移动后过滤。',
        '- T3：运动后本地测量，在完整当前可服务集合上重新一轮Top-K及UE选择。没有补位循环、最大匹配或额外actor调用。',
        '- 两个checkpoint×三个协议×三个动作种子×32场景，共576个400步回合、230,400条闭环联合转移。',
        '- T1参考状态另计算五种独立关联，共76,800个相同物理后状态、384,000条影子结果。',
        '- T1逐步复现原A3，T2逐步复现此前冻结SINR替换：12组历史对照均通过绝对误差≤1e-6检查。',
        '- 所有协议的外生UE轨迹哈希一致，源模型哈希前后不变；同状态分解逐状态相加正确。',
        '- 本地55项相关测试通过，服务器8项时序测试及CUDA短验收通过。验收原始记录见tests/。','',
        '## 后100步主要结果','',
        '后100步为301–400。覆盖标准差先逐轨迹求时间总体标准差，再平均动作重复及场景。',
        '|权重|协议|覆盖%|吞吐Mbps|覆盖时间σ pp|覆盖P05%|终态覆盖%|终态吞吐Mbps|回合回报|联网UAV%|≥10pp骤降次数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in perf:
        lines.append(f"|{r['checkpoint']}|{r['protocol']}|{n(r,'last100_coverage',100):.4f}|{n(r,'last100_throughput_mbps'):.3f}|{n(r,'last100_coverage_std_pp'):.3f}|{n(r,'last100_coverage_p05',100):.3f}|{n(r,'final_coverage',100):.3f}|{n(r,'final_throughput_mbps'):.3f}|{n(r,'episode_return'):.3f}|{n(r,'last100_connected_fraction',100):.3f}|{n(r,'last100_drop_events_10pp'):.3f}|")
    lines+=['','## 闭环配对差值与不确定性','',
        '括号为95%场景配对bootstrap区间：每场景先平均3个动作种子，2000次重采样，种子950001。不是训练种子置信区间。',
        '|权重|比较|后100覆盖变化 pp|后100吞吐变化 Mbps|后100中断变化 次/回合窗口|',
        '|---|---|---:|---:|---:|']
    for c in ['best','last']:
        for comparison in ['T2 - T1','T3 - T2','T3 - T1']:
            rows={r['metric']:r for r in diff if r['checkpoint']==c and r['comparison']==comparison}
            lines.append(f"|{c}|{comparison}|{interval(rows['last100_coverage'],100)}|{interval(rows['last100_throughput_mbps'])}|{interval(rows['last100_service_interruptions_window_sum'])}|")
    lines+=['','## 相同物理布局的即时分解','',
        '以下只使用T1访问的后100步状态，上一服务归属、服务年龄和物理运动均相同，影子分支互不写回环境。',
        '|权重|影子差值|覆盖变化 pp|吞吐变化 Mbps|',
        '|---|---|---:|---:|']
    for c in ['best','last']:
        for comparison in ['T2 - T1','P1 - T2','P2 - P1','T3 - P2','T3 - T2']:
            rows={r['metric']:r for r in immediate if r['checkpoint']==c and r['window']=='last100' and r['comparison']==comparison}
            lines.append(f"|{c}|{comparison}|{interval(rows['coverage'],100)}|{interval(rows['throughput_bps'],1e-6)}|")
    lines+=['',
        'P1固定旧有效提议，仅用新SINR让UE选择UAV，因此服务UE并集不变、覆盖差严格为零是协议结构的必然结果；吞吐差才衡量这一步归属更新的效果。',
        'P2进一步在新旧候选交集重排Top-K；T3再纳入移动后新候选。各项按该顺序相加，不是独立无交互贡献。','',
        '## 候选变化','',
        '|权重|后100步每步指标|均值|95%场景区间|','|---|---|---:|---:|']
    for r in mechanism:
        if r['window']!='last100':continue
        lines.append(f"|{r['checkpoint']}|{r['metric']}|{float(r['mean']):.4f}|[{float(r['low']):.4f}, {float(r['high']):.4f}]|")
    lines+=['','## 服务连续性','',
        '|权重|协议|直接切换 次/步|中断 次/步|中断 次/后100步窗口|重接入 次/步|已结束服务段平均时长 步|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in perf:
        lines.append(f"|{r['checkpoint']}|{r['protocol']}|{n(r,'last100_direct_switches_per_step'):.3f}|{n(r,'last100_service_interruptions_per_step'):.3f}|{n(r,'last100_service_interruptions_window_sum'):.3f}|{n(r,'last100_service_reconnections_per_step'):.3f}|{n(r,'last100_completed_service_duration_mean'):.3f}|")
    lines+=['','服务段统计只包括窗口内结束的段；仍在服务的UE存在右删失，另存service_age_mean，不能把已结束段均值当成所有连接的无偏寿命。','',
        '## 结论与下一步','',
        f'T2相对T1：best/last的后100步覆盖分别增加{score_coverage}个百分点，吞吐增加{score_rate} Mbps；两组差值区间均为正。中断：{service_change}。替换评分规则的作用可以在同一冻结actor上观察到，不需要诉诸额外训练步数。','',
        f"T3相对T2：覆盖只增加{timing_coverage}个百分点，两个区间均跨0；吞吐变化为{timing_rate} Mbps，区间同样跨0。last中断反而增加{n(interrupted,'difference'):.2f}次/窗口，区间[{n(interrupted,'low'):.2f},{n(interrupted,'high'):.2f}]。因此当前冻结策略下，不能宣布运动后更新具有明确净优势。",'',
        f"即时分解中，best/last的T3−T2覆盖增加{pair_values(now,'T3 - T2','coverage',100)}个百分点；新增候选贡献{pair_values(now,'T3 - P2','coverage',100)}个百分点，约占99.6%。旧候选重新Top-K的覆盖恢复只有{pair_values(now,'P2 - P1','coverage',100)}个百分点。已有提议改用新SINR选择归属的吞吐增益仅{pair_values(now,'P1 - T2','throughput_bps',1e-6)} Mbps。**在本状态分布中，“旧分数过时”和“旧候选不重排”没有解释主要性能损失。**",'',
        '1. 评分来源与时序更新需要分开判断。T2−T1检验当前学习评分相对SINR规则的执行表现，T3−T2才检验后运动测量、候选更新和重新Top-K的整体效应。不能把两者收益都归为“解决评分过时”。',
        '2. 从逐状态分解判断旧分数更新、旧集合重排及新增候选分别有多大即时价值；从闭环区间判断这些收益是否在后续部署中保持。闭环关联会改变下一观察和运动，不能声称闭环三条轨迹运动相同。',
        '3. 当前数据支持优先研究学习评分的排序质量与时间稳定性，以及评分／运动联合采样和观察反馈的耦合；后运动更新不应在没有明显闭环增益的情况下被当成已经解决主要瓶颈。',
        '4. 本轮不训练新策略，因此不能由冻结策略对T3的适应表现直接断言“以T3重新训练也不会有效”；也不能把32个已用于选模的开发场景当成未接触的最终泛化测试。',
        '5. T3假设运动后可以即时进行本地测量与服务交换，没有计入额外信令和测量时延，部署结论受此限制。','',
        '## 文件与复现','',
        '- 主表：summary/performance.csv；闭环区间：summary/paired_differences.csv。',
        '- 即时配对效应：summary/immediate_paired_effects.csv；逐场景差值：one_step/scene_paired_effects.csv。',
        '- 每步/每场景：rollout/per_step.csv、rollout/per_episode.csv及各协议子目录。',
        '- 图：figures/中的覆盖吞吐曲线、分解图、即时/闭环对比、服务连续性和固定场景拓扑。',
        '- 完整物理快照、扩散链及闭环关联块保存在服务器同名结果目录的各case raw/与closed_loop/；本地同步指标、轨迹和图表。',
        '- 服务器目录：`/mnt/disk2/dym/uav_sac/analysis/t1_t2_t3_post_motion_association/`。',
        '- 重建：`.venv/bin/python scripts/diagnose_association_timing.py --rebuild`；解释报告：`.venv/bin/python scripts/summarize_association_timing.py`。']
    text='\n'.join(lines)+'\n'
    (root/'analysis.md').write_text(text)
    (root/'summary/conclusions.md').write_text(text)
    print(root/'analysis.md')


if __name__=='__main__':main()
