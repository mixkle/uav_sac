"""Read-only, trajectory-first re-evaluation following the user's A-F history."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from report_all_tail100 import make_registry, metadata, sha, write_csv
from uav_research.post_deployment_metrics import (
    METRICS, METRIC_LABELS, load_trajectory, trajectory_metrics, summarize,
)

DEFAULT_OUT = Path('analysis/uav_evolution_af_reevaluation_20261110')
HISTORY = Path('UAV_实验演进_后100步重新评估.md')
OLD = Path('analysis/tail100_all_completed_results_20261110')
TITLES = {
    'A': '纯覆盖、容量20：共享与独立 Gaussian PPO',
    'B': '容量10：局部邻居数3–8，1M Best',
    'C': '容量10：断联惩罚权重，1M Last',
    'D': '冻结策略的运动执行干预，未重新训练',
    'E': '原10M Gaussian集中式关联与Diffusion Top-K，无Hop缩放',
    'E.1': '同一无缩放Diffusion续训链：1M→3M→5M→10M',
    'F': '新10M联合评分：Hop Scaling与真实位移历史',
}

PROTOCOL = r'''# 指标与数据口径

输入为已保存的完整400步、动作后状态序列。核心表按用户文档A–F及E.1逐项对应，共33行；E和E.1重复展示同一个DPPO 10M Best，不是新的独立实验。没有重新训练、重新选Best或修改历史结果。

- 后100步为301–400，后50步为351–400；终态为第400步。覆盖与联网比例用百分数，吞吐为系统总吞吐Mbps，不是单UE速率。
- 先在每条“场景×动作采样种子”轨迹上算所有指标，再在同一场景内平均动作重复，最后等权平均场景。A组沿用三个动作种子880001/880002/880003；其余核心组沿用原32验证场景与单动作种子。CPU/CUDA口径保留原样，不混合补评。
- **平均时间标准差**：每条轨迹窗口内`std(ddof=0)`，再算术平均。不会先平均曲线，不包括场景间均值差异，也不是之前报告的`sqrt(mean(variance))` RMS标准差。
- 回合回报：该记录实际环境奖励的400步**不折扣求和**，再对场景平均；不含策略熵。不同奖励定义的回报不能跨组直接排名。影子关联的回报是该影子方案的记录奖励，并非执行教师的真实回报。
- 联网比例：`1-disconnection_ratio`，记录的是能到达任一BS的UAV占比，不是用户覆盖率。没有网络字段时留空，不从服务人数推算。
- 骤降：`rho[t-1]-rho[t] >= 0.10`算一次，阈值容差1e-12；连续多个达标下降分别计数。报告是**每回合次数的场景平均**，不是全部场景事件总数。后100窗口包括300→301，后50包括350→351，分别100/50个转移。
- 最大单步跌幅：对上述转移计算`max(0,rho[t-1]-rho[t])`的最大值，再对场景平均，单位百分点；不是所有场景中的全局最大值。
- 后期5%分位数：先对每条轨迹的指定窗口用线性插值算P05，再对动作重复和场景平均。不是将所有场景时间点混在一起的P05，也不是覆盖均值的置信下界。
- 部署形成时间：每条轨迹首次达到**自身后100步平均覆盖**的动作后时刻，再对动作重复及场景平均。从t=1开始，因并非全部场景保存了t=0，不补造初始状态。初始就达标或后期均值为0时会得到1。**首次越线不意味着稳定部署**，不能单靠该指标比较不同覆盖水平的优劣。
- 缺失字段输出空CSV单元格或表中的“—”，并在missing_fields.csv说明。不能用代表拓扑、均值曲线或不完整分支推算全场景指标。
- 场景级bootstrap 2000次，种子950001，先平均场景内动作重复；区间保存在CSV，是场景区间，**不是训练种子置信区间**。

核心A组容量20、纯覆盖；B/C/D/E/F容量10，但训练预算、关联协议、奖励、历史与动作分布不同；B用Best而C用Last；D是冻结执行规则干预；E.1是同一续训链；F的Diffusion Step0已经过示范预训练，Gaussian Step0为随机初始化。不能拼成单一公平算法排行榜。

本轮保留原Best的全回合奖励选模定义。补充表包含此前保留的其他Best/Last、SAC、1.2M、数量泛化、教师、影子关联及定位实验；它们按A–F的研究阶段归类，但不冒充用户核心表中的实验。已经删除且没有完整数组的早期运行仅列缺失清单，不能恢复的数值留空。
'''


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def core_registry(entries):
    result = []
    def add(group, label, source_label, kind):
        match = [e for e in entries if e['label'] == source_label and e['kind'] == kind]
        if len(match) != 1:
            raise ValueError((source_label, kind, len(match)))
        result.append(dict(match[0], group=group, display_label=label, scope='core'))
    for kind in ['best', 'last']:
        for mode, zh in [('Shared', '共享'), ('Independent', '独立')]:
            add('A', f'{zh} Gaussian PPO {kind.upper()} / 1M',
                f'{mode} PPO / 1M（三动作种子）', kind)
    for n in range(3, 9):
        add('B', f'Gaussian N{n} / 1M BEST', f'Baseline N{n} / 1M', 'best')
    for mu, src in [('0', 'Baseline N3 / 1M'), ('0.1', '断联μ=0.1'),
                    ('0.5', '断联μ=0.5'), ('2', '断联μ=2'), ('5', '断联μ=5')]:
        add('C', f'Gaussian N3 μ={mu} / 1M LAST', src, 'last')
    for label in ['原始运动', '跳数速度缩放']:
        add('D', label+' / 冻结BEST', label, 'best')
    add('E', 'Gaussian 10M BEST / 集中式关联', 'Baseline N3 / 10M', 'best')
    add('E', 'Diffusion 10M BEST / Top-K', 'DPPO无缩放 / 10M', 'best')
    for m in [1, 3, 5, 10]:
        for kind in ['best', 'last']:
            add('E.1', f'Diffusion {m}M {kind.upper()} / 无缩放', f'DPPO无缩放 / {m}M', kind)
    for kind in ['step0', 'best', 'last']:
        for mode in ['Diffusion', 'Gaussian']:
            add('F', f'{mode} {kind.upper()} / Hop', f'{mode} Hop 10M', kind)
    assert len(result) == 33
    return result


def key(entry):
    return (tuple(entry.get('files') or [entry['path']]),
            tuple(sorted((entry.get('columns') or {}).items())))


def supplementary_group(entry):
    return {'legacy': 'A', 'baseline': 'B', 'generalization': 'B',
            'reward': 'C', 'motion': 'D', 'dppo': 'E.1', 'teacher': 'E',
            'other': 'B', 'shadow': 'E', 'diagnosis': 'F', 'hop': 'F'}[entry['section']]


def reason_for_missing(key):
    if 'throughput' in key:
        return '没有定义/记录此关联的吞吐序列；最大人数匹配不等于最优吞吐'
    if 'connected' in key:
        return '缺少逐步UAV断联比例，不能用用户覆盖率推算'
    if key == 'episode_return':
        return '缺少该评估方案的逐步实际奖励序列'
    return '缺少必要逐步数据'


def process(entry, ident, hashes):
    meta = metadata(entry)
    repetitions = []
    for path in entry.get('files') or [entry['path']]:
        path = str(path)
        hashes.setdefault(path, sha(path))
        columns = dict(entry.get('columns') or {})
        if columns.get('coverage_name') == 'sinr_score_coverage':
            columns['reward_name'] = 'sinr_score_reward'
        reps = load_trajectory(path, throughput_scale_bps=meta['throughput_scale_bps'], **columns)
        repetitions.append(reps)
    return aggregate(entry, ident, meta, repetitions)


def aggregate(entry, ident, meta, repetitions):
    result, intervals, values = summarize(repetitions)
    row = dict(evaluation_id=ident, group=entry['group'], scope=entry['scope'],
               label=entry.get('display_label', entry['label']), checkpoint=entry['kind'],
               source_label=entry['label'], source_path=entry['path'],
               **{k: v for k, v in meta.items() if k not in ['throughput_scale_bps', 'provenance']},
               **result, note=entry.get('note', ''))
    if entry.get('action_repeats'):
        row['action_seed'] = ';'.join(map(str, entry['action_repeats']))
    for k in METRICS:
        ci = intervals[k]
        row[k+'_ci_low'], row[k+'_ci_high'] = (None, None) if ci is None else ci
    scene_rows, repeat_rows, missing = [], [], []
    for i, scene in enumerate(repetitions[0]['scene_seed']):
        scene_rows.append(dict(evaluation_id=ident, group=entry['group'], scene_seed=int(scene),
                               **{k: None if values[k] is None else float(values[k][i]) for k in METRICS}))
    for repeat, rep in enumerate(repetitions):
        for i, scene in enumerate(rep['scene_seed']):
            repeat_rows.append(dict(evaluation_id=ident, repetition=repeat,
                                    scene_seed=int(scene),
                                    **{k: None if rep[k] is None else float(rep[k][i]) for k in METRICS}))
    for k in METRICS:
        if result[k] is None:
            missing.append(dict(evaluation_id=ident, group=entry['group'], label=row['label'],
                                metric=k, metric_name=METRIC_LABELS[k], reason=reason_for_missing(k)))
    detail = dict(entry=entry, meta=meta, intervals=intervals)
    return row, scene_rows, repeat_rows, missing, detail


def shadow_entries(hashes):
    """16 retained shadow comparisons, not extra closed-loop experiments."""
    for root in [Path('analysis/topk_capacity10_shared_best_diagnostic'),
                 Path('analysis/topk_capacity10_coverage_throughput_best_diagnostic')]:
        data, seeds, upper = [], [], []
        for p in sorted((root/'audit').glob('scene_*.npz')):
            hashes.setdefault(str(p), sha(p))
            with np.load(p) as z:
                data.append(z['metrics'])
                names = z['metric_names'].tolist()
                seeds.append(int(z['scene_seed']))
                upper.append((z['optima'] >= 0).sum(-1)/z['optima'].shape[-1])
                trajectory_capacity = int(z['trajectory_capacity'])
                capacity = int(z['audit_capacity'])
        a, seeds, upper = np.stack(data), np.array(seeds), np.stack(upper)
        def field(i, j, name):
            return a[:, :, i, j, names.index(name)]
        meta = dict(device='CPU shadow', action_seed=970001, checkpoint_step=None,
                    training_budget=None, num_users=150, num_uavs=18, capacity=capacity,
                    trajectory_capacity=trajectory_capacity)
        for score, indices in [('SINR教师', [0]), ('常数', [1]),
                                ('随机20重复', list(range(2, a.shape[2])))]:
            for j, scheme in enumerate(['P0', 'P1']):
                reps = [trajectory_metrics(field(i,j,'coverage'), seeds,
                            throughput=field(i,j,'throughput_bps'), rewards=field(i,j,'task_reward'),
                            disconnection=field(i,j,'disconnection')) for i in indices]
                label = f'{root.name} / {score}/{scheme}'
                entry = dict(group='E', scope='supplementary', kind='shadow', label=label,
                             path=str(root/'audit'), note='冻结轨迹影子关联，不改变后续状态；回报是影子task_reward')
                yield entry, meta, reps
        for j, scheme in enumerate(['O1 known maximum', 'O2 full maximum']):
            reps = [trajectory_metrics(upper[:,:,j], seeds,
                                      disconnection=field(0,0,'disconnection'))]
            entry = dict(group='E', scope='supplementary', kind='upper_bound',
                         label=f'{root.name} / {scheme}', path=str(root/'audit'),
                         note='只定义服务人数上界；联网比来自同一物理轨迹，不定义最优吞吐/回报')
            yield entry, meta, reps


def format_num(x):
    return '—' if x is None else f'{x:.3f}'


def core_observations(rows):
    r = {row['evaluation_id']: row for row in rows}
    def n(ident, field):
        return format_num(r[ident][field])
    return [
        '## 分组结论速览', '',
        f"- **A**：同预算Last，共享组后100步覆盖{n('A-03','tail100_coverage_pct')}%，"
        f"独立组{n('A-04','tail100_coverage_pct')}%；平均时间σ分别"
        f"{n('A-03','tail100_coverage_std_pp')}、{n('A-04','tail100_coverage_std_pp')}pp。"
        '在这套三动作种子、容量20口径下，共享组后期覆盖更高且波动更小。',
        f"- **B**：N8后100步覆盖最高，为{n('B-06','tail100_coverage_pct')}%；"
        f"N7的平均时间σ最小，为{n('B-05','tail100_coverage_std_pp')}pp。"
        '覆盖水平与波动程度不是同一排名，增加邻居数的收益也不是严格单调。',
        f"- **C**：μ=0.1相对μ=0，后100步覆盖从{n('C-01','tail100_coverage_pct')}%"
        f"升至{n('C-02','tail100_coverage_pct')}%，≥10pp骤降从"
        f"{n('C-01','tail100_drop_count')}降至{n('C-02','tail100_drop_count')}次/回合。"
        '更高权重进一步减少波动，却明显牺牲覆盖；低SD本身不等于好部署。',
        f"- **D**：冻结速度缩放后，后100步覆盖从{n('D-01','tail100_coverage_pct')}%"
        f"升至{n('D-02','tail100_coverage_pct')}%，平均σ从"
        f"{n('D-01','tail100_coverage_std_pp')}降至{n('D-02','tail100_coverage_std_pp')}pp。"
        '这是后期改善；全回合回报包含展开阶段，不一定同步上升。',
        f"- **E**：Gaussian的后100步覆盖{n('E-01','tail100_coverage_pct')}%高于"
        f"DPPO的{n('E-02','tail100_coverage_pct')}%，平均σ也更小；"
        f"DPPO后100步吞吐{n('E-02','tail100_throughput_mbps')}Mbps高于"
        f"Gaussian的{n('E-01','tail100_throughput_mbps')}Mbps，体现覆盖与吞吐的取舍。",
        f"- **E.1**：同一续训链Last的后100步覆盖从1M的{n('E.1-02','tail100_coverage_pct')}%"
        f"提高到10M的{n('E.1-08','tail100_coverage_pct')}%；"
        '这支持该旧版本随训练改善，不能据此认定所有扩散版本均优于Gaussian。',
        f"- **F**：新Hop联合评分Best中，Gaussian后100步覆盖"
        f"{n('F-04','tail100_coverage_pct')}%、平均σ{n('F-04','tail100_coverage_std_pp')}pp；"
        f"Diffusion为{n('F-03','tail100_coverage_pct')}%、{n('F-03','tail100_coverage_std_pp')}pp。"
        'Gaussian在后期覆盖、吞吐与稳定性上均更好；同预算Last的方向相同。',
        '',
        '**首次达标时间需结合覆盖水平解读**：例如强断联惩罚组只需达到较低的自身后期覆盖均值，'
        '因此看起来“形成更快”；Gaussian Hop Best的目标覆盖更高，首次达标更晚不能直接解释为部署能力更差。', '',
    ]


def table(rows, fields):
    lines = ['| 实验 | '+' | '.join(METRIC_LABELS[k] for k in fields)+' |',
             '|---|'+'---:|'*len(fields)]
    for r in rows:
        lines.append('| '+r['label']+' | '+' | '.join(format_num(r[k]) for k in fields)+' |')
    return lines


def publish(out, rows, supplementary=False):
    title = 'A–F补充：其他现存实验的19项指标' if supplementary else 'A–F实验演进：后期部署19项指标完整复评'
    lines = ['# '+title, '',
             '按[用户实验演进文档](../../UAV_实验演进_后100步重新评估.md)分组。', '',
             '[完整口径与计算公式](metric_definitions.md) · [核心CSV](core_summary.csv) · '
             '[补充CSV](supplementary_summary.csv) · [逐场景数据](per_scene.csv) · [缺失字段](missing_fields.csv)', '',
             '**标准差是各场景时间标准差的算术平均**。骤降阈值10个百分点，次数为次/回合；最大跌幅、P05也先逐场景计算再平均。'
             '形成时间仅为首次越过自身后100步均值，不是稳定维持时长。所有Best保留原选模结果。', '']
    if not supplementary:
        lines += [f'核心表共{len(rows)}条记录，严格对应A、B、C、D、E、E.1、F；全部19项均有逐步数据支持。'
                  'E与E.1中的10M Diffusion Best是同一份结果，重复仅为符合分组。其他已保留实验另见[分组补充报告](supplementary_report.md)。', '',
                  'A组三套动作种子先分别算指标再平均；其余核心组沿用单种子32场景口径。'
                  '不同奖励的回合回报不可直接横比；跨组算法差异还伴随容量、协议、预算和观察等变化。', '']
        lines += core_observations(rows)
    fields = [list(METRICS[:6]),
              ['tail100_coverage_std_pp','tail50_coverage_std_pp','tail100_drop_count','tail50_drop_count',
               'tail100_max_drop_pp','tail50_max_drop_pp','tail100_coverage_p05_pct','tail50_coverage_p05_pct'],
              ['episode_return','tail100_connected_pct','tail50_connected_pct','final_connected_pct','formation_step']]
    for group, title in TITLES.items():
        chosen = [r for r in rows if r['group'] == group]
        if not chosen:
            continue
        lines += [f'## {group}. {title}', '']
        for section, fs in zip(['覆盖与吞吐', '后期波动与骤降', '回报、联网与首次达标'], fields):
            lines += [f'### {section}', '']+table(chosen, fs)+['']
        if group == 'A':
            lines += ['A组容量20、纯覆盖；共享/独立1M按三个动作种子聚合。补充SAC或1.2M保持各自预算和CPU/CUDA口径。', '']
        elif group == 'C':
            lines += ['核心表统一用1M Last，不把Best与Last混比。回报包括各自训练时的奖励权重，较大的惩罚会改变回报尺度。', '']
        elif group == 'D':
            lines += ['这是同一冻结策略的执行干预，使用旧指令历史与集中式关联，不等于新Hop模型的重新训练收益。', '']
            if not supplementary:
                lines += ['缩放减原始的全部19项配对差值及场景区间见[D_paired_delta.json](D_paired_delta.json)。'
                          '差值的单位沿用原指标；覆盖增大为正，SD/跌幅/次数减小为负。', '']
        elif group == 'E':
            lines += ['Gaussian为集中式关联，DPPO为评分Top-K；算法与关联都不同。补充表中的影子P0/P1及最大匹配仅适用于冻结轨迹。', '']
        elif group == 'E.1':
            lines += ['1M、3M、5M、10M属于同一续训链，非四个独立训练种子；10M Best与E组重复引用。', '']
        elif group == 'F':
            lines += ['两组都用Hop缩放及联合评分；Diffusion Step0已预训练，Gaussian Step0是随机模型，不能把两者初始成本视为相同。', '']
    lines += ['## 来源与完整性', '',
              '| ID | 分组 | 实验 | 保存步数 | 预算 | 设备 | 场景×动作重复 |',
              '|---|---|---|---:|---:|---|---|']
    for r in rows:
        lines.append('| '+' | '.join(str(x) if x is not None else '—' for x in [
            r['evaluation_id'],r['group'],r['label'],r.get('checkpoint_step'),r.get('training_budget'),
            r.get('device'),f"{r['scene_count']}×{r['action_repetitions']}"])+' |')
    lines += ['', '输入数组、配置、源报告及SHA256见[manifest.json](manifest.json)和[provenance.json](provenance.json)。'
              '缺失历史见[unavailable_history.json](unavailable_history.json)；没有数据的已清理早期实验不编造结果。', '',
              '重建命令（只读已有数组，不采样、不训练）：', '', '```bash',
              '.venv-dppo-compat/bin/python scripts/report_uav_evolution_af.py', '```']
    (out/('supplementary_report.md' if supplementary else 'report.md')).write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    entries = make_registry()
    core = core_registry(entries)
    selected = {key(e) for e in core}
    supplement = [dict(e, group=supplementary_group(e), scope='supplementary')
                  for e in entries if key(e) not in selected]
    for entry in supplement:
        entry['display_label'] = entry['label']+' / '+entry['kind'].upper()
    rows, per_scene, per_repeat, missing, details, hashes = [], [], [], [], {}, {}
    def append(entry, result):
        row, scene, repeats, gaps, detail = result
        rows.append(row)
        per_scene.extend(scene)
        per_repeat.extend(repeats)
        missing.extend(gaps)
        details[row['evaluation_id']] = detail
    counters = {}
    for entry in core+supplement:
        prefix = entry['group'] if entry['scope'] == 'core' else 'S-'+entry['group']
        counters[prefix] = counters.get(prefix, 0)+1
        ident = f'{prefix}-{counters[prefix]:02d}'
        append(entry, process(entry, ident, hashes))
    for entry, meta, reps in shadow_entries(hashes):
        prefix = 'S-E'
        counters[prefix] = counters.get(prefix, 0)+1
        append(entry, aggregate(entry, f'{prefix}-{counters[prefix]:02d}', meta, reps))
    main_rows = [r for r in rows if r['scope'] == 'core']
    other_rows = [r for r in rows if r['scope'] != 'core']
    assert len(main_rows) == 33 and len(other_rows) == 78
    assert all(r[k] is not None for r in main_rows for k in METRICS)
    # Regression against every retained earlier result. Temporal SD now uses
    # its arithmetic-mean field, deliberately NOT the old primary RMS field.
    previous = list(csv.DictReader((OLD/'summary.csv').open()))
    checked = 0
    for row in rows:
        kind = row['checkpoint']
        matches = [x for x in previous if x['label'] == row['source_label'] and x['checkpoint'] == kind]
        if len(matches) != 1:
            raise ValueError(('Ambiguous historical reference', row['label'], len(matches)))
        ref = matches[0]
        for field in ['tail100_coverage_pct','final_coverage_pct','tail100_throughput_mbps','final_throughput_mbps']:
            if row[field] is not None:
                np.testing.assert_allclose(row[field], float(ref[field]), rtol=1e-9, atol=1e-7)
                checked += 1
        if ref.get('tail100_scene_std_pp'):
            np.testing.assert_allclose(row['tail100_coverage_std_pp'], float(ref['tail100_scene_std_pp']), atol=1e-9)
            checked += 1
    # User-document spot checks detect wrong checkpoint/repetition mapping.
    checks = {'A-01':69.54, 'A-02':66.14, 'B-01':58.107, 'C-01':59.933,
              'D-01':61.021, 'E-01':68.841, 'E.1-01':57.513, 'F-03':65.243, 'F-04':75.104}
    for ident, expected in checks.items():
        actual = next(r for r in main_rows if r['evaluation_id']==ident)['tail100_coverage_pct']
        assert abs(actual-expected)<.006, (ident,actual,expected)
    # Paired D contrast, with the same scene-level bootstrap as raw metrics.
    d = [r for r in main_rows if r['group']=='D']
    scene_lookup = {r['evaluation_id']: [x for x in per_scene if x['evaluation_id']==r['evaluation_id']] for r in d}
    left, right = (scene_lookup[r['evaluation_id']] for r in d)
    assert [x['scene_seed'] for x in left] == [x['scene_seed'] for x in right]
    difference = dict(scene_seed=np.array([x['scene_seed'] for x in left]),
                      **{k:np.array([b[k]-a[k] for a,b in zip(left,right)]) for k in METRICS})
    delta, intervals, _ = summarize([difference])
    dump(out/'D_paired_delta.json', dict(contrast='Hop-scaled minus original', values=delta, intervals=intervals))
    write_csv(out/'core_summary.csv', main_rows)
    write_csv(out/'supplementary_summary.csv', other_rows)
    write_csv(out/'all_summary.csv', rows)
    write_csv(out/'per_scene.csv', per_scene)
    write_csv(out/'per_trajectory.csv', per_repeat)
    write_csv(out/'missing_fields.csv', missing)
    zh_rows = [{ '分组':r['group'], '实验':r['label'], '范围':r['scope'],
                 **{METRIC_LABELS[k]:r[k] for k in METRICS}} for r in rows]
    write_csv(out/'全部指标_中文.csv', zh_rows)
    dump(out/'provenance.json', details)
    dump(out/'unavailable_history.json', json.loads((OLD/'deleted_historical_results.json').read_text()))
    (out/'metric_definitions.md').write_text(PROTOCOL)
    for entry in core+supplement:
        paths = [Path(entry['run'])/'config.yaml'] if entry.get('run') else []
        paths += [Path(entry['path']).parent/n for n in ['provenance.json','evaluation_config.json','evaluation_source.json']]
        for path in paths:
            if path.is_file():
                hashes.setdefault(str(path),sha(path))
    for path in [HISTORY, Path(__file__), Path('src/uav_research/post_deployment_metrics.py'),
                 Path('tests/test_post_deployment_metrics.py'), OLD/'summary.csv', OLD/'provenance.json']:
        hashes.setdefault(str(path),sha(path))
    assert all(sha(path)==digest for path,digest in hashes.items())
    manifest = dict(status='complete', core_rows=len(main_rows), supplemental_rows=len(other_rows),
                    metrics=len(METRICS), source_files=len(hashes), regression_checks=checked,
                    missing_fields=len(missing), missing_core_fields=0,
                    bootstrap=dict(repeats=2000,seed=950001,unit='scene'),
                    trained=False, resampled=False, historical_best_unchanged=True,
                    history_document=str(HISTORY), source_sha256=hashes,
                    elapsed_seconds=time.perf_counter()-started)
    dump(out/'manifest.json', manifest)
    publish(out, main_rows)
    publish(out, other_rows, supplementary=True)
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
