"""Rebuild a DPPO training diagnosis from downloaded evaluation and shadow-audit data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from uav_research.plotting import style, save_figure, COLORS


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def main(run, output):
    run, output = Path(run), Path(output)
    comparison = json.loads((output/'comparison.json').read_text())
    models = comparison['models']
    train, validation = rows(run/'metrics.csv'), rows(run/'validation.csv')
    best_row = max(validation, key=lambda r: float(r['return']))
    best_step = int(best_row['env_steps'])
    low = min(validation, key=lambda r: float(r['coverage']))
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['env_steps'] == 1000000 and manifest['updates'] == 489
    assert manifest['optimizer_steps'] == 19560 and manifest['best_step'] == best_step
    reproducibility = {}
    for name, step in [('step0', 0), ('best', best_step), ('last', manifest['env_steps'])]:
        row = next(r for r in validation if int(r['env_steps']) == step)
        reproducibility[name] = {k: abs(float(row[k])-models[name][k]) for k in
                                ['return', 'coverage', 'last100_coverage', 'final_coverage']}
        assert max(reproducibility[name].values()) < 1e-6
    shadow = {}
    for name in ['best', 'last']:
        shadow[name] = json.loads((output/'score_audit'/name/'summary.json').read_text())
        assert abs(shadow[name]['metrics']['coverage']-models[name]['coverage']) < 1e-12
        assert abs(shadow[name]['metrics']['reward']*400-models[name]['return']) < 1e-8
    moments, series = {}, {}
    for name in ['step0', 'best', 'last']:
        with np.load(output/name/'action_moments.npz') as z:
            x = z['moments']
            moments[name] = dict(mean_command_norm=float(np.linalg.norm(x[:, :, :, 0], axis=-1).mean()),
                                 covariance_trace=float(x[:, :, :, 1].sum(-1).mean()))
        with np.load(output/name/'step_metrics.npz') as z:
            series[name] = dict(zip(z['names'].tolist(), z['metrics'].mean(0).T))
    state = torch.load(run/'step0.pt', map_location='cpu', weights_only=False, mmap=True)
    actor = state['learner']['actor']
    sensitivity = (actor['coef_x0']*actor['sqrt_one_minus_abar']/actor['sqrt_abar']/actor['transition_std']).numpy()
    train_metrics = {}
    for k in ['ppo_approx_kl', 'ppo_clip_fraction', 'actor_grad_norm', 'movement_head_grad_norm',
              'score_head_grad_norm', 'value_loss']+[f'denoise_{i}_{s}' for i in range(1,9) for s in ['kl','clip']]:
        x = np.array([float(r[k]) for r in train])
        train_metrics[k] = dict(first=float(x[0]), median=float(np.median(x)), last50=float(x[-50:].mean()))
    diag = output/'diagnosis'
    diag.mkdir(exist_ok=True)
    details = dict(training=manifest, validation_reproduction=reproducibility, moments=moments,
                   training_metrics=train_metrics, epsilon_mean_sensitivity_in_std_units=sensitivity.tolist(),
                   shadow_score_audit=shadow, note='Single training seed; paired validation-scene intervals, not seed confidence.')
    (diag/'details.json').write_text(json.dumps(details, indent=2, ensure_ascii=False)+'\n')
    style()
    x = np.array([int(r['env_steps']) for r in train])
    vx = np.array([int(r['env_steps']) for r in validation])
    fig, axes = plt.subplots(2, 2, figsize=(11,7))
    axes[0,0].plot(vx, [float(r['return']) for r in validation], color=COLORS[0])
    axes[0,0].axhline(models['step0']['return'], ls='--', color='gray', label='Pretrained step 0')
    axes[0,0].axvline(best_step, ls=':', color=COLORS[1], label=f'Best: {best_step:,}')
    axes[0,0].set(ylabel='Validation episode return'); axes[0,0].legend(fontsize=8)
    for key, label in [('coverage','Full episode'),('last100_coverage','Last 100 steps'),('final_coverage','Final step')]:
        axes[0,1].plot(vx, [100*float(r[key]) for r in validation], label=label)
    axes[0,1].set(ylabel='Validation coverage (%)'); axes[0,1].legend(fontsize=8)
    for key, label in [('ppo_approx_kl','Mean over 8 steps'),('denoise_8_kl','Highest-noise step 8'),('denoise_1_kl','Final denoising step 1')]:
        axes[1,0].semilogy(x, np.maximum([float(r[key]) for r in train], 1e-9),label=label)
    axes[1,0].set(ylabel='Sampled approximate KL (log scale)'); axes[1,0].legend(fontsize=8)
    for key, label in [('ppo_clip_fraction','Mean over 8 steps'),('denoise_8_clip','Step 8'),('denoise_1_clip','Step 1')]:
        axes[1,1].plot(x, [100*float(r[key]) for r in train],label=label)
    axes[1,1].set(ylabel='Probability-ratio clip fraction (%)'); axes[1,1].legend(fontsize=8)
    for ax in axes.flat: ax.set_xlabel('Joint transitions'); ax.ticklabel_format(axis='x',style='sci',scilimits=(6,6))
    save_figure(fig,diag,'learning_and_update')
    fig, axes = plt.subplots(2, 2, figsize=(11,7))
    for j,name in enumerate(['step0','best','last']):
        for ax,key in zip(axes.flat,['coverage','discovery_fraction','boundary_within_150_fraction','boundary_reflection_fraction']):
            ax.plot(np.arange(1,401),100*series[name][key],color=COLORS[j],label=name)
    for ax,title in zip(axes.flat,['Service coverage (%)','Instant sensing (%)','Within 150 m of boundary (%)','Reflected commands (%)']):
        ax.set(xlabel='Episode step',ylabel=title)
    axes[0,0].legend();save_figure(fig,diag,'deployment_maintenance')
    def pct(name,key): return f'{models[name][key]*100:.2f}%'
    s,b,l = (models[n] for n in ['step0','best','last'])
    lines = ['# 扩散DPPO 1M：验证与退化诊断', '',
        '**结论：在线早期有增益，但没有保持到1M。主要可见退化是感知与部署范围收缩、贴边及聚集增加；相同布局下的评分误差只解释极小的即时服务损失。**', '',
        f'运行完整结束：1,000,000条联合转移、489轮PPO、19,560次优化。best实际为{best_step:,}步。step0已经完成示范预训练，使用第{state["provenance"]["pretraining_epoch"]}轮预训练best，不是随机初始化网络。', '',
        '## 1. 验证口径与下载', '',
        '32个验证场景10000–10031，400步，批次16，动作采样种子880001，CUDA FP32；每20步独立采样64次估计动作矩。step0/best/last复测与训练验证日志四项关键指标误差均为0。没有使用保留测试集，没有训练或修改任何网络。', '',
        '完整运行已下载至 `runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/`，包括示范、预训练、TensorBoard、best/last/step0和完整恢复检查点。1626个文件、1,263,638,667字节，逐文件SHA256已验证。清单见 `1m_download_manifest.json` 和 `download_verified.json`。', '',
        '本次所有参考策略统一在GPU采样验证。旧baseline历史CPU验证的58.63125%与这里的GPU结果不是同一随机轨迹；相同种子不保证CPU/CUDA随机数相同。下表使用此次统一复测值。', '',
        '| 策略 | 全程覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 |', '|---|---:|---:|---:|---:|---:|']
    labels={'baseline_best':'原MAPPO best（671744步）','baseline_last':'原MAPPO last（1M）','teacher_topk':'冻结运动教师＋SINR评分＋Top-K',
            'step0':'扩散预训练后 step0','best':f'扩散在线best（{best_step}步）','last':'扩散在线last（1M）'}
    for n in ['baseline_best','baseline_last','teacher_topk','step0','best','last']:
        m=models[n]; lines.append(f'| {labels[n]} | {pct(n,"coverage")} | {pct(n,"last100_coverage")} | {pct(n,"final_coverage")} | {m["throughput_bps"]/1e6:.2f} | {m["return"]:.3f} |')
    lines += ['',f'best相对step0覆盖提高{100*(b["coverage"]-s["coverage"]):.3f}个百分点、回报提高{b["return"]-s["return"]:.3f}；last相对best覆盖下降{100*(b["coverage"]-l["coverage"]):.3f}个百分点，终态下降{100*(b["final_coverage"]-l["final_coverage"]):.3f}个百分点。不能说“从未学到任何东西”，但此次1M在线微调确实没有得到持续增益。', '',
        '场景配对bootstrap的last−step0全程覆盖差为−2.025个百分点，95%区间[−5.407,+1.072]，包含0；后100步差为−8.175个百分点，区间[−14.175,−2.436]，终态差为−12.125个百分点，区间[−20.188,−3.748]。后期维持退化的证据比全程均值下降更清楚。best在同一验证集挑选，不能把best的区间当成独立泛化证明；这些都不是训练种子置信区间。', '',
        '![训练过程与更新幅度](diagnosis/learning_and_update.png)', '',
        f'曲线也存在明显的中期下跌与后期恢复：验证覆盖率在{int(low["env_steps"]):,}步为{100*float(low["coverage"]):.2f}%，到1M恢复为{100*l["coverage"]:.2f}%。因此不能把1M称为已经稳定收敛。后续按用户决定从完整1M检查点续训到累计3M，保持学习设置不变；能否超过早期best需由新增数据检验。', '',
        '## 2. 具体退化了什么', '',
        '| 指标（全回合均值） | step0 | best | last |', '|---|---:|---:|---:|']
    for key,label,scale in [('boundary_within_150_fraction','距边界150m内比例',100),('boundary_reflection_fraction','反射比例',100),
        ('uav_close_neighbor_fraction','有近邻的UAV比例',100),('disconnection_ratio','断联UAV比例',100),('actual_speed_mean','实际速度m/s',1),('near_hover_fraction','位移小于1m的比例',100)]:
        lines.append('| '+label+' | '+' | '.join(f'{models[n][key]*scale:.3f}'+('%' if scale==100 else '') for n in ['step0','best','last'])+' |')
    lines += ['',
        'last并不是主要因为更多UAV断网：其平均断联比例反而更低。后100步感知率从best的85.93%降到last的69.36%，边界150m内比例从16.08%升至34.34%；说明可发现UE的空间范围本身在缩小。', '',
        '条件动作统计（各模型自身访问状态，每20步64次采样）：', '',
        '| 模型 | 条件平均指令位移模长m | 位移协方差迹m² |', '|---|---:|---:|']
    for n in moments: lines.append(f'| {n} | {moments[n]["mean_command_norm"]:.3f} | {moments[n]["covariance_trace"]:.3f} |')
    lines += ['',
        '因此“采样越来越随机”不符合这次观察。last的随机分量更小、方向偏置更强，却没有形成稳定保持动作。这里比较的是各自on-policy状态上的统计，不是同一组观察上的策略函数对照。', '',
        '![回合内部署维持](diagnosis/deployment_maintenance.png)', '',
        '固定场景10000：last在150步覆盖84.0%，400步65.3%；场景10001：150步76.0%，400步64.7%。这两场景高于整体终态41.27%，不能仅凭它们代表全部场景。完整32场景中last的终态覆盖中位数41.0%，7场低于30%，1场为0。', '',
        '[last场景10000拓扑](figures/last_trace_10000.png) · [last场景10001拓扑](figures/last_trace_10001.png) · [best场景10001拓扑](figures/best_trace_10001.png)', '',
        '## 3. 服务评分是不是主要原因', '',
        '补做了固定转移影子诊断：冻结best/last，在同一真实动作、位置、无线、回传与候选图上，把网络评分替换成动作前SINR教师评分；仍执行原Top-K、移动后过滤和UE选择。影子关联不写回环境，不改变之后的动作或历史。实际轨迹指标与主评估一致，源模型哈希保持不变。', '',
        '| 模型 | 学习评分覆盖 | SINR影子评分覆盖 | 覆盖增量（百分点） | 吞吐增量Mbps |', '|---|---:|---:|---:|---:|']
    for n in ['best','last']:
        a=shadow[n]; v=a['metrics']
        lines.append(f'| {n} | {100*v["coverage"]:.5f}% | {100*v["sinr_score_coverage"]:.5f}% | {100*a["paired"]["coverage"]["difference"]:.5f} | {a["paired"]["throughput_bps"]["difference"]/1e6:.4f} |')
    lines += ['',
        '这一恢复量远小于best到last约4.91个百分点的覆盖损失；评分没有饱和（极端sigmoid评分比例为0），step0/best/last评分latent标准差约0.76–0.77。预训练评分分支选模MSE约3.29e-5，也不支持“评分网络根本没学会教师规则”。', '',
        '这个实验只能说明：在已经访问的布局上，换回SINR教师评分几乎不能立即恢复服务；不能证明评分梯度对运动头没有影响，也不能证明闭环更换评分后会有同样结果，更不是全局最优关联上界。', '',
        '另一个独立限制来自协议：同一冻结高斯运动教师从旧集中式关联切换到Top-K闭环，覆盖由56.44%变为52.89%。这包含协议改变后的观察与动作反馈，不能当作同一布局上的纯Top-K损失。预训练扩散step0为52.00%，距离新协议教师不大；在线last的进一步退化是另一个问题。', '',
        '## 4. Loss很小，为什么策略还会变差', '',
        'PPO actor loss使用中心化团队优势，采样策略与更新策略接近时其均值本来就接近0；每轮又换了轨迹和优势，它不是应当持续下降的固定监督损失。critic loss拟合归一化GAE目标，也不等于原始尺度MC误差。', '',
        '训练日志最后一条覆盖61.97%，不能直接和完整验证49.97%比较：最后一轮仅36个向量步，对应各环境当前回合的第65–100步，偏向展开阶段。最后50条训练记录的覆盖均值约52.14%，也仍混合不同场景和回合片段，不能作为同模型完整回合泛化检查。', '',
        '| Critic（各自on-policy轨迹） | 累计回报RMSE | 除以标签std的RMSE | 按时刻去均值后的场景相关性 |', '|---|---:|---:|---:|']
    for n in ['best','last']:
        c=comparison['critic'][n]['cumulative'];lines.append(f'| {n} | {c["all"]["rmse"]:.3f} | {c["normalized_rmse"]:.3f} | {c["time_centered_scene_correlation"]:.3f} |')
    lines += ['',
        'last的平均价值精度略好，策略仍退化。这不支持再次把全部问题归为“critic完全不会预测”，也不能由整体MC误差排除critic/GAE：本报告没有检验同一起点不同动作或保持干预之间的小价值差。', '',
        '[Loss与策略诊断](figures/metrics.png) · [Critic与MC](figures/critic_mc.png) · [预训练MSE](figures/pretraining.png)', '',
        '## 5. 在线更新的具体风险证据', '',
        f'第一轮PPO平均近似KL={train_metrics["ppo_approx_kl"]["first"]:.3f}，最高噪声第8步KL={train_metrics["denoise_8_kl"]["first"]:.3f}，该步clip比例={100*train_metrics["denoise_8_clip"]["first"]:.1f}%。最后50轮总体clip均值约{100*train_metrics["ppo_clip_fraction"]["last50"]:.1f}%，但第8步仍约{100*train_metrics["denoise_8_clip"]["last50"]:.1f}%，第1步约{100*train_metrics["denoise_1_clip"]["last50"]:.1f}%；总体平均掩盖了去噪步骤的不均衡。', '',
        '代码中反向均值对噪声预测的导数为 `−coef_x0[k] * sqrt(1−abar[k]) / sqrt(abar[k])`。再除以该步固定反向std，得到以下单坐标敏感度：', '',
        '| 扩散步k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| 标准化均值敏感度 | '+' | '.join(f'{v:.2f}' for v in sensitivity)+' |', '',
        '第8步位于最大β=0.999处，噪声头变化会被明显放大。固定方差高斯的解析KL为均值变化平方和除以2σ²，因此相同噪声预测扰动在该步更容易引起大的策略分布变化。当前代码的经验KL包含采样噪声及所有有效评分坐标，不能把表中的导数直接当成实际KL；但它解释了为什么需要分步检查。PPO clip也不是硬KL约束，没有自动保证更新仍留在近邻策略。', '',
        f'actor梯度裁剪前范数中位数约{train_metrics["actor_grad_norm"]["median"]:.2f}，阈值为0.5；最后50轮评分头与运动头裁剪后梯度范数均值约{train_metrics["score_head_grad_norm"]["last50"]:.3f}/{train_metrics["movement_head_grad_norm"]["last50"]:.3f}。评分坐标较多、两头共享编码并一起裁剪，可能影响运动学习的有效步长，但这不是已经证实的因果结论。', '',
        '预训练后在线critic从头初始化，step0累计MC RMSE约75.74；actor同时开始更新。早期优势可能受尚未校准的critic影响。随后仍出现best增益，所以不能仅凭这个初始误差解释全部后期退化。在线BC权重为0，也没有约束策略维持预训练分布；这些都是后续需分别控制的候选因素。', '',
        '## 6. 建议下一步', '',
        '1. 保留本轮全部结果。用户已决定在独立目录从1M续训到累计3M，以检查后期回升能否持续；仅延长预算，不同时调整学习率、奖励或协议。进一步的机制诊断可审计逐minibatch、逐去噪步、按候选数分组的KL与更新后的回报方向。',
        '2. 优先做一个独立更新幅度对照，例如只降低actor学习率，或单独加入逐步KL早停；每次只改一项，critic、协议和奖励保持原配置。不要用裁剪log概率或坐标平均去伪装更小KL。',
        '3. 若要隔离联合评分训练的影响，固定SINR评分，只让扩散策略学习运动，和本轮联合策略比较。影子审计说明这项对照有价值，但目前并未执行正式训练。',
        '4. critic预热作为另一项独立对照：同一个预训练actor先冻结，用其新协议轨迹校准critic后再开始在线更新。不要与学习率、扩散日程同时修改。', '',
        '当前最有依据的结论是：**预训练与早期微调有效；后续在线优化未能保持部署质量，评分的即时选择错误不是主要解释。** 最高噪声步骤更新敏感、critic初期失配和联合梯度分配值得查，但尚不能认定其中任何一项是唯一根因。', '',
        '## 7. 成本与复现', '',
        f'在线耗时{manifest["elapsed_seconds"]/3600:.2f}小时，其中验证约{manifest["evaluation_seconds"]/3600:.2f}小时；预训练约{comparison["costs"]["pretraining"]["elapsed_seconds"]/60:.1f}分钟，示范采集约{comparison["costs"]["demonstrations"]["elapsed_seconds"]/60:.1f}分钟。', '',
        '报告、PDF/SVG/PNG图、逐步指标、MC标签、条件动作矩和固定拓扑均在本目录。`scripts/analyze_dppo_training.py` 只读取保存数据重建本诊断；`scripts/audit_dppo_service_scores.py` 是独立冻结策略影子审计入口，不修改生产环境或训练算法。', '',
        '```bash', '.venv-dppo-compat/bin/python scripts/analyze_dppo_training.py \\',
        '  --run runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0 \\',
        '  --output analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0', '```', '']
    (output/'diagnosis.md').write_text('\n'.join(lines))
    print(output/'diagnosis.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.run,a.output)
