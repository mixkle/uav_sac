"""Rebuild the 3M continuation diagnosis from saved evaluations and CSV logs.

No training, trajectory sampling or checkpoint modification occurs here.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from uav_research.plotting import style, save_figure, COLORS
from uav_research.diffusion_dppo.report import error_metrics


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def rolling(values, n):
    return np.convolve(values, np.ones(n)/n, mode='valid')


def temporal_trend(rows, lower, key, seed=950003):
    selected = [r for r in rows if int(r['env_steps']) > lower]
    x = np.array([int(r['env_steps'])/1e6 for r in selected])
    y = np.array([float(r[key]) for r in selected])
    xc = x-x.mean()
    slope = float(xc@y/(xc@xc))
    fitted = y.mean()+slope*xc
    residual = y-fitted
    n, block = len(x), min(5, len(x))
    rng = np.random.default_rng(seed)
    starts = rng.integers(n-block+1, size=(2000, int(np.ceil(n/block))))
    indices = (starts[:, :, None]+np.arange(block)).reshape(2000, -1)[:, :n]
    slopes = (fitted[None, :]+residual[indices])@xc/(xc@xc)
    ci = np.quantile(slopes, [.025, .975])
    return dict(n=n, lower_step=lower, slope_per_million=slope,
        moving_block_residual_bootstrap_95=ci.tolist(), block_evaluations=block,
        first10_mean=float(y[:10].mean()), last10_mean=float(y[-10:].mean()),
        minimum=float(y.min()), maximum=float(y.max()), std=float(y.std()))


def main(run, output):
    run, output = Path(run), Path(output)
    saved = json.loads((output/'comparison.json').read_text())
    models = saved['models']
    train, valid = read(run/'metrics.csv'), read(run/'validation.csv')
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['env_steps'] == 3_000_000
    assert len(train) == manifest['updates'] == 1466
    assert manifest['optimizer_steps'] == 58640
    best_step = int(manifest['best_step'])
    reproduction = {}
    for label, step in [('best', best_step), ('last', 3_000_000), ('step0', 0)]:
        row = next(r for r in valid if int(r['env_steps']) == step)
        reproduction[label] = {k: abs(models[label][k]-float(row[k])) for k in
            ['return', 'coverage', 'last100_coverage', 'final_coverage', 'throughput_bps']}
        assert max(reproduction[label].values()) < 1e-5, reproduction[label]
    old_run = Path(json.loads((run/'continuation.json').read_text())['source_run']).name
    old_local = run.parent/old_run
    old_valid = read(old_local/'validation.csv')
    assert valid[:len(old_valid)] == old_valid, '1M validation history changed'
    metrics = ['return', 'coverage', 'last100_coverage', 'final_coverage',
               'maintain_coverage', 'throughput_bps', 'boundary_within_150_fraction',
               'boundary_reflection_fraction', 'disconnection_ratio']
    pairs = []
    rng = np.random.default_rng(950001)
    for left, right in [('last', 'last_1m'), ('best', 'best_1m'), ('last', 'step0'),
                        ('last', 'baseline_last'), ('best', 'baseline_best'), ('last', 'best')]:
        a, b = read(output/left/'episodes.csv'), read(output/right/'episodes.csv')
        assert [r['scene_seed'] for r in a] == [r['scene_seed'] for r in b]
        index = rng.integers(len(a), size=(2000, len(a)))
        for key in metrics:
            delta = np.array([float(x[key])-float(y[key]) for x, y in zip(a,b)])
            low, high = np.quantile(delta[index].mean(1), [.025,.975])
            pairs.append(dict(comparison=f'{left} - {right}', metric=key,
                difference=float(delta.mean()), low=float(low), high=float(high)))
    with (output/'continuation_paired_scenes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    windows = []
    for lo in range(0, 3_000_000, 500_000):
        rows = [r for r in valid if lo < int(r['env_steps']) <= lo+500_000]
        windows.append(dict(lo=lo, hi=lo+500_000, n=len(rows), **{
            key: float(np.mean([float(r[key]) for r in rows])) for key in
            ['return','coverage','last100_coverage','final_coverage']}))
    trends = {str(lo): {key: temporal_trend(valid, lo, key) for key in
              ['return','coverage','last100_coverage','final_coverage']}
              for lo in [2_000_000, 2_500_000, 2_800_000]}
    moments, critic, deployment = {}, {}, {}
    for name in ['step0','best_1m','last_1m','best','last','baseline_last']:
        with np.load(output/name/'step_metrics.npz') as z:
            deployment[name] = dict(zip(z['names'].tolist(), z['metrics'].mean(0).T))
        with np.load(output/name/'critic_predictions.npz') as z:
            critic[name] = dict(cumulative=error_metrics(z['predicted'],z['mc_return']),
                                average=error_metrics(z['predicted_average'],z['mc_average']))
        p = output/name/'action_moments.npz'
        if p.exists():
            with np.load(p) as z:
                moments[name] = dict(mean_displacement_norm=float(np.linalg.norm(z['moments'][:,:,:,0],axis=-1).mean()),
                    covariance_trace=float(z['moments'][:,:,:,1].sum(-1).mean()))
    tail = [r for r in train if int(r['env_steps'])>2_500_000]
    optimization = {key: dict(first=float(train[0][key]),
        last500k_mean=float(np.mean([float(r[key]) for r in tail]))) for key in
        ['actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction',
         'denoise_8_kl','denoise_8_clip','denoise_1_clip','actor_grad_norm',
         'movement_head_grad_norm','score_head_grad_norm']}
    series = deployment
    style()
    figdir = output/'figures'
    fig, axes = plt.subplots(3,2,figsize=(12,10))
    x=np.array([int(r['env_steps'])/1e6 for r in train])
    vx=np.array([int(r['env_steps'])/1e6 for r in valid])
    for ax, key, label in [(axes[0,0],'reward','Training rollout reward'),
                           (axes[0,1],'coverage','Training rollout coverage')]:
        y=np.array([float(r[key]) for r in train]); scale=100 if key=='coverage' else 1
        ax.plot(x,y*scale,alpha=.2,color=COLORS[0],lw=.7)
        ax.plot(x[49:],rolling(y,50)*scale,color=COLORS[0],label='50-update trailing mean')
        ax.set(ylabel=label+(' (%)' if key=='coverage' else ''));ax.legend(fontsize=7)
    for ax,key,label in [(axes[1,0],'return','Validation episode return'),
                         (axes[1,1],'coverage','Validation coverage (%)')]:
        y=np.array([float(r[key]) for r in valid]);scale=100 if key=='coverage' else 1
        ax.plot(vx,y*scale,alpha=.35,color=COLORS[0],lw=.8)
        ax.plot(vx[9:],rolling(y,10)*scale,color=COLORS[0],label='10-evaluation trailing mean')
        ax.axhline(models['step0'][key]*scale,ls='--',color='gray',label='Pretrained step 0')
        ax.axhline(models['baseline_last'][key]*scale,ls=':',color=COLORS[2],label='Gaussian MAPPO 1M last')
        ax.scatter([best_step/1e6],[models['best'][key]*scale],color=COLORS[1],s=30,zorder=4)
        ax.set(ylabel=label);ax.legend(fontsize=7)
    for key,label in [('coverage','Full episode'),('last100_coverage','Last 100'),('final_coverage','Final')]:
        axes[2,0].plot(vx[9:],rolling([100*float(r[key]) for r in valid],10),label=label)
    axes[2,0].set(ylabel='Validation coverage, trailing mean (%)');axes[2,0].legend(fontsize=7)
    for key,label in [('ppo_clip_fraction','All denoising steps'),('denoise_8_clip','Step 8'),('denoise_1_clip','Step 1')]:
        axes[2,1].plot(x[49:],rolling([100*float(r[key]) for r in train],50),label=label)
    axes[2,1].set(ylabel='PPO clipping, trailing mean (%)');axes[2,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.set(xlabel='Joint transitions (millions)');ax.axvline(1,color='gray',ls=':',lw=.8)
    save_figure(fig,figdir,'continuation_learning')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    keys=['coverage','discovery_fraction','boundary_within_150_fraction',
          'boundary_reflection_fraction','disconnection_ratio','actual_speed_mean']
    for ax,key in zip(axes.flat,keys):
        for j,name in enumerate(['last_1m','best','last','baseline_last']):
            ax.plot(np.arange(1,401),series[name][key]*(1 if key=='actual_speed_mean' else 100),
                    color=COLORS[j],label=name)
        ax.set(xlabel='Episode step',ylabel=key+(' (m/s)' if key=='actual_speed_mean' else ' (%)'))
    axes[0,0].legend(fontsize=8);save_figure(fig,figdir,'continuation_deployment')
    result=dict(manifest=manifest,validation_reproduction=reproduction,windows=windows,trends=trends,
        paired=pairs,moments=moments,critic=critic,optimization=optimization,
        trend_note='Descriptive OLS slope; residual moving-block bootstrap, 5 evaluations/block, 2000 draws. '
                   'Conditional on this single correlated training path; not a forecast or seed confidence interval.')
    (output/'continuation_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    def pair(a,b,key):
        return next(r for r in pairs if r['comparison']==f'{a} - {b}' and r['metric']==key)
    def ci(a,b,key,scale=100):
        r=pair(a,b,key)
        return f'{r["difference"]*scale:+.3f}，95%场景区间[{r["low"]*scale:+.3f}, {r["high"]*scale:+.3f}]'
    names={'step0':'预训练后step0','best_1m':'1M运行best（40,960）','last_1m':'1M last',
           'best':f'3M运行best（{best_step:,}）','last':'3M last',
           'baseline_best':'原Gaussian MAPPO best（671,744）','baseline_last':'原Gaussian MAPPO 1M last'}
    lines=['# 扩散DPPO 3M续训：验证与收敛分析','',
        '**结论：从1M继续训练有效，last的全程和后期覆盖均改善；当前仍有上升迹象，但尚未稳定。新best只是小幅超过早期best，不能把“best出现在2.79M”直接解释为还会持续大幅提升。**','',
        f'训练完整结束于3,000,000条联合转移，累计{manifest["updates"]:,}轮PPO、{manifest["optimizer_steps"]:,}次优化；best实际为{best_step:,}步。从原1M完整状态恢复，未重置actor、critic、Adam或归一化，未重新示范/预训练。物理、奖励和学习设置保持原实验；本次实际采用单GPU、微批16384。','',
        '## 1. 下载与验证口径','',
        '完整运行保存于 `runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0/`：15个文件、168,163,701字节，含完整恢复检查点、best/last/step0、CSV和TensorBoard。逐文件SHA256已验证，原1M结果保留。best/last重新验证的数据也逐文件校验。','',
        '32个固定验证场景10000–10031，400步，批次16，动作采样种子880001，CUDA FP32；每20步独立采样64次统计动作矩。新best/last与训练验证的覆盖、后100步、终态、吞吐及回报一致。旧1M和baseline复用此前同口径GPU验证，来源哈希保留；没有重训或使用保留测试集。','',
        '## 2. 增加训练是否有效','',
        '| 模型 | 全程覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回合回报 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['step0','best_1m','last_1m','best','last','baseline_last']:
        m=models[name]
        lines.append(f'| {names[name]} | {100*m["coverage"]:.3f}% | {100*m["last100_coverage"]:.3f}% | {100*m["final_coverage"]:.3f}% | {m["throughput_bps"]/1e6:.2f} | {m["return"]:.3f} |')
    lines += ['',
        f'3M last相对1M last：全程覆盖差为{ci("last","last_1m","coverage")}个百分点；后100步为{ci("last","last_1m","last100_coverage")}个百分点；终态为{ci("last","last_1m","final_coverage")}个百分点。此次续训带来的恢复有配对场景支持，不能继续说在线训练完全没有作用。','',
        f'但3M best相对原早期best的全程覆盖差仅{ci("best","best_1m","coverage")}个百分点；后100步反而由{100*models["best_1m"]["last100_coverage"]:.2f}%降至{100*models["best"]["last100_coverage"]:.2f}%。best按覆盖＋数据率的全回合回报选取，不按后期覆盖选取。换言之，主要证据是从中期退化恢复，尚不是明确突破早期性能上限。','',
        '区间为2000次配对验证场景bootstrap，不是训练种子区间。best在同一验证集反复选取，其区间未校正选模偏差，不能当成独立泛化证明。各方法采用各自on-policy轨迹；3M与baseline 1M预算及关联协议不同，表格用于定位，不是严格同预算算法优劣证明。','',
        '## 3. 是否还没收敛','',
        '![完整训练曲线](figures/continuation_learning.png)','',
        '以下为各500k阶段内验证记录的平均值，不是每个阶段末尾单个模型的指标：','',
        '| 在线区间 | 验证次数 | 平均回报 | 全程覆盖 | 后100步覆盖 | 终态覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    for w in windows:
        lines.append(f'| {w["lo"]/1e6:g}–{w["hi"]/1e6:g}M | {w["n"]} | {w["return"]:.3f} | {100*w["coverage"]:.3f}% | {100*w["last100_coverage"]:.3f}% | {100*w["final_coverage"]:.3f}% |')
    lines += ['',
        '最近窗口仍向上：2.5M之后最初10次验证的平均覆盖为'+f'{100*trends["2500000"]["coverage"]["first10_mean"]:.2f}%，最后10次为{100*trends["2500000"]["coverage"]["last10_mean"]:.2f}%。但这段起点靠近一次低谷，窗口选择影响很大。','',
        '| 趋势窗口 | 全程覆盖斜率（百分点/M） | 描述性95%时间块区间 |',
        '|---|---:|---:|']
    for lo in ['2000000','2500000','2800000']:
        tr=trends[lo]['coverage'];low,high=tr['moving_block_residual_bootstrap_95']
        lines.append(f'| {int(lo)/1e6:g}–3M | {100*tr["slope_per_million"]:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] |')
    lines += ['',
        '斜率由当前单条训练轨迹的验证记录作OLS拟合；去趋势残差按连续5次验证成块重抽样2000次。其区间依赖局部线性和残差块近似假设，只是描述相关时间序列，不是训练种子置信区间，更不是未来收益预测。最后500k呈上升，但最后1M的趋势区间包含0；因此合理表述是“仍在波动恢复，尚未稳定”，不能外推再训1M一定提高若干百分点。','',
        '训练rollout覆盖在1.5M后主要在相近水平摆动；固定完整回合验证的改善也不是持续单调上升。actor loss接近0并不意味着性能收敛：PPO使用中心化优势和每轮变化的目标；小的归一化critic loss也不等于部署维持已经正确。','',
        '## 4. 展开和贴边是否改善','',
        '![回合内部署](figures/continuation_deployment.png)','',
        '| 指标 | 1M last | 3M best | 3M last | 原MAPPO 1M last |',
        '|---|---:|---:|---:|---:|']
    for key,label,scale in [('discovery_fraction','全程瞬时感知',100),
            ('boundary_within_150_fraction','全程距边界150m内',100),
            ('last100_boundary_within_150_fraction','后100步距边界150m内',100),
            ('boundary_reflection_fraction','全程反射',100),
            ('disconnection_ratio','全程断联UAV',100),
            ('uav_close_neighbor_fraction','全程近邻聚集',100),
            ('near_hover_fraction','位移小于1m',100),
            ('actual_speed_mean','平均实际速度m/s',1),
            ('maintain_coverage','后100步−第51至150步覆盖（百分点）',100)]:
        lines.append('| '+label+' | '+' | '.join(f'{models[n][key]*scale:.3f}'+('%' if scale==100 and key!='maintain_coverage' else '') for n in ['last_1m','best','last','baseline_last'])+' |')
    lines += ['',
        '3M确实改善了感知、后期服务和部分聚集，但仍大量持续移动、向边界反射。部署维持差值虽比1M改善，仍为负，且该差值改善的场景区间包含0，不能宣布保持问题已解决。原高斯baseline的边界与反射比例明显更低，后期覆盖也更高。','',
        f'32场景平均覆盖曲线中，3M last约在第{np.argmax(series["last"]["coverage"])+1}步达到{100*series["last"]["coverage"].max():.2f}%，终态为{100*models["last"]["final_coverage"]:.2f}%，仍有展开后回落。1M last的均值曲线峰值则在第{np.argmax(series["last_1m"]["coverage"])+1}步，峰值时刻并不是所有模型都固定在同一步；不能仅凭它认定某种时钟机制是根因。','',
        '固定场景10000的3M last在150步为75.33%，200步为84.00%，400步仍84.00%，说明某些布局中已经能展开并维持较高覆盖；场景10001对应75.33%、65.33%、65.33%，仍存在回落。两者终态均高于32场景均值55.50%，不能只看这两张图判断整体。','',
        '![last场景10000](figures/last_trace_10000.png)','',
        '[last场景10001](figures/last_trace_10001.png) · [best场景10000](figures/best_trace_10000.png) · [best场景10001](figures/best_trace_10001.png)','',
        '条件动作方差按各模型自身访问状态计算：1M last位移协方差迹为'+f'{moments["last_1m"]["covariance_trace"]:.2f}m²，3M best为{moments["best"]["covariance_trace"]:.2f}m²，3M last为{moments["last"]["covariance_trace"]:.2f}m²；3M last条件平均指令位移模长约{moments["last"]["mean_displacement_norm"]:.2f}m。随机分量没有明显扩大，也没有学成大面积悬停；这些不是同一固定观察上的策略分布比较。','',
        '## 5. Critic和策略更新','',
        '| 模型 | 累计MC RMSE | RMSE/标签std | 按时刻去均值后的场景相关性 |',
        '|---|---:|---:|---:|']
    for name in ['best_1m','last_1m','best','last']:
        c=critic[name]['cumulative'];lines.append(f'| {names[name]} | {c["all"]["rmse"]:.3f} | {c["normalized_rmse"]:.3f} | {c["time_centered_scene_correlation"]:.3f} |')
    lines += ['',
        'best的价值预测优于1M，但3M last并没有随覆盖提高而同步降低全程MC误差。各模型轨迹与回报分布不同，这不是同一状态上的网络精度对照；也没有检查同一部署下移动/保持之间的反事实价值差。不能据此说critic已完全无问题，或把全部改善归因于critic。','',
        f'最后500k总体PPO clip均值为{100*optimization["ppo_clip_fraction"]["last500k_mean"]:.2f}%，最高噪声第8步仍为{100*optimization["denoise_8_clip"]["last500k_mean"]:.2f}%，第1步为{100*optimization["denoise_1_clip"]["last500k_mean"]:.2f}%；actor裁剪前梯度范数均值{optimization["actor_grad_norm"]["last500k_mean"]:.2f}，阈值0.5。网络仍在更新，更新尺度的不均衡仍存在，不能把后期曲线形状只归结为参数太多或训练次数不足。','',
        '[Loss曲线](figures/metrics.png) · [逐去噪步KL/clip](figures/per_denoising_step.png) · [Critic与MC](figures/critic_mc.png)','',
        '## 6. 下一步建议','',
        '可以继续做一个预先限定预算的同设置续训段，例如再增加1M，并持续观察完整验证回报、后100步覆盖和维持差值；这应当作为“额外预算能否持续带来收益”的实验，而不是预设只要训练更久就能解决贴边。当前没有自动追加训练，也没有修改正式训练配置。','',
        '重点看多次验证的滚动表现能否持续超越原早期best、后期覆盖是否同步改善、反射是否下降。若只是反复回到约52%–55%的区域而维持差值仍为负，优先回到更新稳定性和保持动作的价值差诊断；不要只以某次新best出现的步数决定无限延长。若之后改变学习率或KL约束，应另开分支并保留当前3M作为对照。','',
        '本轮没有重做新模型的评分影子审计，不能把续训增益单独归因于运动、评分或Top-K协议。一个训练种子的结果用于定位问题，尚不能支持算法普遍优于baseline。','',
        '## 7. 成本与重建','',
        f'累计在线耗时{manifest["elapsed_seconds"]/3600:.2f}小时，其中新增2M约{(manifest["elapsed_seconds"]-json.loads((run/"source_manifest.json").read_text())["elapsed_seconds"])/3600:.2f}小时；累计验证约{manifest["evaluation_seconds"]/3600:.2f}小时。示范和预训练成本另计。','',
        'JSON、逐场景CSV、逐步物理指标、MC标签、条件动作矩、固定拓扑及PDF/SVG/PNG均已保存。`3m_download_manifest.json`、`evaluation_download_manifest.json`和校验记录保存下载依据；`report_sources.json`保存新旧评估来源。','',
        '```bash','.venv-dppo-compat/bin/python scripts/analyze_dppo_continuation.py \\',
        '  --run runs/diffusion_dppo_capacity10_coverage_throughput_3m_seed0 \\',
        '  --output analysis/diffusion_dppo_capacity10_coverage_throughput_3m_seed0','```','']
    (output/'diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(best_step=best_step,windows=windows,trends=trends),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(args.run,args.output)
