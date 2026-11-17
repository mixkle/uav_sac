"""Build Chinese A--F interpretation tables and overview plots from saved evaluation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from uav_research.system_ablation.report import GROUPS, write_csv
from uav_research.plotting import style, save_figure


DESCRIPTIONS = {
 'G0':'Gaussian联合动作＋指数缩放', 'A1':'随机扩散、无BC',
 'A3':'扩散BC＋在线微调', 'A4':'G0 best actor新阶段', 'A5':'同示范Gaussian BC＋微调',
 'B2':'线性跳数缩放', 'B3':'指令历史', 'B4':'不缩放', 'B5':'恒定0.5缩放',
 'C2':'前10 UE展平MLP', 'C3':'前10 UE集合编码', 'C4':'全集合统计编码',
 'D2':'仅运动＋移动后集中贪婪', 'D3':'联合动作＋SINR规则评分',
 'D4':'仅运动＋SINR规则评分', 'E2':'全局信息＋局部Top-K',
 'E3':'全局信息＋移动后集中贪婪', 'F1a':'扩散BC、不缩放', 'F2a':'扩散仅运动＋SINR评分'}
DIFFUSION = {'A1','A3','F1a','F2a'}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',default='analysis/diffusion_gaussian_ablation_1m')
 a=p.parse_args();root=Path(a.output)
 with (root/'metrics/unified_best_last.csv').open() as f:rows=list(csv.DictReader(f))
 lookup={(r['run_id'],r['checkpoint']):r for r in rows if r['split']=='validation'}
 sources=json.loads((root/'sources.json').read_text())
 def value(group,kind,key):
  v=lookup.get((group,kind),{}).get(key)
  return float(v) if v not in (None,'') else float('nan')
 def number(group,kind,key,scale=1,digits=2):
  x=value(group,kind,key)*scale
  return f'{x:.{digits}f}' if np.isfinite(x) else '—'
 ids=[g for g in DESCRIPTIONS if (g,'last') in lookup]
 clean_fields={
  'last100_coverage_pct':('last100_coverage',100),
  'last50_coverage_pct':('last50_coverage',100),'final_coverage_pct':('final_coverage',100),
  'last100_throughput_mbps':('last100_throughput_mbps',1),
  'last50_throughput_mbps':('last50_throughput_mbps',1),'final_throughput_mbps':('final_throughput_mbps',1),
  'last100_coverage_std_pp':('last100_coverage_std_pp',1),'last50_coverage_std_pp':('last50_coverage_std_pp',1),
  'episode_return':('episode_return',1),'last100_connected_pct':('last100_connected_fraction',100),
  'last50_connected_pct':('last50_connected_fraction',100),'final_connected_pct':('final_connected_fraction',100),
  'last100_drop_events_10pp':('last100_drop_events_10pp',1),'last50_drop_events_10pp':('last50_drop_events_10pp',1),
  'last100_max_drop_pp':('last100_max_drop_pp',1),'last50_max_drop_pp':('last50_max_drop_pp',1),
  'last100_coverage_p05_pct':('last100_coverage_p05',100),'last50_coverage_p05_pct':('last50_coverage_p05',100),
  'formation_step':('formation_step',1)}
 for kind in ('step0','best','last'):
  table=[]
  for g in ids:
   r=lookup.get((g,kind))
   if not r:continue
   line=dict(group=g,description=DESCRIPTIONS[g],checkpoint=kind,
     online_steps=int(r['online_steps']),scenes=int(r['scenes']),action_repetitions=int(r['action_repetitions']))
   for name,(key,scale) in clean_fields.items():
    v=value(g,kind,key)*scale;line[name]=float(v) if np.isfinite(v) else None
   table.append(line)
  write_csv(root/f'metrics/{kind}_late_metrics.csv',table)
 lines=['# 1M系统消融：统一验证与A–F对比分析','',
 '验证场景10000–10031，400步完整回合；动作采样种子880001、880002、880003。'+
 '模型冻结，未更新权重，保留测试集未使用。先在每场景内平均动作重复，再平均32场景。',
 '主要比较同预算last；best按训练时固定验证种子880001的后100步覆盖选取，吞吐仅用于数值平局。'+
 '本次三动作种子复评可能改变best/last的均值排序；不据此重新选检查点。',
 '标准差是在每条轨迹的后100/50步计算总体标准差，再平均，不是场景间标准差。'+
 '骤降阈值为单步10个百分点，计入窗口起点的进入转移。区间来自2000次场景级配对bootstrap，不能代替训练种子重复。','',
 '## 全部组：后期表现','',
 '|组|best步数|best后100覆盖%|last后100覆盖%|last后50覆盖%|last终态覆盖%|last后100吞吐Mbps|last终态吞吐Mbps|last后100σ pp|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for g in ids:
  lines.append('|'+ '|'.join([g,number(g,'best','online_steps',digits=0),
   number(g,'best','last100_coverage',100),number(g,'last','last100_coverage',100),
   number(g,'last','last50_coverage',100),number(g,'last','final_coverage',100),
   number(g,'last','last100_throughput_mbps'),number(g,'last','final_throughput_mbps'),
   number(g,'last','last100_coverage_std_pp')])+'|')
 lines+=['','## 各组相对初始化的在线增益','',
 '|组|step0后100覆盖%|best覆盖增益pp|last覆盖增益pp|step0后100吞吐Mbps|last吞吐增益Mbps|',
 '|---|---:|---:|---:|---:|---:|']
 gains=[]
 for g in ids:
  c0=value(g,'step0','last100_coverage');cl=value(g,'last','last100_coverage');cb=value(g,'best','last100_coverage')
  t0=value(g,'step0','last100_throughput_mbps');tl=value(g,'last','last100_throughput_mbps')
  gains.append(dict(group=g,step0_coverage=c0,last_coverage=cl,last_gain_pp=(cl-c0)*100,
                    best_gain_pp=(cb-c0)*100,last_throughput_gain_mbps=tl-t0))
  lines.append(f"|{g}|{c0*100:.2f}|{(cb-c0)*100:+.2f}|{(cl-c0)*100:+.2f}|{t0:.2f}|{tl-t0:+.2f}|")
 write_csv(root/'metrics/online_gains.csv',gains)
 lines+=['','## 部署保持与价值误差（last）','',
 '|组|51–150覆盖%|后100覆盖%|维持变化pp|后100 P05%|后100骤降次/回合|后100最大跌幅pp|后100联网%|critic全程RMSE|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for g in ids:
  lines.append('|'+ '|'.join([g,number(g,'last','early51_150_coverage',100),
   number(g,'last','last100_coverage',100),number(g,'last','maintain_coverage',100),
   number(g,'last','last100_coverage_p05',100),number(g,'last','last100_drop_events_10pp',digits=3),
   number(g,'last','last100_max_drop_pp'),number(g,'last','last100_connected_fraction',100),
   number(g,'last','critic_rmse_all')])+'|')
 names={'A_actor_type':'A：策略与初始化','B_motion_scaling':'B：运动缩放与历史',
  'C_ue_encoding':'C：UE编码','D_service_association':'D：服务评分与关联',
  'E_global_observation':'E：局部／全局信息','F_interactions':'F：跨策略交互'}
 with (root/'metrics/paired_differences.csv').open() as f:paired=list(csv.DictReader(f))
 contrasts={
  'A_actor_type':['A1 - G0','A3 - A1','A3 - G0','A3 - A5','A3 - A4','A4 - G0','A5 - G0'],
  'B_motion_scaling':['G0 - B2','G0 - B3','G0 - B4','G0 - B5'],
  'C_ue_encoding':['C3 - C2','G0 - C3','G0 - C2','G0 - C4'],
  'D_service_association':['G0 - D2','G0 - D3','D4 - D3','G0 - D4'],
  'E_global_observation':['E2 - G0','E3 - E2'],
  'F_interactions':['F1a - A3','F2a - A3','F1a - B4','F2a - D4','I_Hop','I_score']}
 for label,groups in GROUPS.items():
  lines+=['','## '+names[label],'', '; '.join(g+'：'+DESCRIPTIONS[g] for g in groups), '',
   '|last配对对照（左减右）|后100覆盖差pp［95%场景区间］|后100吞吐差Mbps［95%场景区间］|',
   '|---|---:|---:|']
  for comparison in contrasts[label]:
   cells=[]
   for metric,scale in [('last100_coverage',100),('last100_throughput_mbps',1)]:
    found=[r for r in paired if r['comparison']==comparison and r['checkpoint']=='last'
      and r['split']=='validation' and r['metric']==metric]
    cells.append('—' if not found else '{:+.2f} [{:+.2f}, {:+.2f}]'.format(*[
     float(found[0][k])*scale for k in ('difference','low','high')]))
   lines.append('|'+comparison+'|'+'|'.join(cells)+'|')
 lines+=['', '## 成本与结论边界','',
 '每个在线阶段均为1M条联合转移。A4是加载G0 best actor后的额外1M，'+
 'A3/A5/F1a/F2a另含教师训练、64k联合示范及50轮BC；初始化已学能力和在线新增能力必须分开。',
 'A3/A5使用同一份联合latent示范；F2a只监督运动，不能把评分机制与网络容量、概率维度完全分离。',
 'D3/E3保留评分随机变量但执行不使用这些评分。D4/F2a删除评分变量，也改变网络及梯度耦合。',
 'E2/E3是全局信息oracle。全局组较差不证明全局信息无用：输入维度、参数量和优化难度同时增加。',
 'B4/F1a关闭缩放同时提高运动权限，1M下收益不能等同于跳数缩放在所有规模或长期预算下无效。',
 'BC去噪MSE与Gaussian NLL不在同一尺度；不直接比较其绝对loss。MC误差仅反映各自on-policy状态，'+
 '不能单凭平均RMSE排除动作价值差估计问题。',
 '所有结果只有一个训练种子。动作重复只检查策略采样噪声；32个场景也已经用于训练期选模，'+
 '不作为未接触的最终测试证据。','',
 '完整逐场景、逐采样种子结果见`metrics/per_scene_action.csv`；全部指标及区间见'+
 '`metrics/unified_best_last.csv`；来源和检查点SHA256见`sources.json`。']
 (root/'comparison_analysis.md').write_text('\n'.join(lines)+'\n')
 # Training diagnostics retain their own units: Gaussian joint likelihood and
 # DDPM conditional transitions are not the same final-action entropy/KL.
 diagnostics=[]
 for g in ids:
  path=root/g/'training_data/metrics.csv'
  if not path.exists():continue
  with path.open() as f:log=list(csv.DictReader(f))
  row=dict(group=g,updates_averaged=min(50,len(log)))
  for key in ['actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction',
    'movement_conditional_kl','score_conditional_kl','movement_only_ratio_clip',
    'score_only_ratio_clip','movement_head_grad_norm','score_head_grad_norm','encoder_grad_norm',
    'score_saturation_fraction','movement_latent_norm_mean']:
   vals=[float(r[key]) for r in log[-50:] if r.get(key) not in (None,'')]
   if vals:row[key]=float(np.mean(vals))
  diagnostics.append(row)
 write_csv(root/'metrics/final50_updates_diagnostics.csv',diagnostics)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 style()
 palette=['#0072B2' if g in DIFFUSION else '#009E73' for g in ids]
 fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
 for ax,metric,scale,ylabel in zip(axes,
  ['last100_coverage','last100_throughput_mbps','last100_coverage_std_pp'],[100,1,1],
  ['Last-100 coverage (%)','Last-100 throughput (Mbps)','Mean temporal coverage SD (pp)']):
  vals=np.array([value(g,'last',metric)*scale for g in ids])
  low=np.array([value(g,'last',metric+'_ci_low')*scale for g in ids])
  high=np.array([value(g,'last',metric+'_ci_high')*scale for g in ids])
  ax.bar(np.arange(len(ids)),vals,color=palette,yerr=np.vstack([vals-low,high-vals]),capsize=2)
  ax.set(ylabel=ylabel);ax.grid(axis='y',alpha=.2)
 axes[0].legend(handles=[Patch(color='#009E73',label='Gaussian'),Patch(color='#0072B2',label='Diffusion')])
 axes[-1].set_xticks(np.arange(len(ids)),ids)
 save_figure(fig,root/'figures/overview','last_performance')
 fig,ax=plt.subplots(figsize=(9,7))
 for g,c in zip(ids,palette):
  x=value(g,'step0','last100_coverage')*100;y=value(g,'last','last100_coverage')*100
  ax.scatter(x,y,c=c,s=45);ax.annotate(g,(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
 ax.plot([0,100],[0,100],ls='--',c='#888888',lw=1)
 ax.set(xlabel='Online step 0: last-100 coverage (%)',ylabel='Online last: last-100 coverage (%)',
        xlim=(15,85),ylim=(15,85))
 save_figure(fig,root/'figures/overview','initialization_vs_online_gain')
 # Temporal curves are averaged over trajectories before plotting; shaded windows
 # identify fixed evaluation windows, never a selected per-model peak window.
 for label,groups in GROUPS.items():
  fig,axes=plt.subplots(1,2,figsize=(12,4))
  for i,g in enumerate(groups):
   for ax,kind in zip(axes,['best','last']):
    paths=sorted((root/g/'validation'/kind).glob('*/step_metrics.npz'))
    vals=[]
    for path in paths:
     with np.load(path) as z:
      vals.append(z['metrics'][:,:,z['names'].tolist().index('coverage')].mean(0)*100)
    if not vals:continue
    colors=['#009E73','#0072B2','#E69F00','#333333','#56B4E9','#D55E00']
    styles=['-','--','-.',':','-','--']
    ax.plot(np.arange(1,len(vals[0])+1),np.mean(vals,0),label=g,color=colors[i],linestyle=styles[i])
  for ax,kind in zip(axes,['best','last']):
   ax.axvspan(301,400,color='#999999',alpha=.09);ax.set(xlabel='Episode step',ylabel='Coverage (%)',title=kind)
   ax.legend(fontsize=8)
  save_figure(fig,root/'figures'/label,'best_and_last_coverage')
 print(root/'comparison_analysis.md')


if __name__=='__main__':main()
