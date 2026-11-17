"""Frozen checkpoints, per-scene late metrics, paired bootstrap and rebuild."""
import csv,json,copy
from pathlib import Path
import numpy as np
from ..runtime import atomic_json,seed_all
from ..topk_diagnostic import file_hash
from ..diffusion_dppo.storage import load,configure,Progress
from ..diffusion_dppo.config import from_dict
from ..diffusion_dppo.learner import DiffusionLearner
from ..diffusion_dppo.evaluation import evaluate

GROUPS={
 'A_actor_type':['G0','A1','A3','A4','A5'],
 'B_motion_scaling':['G0','B2','B3','B4','B5'],
 'C_ue_encoding':['G0','C2','C3','C4'],
 'D_service_association':['G0','D2','D3','D4'],
 'E_global_observation':['G0','E2','E3'],
 'F_interactions':['G0','B4','A3','F1a','D4','F2a']}


def episode_metrics(data,names,seeds):
 a=dict(zip(names,data.transpose(2,0,1)));c=a['coverage'];h=c.shape[1]
 rows=[]
 for i,scene in enumerate(seeds):
  x=c[i];rate=a['throughput_bps'][i]/1e6
  row=dict(scene_seed=int(scene),episode_return=float(a['reward'][i].sum()),
      final_coverage=float(x[-1]),final_throughput_mbps=float(rate[-1]),
      mean_coverage=float(x.mean()),mean_throughput_mbps=float(rate.mean()))
  row['formation_step']=int(np.flatnonzero(x>=x[-min(100,h):].mean()-1e-12)[0]+1)
  row['early51_150_coverage']=float(x[50:min(150,h)].mean()) if h>50 else None
  row['maintain_coverage']=float(x[-min(100,h):].mean()-x[50:150].mean()) if h>=150 else None
  row['peak_coverage_step']=int(np.argmax(x)+1)
  if 'disconnection_ratio' in a:row['final_connected_fraction']=float(1-a['disconnection_ratio'][i,-1])
  for n in (100,50):
   size=min(n,h);tail=x[-size:];prev=x[max(0,h-size-1):-1];cur=x[max(1,h-size):]
   drops=np.maximum(0,prev-cur)
   row.update({f'last{n}_coverage':float(tail.mean()),
    f'last{n}_coverage_std_pp':float(tail.std(ddof=0)*100),
    f'last{n}_coverage_p05':float(np.quantile(tail,.05)),
    f'last{n}_throughput_mbps':float(rate[-size:].mean()),
    f'last{n}_drop_events_10pp':int((drops>=.1-1e-12).sum()),
    f'last{n}_max_drop_pp':float(drops.max(initial=0)*100)})
   if 'disconnection_ratio' in a:row[f'last{n}_connected_fraction']=float((1-a['disconnection_ratio'][i,-size:]).mean())
  threshold=.9*x[-min(100,h):].mean()
  row['time_to_90pct_own_last100']=int(np.flatnonzero(x>=threshold-1e-12)[0]+1)
  public=np.convolve((x>=.6).astype(int),np.ones(10,dtype=int),'valid') if h>=10 else np.empty(0)
  hit=np.flatnonzero(public==10)
  row['time_to_60pct_10steps']=int(hit[0]+1) if len(hit) else None
  row['not_reached_60pct_10steps']=int(not len(hit))
  for key in ('proposal_acceptance','over_capacity_fraction','multi_candidate_fraction',
              'invalidated_proposals','direct_switches','service_interruptions',
              'boundary_within_150_fraction','boundary_reflection_fraction',
              'near_hover_fraction','speed_factor_mean','uav_speed_mean','actual_speed_mean',
              'uav_nearest_neighbor_mean','uav_close_neighbor_fraction','boundary_distance_mean',
              'discovery_fraction','proposal_count','duplicate_proposals','score_mean','score_std',
              'score_saturation_fraction','movement_latent_norm_mean','movement_latent_std',
              'score_latent_std','observed_count_mean','candidate_count_mean'):
   if key in a:row['last100_'+key]=float(a[key][i,-min(100,h):].mean())
  rows.append(row)
 return rows


def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def collect_run(run,output,device='cpu',split='validation',action_seeds=(880001,),kinds=('step0','best','last'),moments=0):
 run,output=Path(run),Path(output)
 sources=[]
 for kind in kinds:
  path=run/(kind+'.pt')
  if not path.exists():raise FileNotFoundError(path)
  digest=file_hash(path)
  data=load(path,kinds={'dppo_model'})
  cfg=from_dict(data['config']);cfg.run.device=device;cfg.diffusion.actor_update_devices=[]
  chosen=cfg.run.validation_seeds if split=='validation' else cfg.run.test_seeds
  configure(cfg);seed_all(cfg.run.seed)
  learner=DiffusionLearner(cfg,device);learner.load_state_dict(data['learner']);learner.actor.eval()
  for action_seed in action_seeds:
   target=output/split/kind/str(action_seed)
   provenance=dict(checkpoint=str(path),sha256=digest,kind=kind,
    env_steps=data['env_steps'],run_id=cfg.diffusion.ablation['run_id'],
    split=split,scene_seeds=chosen,action_seed=action_seed,
    initialization=data.get('provenance',{}),config=cfg.as_dict(),
    actor_parameters=sum(p.numel() for p in learner.actor.parameters()))
   if (target/'provenance.json').exists():
    saved=json.loads((target/'provenance.json').read_text())
    if any(saved[k]!=provenance[k] for k in ('sha256','scene_seeds','action_seed')):
     raise ValueError('Evaluation source changed; choose a new output directory')
    if (target/'complete.json').exists():
     sources.append(str(target));continue
   evaluate(cfg,learner.actor,chosen,target,critic=learner,moment_repeats=moments,
            action_seed=action_seed,traces=2)
   if file_hash(path)!=digest:raise ValueError('Model changed during frozen evaluation')
   atomic_json(target/'provenance.json',provenance)
   atomic_json(target/'complete.json',dict(status='complete'))
   sources.append(str(target))
 # Save numerical training logs for rebuild; avoid copying model weights.
 import shutil
 for name in ('metrics.csv','validation.csv','metadata.json','manifest.json','config.yaml'):
  if (run/name).exists():
   dest=output/'training_data';dest.mkdir(exist_ok=True);shutil.copy2(run/name,dest/name)
 return sources


def rebuild(root,comparisons_extra=(),groupings=None):
 root=Path(root);records=[];scene_rows=[];series={};provenance=[]
 for path in sorted(root.glob('*/validation/*/*/step_metrics.npz'))+sorted(root.glob('*/test/*/*/step_metrics.npz')):
  target=path.parent
  if not (target/'complete.json').exists():continue
  source=json.loads((target/'provenance.json').read_text());provenance.append(source)
  with np.load(path,allow_pickle=False) as z:
   per=episode_metrics(z['metrics'],z['names'].tolist(),z['scene_seeds'])
   series[(source['run_id'],source['kind'],source['split'],source['action_seed'])]=(z['metrics'].copy(),z['names'].tolist())
  critic=target/'critic_predictions.npz'
  if critic.exists():
   with np.load(critic,allow_pickle=False) as z:
    error=z['predicted']-z['mc_return']
    for i,row in enumerate(per):
     for label,sl in [('all',slice(None)),('last100',slice(-100,None)),('last20',slice(-20,None)),('last1',slice(-1,None))]:
      row['critic_rmse_'+label]=float(np.sqrt(np.mean(error[i,sl]**2)))
      row['critic_bias_'+label]=float(error[i,sl].mean())
  for row in per:scene_rows.append(dict(run_id=source['run_id'],checkpoint=source['kind'],split=source['split'],
    action_seed=source['action_seed'],online_steps=source['env_steps'],**row))
 # Average action repetitions WITHIN scenes, then bootstrap scenes.
 keys=sorted(set((r['run_id'],r['checkpoint'],r['split']) for r in scene_rows))
 per_scene={}
 for group,kind,split in keys:
  subset=[r for r in scene_rows if (r['run_id'],r['checkpoint'],r['split'])==(group,kind,split)]
  by_scene={s:[r for r in subset if r['scene_seed']==s] for s in sorted({r['scene_seed'] for r in subset})}
  metric_keys=[k for k in subset[0] if k not in ('run_id','checkpoint','split','action_seed','scene_seed','online_steps')]
  averaged={k:np.array([np.mean([r[k] for r in rows]) if all(r[k] is not None for r in rows) else np.nan
          for rows in by_scene.values()]) for k in metric_keys}
  per_scene[(group,kind,split)]=(list(by_scene),averaged)
  draws=np.random.default_rng(950001).integers(len(by_scene),size=(2000,len(by_scene)))
  summary=dict(run_id=group,checkpoint=kind,split=split,online_steps=subset[0]['online_steps'],scenes=len(by_scene),
       action_repetitions=len({r['action_seed'] for r in subset}))
  for k,v in averaged.items():
   summary[k]=float(v.mean()) if np.isfinite(v).all() else None
   if np.isfinite(v).all():
    lo,hi=np.quantile(v[draws].mean(1),[.025,.975]);summary[k+'_ci_low']=float(lo);summary[k+'_ci_high']=float(hi)
  records.append(summary)
 write_csv(root/'metrics/unified_best_last.csv',records);write_csv(root/'metrics/per_scene_action.csv',scene_rows)
 paired=[]
 comparisons=[('A1','G0'),('A3','A1'),('A3','G0'),('A3','A5'),('A3','A4'),('A4','G0'),('A5','G0'),
  ('G0','B2'),('G0','B3'),('G0','B4'),('G0','B5'),('C3','C2'),('G0','C3'),
  ('G0','C2'),('G0','C4'),('G0','D2'),('G0','D3'),('D4','D3'),('G0','D4'),
  ('E2','G0'),('E3','E2'),('F1a','A3'),('F2a','A3'),('F1a','B4'),('F2a','D4')]
 comparisons+=list(comparisons_extra)
 for split in ('validation','test'):
  for kind in ('step0','best','last'):
   for left,right in comparisons:
    a=per_scene.get((left,kind,split));b=per_scene.get((right,kind,split))
    if not a or not b:continue
    if a[0]!=b[0]:raise ValueError('Unpaired scenes')
    draws=np.random.default_rng(950001).integers(len(a[0]),size=(2000,len(a[0])))
    for metric in ('last100_coverage','last50_coverage','final_coverage','last100_throughput_mbps',
                   'last50_throughput_mbps','final_throughput_mbps','last100_coverage_std_pp',
                   'last50_coverage_std_pp','last100_coverage_p05','last100_drop_events_10pp',
                   'last100_max_drop_pp','episode_return','maintain_coverage'):
     x=a[1][metric]-b[1][metric];lo,hi=np.quantile(x[draws].mean(1),[.025,.975])
     paired.append(dict(comparison=left+' - '+right,checkpoint=kind,split=split,metric=metric,
       difference=float(x.mean()),low=float(lo),high=float(hi)))
   for label,terms in [('I_Hop',[('A3',1),('F1a',-1),('G0',-1),('B4',1)]),
                       ('I_score',[('A3',1),('F2a',-1),('G0',-1),('D4',1)])]:
    sets=[per_scene.get((g,kind,split)) for g,_ in terms]
    if any(s is None for s in sets):continue
    if any(s[0]!=sets[0][0] for s in sets):raise ValueError('Unpaired interaction scenes')
    draws=np.random.default_rng(950001).integers(len(sets[0][0]),size=(2000,len(sets[0][0])))
    for metric in ('last100_coverage','last100_throughput_mbps'):
     x=sum(s[1][metric]*term[1] for s,term in zip(sets,terms));lo,hi=np.quantile(x[draws].mean(1),[.025,.975])
     paired.append(dict(comparison=label,checkpoint=kind,split=split,metric=metric,
       difference=float(x.mean()),low=float(lo),high=float(hi)))
 write_csv(root/'metrics/paired_differences.csv',paired)
 atomic_json(root/'sources.json',provenance)
 render(root,records,series,groupings)
 lines=['# Diffusion / Gaussian 1M 系统消融','',
 '主要比较last；best按验证后100步覆盖率选取，覆盖数值相等时按后100步吞吐。G0在A–F中只训练一次。',
 '标准差逐轨迹计算，再在场景内平均动作重复，最后平均场景。区间为2000次场景bootstrap，不代表训练种子置信区间。',
 'formation_step为首次达到自身后100步平均覆盖；另报90%自身均值和60%连续10步。回报是400步未折扣奖励之和。',
 'A4只加载G0 best actor并重置critic/Adam；A3/A5额外拥有教师、示范和BC成本。',
 'D2/E3为动作后集中式贪婪参考，非最优上界；E组为全局信息oracle，非可部署局部执行。','',
 '| 组 | checkpoint | 集合 | 后100覆盖 | 后100σ pp | 终态覆盖 | 后100吞吐Mbps | 终态吞吐Mbps | 回报 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in records:
  lines.append(f"| {r['run_id']} | {r['checkpoint']} | {r['split']} | {r['last100_coverage']:.3%} | {r['last100_coverage_std_pp']:.3f} | {r['final_coverage']:.3%} | {r['last100_throughput_mbps']:.3f} | {r['final_throughput_mbps']:.3f} | {r['episode_return']:.3f} |")
 lines+=['','未出现的组是尚未训练或尚未评估，不填造指标。单种子1M用于定位瓶颈，不能作为性能上限证明。']
 (root/'report.md').write_text('\n'.join(lines)+'\n')
 return records


def render(root,records,series,groupings=None):
 import matplotlib.pyplot as plt
 from ..plotting import style,save_figure
 style()
 colors=['#009E73','#0072B2','#E69F00','#333333','#56B4E9','#D55E00']
 styles=['-','--','-.',':','-','--']
 for label,groups in (GROUPS if groupings is None else groupings).items():
  fig,axes=plt.subplots(1,3,figsize=(13,3.7))
  present=False
  for i,g in enumerate(groups):
   hits=[v for (name,k,s,_),v in series.items() if name==g and k=='last' and s=='validation']
   if not hits:continue
   present=True
   for ax,key,scale in zip(axes,('coverage','throughput_bps','reward'),(100,1e-6,1)):
    x=np.mean([v[0][:,:,v[1].index(key)].mean(0) for v in hits],0)
    ax.plot(np.arange(1,len(x)+1),x*scale,label=g,color=colors[i%len(colors)],linestyle=styles[i%len(styles)])
    ax.set(xlabel='Environment step',ylabel=key)
  if present:
   axes[0].legend();save_figure(fig,root/'figures'/label,'last_episode_curves')
  else:plt.close(fig)
  # Separate reward/coverage and policy/critic optimization curves.
  for log,metrics in [('validation.csv',('last100_coverage','last100_throughput_bps','return','coverage')),
                      ('metrics.csv',('actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction'))]:
   fig,axes=plt.subplots(2,2,figsize=(10,6));present=False
   for i,g in enumerate(groups):
    path=root/g/'training_data'/log
    if not path.exists():continue
    with path.open() as f:rows=list(csv.DictReader(f))
    if not rows:continue
    present=True
    for ax,key in zip(axes.flat,metrics):
     if key in rows[0]:ax.plot([float(r['env_steps']) for r in rows],[float(r[key]) for r in rows],label=g,
       color=colors[i%len(colors)],linestyle=styles[i%len(styles)])
     ax.set(xlabel='Online joint transitions',ylabel=key)
   if present:axes.flat[0].legend();save_figure(fig,root/'figures'/label,Path(log).stem)
   else:plt.close(fig)
 # Reuse immutable trace topology helper rather than resampling trajectories.
 # Topology is handled by a dedicated simple renderer for schema stability.
 for path in root.glob('*/validation/*/880001/trace_*.npz'):
  kind=path.parts[-3]
  if kind not in ('best','last'):continue
  with np.load(path,allow_pickle=False) as z:
   steps=[0,min(150,len(z['uav_xy'])-1),min(200,len(z['uav_xy'])-1),len(z['uav_xy'])-1]
   fig,axes=plt.subplots(1,4,figsize=(16,4))
   for ax,t in zip(axes,steps):
    u,m=z['uav_xy'][t],z['user_xy'][t]
    served=z['C'][t].any(0)
    ax.scatter(m[~served,0],m[~served,1],s=6,c='#aaaaaa',label='Unserved UE')
    ax.scatter(m[served,0],m[served,1],s=8,c='#009E73',label='Served UE')
    connected=z['B'][t].min(1)<=len(u)
    ax.scatter(u[connected,0],u[connected,1],s=26,c='#0072B2',marker='^',label='Connected UAV')
    ax.scatter(u[~connected,0],u[~connected,1],s=26,c='#333333',marker='x',label='Disconnected UAV')
    bs=z['bs_positions'];ax.scatter(bs[:,0],bs[:,1],s=65,c='#111111',marker='s',label='BS')
    for i in range(len(u)):ax.annotate(str(i),u[i],xytext=(2,2),textcoords='offset points',fontsize=5)
    for i,g in zip(*np.where(z['B'][t]==1)):
     ax.plot([u[i,0],bs[g,0]],[u[i,1],bs[g,1]],c='#0072B2',lw=.6,ls='--',alpha=.6)
    if 'A' in z:
     for i,j in zip(*np.where(np.triu(z['A'][t],1))):ax.plot(u[[i,j],0],u[[i,j],1],c='#777777',lw=.6)
    ax.set(xlim=(0,float(z['width'])),ylim=(0,float(z['height'])),
      title=f't={t}; coverage={served.mean():.1%}; online={connected.sum()}/{len(u)}',aspect='equal')
   axes[0].legend(fontsize=6,loc='lower right')
   save_figure(fig,root/'figures'/path.parts[-5],path.stem+'_'+kind)
 for path in root.glob('*/validation/*/880001/critic_predictions.npz'):
  with np.load(path,allow_pickle=False) as z:
   fig,axes=plt.subplots(1,2,figsize=(10,3.6))
   for ax,keys in zip(axes,[('predicted','mc_return'),('predicted_average','mc_average')]):
    for k,color,ls in zip(keys,['#009E73','#0072B2'],['-','--']):
     ax.plot(np.arange(1,z[k].shape[1]+1),z[k].mean(0),color=color,linestyle=ls,label=k)
    ax.set(xlabel='State before action t',ylabel='Cumulative value' if keys[0]=='predicted' else 'Average reward')
    ax.legend()
   save_figure(fig,root/'figures'/path.parts[-5],path.parts[-3]+'_critic_mc')
