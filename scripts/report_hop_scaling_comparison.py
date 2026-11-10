"""Frozen Diffusion/Gaussian joint-service comparison; saved-data rebuild."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from uav_research.diffusion_dppo.report import evaluate_checkpoint
from uav_research.diffusion_dppo import report as rendering
from uav_research.coverage_stability import coverage_drop_metrics
from uav_research.runtime import atomic_json
from uav_research.topk_diagnostic import file_hash
from uav_research.plotting import style,save_figure

COLORS=['#009E73','#0072B2']


def read(path):
 with Path(path).open() as f:return list(csv.DictReader(f))


def enrich(directory):
 with np.load(directory/'step_metrics.npz') as z:
  arrays=dict(zip(z['names'].tolist(),z['metrics'].transpose(2,0,1)))
 rows=read(directory/'episodes.csv')
 for i,row in enumerate(rows):
  row.update(coverage_drop_metrics(arrays['coverage'][i]))
  row['coverage_q05']=float(np.quantile(arrays['coverage'][i],.05))
  connected=arrays['reachable_none'][i]
  row['network_loss_events']=int((np.diff(connected)>0).sum())
  lengths=[];start=None
  for t in range(1,len(connected)):
   if start is None and connected[t]>connected[t-1]:start=(t,connected[t-1])
   if start is not None and connected[t]<=start[1]:lengths.append(t-start[0]);start=None
  row['recovered_outages']=len(lengths)
  row['recovery_steps_mean']=float(np.mean(lengths)) if lengths else None
  row['unrecovered_outage']=int(start is not None)
 return rows


def rebuild(root):
 import matplotlib.pyplot as plt
 rendering.COLORS=['#009E73','#0072B2','#444444','#E69F00']
 outputs={};pairs=[]
 for kind in ('step0','best','last'):
  directory=root/kind
  rendering.rebuild(directory)
  # Explicit line styles for the two experimental algorithms.
  fig,axes=plt.subplots(1,3,figsize=(12,3.7))
  for i,g in enumerate(('diffusion','gaussian')):
   with np.load(directory/g/'step_metrics.npz') as z:
    data=dict(zip(z['names'].tolist(),z['metrics'].transpose(2,0,1)))
   for ax,key in zip(axes,('coverage','throughput_bps','reward')):
    ax.plot(np.arange(1,len(data[key][0])+1),data[key].mean(0),label=g,color=COLORS[i],linestyle='-' if i==0 else '--')
    ax.set(xlabel='Environment step',ylabel=key)
  axes[0].legend();save_figure(fig,directory/'figures','task_curves')
  fig,axes=plt.subplots(1,3,figsize=(12,3.7))
  for i,g in enumerate(('diffusion','gaussian')):
   path=directory/g/'action_moments.npz'
   if path.exists():
    with np.load(path) as z:
     for ax,key in zip(axes,('moments','effective_moments','actual_moments')):
      if key in z:
       ax.plot(z['times'],z[key][:,:,:,1].sum(-1).mean((0,2)),color=COLORS[i],linestyle='-' if i==0 else '--',label=g)
      ax.set(xlabel='Environment step',ylabel=key+' covariance trace (m²)')
  axes[0].legend();save_figure(fig,directory/'figures','motion_stage_variance')
  rows={g:enrich(directory/g) for g in ('diffusion','gaussian')}
  outputs[kind]={g:{k:float(np.mean([float(r[k]) for r in rs])) for k in
    ('coverage','last100_coverage','final_coverage','throughput_bps','return','max_drop_1','max_drop_5','drop_events_10pp','coverage_q05','maintain_coverage','network_loss_events')}
    for g,rs in rows.items()}
  atomic_json(directory/'extended_metrics.json',rows)
  if [r['scene_seed'] for r in rows['diffusion']] != [r['scene_seed'] for r in rows['gaussian']]:raise ValueError('Unpaired scenes')
  draws=np.random.default_rng(950001).integers(len(rows['diffusion']),size=(2000,len(rows['diffusion'])))
  for key in outputs[kind]['diffusion']:
   x=np.array([float(a[key])-float(b[key]) for a,b in zip(rows['diffusion'],rows['gaussian'])])
   low,high=np.quantile(x[draws].mean(1),[.025,.975]);pairs.append(dict(checkpoint=kind,metric=key,difference=float(x.mean()),low=float(low),high=float(high)))
 style()
 for filename,keys in [('validation.csv',['return','coverage','throughput_bps','last100_coverage']),
                       ('metrics.csv',['actor_loss','value_loss','ppo_approx_kl','ppo_clip_fraction'])]:
  fig,axes=plt.subplots(2,2,figsize=(10,6))
  for i,g in enumerate(('diffusion','gaussian')):
   rows=read(root/'training_data'/g/filename)
   for ax,key in zip(axes.flat,keys):
    if rows and key in rows[0]:ax.plot([float(r['env_steps']) for r in rows],[float(r[key]) for r in rows],label=g,color=COLORS[i],linestyle='-' if i==0 else '--')
    ax.set(xlabel='Joint transitions',ylabel=key)
  axes.flat[0].legend();save_figure(fig,root/'figures',Path(filename).stem)
 costs={}
 for g in ('diffusion','gaussian'):
  base=root/'training_data'/g;costs[g]={}
  for phase,file in (('online','manifest.json'),('pretraining','pretrain/summary.json'),('demonstrations','demonstrations/dataset.json'),('model','metadata.json')):
   path=base/file
   if path.exists():
    saved=json.loads(path.read_text());costs[g][phase]={k:saved[k] for k in ('elapsed_seconds','env_steps','epoch','batches','expected_joint_transitions','peak_rss_mib','peak_gpu_mib','actor_parameters','critic_parameters') if k in saved}
 atomic_json(root/'comparison.json',dict(metrics=outputs,paired_differences=pairs,costs=costs))
 lines=['# 跳数缩放：Diffusion DPPO 与 Gaussian MAPPO','',
 '两组均学习运动及UE评分，使用相同Top-K、速度缩放与真实位移历史。主要比较同预算last；扩散独有的示范和预训练成本须另计。',
 '场景级bootstrap不代表训练种子置信区间；MC诊断使用各策略自身on-policy轨迹。','',
 '| 模型 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回报 | 最大单步跌幅均值 |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for kind in ('last','best','step0'):
  for g,v in outputs[kind].items():lines.append(f"| {g} {kind} | {v['coverage']:.3%} | {v['last100_coverage']:.3%} | {v['final_coverage']:.3%} | {v['throughput_bps']/1e6:.3f} | {v['return']:.3f} | {100*v['max_drop_1']:.3f}pp |")
 lines+=['','运行成本与预训练成本：','', '```json',json.dumps(costs,ensure_ascii=False,indent=2),'```']
 (root/'report.md').write_text('\n'.join(lines)+'\n')
 return outputs


def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=2);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--moment-repeats',type=int,default=64);p.add_argument('--rebuild',action='store_true');a=p.parse_args()
 if not a.rebuild:
  if not a.runs:p.error('--runs requires diffusion then Gaussian run')
  a.output.mkdir(parents=True,exist_ok=True)
  from uav_research.config import load_config
  cfgs=[load_config(Path(r)/'config.yaml') for r in a.runs]
  if cfgs[0].env!=cfgs[1].env or cfgs[0].run.validation_seeds!=cfgs[1].run.validation_seeds:raise ValueError('Comparison environments/scenes differ')
  for g,run in zip(('diffusion','gaussian'),map(Path,a.runs)):
   target=a.output/'training_data'/g;target.mkdir(parents=True,exist_ok=True)
   for f in ('config.yaml','manifest.json','metrics.csv','validation.csv','metadata.json','pretrain/summary.json','pretrain/history.csv','demonstrations/dataset.json'):
    if (run/f).exists():
     (target/f).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/f,target/f)
  for kind in ('step0','best','last'):
   sources={}
   for g,run in zip(('diffusion','gaussian'),map(Path,a.runs)):
    cp=run/f'{kind}.pt';target=a.output/kind/g;cached=target/'provenance.json'
    old=json.loads(cached.read_text()) if cached.exists() else {}
    if old.get('sha256')!=file_hash(cp) or old.get('device')!=a.device or old.get('moment_repeats')!=a.moment_repeats or old.get('evaluation_data_version')!=2:
     evaluate_checkpoint(cp,target,a.device,moment_repeats=a.moment_repeats)
    sources[g]=json.loads((target/'provenance.json').read_text())
   atomic_json(a.output/kind/'report_sources.json',dict(labels=['diffusion','gaussian'],sources=sources))
  paths=[x for x in (a.output/'training_data').rglob('*') if x.is_file()]
  for kind in ('step0','best','last'):
   paths.append(a.output/kind/'report_sources.json')
   for g in ('diffusion','gaussian'):
    paths.extend(x for x in (a.output/kind/g).iterdir() if x.is_file())
  atomic_json(a.output/'source_hashes.json',{str(x.relative_to(a.output)):file_hash(x) for x in paths})
 else:
  for path,digest in json.loads((a.output/'source_hashes.json').read_text()).items():
   if file_hash(a.output/path)!=digest:raise ValueError(f'Saved report source changed: {path}')
 rebuild(a.output)

if __name__=='__main__':main()
