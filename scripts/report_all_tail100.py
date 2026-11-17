"""Recompute five late-deployment metrics for every retained complete evaluation.

Reads historical arrays only. Never changes models, historical reports or best
selection. Saved figures and per-scene tables are reproducible without rollout.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import yaml
from uav_research.tail_metrics import read_series,per_scene,summarize,METRICS
from uav_research.runtime import atomic_json
from uav_research.plotting import style,save_figure

ROOT=Path('analysis');OUT=ROOT/'tail100_all_completed_results_20261110'
FIELDS=list(METRICS)
SECTIONS={
 'hop':'跳数缩放：Diffusion与Gaussian联合运动/评分，10M',
 'baseline':'容量10 Gaussian MAPPO原baseline：邻居3–8与10M',
 'reward':'容量10 baseline奖励消融：断联与鲁棒性',
 'dppo':'原无跳数缩放DPPO：1M/3M/5M/10M',
 'legacy':'容量20历史基准：共享/独立PPO与SAC，纯覆盖',
 'teacher':'冻结baseline教师＋Top-K关联',
 'motion':'冻结baseline：速度缩放与局部链路安全干预',
 'generalization':'邻居3/4/5的UE/UAV数量泛化（64诊断场景）',
 'diagnosis':'最新定位实验：噪声干预与200k微调',
 'other':'其他完整评估与初始化参考',
 'shadow':'固定布局影子关联：不属于闭环策略性能'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def make_registry():
 rows=[]
 def add(section,label,path,run=None,kind='best',budget=None,step=None,note='',files=None,action_repeats=None,env_override=None,columns=None):
  rows.append(dict(section=section,label=label,path=str(path),run=run,kind=kind,budget=budget,step=step,note=note,files=files,action_repeats=action_repeats,env_override=env_override,columns=columns))
 base='analysis/hop_scaling_joint_service_10m_seed0'
 for g in ['diffusion','gaussian']:
  for ck in ['best','last','step0']:
   add('hop',f'{g.capitalize()} Hop 10M',f'{base}/{ck}/{g}/step_metrics.npz',kind=ck,budget=10000000)
 base='analysis/mappo_neighbors_1m_vs_10m_seed0'
 for n in range(3,9):
  run=f'runs/mappo_capacity10_neighbors{n}_1m_seed0' if n>3 else 'runs/mappo_coverage_throughput_capacity10_1m_seed0'
  for ck in ['best','last']:
   if ck=='best':path=f'analysis/mappo_neighbors3to8_1m_vs_dppo10m_seed0/best/neighbors{n}/step_metrics.npz'
   elif n<=5:path=f'{base}/last/neighbors{n}/step_metrics.npz'
   else:path=f'{OUT}/supplemental_evaluations/{Path(run).name}/last/step_metrics.npz'
   add('baseline',f'Baseline N{n} / 1M',path,run,ck,1000000)
 run='runs/mappo_coverage_throughput_capacity10_10m_seed0'
 add('baseline','Baseline N3 / 10M','analysis/baseline_dppo_10m_best_teacher_seed0/baseline_best/step_metrics.npz',run,budget=10000000)
 add('baseline','Baseline N3 / 10M',f'{OUT}/supplemental_evaluations/{Path(run).name}/last/step_metrics.npz',run,'last',10000000)
 for key,title in [('disconnection','断联μ=0.1'),('robustness','鲁棒λ=0.5'),('combined','断联μ=0.1＋鲁棒λ=0.5')]:
  for ck in ['best','last']:
   add('reward',title,f'analysis/mappo_stability_six_group_1m_vs_10m_seed0/{ck}/{key}/reward_curves.npz',f'runs/mappo_capacity10_stability_{key}_1m_seed0',ck,1000000)
 for key,title in [('mu05','断联μ=0.5'),('mu2','断联μ=2'),('mu5','断联μ=5')]:
  for ck in ['best','last']:
   add('reward',title,f'analysis/mappo_disconnection_weight_sweep_1m_seed0/{ck}/disconnection_{key}/reward_curves.npz',f'runs/mappo_capacity10_stability_disconnection_{key}_1m_seed0',ck,1000000)
 for m in [1,3,5,10]:
  run=f'runs/diffusion_dppo_capacity10_coverage_throughput_{m}m_seed0'
  for ck in ['best','last']:
   add('dppo',f'DPPO无缩放 / {m}M',f'analysis/{Path(run).name}/{ck}/step_metrics.npz',run,ck,m*1000000)
 # Original actor-sharing reports retain independent 1M even though its run is now 1.2M.
 for report in ['mappo_actor_sharing_coverage_1m_seed0','mappo_actor_sharing_independent_1p2m_vs_shared_1m_seed0']:
  d=read(ROOT/report/'comparison.json')
  for mode,g in d['groups'].items():
   if report.endswith('shared_1m_seed0') and mode=='shared':continue
   if report.startswith('mappo_actor_sharing_independent') and mode!='independent':continue
   if report=='mappo_actor_sharing_coverage_1m_seed0':budget=1000000
   else:budget=1200000
   for ck in ['initial','best','last']:
    v=g['checkpoints'][ck];files=v['files']
    add('legacy',f'{mode.capitalize()} PPO / {budget/1e6:g}M',files[0],g['run'],ck,budget,v['step'],note='CPU，主表动作种子880001',env_override=g['config']['env'])
    if len(files)>1:add('legacy',f'{mode.capitalize()} PPO / {budget/1e6:g}M（三动作种子）',files[0],g['run'],ck,budget,v['step'],files=files,action_repeats=[880001,880002,880003],note='先每条轨迹计算方差，再平均重复；不平均轨迹后再求方差',env_override=g['config']['env'])
 run='runs/sac_coverage_time_factor_relative_central_history_server_1m_seed0'
 for ck in ['best','last']:add('legacy','SAC / 1M',f'{OUT}/supplemental_evaluations/{Path(run).name}/{ck}/step_metrics.npz',run,ck,1000000,note='补评估CUDA，与历史纯覆盖PPO CPU结果分口径展示')
 for ck in ['best','last']:
  cpu_files=[str(OUT/'supplemental_evaluations'/'sac_coverage_time_factor_relative_central_history_server_1m_seed0'/(ck+'_cpu_seed'+str(seed))/'step_metrics.npz') for seed in [880001,880002,880003]]
  add('legacy','SAC / 1M CPU',cpu_files[0],run,ck,1000000,note='与历史PPO CPU相同动作种子880001')
  add('legacy','SAC / 1M CPU（三动作种子）',cpu_files[0],run,ck,1000000,files=cpu_files,action_repeats=[880001,880002,880003],note='与历史共享/独立PPO相同CPU、三套动作种子；先逐轨迹计算后平均')
 for m in [1,10]:
  path='analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/teacher_topk/step_metrics.npz' if m==1 else 'analysis/baseline_dppo_10m_best_teacher_seed0/teacher_topk/step_metrics.npz'
  add('teacher',f'Baseline {m}M教师＋Top-K',path,f'runs/mappo_coverage_throughput_capacity10_{m}m_seed0',budget=m*1000000)
 for key,label,path in [('original','原始运动','analysis/baseline3_best_hop_speed_025_k04/original/step_metrics.npz'),('hop_scaled','跳数速度缩放','analysis/baseline3_best_hop_speed_025_k04/hop_scaled/step_metrics.npz'),('link_safe','缩放＋5m局部链路约束','analysis/baseline3_best_hop_speed_link_safety_delta5/link_safe/step_metrics.npz')]:
  add('motion',label,path,'runs/mappo_coverage_throughput_capacity10_1m_seed0',budget=1000000)
 for n in [3,4,5]:
  run=f'runs/mappo_capacity10_neighbors{n}_1m_seed0' if n>3 else 'runs/mappo_coverage_throughput_capacity10_1m_seed0'
  for key,label,u,m in [('original','原场景',18,150),('ue60','UE60%',18,90),('ue80','UE80%',18,120),('uav14','14 UAV',14,150),('uav10','10 UAV',10,150)]:
   add('generalization',f'N{n} / {label}',f'analysis/mappo_neighbors_population_generalization_1m_seed0/neighbors{n}/{key}/step_metrics.npz',run,budget=1000000,env_override={'num_uavs':u,'num_users':m},note='冻结actor；64个660000起诊断场景，不能与32个验证场景直接排序')
 for ck in ['best','last']:
  add('other','Baseline N3 / 1M（CPU复现）',f'analysis/mappo_coverage_throughput_capacity10_1m_seed0/critic_{ck}/audit.npz','runs/mappo_coverage_throughput_capacity10_1m_seed0',ck,1000000,note='同一模型CPU采样验证，与CUDA原口径分开展示')
  add('shadow',f'DPPO 1M {ck}：事后SINR评分',f'analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/score_audit/{ck}/step_metrics.npz','runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0',kind='shadow',budget=1000000,step=40960 if ck=='best' else 1000000,note='在同一冻结物理轨迹上改变评分，非闭环重采样',columns={'coverage_name':'sinr_score_coverage','throughput_name':'sinr_score_throughput_bps'})
 add('dppo','DPPO无缩放：预训练后step0','analysis/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/step0/step_metrics.npz','runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0',kind='step0',budget=1000000,step=0)
 for kind in ['best','last']:
  for g in ['diffusion','gaussian']:
   add('diagnosis',f'{g} {kind}，噪声×0.8',f'analysis/diffusion_gaussian_diagnosis_20261110/exp2_action_distribution/noise_{kind}_{g}/step_metrics.npz',kind=kind,budget=10000000)
 for g in ['control','conservative']:
  for ck in ['best','last']:
   add('diagnosis',f'Diffusion best微调 / {g}',f'analysis/diffusion_gaussian_diagnosis_20261110/exp4_conservative_finetuning/evaluation/{g}/{ck}/step_metrics.npz',kind=ck,budget=200000,note='从7,501,824步源best微调；步数为本次额外转移，Adam重置')
 return rows


def local_path(path):
 p=Path(path)
 if p.exists():return p
 s=str(path)
 for prefix in ['runs/','analysis/']:
  i=s.find('/'+prefix)
  if i>=0 and Path(s[i+1:]).exists():return Path(s[i+1:])
 return None


def metadata(entry):
 path=Path(entry['path']);meta={};env={};device='unknown';action=880001;batch=None;step=entry.get('step');budget=entry.get('budget')
 if entry.get('run') and (Path(entry['run'])/'config.yaml').exists():
  cfg=yaml.safe_load((Path(entry['run'])/'config.yaml').read_text());env.update(cfg['env']);device=cfg['run'].get('device','unknown');action=cfg['run'].get('evaluation_action_seed',880001);batch=cfg['run'].get('num_envs')
  if step is None:
   name='best_selection.json' if entry['kind']=='best' else 'manifest.json'
   if (Path(entry['run'])/name).exists():step=read(Path(entry['run'])/name).get('env_steps')
 for name in ['evaluation_config.json','provenance.json','evaluation_source.json']:
  p=path.parent/name
  if not p.exists():continue
  d=read(p);meta[name]=d
  cfg=d.get('training_config')
  if name=='evaluation_config.json':cfg=d
  if cfg:
   env.update(cfg['env']);device=cfg['run'].get('device',device);action=cfg['run'].get('evaluation_action_seed',action);batch=cfg['run'].get('num_envs',batch)
  if name=='provenance.json':
   device=d.get('device',device);action=d.get('action_seed',action);batch=d.get('batch_size',batch);step=d.get('env_steps',d.get('checkpoint_step',step));budget=d.get('training_budget',budget)
   source=local_path(d.get('checkpoint',''))
   if source and (source.parent/'config.yaml').exists() and not env:env.update(yaml.safe_load((source.parent/'config.yaml').read_text())['env'])
  elif name=='evaluation_source.json':step=d.get('checkpoint_env_steps',step)
 if entry.get('env_override'):env.update(entry['env_override'])
 # Audit files have no provenance.json but are CPU-only by construction.
 if path.name=='audit.npz':device='cpu'
 if 'snapshot' not in str(path) and 'hop_scaling_joint_service_10m_seed0' in str(path) and not env:
  group=path.parts[-2];cfg=yaml.safe_load((ROOT/'hop_scaling_joint_service_10m_seed0/snapshots'/group/'config.yaml').read_text());env=cfg['env']
 if not env and 'diffusion_gaussian_diagnosis' in str(path):env=yaml.safe_load((ROOT/'hop_scaling_joint_service_10m_seed0/snapshots/diffusion/config.yaml').read_text())['env']
 if not env:
  # A reused evaluation can still resolve configuration from source provenance.
  d=meta.get('provenance.json',{});src=local_path(d.get('checkpoint',''))
  if src and src.parent.name in ['diffusion','gaussian'] and (src.parent/'config.yaml').exists():env=yaml.safe_load((src.parent/'config.yaml').read_text())['env']
 return dict(device=device,action_seed=action,batch_size=batch,checkpoint_step=step,training_budget=budget,
             num_users=env.get('num_users'),num_uavs=env.get('num_uavs'),capacity=env.get('capacity'),neighbor_slots=env.get('neighbor_slots'),
             throughput_scale_bps=(env['num_users']*env['rate_reference_bps'] if 'num_users' in env and 'rate_reference_bps' in env else None),provenance=meta)


def process(entry,ident):
 meta=metadata(entry);files=entry.get('files') or [entry['path']];repetitions=[];hashes={};curves=[]
 for path in files:
  c,r,seeds=read_series(path,throughput_scale_bps=meta['throughput_scale_bps'],**(entry.get('columns') or {}))
  if c.shape[1]!=400:raise ValueError('A full current task is exactly 400 steps: '+path)
  repetitions.append(per_scene(c,r,seeds));hashes[str(path)]=sha(path);curves.append((c,r))
 result,ci,scene=summarize(repetitions)
 row=dict(evaluation_id=ident,section=entry['section'],label=entry['label'],checkpoint=entry['kind'],
          **{k:v for k,v in meta.items() if k not in ['throughput_scale_bps','provenance']},
          **result,note=entry.get('note',''),source_path=entry['path'])
 if entry.get('action_repeats'):row['action_seed']=';'.join(map(str,entry['action_repeats']))
 for k in METRICS:row[k+'_ci_low'],row[k+'_ci_high']=ci[k]
 per=[]
 for i,s in enumerate(seeds):per.append(dict(evaluation_id=ident,scene_seed=int(s),**{k:float(v[i]) for k,v in scene.items()}))
 digest=hashlib.sha256()
 for c,r in curves:
  for x in [seeds,np.round(c,12),np.round(r,1)]:digest.update(x.tobytes())
 return row,per,hashes,digest.hexdigest(),dict(meta=meta,entry=entry,intervals=ci)


def table(rows,out):
 lines=['| 实验 | 模型 | 模型步数 | 后100步覆盖 % | 终态覆盖 % | 后100步吞吐 Mbps | 终态吞吐 Mbps | 后100步覆盖σ pp | 口径 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  lines.append('| '+ ' | '.join([r['label'],r['checkpoint'],f"{r['checkpoint_step']:,}" if r.get('checkpoint_step') is not None else '未记录',
      *[f"{r[k]:.3f}" if r.get(k) is not None else '不适用' for k in METRICS],f"{r.get('device','unknown')}; {r['scene_count']}场景×{r['action_repetitions']}重复"])+ ' |')
 return lines


def shadow_groups(start_index):
 all_rows=[];per_rows=[];hashes={};details={}
 for root in [ROOT/'topk_capacity10_shared_best_diagnostic',ROOT/'topk_capacity10_coverage_throughput_best_diagnostic']:
  paths=sorted((root/'audit').glob('scene_*.npz'))
  if not paths:continue
  arrays=[];seeds=[];upper=[]
  for p in paths:
   with np.load(p) as z:
    names=z['metric_names'].tolist();data=z['metrics'];arrays.append(data);seeds.append(int(z['scene_seed']));upper.append((z['optima']>=0).sum(-1)/z['optima'].shape[-1]);capacity=int(z['audit_capacity']);trajectory_capacity=int(z['trajectory_capacity'])
   hashes[str(p)]=sha(p)
  a=np.stack(arrays);seeds=np.array(seeds);u=np.stack(upper)
  for score,idx in [('SINR教师',[0]),('常数',[1]),('随机20重复',list(range(2,a.shape[2])))]:
   for scheme,k in [('P0',0),('P1',1)]:
    reps=[per_scene(a[:,:,i,k,names.index('coverage')],a[:,:,i,k,names.index('throughput_bps')],seeds) for i in idx]
    result,ci,values=summarize(reps);ident=f'E{start_index+len(all_rows):03d}'
    row=dict(evaluation_id=ident,section='shadow',label=f'{root.name} / {score}/{scheme}',checkpoint='shadow',checkpoint_step=None,training_budget=None,
             device='CPU shadow',action_seed=970001,batch_size=None,num_users=150,num_uavs=18,capacity=capacity,trajectory_capacity=trajectory_capacity,**result,note='在冻结实际轨迹上重新关联，影子结果不影响后续状态',source_path=str(root/'audit'))
    for key in METRICS:row[key+'_ci_low'],row[key+'_ci_high']=ci[key]
    all_rows.append(row)
    for i,s in enumerate(seeds):per_rows.append(dict(evaluation_id=ident,scene_seed=int(s),**{k:float(v[i]) for k,v in values.items()}))
  for k,scheme in enumerate(['O1 known maximum','O2 full maximum']):
   result,ci,values=summarize([per_scene(u[:,:,k],np.zeros_like(u[:,:,k]),seeds)])
   ident=f'E{start_index+len(all_rows):03d}';row=dict(evaluation_id=ident,section='shadow',label=f'{root.name} / {scheme}',checkpoint='upper_bound',checkpoint_step=None,device='CPU shadow',**result,
    note='最大服务人数上界；未将某个最大匹配的吞吐定义为最优吞吐',source_path=str(root/'audit'))
   for name in list(row):
    if 'throughput' in name:row[name]=None
   all_rows.append(row)
   for i,seed in enumerate(seeds):
    per_rows.append(dict(evaluation_id=ident,scene_seed=int(seed),**{name:(None if 'throughput' in name else float(v[i])) for name,v in values.items()}))
 return all_rows,per_rows,hashes,details


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--allow-missing',action='store_true');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();entries=make_registry();rows=[];scene_rows=[];inputs={};details={};fingerprints={};missing=[];registered=set();aliases=[]
 for entry in entries:
  if not Path(entry['path']).exists():missing.append(entry);continue
  ident=f'E{len(rows)+1:03d}';row,per,hashes,fp,detail=process(entry,ident);rows.append(row);scene_rows+=per;inputs.update(hashes);details[ident]=detail
  fingerprints.setdefault(fp,ident);registered.update(entry.get('files') or [entry['path']])
 print('CURATED',len(rows),'missing',len(missing),flush=True)
 if missing and not args.allow_missing:raise FileNotFoundError([x['path'] for x in missing])
 # Audit every other scalar full-episode dataset. Reused identical evaluations
 # are aliases, not extra independent evidence; different CPU/CUDA trajectories remain distinct.
 for path in sorted(ROOT.rglob('*.npz')):
  name=str(path)
  if name in registered or 'code_snapshot' in path.parts or out in path.parents:continue
  try:
   with np.load(path,allow_pickle=False) as z:
    names=z['diagnostic_names'].tolist() if 'diagnostics' in z else z['names'].tolist() if 'names' in z else []
    key='diagnostics' if 'diagnostics' in z else 'metrics'
    if 'coverage' not in names or key not in z or z[key].ndim!=3 or z[key].shape[1]!=400 or 'scene_seeds' not in z:continue
   entry=dict(section='other',label=str(path.parent.relative_to(ROOT)),path=name,kind='saved',run=None,budget=None,step=None,note='独立保留的完整评估；若配置不同不可直接作算法因果比较')
   ident=f'E{len(rows)+1:03d}';row,per,hashes,fp,detail=process(entry,ident);inputs.update(hashes)
   if fp in fingerprints:
    aliases.append(dict(source_path=name,canonical_evaluation_id=fingerprints[fp],reason='同一场景顺序下覆盖/吞吐完整序列一致（用于索引，不重复作为独立证据）'))
   else:
    rows.append(row);scene_rows+=per;details[ident]=detail;fingerprints[fp]=ident
  except ValueError as e:
   missing.append(dict(path=name,reason=str(e),category='unsupported_or_missing_metadata'))
 shadow,shadow_per,shadow_hashes,shadow_details=shadow_groups(len(rows)+1);rows+=shadow;scene_rows+=shadow_per;inputs.update(shadow_hashes);details.update(shadow_details)
 # Verify source arrays stayed immutable throughout reporting.
 assert all(sha(path)==h for path,h in inputs.items())
 write_csv(out/'summary.csv',rows);write_csv(out/'per_scene.csv',scene_rows);write_csv(out/'evaluation_aliases.csv',aliases)
 atomic_json(out/'missing_or_unavailable.json',missing);atomic_json(out/'provenance.json',details)
 legacy=read('docs/cleanup_20261029.json')
 atomic_json(out/'deleted_historical_results.json',dict(reason='Previously deleted on user request; cannot reconstruct tail SD without arrays or frozen checkpoints',removed_runs=legacy['removed_runs'],removed_analysis=legacy['removed_analysis']))
 covered={r.get('run') for r in entries if Path(r['path']).exists()}
 formal=read('docs/cleanup_20261110.json')['formal_runs_retained']
 assert all(r['path'] in covered for r in formal),[r['path'] for r in formal if r['path'] not in covered]
 source_metadata={str(p):sha(p) for p in [Path(__file__),Path('src/uav_research/tail_metrics.py'),Path('docs/cleanup_20261029.json'),Path('docs/cleanup_20261110.json'),Path('scripts/evaluate_tail100_missing.py'),Path('tests/test_tail_metrics.py')]}
 for e in entries:
  files=[Path(e['run'])/'config.yaml'] if e.get('run') else []
  files += [Path(e['path']).parent/n for n in ['provenance.json','evaluation_config.json','evaluation_source.json']]
  for f in files:
   if f.exists():source_metadata[str(f)]=sha(f)
 manifest=dict(status='complete' if not missing else 'complete_with_documented_gaps',window='post-action steps301..400',horizon=400,
    std_definition='sqrt(mean over scenarios and action repetitions of population temporal variance within final100; ddof0)',
    other_std='tail100_scene_std_pp is mean of individual trajectory temporal SD, not primary RMS SD',
    bootstrap=dict(repeats=2000,seed=950001,unit='scene; action repetitions averaged within scene after computing trajectory statistics'),
    reevaluated_saved_models=True,supplemental_frozen_evaluations=12,supplemental_distinct_model_checkpoints=6,historical_best_selection_unchanged=True,reserved_test_set_used=False,
    formal_runs_covered=len(formal),metadata_sha256=source_metadata,evaluations=len(rows),source_arrays=len(inputs),alias_paths=len(aliases),gaps=len(missing),source_sha256=inputs,elapsed_seconds=time.perf_counter()-start)
 atomic_json(out/'manifest.json',manifest)
 publish(out,rows,missing,aliases,manifest)
 print('COMPLETE',json.dumps({k:v for k,v in manifest.items() if k not in ['source_sha256','metadata_sha256']}),flush=True)


def publish(out,rows,missing,aliases,manifest):
 import matplotlib.pyplot as plt
 style();lines=['# 已完成实验：统一后100步与终态指标复评','',
 '本报告读取已有逐步验证数据；缺失完整数组的六份冻结模型另补CUDA验证，再补SAC best/last各三套CPU动作种子，共12次完整验证。未训练任何网络、未改奖励或历史best选择，旧报告及模型保持不变。',
 '', '## 指标定义','',
 '1. 后100步平均覆盖：每条400步轨迹的第301–400步平均，再对场景平均。',
 '2. 终态覆盖：第400步覆盖，再对场景平均。',
 '3. 后100步平均吞吐：第301–400步系统总吞吐量平均，单位Mbps；非每UE速率。',
 '4. 终态吞吐：第400步系统总吞吐量的场景平均。',
 '5. 后100步覆盖时间标准差σ：先在每条轨迹内计算100步总体方差（ddof=0），再平均场景/动作重复的方差并开根号，单位百分点（pp）。该RMS标准差延续上轮报告5.995/4.594pp的口径。另在CSV保留“各场景标准差的算术平均”，两者不能混称。',
 '', r'$$\sigma_{100}=\sqrt{\frac1S\sum_s\frac1R\sum_r\frac1{100}\sum_{t=301}^{400}(100\rho_{s,r,t}-100\bar\rho_{s,r,100})^2}.$$',
 '', '多个动作种子时，先逐轨迹求方差再平均，禁止先平均覆盖曲线来人为消除波动。也不把场景之间均值差异算入时间不稳定性。100步固定窗口不预设系统已达到最大覆盖或稳态。',
 '', f'共整理{len(rows)}条评估记录，读取{manifest["source_arrays"]}份数组；{len(aliases)}处复用数据记为别名，不作为独立验证重复。',
 '每张表保留模型实际保存步数。best仍按当时训练的原选模标准选择，未按本次后100步指标重新挑模型。各组预算、容量、奖励、关联、观察、CPU/CUDA或场景不同，不能合并做不加条件的算法排名。',
 '', '完整机器可读结果：[summary.csv](summary.csv)，逐场景：[per_scene.csv](per_scene.csv)，场景级2000次bootstrap区间包含在CSV中，不是训练种子区间。原始文件与SHA256见manifest.json和provenance.json。']
 highlights=[]
 for label in ['Baseline N3 / 1M','Baseline N3 / 10M','Baseline N8 / 1M','DPPO无缩放 / 10M','Diffusion Hop 10M','Gaussian Hop 10M']:
  highlights.append(next(r for r in rows if r['label']==label and r['checkpoint']=='best'))
 lines+=['','## 主要best结果速览','']+table(highlights,out)
 lines+=['','新口径下，原邻居8 baseline的后100步覆盖66.461%，原无缩放DPPO 10M为66.338%；后者的后100步吞吐更高（616.788 vs 547.967 Mbps）。这说明覆盖与吞吐应分开判断，不能沿用全回合均值的名次推断后期覆盖优势。预算与关联机制等差异仍存在，这不是纯算法因果对照。',
         '最新跳数缩放两组中，Gaussian best的后100步覆盖75.104%、σ4.594pp；Diffusion best为65.243%、σ5.995pp。改用后期指标后，两组的描述性排序没有反转。',
         '高断联惩罚的低σ伴随低覆盖：μ=2、5的best后100步覆盖分别40.358%、41.445%，σ为3.900、3.020pp。应把“覆盖水平”和“时间波动”一起看，不能把较小标准差单独当作成功。']
 for section,title in SECTIONS.items():
  chosen=[r for r in rows if r['section']==section]
  if not chosen:continue
  lines+=['',f'## {title}','']+table(chosen,out)
  if section=='legacy':lines+=['','容量20纯覆盖PPO与SAC均提供CPU、动作种子880001/880002/880003的可比口径。SAC CUDA单独保留以复现原训练验证日志，不把CPU与CUDA采样轨迹混合平均。']
  if section=='generalization':lines+=['','此组使用660000–660063，共64个诊断场景，UE减少时总吞吐会受总需求人数变化影响。冻结actor的UAV数量适配保持训练编号/跳数尺度，不能与32个原验证场景混排。']
  if section=='shadow':lines+=['','这些是教师真实轨迹上的影子关联，动作和下一状态仍由原教师决定。P1收益不是闭环补位策略的长期收益；O1/O2只报告覆盖上界，不虚构最优吞吐。历史容量20来源轨迹上的capacity10反事实与真正capacity10训练分开展示。']
  # Reproducible two-panel performance/stability plots, separate families.
  if section in ['hop','baseline','reward','dppo','legacy']:
   selected=[r for r in chosen if r['checkpoint'] in ['best','last'] and '三动作种子' not in r['label']]
   y=np.arange(len(selected));labels=[f"{r['label']} / {r['checkpoint']}".replace('断联','Disconnection ').replace('鲁棒','Robustness ').replace('＋',' + ').replace('无缩放','no scaling') for r in selected]
   specifications=[('late',[('tail100_coverage_pct','Last100 coverage (%) ↑'),('tail100_throughput_mbps','Last100 throughput (Mbps) ↑'),('tail100_coverage_std_pp','Last100 temporal SD (pp) ↓')]),
                   ('terminal',[('final_coverage_pct','Final coverage (%) ↑'),('final_throughput_mbps','Final throughput (Mbps) ↑')])]
   for suffix,spec in specifications:
    fig,axes=plt.subplots(1,len(spec),figsize=(15 if len(spec)==3 else 12,max(3,len(selected)*.32)))
    for ax,(key,axis_title) in zip(axes,spec):
     for i,r in enumerate(selected):
      val=r[key];lo=r.get(key+'_ci_low',val);hi=r.get(key+'_ci_high',val)
      ax.errorbar(val,i,xerr=np.array([[max(0,val-lo)],[max(0,hi-val)]]),fmt='o' if r['checkpoint']=='best' else 's',color='#009E73' if r['checkpoint']=='best' else '#0072B2',capsize=2)
     ax.set(yticks=y,yticklabels=labels if ax is axes[0] else ['']*len(y),title=axis_title);ax.invert_yaxis()
    save_figure(fig,out/'figures',section+'_'+suffix+'_metrics')
    lines+=['',f'![{title}: {suffix}](figures/{section}_{suffix}_metrics.png)']
 lines+=['','## 完整性与不能恢复的历史结果','',
 '保留的20个正式运行已逐一覆盖best/last；其中文件名100k并不代表实际预算。独立PPO的1M历史模型和1.2M模型分别来自各自保存的报告，不把当前1.2M权重冒充旧1M。最新两组跳数缩放10M和最新20万步诊断也单列。',
 '此前已清理的250k/100k旧实验（例如最早A–F、早期critic实验）若没有逐步验证数据与对应检查点，不能从训练均值或代表场景图恢复这五项指标。已清理验收摘要也不能替代全场景验证；本报告不编造这些结果。SAC旧逐步报告缺失，已用保留best/last补评估。',
 '清理过的旧实验路径见[历史删除清单](deleted_historical_results.json)，它们与本报告零缺口的“现存20个正式运行”是不同范围。',
 '仅有两张代表拓扑轨迹、训练日志、均值曲线或短分叉的数据，不冒充完整32场景400步评估。单步动作干预/50步fork不是完整回合，不纳入本表。',
 f'数据缺口/无法解析项共{len(missing)}条，见[缺口清单](missing_or_unavailable.json)。',
 '', '主报告中的数据重用别名只用于查找来源；完整别名映射见evaluation_aliases.csv。旧结果文件在计算前后核验SHA256，未写回任何旧结果或改变best。',
 '', '## 重建命令','', '```bash',
 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\',
 '.venv-dppo-compat/bin/python scripts/report_all_tail100.py','```',
 '', '只读取已保存数据，不重新采样。所有新输出在本目录；本轮未自动上传GitHub。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
