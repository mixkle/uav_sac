"""Frozen follow-up comparison; references reused only after source hash checks."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from uav_research.runtime import resolve_output,atomic_json
from uav_research.topk_diagnostic import file_hash,atomic_npz
from uav_research.diffusion_followup.runtime import resolve
from uav_research.diffusion_followup.audits import sample_scale
from uav_research.diffusion_dppo.storage import load,configure
from uav_research.diffusion_dppo.config import from_dict
from uav_research.diffusion_dppo.policy import make_service_actor
from uav_research.system_ablation.report import collect_run,rebuild,write_csv


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--device',default='cpu');p.add_argument('--rebuild',action='store_true')
 p.add_argument('--output',default='analysis/diffusion_followup_f2b_f2c_a1b_20261113')
 p.add_argument('--no-scale-audit',action='store_true')
 a=p.parse_args();out=resolve_output(a.output);out.mkdir(parents=True,exist_ok=True)
 seeds=[880001,880002,880003]
 if not a.rebuild:
  for g in ['A1','G0','A3','A5','F2a']:
   cache=resolve_output('analysis/diffusion_gaussian_ablation_1m')/g
   for kind in ('step0','best','last'):
    sha=file_hash(resolve_output('runs/system_ablation_1m')/(g+'_seed0')/(kind+'.pt'))
    for seed in seeds:
     src=cache/'validation'/kind/str(seed)
     if not (src/'complete.json').exists():raise FileNotFoundError('Reference evaluation incomplete: '+str(src))
     if json.loads((src/'provenance.json').read_text())['sha256']!=sha:raise ValueError('Reference model changed')
   shutil.copytree(cache,out/g,dirs_exist_ok=True)
  for g in ['F2b','F2c','A1b']:
   _,run,_=resolve(g,a.device)
   collect_run(run,out/g,a.device,action_seeds=seeds)
  if not a.no_scale_audit:
   import torch
   pool=torch.load(resolve_output('runs/diffusion_followup_1m/audits/A1b_initialization/observation_pool.pt'),
                   map_location='cpu',weights_only=False)
   # observation_pool is a trusted local torch artifact, not an online checkpoint.
   for name in ('step0','audit_100k','best','last'):
    _,run,_=resolve('A1b',a.device);source=run/(name+'.pt')
    data=load(source,kinds={'dppo_model'});cfg=from_dict(data['config']);cfg.run.device=a.device;configure(cfg)
    actor=make_service_actor(cfg).to(a.device);actor.load_state_dict(data['learner']['actor']);actor.eval()
    target=out/'A1b_scale_progress'/name;target.mkdir(parents=True,exist_ok=True)
    sha=file_hash(source)
    if (target/'summary.json').exists() and json.loads((target/'summary.json').read_text()).get('checkpoint_sha256')==sha:continue
    summary,arrays=sample_scale(actor,pool['observation'],pool['pool_ids'],64)
    summary.update(checkpoint_sha256=sha,online_steps=data['env_steps'],observations='fixed initialization pools, not on-policy states')
    atomic_json(target/'summary.json',summary);atomic_npz(target/'samples.npz',**arrays)
 groups={'F2_followup':['A3','F2b','F2c','F2a'], 'A1_initialization':['A1','A1b','G0'],
         'A_BC_references':['A3','F2b','F2c','A5']}
 rows=rebuild(out,comparisons_extra=[('F2b','A3'),('F2c','A3'),('A1b','A1'),('A1b','G0'),
     ('F2c','F2b'),('F2a','F2b'),('F2a','F2c')],groupings=groups)
 shutil.copy2(out/'metrics/unified_best_last.csv',out/'metrics/unified_step0_best_last.csv')
 gains=[];lookup={(r['run_id'],r['checkpoint']):r for r in rows}
 for g in sorted({r['run_id'] for r in rows}):
  s=lookup[g,'step0'];b=lookup[g,'best'];r=lookup[g,'last']
  gains.append(dict(group=g,step0_coverage=s['last100_coverage'],best_coverage=b['last100_coverage'],
      last_coverage=r['last100_coverage'],last_gain_pp=100*(r['last100_coverage']-s['last100_coverage']),
      best_gain_pp=100*(b['last100_coverage']-s['last100_coverage'])))
 write_csv(out/'metrics/online_gains.csv',gains)
 diagnostics=[]
 for g in ('A1','A1b','A3','F2b','F2c','F2a','G0'):
  with (out/g/'training_data/metrics.csv').open() as f:log=list(csv.DictReader(f))
  line=dict(group=g,rollout_updates_averaged=min(50,len(log)))
  for key in ['actor_loss','value_loss','movement_conditional_kl','score_conditional_kl',
        'movement_only_ratio_clip','score_only_ratio_clip','movement_head_grad_norm',
        'score_head_grad_norm','encoder_grad_norm','score_saturation_fraction']:
   vals=[float(r[key]) for r in log[-50:] if r.get(key) not in ('',None)]
   if vals:line[key]=float(np.mean(vals))
  diagnostics.append(line)
 write_csv(out/'metrics/internal_diagnostics.csv',diagnostics)
 notes=['# F2b / F2c / A1b追加定位实验','',
  '冻结32验证场景×3动作种子；last为主，best不按复评重选。场景bootstrap不是训练种子置信区间。',
  'F2b：A3 step0同权重，只更改执行评分；F2c：原形状Gate=0、重新BC；A1b：无BC参考残差初始化。',
  'F2b frozen replay在runs/diffusion_followup_1m/audits/F2b_frozen下，与重新训练F2b分开解释。',
  'direct_switches/service_interruptions的last100字段为每步平均事件数，乘100才是该窗口每回合总次数。',
  '历史score_saturation_fraction为|latent|>10。A1b_scale_progress中明确记录sigmoid0.01/0.99及0.05/0.95阈值，不能混用。','',
  '|组|step0后100覆盖%|best后100覆盖%|last后100覆盖%|last净增益pp|last后100吞吐Mbps|last后100σ pp|',
  '|---|---:|---:|---:|---:|---:|---:|']
 for g in gains:
  r=lookup[g['group'],'last']
  notes.append(f"|{g['group']}|{g['step0_coverage']*100:.2f}|{g['best_coverage']*100:.2f}|{g['last_coverage']*100:.2f}|{g['last_gain_pp']:+.2f}|{r['last100_throughput_mbps']:.2f}|{r['last100_coverage_std_pp']:.2f}|")
 (out/'README.md').write_text('\n'.join(notes)+'\n')
 atomic_json(out/'manifest.json',dict(groups=sorted({g for g,_ in lookup}),source_records='sources.json',
     action_seeds=seeds,scene_seeds=list(range(10000,10032)),heldout_test_used=False,
     old_F2b_alias_preserved=True))
 print(out/'README.md')


if __name__=='__main__':main()
