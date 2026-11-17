"""Freeze A--F configs from the latest joint Gaussian baseline; no training."""
from pathlib import Path
import json,copy,yaml,hashlib
from uav_research.config import load_config
from uav_research.system_ablation.settings import DEFAULTS

ROOT=Path(__file__).resolve().parents[1]
DATA='/mnt/disk2/dym/uav_sac'
RUNS=DATA+'/runs/system_ablation_1m'
BASE='configs/gaussian_mappo_topk_hop_scaling_capacity10_10m_server.yaml'
# One physical run for all aliases.
ALIASES={'A2':'G0','B1':'G0','C1':'G0','D1':'G0','E1':'G0','F1b':'A3','F2b':'A3'}
SPECS={
 'G0':{},
 'A1':dict(actor='diffusion'),
 'A3':dict(actor='diffusion',initialization='bc'),
 'A4':dict(initialization='actor'),
 'A5':dict(initialization='bc'),
 'B2':dict(motion_rule='linear'),
 'B3':dict(history='command'),
 'B4':dict(motion_rule='none'),
 'B5':dict(motion_rule='constant'),
 'C2':dict(ue_context='top10_flat'),
 'C3':dict(ue_context='top10_set'),
 'C4':dict(ue_context='statistics',optional=True),
 'D2':dict(score_actions=False,association='centralized_greedy'),
 'D3':dict(association='sinr_topk'),
 'D4':dict(score_actions=False,association='sinr_topk'),
 'E2':dict(observation_scope='global'),
 'E3':dict(observation_scope='global',association='centralized_greedy'),
 'F1a':dict(actor='diffusion',motion_rule='none',initialization='bc'),
 'F2a':dict(actor='diffusion',score_actions=False,association='sinr_topk',initialization='bc'),
}


def build():
 base=load_config(ROOT/BASE).as_dict()
 entries={}
 target=ROOT/'configs/system_ablation_1m';target.mkdir(exist_ok=True)
 for key,change in SPECS.items():
  change=dict(change);c=copy.deepcopy(base)
  actor=change.pop('actor','gaussian');history=change.pop('history',None)
  optional=change.pop('optional',False)
  c['algo']['name']='diffusion_dppo' if actor=='diffusion' else 'gaussian_service_mappo'
  c['run'].update(total_steps=1_000_000,eval_interval=50_000,best_metric='last100_coverage_then_throughput',test_seeds=list(range(20000,20064)))
  o=dict(DEFAULTS,run_id=key,bc_objective='denoising' if actor=='diffusion' else 'nll',**change)
  if history:c['env']['history_motion_source']=history
  if o['motion_rule']=='none':c['env']['motion_scaling']['enabled']=False
  if o['initialization']=='actor':o['initial_checkpoint']=RUNS+'/G0_seed0/best.pt'
  if o['initialization']=='bc':
   stage='A3' if key=='F1a' else key
   o['initial_checkpoint']=RUNS+f'/pretraining/{stage}/best.pt'
  c['diffusion'].update(ablation=o,teacher_checkpoint=RUNS+'/G0_seed0/best.pt',
      teacher_step=0,pretrained_checkpoint='',microbatch_agents=4096,actor_update_devices=[])
  path=target/f'{key}.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
  load_config(path)
  dependencies=['G0','demos','pretrain:'+('A3' if key=='F1a' else key)] if o['initialization']=='bc' else ['G0'] if key=='A4' else []
  entries[key]=dict(config=str(path.relative_to(ROOT)),output=RUNS+f'/{key}_seed0',
      dependencies=dependencies,optional=optional,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),options=o)
 manifest=dict(version=DEFAULTS['suite'],baseline=BASE,aliases=ALIASES,runs=entries,
  validation_seeds=list(range(10000,10032)),test_seeds=list(range(20000,20064)),
  action_seeds=[880001,880002,880003],online_steps_per_run=1_000_000,
  pretraining=dict(fit_scenes=list(range(300000,300128)),selection_scenes=list(range(301000,301032)),epochs=50,batch=1024),
  selection='lexicographic last100 coverage, then last100 throughput; coverage equality tolerance 1e-12',
  A4='load G0 best actor ONLY; fresh critic, Adam, online RNG and 1M additional online budget',
  B5='constant ratio 0.5 fixed a priori, NOT fitted to final performance')
 (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(f'Generated {len(entries)} configs (18 required, C4 optional), aliases deduplicated.')
 return manifest

if __name__=='__main__':build()
