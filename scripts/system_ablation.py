"""User-controlled training stages for the A--F suite. Never auto-launch jobs."""
import argparse,json,os
from pathlib import Path
from uav_research.config import load_config
from uav_research.runtime import resolve_output

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'configs/system_ablation_1m/manifest.json'


def resolve(group,device=None):
 m=json.loads(MANIFEST.read_text()); group=m['aliases'].get(group,group)
 if group not in m['runs']:raise ValueError('Unknown group: '+group)
 entry=m['runs'][group];cfg=load_config(ROOT/entry['config'])
 data=resolve_output('runs/system_ablation_1m')
 # Canonical storage root is movable; rewrite paths before contracts are saved.
 for key in ('teacher_checkpoint','pretrained_checkpoint'):
  old=getattr(cfg.diffusion,key)
  if old and '/runs/system_ablation_1m/' in old:
   setattr(cfg.diffusion,key,str(data/old.split('/runs/system_ablation_1m/',1)[1]))
 old=cfg.diffusion.ablation['initial_checkpoint']
 if old:cfg.diffusion.ablation['initial_checkpoint']=str(data/old.split('/runs/system_ablation_1m/',1)[1])
 if device:cfg.run.device=device
 cfg.run.cpu_threads=int(os.environ.get('UAV_TORCH_THREADS',cfg.run.cpu_threads))
 cfg.run.numba_threads=int(os.environ.get('UAV_NUMBA_THREADS',cfg.run.numba_threads))
 return cfg.validate(),data/f'{group}_seed0',data


def check(cfg,output,resume=None):
 from uav_research.diffusion_dppo.storage import ensure_budget,load
 from uav_research.system_ablation.settings import options
 ensure_budget(cfg,output)
 if not resume and any((output/x).exists() for x in ('step0.pt','last.pt','manifest.json')):
  raise FileExistsError(f'Existing run; explicitly resume: {output}')
 if resume:load(resume,cfg,{'dppo_full'})
 else:
  o=options(cfg)
  if o['initialization']!='random':
   from uav_research.diffusion_dppo.learner import DiffusionLearner
   from uav_research.diffusion_dppo.storage import configure
   from uav_research.system_ablation.initialization import initialize
   learner=DiffusionLearner(cfg,configure(cfg));initialize(learner)
 return dict(group=options(cfg)['run_id'],output=str(output),online_steps=cfg.run.total_steps,
  initialization=options(cfg)['initialization'],status='ready',resume=str(resume) if resume else None)


def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='phase',required=True)
 sp.add_parser('list')
 for name in ('train','check','collect','pretrain'):
  s=sp.add_parser(name);s.add_argument('--group',default='G0' if name in ('train','check') else 'A3')
  s.add_argument('--device');s.add_argument('--resume');s.add_argument('--auto-resume',action='store_true')
 args=p.parse_args()
 if args.phase=='list':print(MANIFEST.read_text());return
 cfg,out,data=resolve(args.group,args.device)
 if args.phase in ('train','check'):
  resume=args.resume
  if args.auto_resume:
   if resume:p.error('Choose --resume OR --auto-resume')
   checkpoints=sorted((out/'checkpoints').glob('full_[0-9]*'))
   if not checkpoints:raise FileNotFoundError('No full checkpoint; auto-resume never silently starts fresh')
   resume=checkpoints[-1]
  result=check(cfg,out,resume)
  print(json.dumps(result,indent=2),flush=True)
  if args.phase=='train':
   from uav_research.diffusion_dppo.runner import train_online
   print(json.dumps(train_online(cfg,out,resume=resume),indent=2))
 elif args.phase=='collect':
  if args.group!='A3':p.error('Collect exactly once using --group A3; all students share the same data')
  from uav_research.diffusion_dppo.data import collect_demonstrations
  print(json.dumps(collect_demonstrations(cfg,data/'demonstrations'),indent=2))
 else:
  if args.group not in ('A3','A5','F2a'):p.error('Only A3 A5 F2a need pretraining; F1a reuses A3')
  if args.auto_resume:p.error('Pass --resume for pretraining')
  from uav_research.diffusion_dppo.data import pretrain
  print(json.dumps(pretrain(cfg,data/'demonstrations',data/'pretraining'/args.group,args.resume),indent=2))

if __name__=='__main__':main()
