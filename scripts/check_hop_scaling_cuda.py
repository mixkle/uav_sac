"""Two single-GPU acceptance jobs, bounded to 64 transitions per algorithm."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
import numpy as np
from uav_research.runtime import atomic_json

p=argparse.ArgumentParser();p.add_argument('--pretrained',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0,2');p.add_argument('--steps',type=int,default=64);a=p.parse_args()
if not 0<a.steps<=2048 or a.steps%16:raise ValueError('Acceptance steps must be divisible by16 and at most2048')
root=Path.cwd().resolve()
if not str(root).startswith('/home/dym/'):raise ValueError('Server acceptance writes only below /home/dym')
a.output=a.output.resolve()
if not a.output.is_relative_to(root):raise ValueError('Output must remain inside server project')
a.output.mkdir(parents=True,exist_ok=False)
gpus=a.gpus.split(',')
if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Two distinct GPUs required')
processes=[];handles=[];samples=[];start=time.perf_counter()
try:
 for group,gpu,cpus in zip(('diffusion','gaussian'),gpus,('0-11','12-23')):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
  command=['taskset','-c',cpus,str(root/'.venv-dppo/bin/python'),'-u','scripts/validate_hop_scaling.py','--group',group,'--output',str(a.output/group),'--device','cuda:0','--steps',str(a.steps)]
  if group=='diffusion':command+=['--pretrained',str(Path(a.pretrained).resolve())]
  handle=(a.output/f'{group}.log').open('w');handles.append(handle)
  processes.append(subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT))
 while any(p.poll() is None for p in processes):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
  samples.append(dict(seconds=time.perf_counter()-start,gpu_rows=raw.strip().splitlines()))
  time.sleep(1)
 codes=[p.returncode for p in processes]
 result=dict(exit_codes=codes,elapsed_seconds=time.perf_counter()-start,gpu_samples=samples,groups={})
 for group in ('diffusion','gaussian'):
  run=a.output/group
  if (run/'acceptance.json').exists():
   value=json.loads((run/'acceptance.json').read_text());value['metrics']=list(csv.DictReader((run/'metrics.csv').open()))
   assert value['env_steps']==a.steps and value['optimizer_steps']==40
   result['groups'][group]=value
 atomic_json(a.output/'cuda_acceptance.json',result)
 if any(codes):raise RuntimeError(f'CUDA acceptance failures: {codes}; inspect logs')
 print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'],exit_codes=codes)))
finally:
 for p in processes:
  if p.poll() is None:p.terminate()
 for h in handles:h.close()
