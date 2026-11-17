import argparse,json
from pathlib import Path
from uav_research.runtime import resolve_output
from uav_research.system_ablation.report import collect_run,rebuild


def main():
 p=argparse.ArgumentParser(description='Frozen A--F late-coverage evaluation; no training')
 p.add_argument('--groups',nargs='+');p.add_argument('--device',default='cpu')
 p.add_argument('--split',choices=['validation','test'],default='validation')
 p.add_argument('--action-seeds',nargs='+',type=int,default=[880001])
 p.add_argument('--checkpoints',nargs='+',choices=['step0','best','last'],default=['step0','best','last'])
 p.add_argument('--moment-repeats',type=int,default=0)
 p.add_argument('--output',default='analysis/diffusion_gaussian_ablation_1m')
 p.add_argument('--rebuild',action='store_true')
 p.add_argument('--collect-only',action='store_true',help='Save frozen evaluation without rebuilding shared report (parallel workers)')
 a=p.parse_args();out=resolve_output(a.output);out.mkdir(parents=True,exist_ok=True)
 if not a.rebuild:
  if not a.groups:p.error('Explicit --groups required; use --rebuild for saved data only')
  manifest=json.loads(Path('configs/system_ablation_1m/manifest.json').read_text())
  groups=list(dict.fromkeys(manifest['aliases'].get(g,g) for g in a.groups))
  for g in groups:
   if g not in manifest['runs']:p.error('Unknown group '+g)
   collect_run(resolve_output('runs/system_ablation_1m')/(g+'_seed0'),out/g,a.device,a.split,
      a.action_seeds,a.checkpoints,a.moment_repeats)
 if not a.collect_only:print(json.dumps(rebuild(out),indent=2))

if __name__=='__main__':main()
