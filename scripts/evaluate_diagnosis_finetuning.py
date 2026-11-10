"""Frozen final A/B evaluation, never resumes/changes a training process."""
import argparse
from pathlib import Path
from uav_research.diffusion_dppo.report import evaluate_checkpoint

p=argparse.ArgumentParser();p.add_argument('--group',choices=['control','conservative'],required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
root=Path('analysis/diffusion_gaussian_diagnosis_20261110/exp4_conservative_finetuning')
for kind in ['best','last']:
    print('BEGIN',args.group,kind,flush=True)
    result=evaluate_checkpoint(root/args.group/(kind+'.pt'),root/'evaluation'/args.group/kind,device=args.device,seeds=list(range(10000,10032)),moment_repeats=0)
    print('COMPLETE',args.group,kind,result['summary']['coverage'],result['summary']['critic_rmse'],flush=True)
