#!/usr/bin/env python3
"""Frozen S1/S2 association diagnostics; never constructs a training learner."""
import argparse
import fcntl
from pathlib import Path
from uav_research.runtime import resolve_output
from uav_research.score_rule_experiment import DEFAULT_OUTPUT, calibrate, closed_case

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['calibrate','closed','s1','report','verify'])
    p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--rule',choices=['R0','R1','R2','R3'],default='R1')
    p.add_argument('--split',choices=['development','confirmation'],default='development')
    p.add_argument('--action-seed',type=int,default=880001)
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=resolve_output(args.output)
    if args.smoke and Path(args.output).name==Path(DEFAULT_OUTPUT).name:p.error('Smoke requires an independent output directory')
    # Independent cases may be distributed across otherwise idle GPUs. Serialize
    # duplicate invocations so a queue and a manual retry cannot race to publish.
    lock=root/'audit/locks'/f'{args.stage}_{args.checkpoint}_{args.rule}_{args.action_seed}_{args.split}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    handle=lock.open('a')
    fcntl.flock(handle,fcntl.LOCK_EX)
    if args.stage=='calibrate':calibrate(root,args.device,args.smoke)
    elif args.stage=='closed':closed_case(args.checkpoint,args.rule,args.action_seed,root,args.device,args.split,args.smoke)
    elif args.stage=='s1':
        from uav_research.score_rule_experiment import audit_s1
        audit_s1(args.checkpoint,args.action_seed,root,args.device,args.smoke)
    else:
        from uav_research.score_rule_report import rebuild, verify
        (rebuild if args.stage=='report' else verify)(root)
if __name__=='__main__':main()
