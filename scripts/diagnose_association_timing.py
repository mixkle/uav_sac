"""Frozen T1/T2/T3 association timing experiment; no training entry point."""
import argparse
from pathlib import Path
from uav_research.runtime import resolve_output
from uav_research.association_timing_experiment import run_case,rebuild,DEFAULT_OUTPUT,ACTION_SEEDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['T1','T2','T3'])
    p.add_argument('--checkpoints',nargs='+',choices=['best','last'],default=['best','last'])
    p.add_argument('--device',default='cpu');p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--rebuild',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=resolve_output(a.output)
    if a.rebuild:
        rebuild(out);return
    if a.protocol is None:p.error('--protocol or --rebuild required')
    if a.smoke and a.output==DEFAULT_OUTPUT:p.error('Smoke requires a separate --output')
    for checkpoint in a.checkpoints:
        for seed in (ACTION_SEEDS[:1] if a.smoke else ACTION_SEEDS):
            run_case(checkpoint,a.protocol,seed,out,a.device,
                     scenes=list(range(10000,10004)) if a.smoke else None,
                     steps=5 if a.smoke else None)


if __name__=='__main__':main()
