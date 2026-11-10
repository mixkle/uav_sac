"""Preserve original 1M artifacts and seed a separate cumulative-budget run."""
import argparse
from uav_research.config import load_config
from uav_research.diffusion_dppo.continuation import prepare_continuation

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(prepare_continuation(load_config(args.config), args.resume, args.output), flush=True)
