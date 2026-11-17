"""Read-only model evaluation on four GPUs; each group writes a separate subtree."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='analysis/diffusion_gaussian_ablation_1m')
    parser.add_argument('--action-seeds', nargs='+', type=int, default=[880001, 880002, 880003])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    groups = list(json.loads((root/'configs/system_ablation_1m/manifest.json').read_text())['runs'])
    data = Path(os.environ['UAV_DATA_ROOT'])
    logs = data/'logs/system_ablation_evaluation_20261113'
    logs.mkdir(parents=True, exist_ok=True)
    slots = {}
    failures = []
    masks = ['0-5', '6-11', '12-17', '18-23']
    while groups or slots:
        for slot in range(4):
            if slot in slots or not groups:
                continue
            group = groups.pop(0)
            log = (logs/f'{group}.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), MPLBACKEND='Agg')
            cmd = ['taskset', '-c', masks[slot], sys.executable, '-u',
                   'scripts/report_system_ablation.py', '--collect-only', '--groups', group,
                   '--device', 'cuda:0', '--output', args.output, '--action-seeds',
                   *map(str, args.action_seeds)]
            p = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            slots[slot] = (group, p, log)
            print(datetime.datetime.now().isoformat(), 'START', group, 'GPU', slot, flush=True)
        for slot, (group, p, log) in list(slots.items()):
            result = p.poll()
            if result is None:
                continue
            log.close()
            del slots[slot]
            print(datetime.datetime.now().isoformat(), 'DONE', group, 'exit', result, flush=True)
            if result:
                failures.append(group)
        if groups or slots:
            time.sleep(5)
    if failures:
        raise RuntimeError('Evaluation failed: '+', '.join(failures))
    subprocess.run([sys.executable, '-u', 'scripts/report_system_ablation.py',
                    '--rebuild', '--output', args.output], cwd=root, check=True)


if __name__ == '__main__':
    main()
