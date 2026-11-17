"""Four independent single-GPU workers; no changes to learning settings."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
REMAINING = ['B2', 'B3', 'B4', 'B5', 'C2', 'C3', 'D2', 'F1a', 'E2', 'E3']


def completed(output):
    path = output / 'manifest.json'
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    return state.get('status') == 'complete' and state.get('env_steps') == 1_000_000


def say(message):
    print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-optional', action='store_true', help='Also run C4')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data = Path(os.environ['UAV_DATA_ROOT'])
    runs = data / 'runs/system_ablation_1m'
    logs = data / 'logs/system_ablation_1m'
    groups = REMAINING + (['C4'] if args.include_optional else [])
    gpu_ids = os.environ.get('UAV_ABLATION_GPUS', '0,1,2,3').split(',')
    cpu_masks = os.environ.get('UAV_ABLATION_CPUSETS', '0-5|6-11|12-17|18-23').split('|')
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(not x.isdecimal() for x in gpu_ids):
        parser.error('Choose four distinct numeric GPU IDs in UAV_ABLATION_GPUS')
    if len(cpu_masks) != 4:
        parser.error('Provide four CPU masks in UAV_ABLATION_CPUSETS')
    seen = set()
    for mask in cpu_masks:
        cpus = set()
        for part in mask.split(','):
            bounds = list(map(int, part.split('-')))
            if len(bounds) not in (1, 2) or bounds[-1] < bounds[0]:
                parser.error('Invalid CPU affinity')
            cpus.update(range(bounds[0], bounds[-1] + 1))
        if len(cpus) < 4 or cpus & seen or not cpus.issubset(os.sched_getaffinity(0)):
            parser.error('CPU sets must be available, disjoint, with >=4 CPUs each')
        seen.update(cpus)
    # The user has finished the first two batches. Never schedule over them.
    for group in ['G0', 'A1', 'D3', 'D4', 'A3', 'A4', 'A5', 'F2a']:
        if not completed(runs / f'{group}_seed0'):
            raise RuntimeError(f'First/second batch incomplete: {group}')
    pending = []
    resume_groups = set()
    for group in groups:
        output = runs / f'{group}_seed0'
        if completed(output):
            say(f'SKIP complete {group}')
            continue
        if output.exists() and any(output.iterdir()):
            full = sorted((output / 'checkpoints').glob('full_[0-9]*'))
            if not args.resume_incomplete or not full:
                raise RuntimeError(f'{group}: existing incomplete output; use --resume-incomplete '
                                   'only when a full checkpoint is available')
            resume_groups.add(group)
        pending.append(group)
    say(f"QUEUE {pending}; GPUs={gpu_ids}; include_optional={args.include_optional}")
    if args.dry_run or not pending:
        return
    if shutil.disk_usage(data).free < 23 * 2**30:
        raise OSError('Need >=23GiB free before multi-job launch')
    # Refuse to occupy GPUs already running training or another compute job.
    gpu_uuids = {}
    for gpu in gpu_ids:
        gpu_uuids[gpu] = subprocess.check_output([
            'nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(uuid in apps for uuid in gpu_uuids.values()):
        raise RuntimeError('Selected GPU has an active compute process; wait until it is free')
    # Validate all required artifacts before starting the first training process.
    for group in pending:
        command = [str(ROOT / '.venv-dppo/bin/python'), '-u', 'scripts/system_ablation.py',
                   'check', '--group', group, '--device', 'cpu']
        if group in resume_groups:
            command.append('--auto-resume')
        subprocess.run(command, cwd=ROOT, check=True)
    logs.mkdir(parents=True, exist_ok=True)
    active = {}
    failed = []
    stopped = False
    try:
        while pending or active:
            for slot in range(4):
                if slot in active or not pending or stopped:
                    continue
                group = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids[slot], UAV_CPU_AFFINITY=cpu_masks[slot])
                log_path = logs / f'{group}.log'
                log = log_path.open('a' if group in resume_groups else 'w')
                command = ['bash', 'scripts/train_system_ablation_server.sh', 'train', '--group', group]
                if group in resume_groups:
                    command.append('--auto-resume')
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active[slot] = (group, process, log)
                say(f'START {group} GPU={gpu_ids[slot]} pid={process.pid} log={log_path}')
            for slot, (group, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[slot]
                if code == 0 and completed(runs / f'{group}_seed0'):
                    say(f'DONE {group} GPU={gpu_ids[slot]}')
                else:
                    say(f'FAILED {group} exit={code}; no new groups will start; active groups continue')
                    failed.append(group)
                    stopped = True
            if stopped and not active:
                break
            if pending or active:
                time.sleep(5)
    except KeyboardInterrupt:
        say('Queue interrupted; no process is explicitly terminated by this script')
        raise
    if failed:
        raise RuntimeError(f'Failed={failed}; not started={pending}. Check group logs before resuming.')
    say('ALL REQUESTED EXPERIMENTS COMPLETE')


if __name__ == '__main__':
    main()
