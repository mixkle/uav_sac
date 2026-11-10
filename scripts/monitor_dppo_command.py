"""Monitor one explicitly launched child; no system configuration changes."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command or args.interval <= 0:
        parser.error('A child command and positive interval are required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'command.json').write_text(json.dumps(command, indent=2))
    rows = []
    start = time.perf_counter()
    with (output/'command.log').open('w') as log, (output/'monitor.csv').open('w', newline='') as file:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        process.cpu_percent()
        writer = csv.DictWriter(file, fieldnames=['elapsed_seconds','pid','cpu_percent','rss_mib',
            'gpu_util_percent','gpu_memory_mib','gpu_power_w','gpu_temperature_c'])
        writer.writeheader()
        while child.poll() is None:
            try:
                raw = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                    '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                gpu = [float(x.strip()) for x in raw.split(',')]
                row = dict(elapsed_seconds=time.perf_counter()-start, pid=child.pid,
                    cpu_percent=process.cpu_percent(), rss_mib=process.memory_info().rss/2**20,
                    **dict(zip(['gpu_util_percent','gpu_memory_mib','gpu_power_w','gpu_temperature_c'], gpu)))
                writer.writerow(row)
                file.flush()
                rows.append(row)
            except (psutil.NoSuchProcess, subprocess.SubprocessError, ValueError):
                if child.poll() is not None:
                    break
            time.sleep(args.interval)
        code = child.wait()
    result = dict(exit_code=code, elapsed_seconds=time.perf_counter()-start, samples=len(rows),
        gpu_util_mean=sum(r['gpu_util_percent'] for r in rows)/max(1,len(rows)),
        gpu_util_peak=max((r['gpu_util_percent'] for r in rows),default=0),
        gpu_memory_peak_mib=max((r['gpu_memory_mib'] for r in rows),default=0),
        cpu_percent_mean=sum(r['cpu_percent'] for r in rows)/max(1,len(rows)),
        rss_peak_mib=max((r['rss_mib'] for r in rows),default=0))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
