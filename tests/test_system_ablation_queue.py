"""Scheduler tests use fake workers; never start training or require GPUs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def setup_queue(tmp_path, monkeypatch, args=()):
    path = Path(__file__).resolve().parents[1] / 'scripts/queue_system_ablation.py'
    spec = importlib.util.spec_from_file_location('ablation_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('UAV_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('UAV_ABLATION_GPUS', '0,1,2,3')
    monkeypatch.setenv('UAV_ABLATION_CPUSETS', '0-5|6-11|12-17|18-23')
    monkeypatch.setattr(sys, 'argv', ['queue', *args])
    monkeypatch.setattr(module.os, 'sched_getaffinity', lambda _: set(range(24)))
    runs = tmp_path / 'runs/system_ablation_1m'

    def complete(group):
        out = runs / f'{group}_seed0'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'manifest.json').write_text(json.dumps(dict(status='complete', env_steps=1_000_000)))

    for group in ['G0', 'A1', 'D3', 'D4', 'A3', 'A4', 'A5', 'F2a']:
        complete(group)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda cmd, **_: '' if '--query-compute-apps=gpu_uuid,pid' in cmd else f'GPU-{cmd[2]}')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    return module, runs, complete


def test_queue_all_slots_then_reuses_free_gpu_and_skips_complete(tmp_path, monkeypatch):
    module, _, complete = setup_queue(tmp_path, monkeypatch, ['--include-optional'])
    complete('B2')
    starts = []

    class Worker:
        pid = 123

        def __init__(self, cmd, **kwargs):
            self.group = cmd[-1]
            starts.append((self.group, kwargs['env']['CUDA_VISIBLE_DEVICES']))

        def poll(self):
            complete(self.group)
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Worker)
    module.main()
    assert [g for g, _ in starts] == module.REMAINING[1:] + ['C4']
    assert [gpu for _, gpu in starts[:4]] == ['0', '1', '2', '3']
    assert starts[4][1] == '0'


def test_failure_does_not_dispatch_more_or_stop_other_workers(tmp_path, monkeypatch):
    module, _, complete = setup_queue(tmp_path, monkeypatch)
    started, finished = [], []

    class Worker:
        pid = 456

        def __init__(self, cmd, **kwargs):
            self.group = cmd[-1]
            started.append(self.group)

        def poll(self):
            finished.append(self.group)
            if self.group == 'B2':
                return 1
            complete(self.group)
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Worker)
    with pytest.raises(RuntimeError, match='Failed='):
        module.main()
    assert started == module.REMAINING[:4]
    assert finished == started


def test_incomplete_output_never_silently_restarts(tmp_path, monkeypatch):
    module, runs, _ = setup_queue(tmp_path, monkeypatch, ['--dry-run'])
    out = runs / 'B2_seed0'
    out.mkdir()
    (out / 'last.pt').touch()
    with pytest.raises(RuntimeError, match='existing incomplete'):
        module.main()


def test_resume_requires_full_checkpoint(tmp_path, monkeypatch):
    module, runs, _ = setup_queue(tmp_path, monkeypatch, ['--dry-run', '--resume-incomplete'])
    out = runs / 'B2_seed0'
    (out / 'checkpoints/full_0000000100').mkdir(parents=True)
    module.main()


def test_dry_run_starts_no_process(tmp_path, monkeypatch):
    module, _, _ = setup_queue(tmp_path, monkeypatch, ['--dry-run'])

    def unexpected(*args, **kwargs):
        raise AssertionError('dry-run started a process')

    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    module.main()
