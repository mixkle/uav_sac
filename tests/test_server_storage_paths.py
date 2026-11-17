from pathlib import Path
import pytest
from uav_research import runtime


def test_bare_server_cli_defaults_to_disk2(monkeypatch):
    monkeypatch.delenv('UAV_DATA_ROOT', raising=False)
    monkeypatch.setattr(Path, 'is_mount', lambda self: self == Path('/mnt/disk2'))
    assert runtime._default_data_root('/home/dym/uav_sac') == '/mnt/disk2/dym/uav_sac'
    monkeypatch.setattr(runtime, '_default_data_root', lambda: '/mnt/disk2/dym/uav_sac')
    assert runtime.resolve_output('runs/new_run') == Path('/mnt/disk2/dym/uav_sac/runs/new_run')
    assert runtime.resolve_output('analysis/validation') == Path('/mnt/disk2/dym/uav_sac/analysis/validation')


def test_missing_server_mount_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(Path, 'is_mount', lambda self: False)
    with pytest.raises(RuntimeError, match='not mounted'):
        runtime._default_data_root('/home/dym/uav_sac')


def test_local_output_and_explicit_paths_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv('UAV_DATA_ROOT', raising=False)
    monkeypatch.setattr(runtime, '_default_data_root', lambda: None)
    assert runtime.resolve_output('runs/probe') == Path('runs/probe')
    monkeypatch.setenv('UAV_DATA_ROOT', str(tmp_path/'data'))
    assert runtime.resolve_output('runs/probe') == tmp_path/'data/runs/probe'
    assert runtime.resolve_output(tmp_path/'explicit') == tmp_path/'explicit'
