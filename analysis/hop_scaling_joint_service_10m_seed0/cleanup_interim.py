"""Remove only the two superseded evaluation folders after final-data checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--location', choices=['local', 'server'], required=True)
    a = p.parse_args()
    base = a.base.resolve(strict=True)
    expected = Path('/home/mixkle/uav_sac' if a.location == 'local' else '/home/dym/uav_sac')
    assert base == expected, base
    final = base / 'analysis/hop_scaling_joint_service_10m_seed0'
    assert final.is_dir() and not final.is_symlink()
    assert '## 结果解读' in (final / 'report.md').read_text()
    audit = json.loads((final / 'validation_audit.json').read_text())
    assert len(audit) == 6 and all(x['status'] == 'passed' for x in audit.values())
    for marker in ['diffusion_last', 'diffusion_best', 'diffusion_step0', 'gaussian_last']:
        assert (final / ('complete_' + marker + '.json')).is_file()
    for rel, wanted in json.loads((final / 'raw_data_hashes.json').read_text()).items():
        target = final / rel
        assert final in target.resolve().parents and digest(target) == wanted, target
    original_hashes = {}
    if a.location == 'server':
        manifest = json.loads((final / 'snapshot_manifest.json').read_text())
        for item in manifest['files'].values():
            src = Path(item['source'])
            assert base in src.resolve().parents
            original_hashes[str(src)] = digest(src)
            assert original_hashes[str(src)] == item['sha256'], src

    names = ['hop_scaling_interim_20261109', 'hop_scaling_interim_20261109_2121']
    targets = [base / 'analysis' / name for name in names]
    if a.location == 'local':
        targets += [base / '.cache' / (stem + '.py') for stem in [
            'evaluate_hop_interim_20261109', 'build_hop_interim_20261109',
            'evaluate_hop_interim_20261109_2121', 'build_hop_interim_20261109_2121']]
    entries = []
    for target in targets:
        assert not target.is_symlink(), target
        assert target.parent.resolve() in [base / 'analysis', base / '.cache']
        if not target.exists():
            entries.append(dict(path=str(target), status='already_absent', files=0, bytes=0))
            continue
        assert target.resolve() != final
        files = [x for x in target.rglob('*') if x.is_file()] if target.is_dir() else [target]
        assert not any(x.is_symlink() for x in target.rglob('*')) if target.is_dir() else True
        entry = dict(path=str(target), status='pending', files=len(files),
                     bytes=sum(x.stat().st_size for x in files))
        entries.append(entry)
    record = dict(location=a.location, checked_at=datetime.now(timezone.utc).isoformat(),
                  final_raw_hashes='raw_data_hashes.json',
                  retained=['training runs and checkpoints', str(final)], entries=entries)
    output = final / ('cleanup_' + a.location + '.json')
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for target, entry in zip(targets, entries):
        if entry['status'] == 'pending':
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            assert not target.exists()
            entry['status'] = 'deleted'
            output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for src, wanted in original_hashes.items():
        assert digest(Path(src)) == wanted, src
    record['retained_server_training_sources_unchanged'] = len(original_hashes)
    record['completed_at'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
