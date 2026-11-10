"""Verify downloaded snapshot files and optionally detach archive hard links."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/20261110/results_manifest.json'))
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--detach-hardlinks', action='store_true',
                        help='Make duplicate archive members independent files before continuing training.')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    checked = []
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in {'runs', 'analysis'}:
            raise ValueError('Unexpected result path: ' + str(rel))
        path = root / rel
        if path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise ValueError('Result path escapes the chosen root: ' + str(rel))
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Result checksum mismatch: ' + str(rel))
        checked.append((path, row))
    detached = 0
    if args.detach_hardlinks:
        for path, row in checked:
            if row.get('archive_hardlink_to') and path.stat().st_nlink > 1:
                fd, temporary = tempfile.mkstemp(prefix='.snapshot-restore-', dir=path.parent)
                os.close(fd)
                temporary = Path(temporary)
                try:
                    shutil.copy2(path, temporary)
                    if digest(temporary) != row['sha256']:
                        raise ValueError('Copied result checksum mismatch: ' + row['path'])
                    os.replace(temporary, path)
                    detached += 1
                finally:
                    temporary.unlink(missing_ok=True)
    print(json.dumps({'verified_files': len(checked), 'detached_hardlinks': detached}))


if __name__ == '__main__':
    main()
