"""Copy, SHA256-verify, then relocate idle server artifacts with legacy links.

No training is started. Code, Python installations and tmux sockets stay put.
The default mode prints a plan; --execute performs the explicitly requested move.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(path):
    """Include directories, permissions and symlinks; never traverse a link."""
    result = {}
    paths = [path]
    if path.is_dir() and not path.is_symlink():
        for directory, dirs, files in os.walk(path, followlinks=False):
            paths.extend(Path(directory)/name for name in dirs+files)
    for p in sorted(paths):
        rel = str(p.relative_to(path))
        st = p.lstat()
        if p.is_symlink():
            result[rel] = dict(type='symlink', target=os.readlink(p))
        elif p.is_dir():
            result[rel] = dict(type='directory', mode=st.st_mode & 0o7777)
        elif p.is_file():
            result[rel] = dict(type='file', size=st.st_size, sha256=digest(p),
                               mode=st.st_mode & 0o7777, mtime_ns=st.st_mtime_ns)
        else:
            raise RuntimeError('Unexpected non-artifact file: '+str(p))
    return result


def ensure_idle(project, sources):
    """Refuse a live trainer/evaluator or any open file within moved artifacts."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if ('python' in cmd and ('-m uav_research' in cmd or
                    any(x in cmd for x in ['scripts/report_', 'scripts/evaluate_', 'scripts/diagnose_']))):
                raise RuntimeError('Training/evaluation still running: '+cmd)
            for fd in (proc/'fd').iterdir():
                try:
                    target = Path(os.readlink(fd))
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if any(target == p or p in target.parents for p in sources):
                    raise RuntimeError(f'Artifact still open by PID {proc.name}: {target}')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def save(path, state):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('/home/dym/uav_sac'))
    p.add_argument('--data-root', type=Path, default=Path('/mnt/disk2/dym/uav_sac'))
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    project, data = args.project_root.resolve(), args.data_root.resolve()
    if project != Path('/home/dym/uav_sac') or data != Path('/mnt/disk2/dym/uav_sac'):
        raise ValueError('This migration is restricted to the authorized server directories')
    if not Path('/mnt/disk2').is_mount() or not data.is_dir():
        raise RuntimeError('Data disk must already be mounted and /mnt/disk2/dym/uav_sac must exist')
    pairs = [(project/name, data/name) for name in ['runs','analysis','output','deployment','logs']]
    # Completed result packages are also artifacts, not the Python/uv cache.
    pairs += [(f, data/'.cache'/'legacy_artifacts'/f.name)
              for f in sorted((project/'.cache').glob('*.tar.gz'))]
    if (project/'.cache'/'tail100_sources').exists():
        pairs.append((project/'.cache'/'tail100_sources', data/'.cache'/'tail100_sources'))
    pairs = [(s,t) for s,t in pairs if s.exists() or s.is_symlink()]
    ensure_idle(project, [s for s,t in pairs])
    for source, target in pairs:
        if source.is_symlink() and source.resolve() == target:
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError('Refusing to merge an existing destination: '+str(target))
    print(json.dumps(dict(project=str(project), data=str(data), execute=args.execute,
                         moves=[dict(source=str(s),target=str(t)) for s,t in pairs]),
                     ensure_ascii=False), flush=True)
    if not args.execute:
        return
    receipt = data/'migration_20261111'
    receipt.mkdir(exist_ok=False)
    state = dict(status='in_progress', started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 project=str(project), data=str(data), operations=[],
                 before_disk=dict(system_free=shutil.disk_usage(project).free,
                                  data_free=shutil.disk_usage(data).free))
    save(receipt/'manifest.json',state)
    for number,(source,target) in enumerate(pairs):
        if source.is_symlink() and source.resolve() == target:
            continue
        print('Hashing source: '+str(source), flush=True)
        before = inventory(source)
        size = sum(x.get('size',0) for x in before.values())
        if shutil.disk_usage(data).free < size+15*2**30:
            raise OSError('Insufficient data disk budget, preserving original')
        op = dict(source=str(source),target=str(target),bytes=size,
                  files=sum(x['type']=='file' for x in before.values()),status='copying')
        state['operations'].append(op)
        save(receipt/'manifest.json',state)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():
            subprocess.run(['rsync','-aH','--',str(source)+'/',str(target)+'/'],check=True)
        else:
            subprocess.run(['rsync','-a','--',str(source),str(target)],check=True)
        print('Verifying both copies: '+str(target), flush=True)
        if inventory(source) != before or inventory(target) != before:
            raise RuntimeError('Source changed or copy verification failed; original retained')
        save(receipt/f'{number:02d}_{source.name}_files.json', before)
        ensure_idle(project,[s for s,t in pairs if not s.is_symlink()])
        backup = source.with_name(source.name+'.migration-verified-'+uuid.uuid4().hex)
        source.rename(backup)
        try:
            source.symlink_to(target,target_is_directory=target.is_dir())
        except BaseException:
            backup.rename(source)
            raise
        # Delete only the verified, uniquely named old copy, never the new data.
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
        op['status']='moved_verified_linked'
        save(receipt/'manifest.json',state)
        print(f'Moved {op["files"]} files ({size/2**30:.3f} GiB): {source} -> {target}',flush=True)
    state.update(status='complete', finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 after_disk=dict(system_free=shutil.disk_usage(project).free,
                                 data_free=shutil.disk_usage(data).free))
    state['total_bytes']=sum(op['bytes'] for op in state['operations'])
    state['total_files']=sum(op['files'] for op in state['operations'])
    save(receipt/'manifest.json',state)
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
