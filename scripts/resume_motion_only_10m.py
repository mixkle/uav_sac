"""Preserve 1M runs; continue complete M1/M2 states to a cumulative 10M."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

from motion_only_comparison import resolve
from uav_research.diffusion_dppo.continuation import prepare_continuation
from uav_research.diffusion_dppo.storage import ensure_budget,load
from uav_research.runtime import resolve_output


def select_checkpoint(cfg, original, target, source_steps=1_000_000):
    """Never silently restart, fall back behind published logs, or load best.pt."""
    original,target=Path(original),Path(target)
    candidates=sorted(p for p in (target/'checkpoints').glob('full_[0-9]*')
                      if (p/'index.json').is_file() and (p/'metadata.pt').is_file())
    if candidates:
        if not (target/'continuation.json').is_file():
            raise ValueError('Existing target checkpoint lacks continuation provenance')
        marker=json.loads((target/'continuation.json').read_text())
        if Path(marker['source_run']).resolve()!=original.resolve() or marker['source_steps']!=source_steps:
            raise ValueError('Target belongs to another source run')
        checkpoint=candidates[-1]
    else:
        checkpoint=original/f'checkpoints/full_{source_steps:09d}'
        if target.exists():
            marker=target/'continuation.json'
            if not marker.is_file():
                raise FileExistsError('Target already exists without continuation provenance')
            saved_marker=json.loads(marker.read_text())
            if Path(saved_marker['source_checkpoint']).resolve()!=checkpoint.resolve():
                raise ValueError('Target belongs to another source checkpoint')
    state=load(checkpoint,cfg,{'dppo_full'})
    if checkpoint.parent.parent.resolve()==original.resolve() and state['env_steps']!=source_steps:
        raise ValueError('Original final checkpoint must match the completed source budget')
    if state['env_steps']>=cfg.run.total_steps:
        raise ValueError('Target budget already completed')
    if state['num_envs']!=cfg.run.num_envs:
        raise ValueError('Environment count changed')
    # Catch partial updates written after the last resumable safe point.
    for name in ('metrics.csv','validation.csv'):
        if (target/name).exists():
            with (target/name).open(newline='') as stream:
                if any(int(row['env_steps'])>state['env_steps'] for row in csv.DictReader(stream)):
                    raise ValueError('Target logs are newer than the full checkpoint; refusing rollback')
    return checkpoint,state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['M1','M2'],required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    cfg,target=resolve(a.group,a.device,10_000_000)
    original=resolve_output(f'runs/motion_only_sinr_topk_1m/{a.group}_seed0')
    target.parent.mkdir(parents=True,exist_ok=True)
    with (target.parent/('.'+target.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ensure_budget(cfg,target)
        checkpoint,state=select_checkpoint(cfg,original,target)
        result=dict(group=a.group,checkpoint=str(checkpoint),source_steps=state['env_steps'],
            target_steps=cfg.run.total_steps,additional_steps=cfg.run.total_steps-state['env_steps'],
            source_updates=state['learner']['updates'],best_step=state['best_step'],
            output=str(target),original_1m_preserved=True,check_only=a.check)
        print(json.dumps(result,indent=2),flush=True)
        del state
        if a.check:return
        if not (target/'continuation.json').exists():
            prepare_continuation(cfg,checkpoint,target)
        from uav_research.diffusion_dppo.runner import train_online
        print(json.dumps(train_online(cfg,target,resume=checkpoint),indent=2))


if __name__=='__main__':main()
