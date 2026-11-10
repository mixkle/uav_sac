#!/usr/bin/env python
"""Frozen-layout scoring/Top-K audit. No training or production env changes."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

from uav_research.config import load_config, config_from_dict
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import atomic_json, configure_runtime, isolated_rng
from uav_research.topk_diagnostic import audit_scene, collect, file_hash
from uav_research.topk_report import build_report

DEFAULT_CHECKPOINT = 'runs/continuous_mappo_coverage_shared_central_history_resumed_400k_seed0/best.pt'
DEFAULT_OUTPUT = 'analysis/topk_capacity_shared_best_diagnostic'
BASELINE = 'configs/mappo_full_reward_baseline_1m.yaml'


def parameter_hash(policy):
    h = hashlib.sha256()
    for name, value in policy.actor.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def code_hashes(baseline=BASELINE):
    paths = sorted(Path('src/uav_research').glob('*.py')) + [Path(__file__), Path(baseline)]
    return {str(path): file_hash(path) for path in paths}


def validate_environment(source, baseline):
    a, b = copy.deepcopy(source.as_dict()['env']), copy.deepcopy(baseline.as_dict()['env'])
    allowed = ('throughput_weight', 'robustness_weight', 'disconnection_weight')
    differences = {}
    for key in allowed:
        if a[key] != b[key]:
            differences[key] = dict(source=a[key], baseline=b[key])
        a.pop(key); b.pop(key)
    if a != b:
        keys = sorted(k for k in set(a)|set(b) if a.get(k) != b.get(k))
        raise ValueError(f'Source differs from baseline outside reward weights: {keys}')
    if not (source.algo.name == 'mappo' and source.algo.actor_sharing == 'shared'
            and source.env.action_space == 'continuous'
            and source.env.observation_history == 'one_step'
            and source.env.reward_primary == 'coverage'
            and source.run.evaluation_policy == 'sample'):
        raise ValueError('Expected shared continuous coverage-primary PPO with history and sampled evaluation')
    return differences


def reward_weights(cfg):
    return [1., cfg.env.throughput_weight, cfg.env.robustness_weight, -cfg.env.disconnection_weight]


def check_source_coverage(out, provenance):
    seeds = provenance['settings']['scenes']
    steps = provenance['settings']['steps']
    means, returns = [], []
    for seed in seeds:
        with np.load(out/'trajectories'/f'scene_{seed}.npz') as d:
            if len(d['actions']) != steps or int(d['scene_seed']) != seed:
                raise ValueError('Trajectory scene or length mismatch')
            means.append(float(d['source_metrics'][:,0].mean()))
            returns.append(float(d['source_metrics'][:,1].sum()))
    mean = float(np.mean(means))
    full = seeds == list(range(10000,10032)) and steps == 400
    expected = provenance['expected_source_coverage']
    if full and abs(mean-expected)>1e-6:
        raise AssertionError(f'Source coverage mismatch: {mean:.12f} vs {expected:.12f}')
    expected_return = provenance.get('expected_source_return')
    if full and expected_return is not None:
        np.testing.assert_allclose(np.mean(returns), expected_return, rtol=1e-9, atol=1e-9)
    return dict(coverage=mean, expected=expected, tolerance=1e-6,
                episode_return=float(np.mean(returns)), expected_return=expected_return,
                full_protocol=full, passed=bool(abs(mean-expected)<=1e-6) if full else None)


def reaudit_recorded(args):
    """Counterfactual capacity on existing positions; never construct an actor."""
    reference=Path(args.reuse_trajectories).resolve()
    original=json.loads((reference/'provenance.json').read_text())
    completed=json.loads((reference/'execution.json').read_text())
    if completed['status']!='completed':
        raise ValueError('Reuse requires a completed source diagnostic')
    if args.phase=='collect':
        raise ValueError('Reusing fixed trajectories cannot collect new trajectories')
    audit_path=Path(args.audit_config)
    cfg=load_config(audit_path)
    baseline=load_config(args.baseline)
    target=copy.deepcopy(cfg.as_dict())
    target['env']['capacity']=baseline.env.capacity
    if target!=baseline.as_dict():
        raise ValueError('Audit config may differ from baseline only in env.capacity')
    source_cfg=config_from_dict(original['source_config'])
    validate_environment(source_cfg,baseline)
    settings=copy.deepcopy(original['settings'])
    if (args.scene_count!=len(settings['scenes']) or args.steps!=settings['steps']
            or args.random_repeats!=settings['random_repeats']):
        raise ValueError('Scene count, steps and random repeats must match source diagnostic')
    source_check=check_source_coverage(reference,original)
    configure_runtime(cfg)
    out=Path(args.output)
    if out.resolve()==reference:
        raise ValueError('Use a separate output directory for capacity re-audit')
    checkpoint=Path(original['checkpoint'])
    if file_hash(checkpoint)!=original['checkpoint_sha256']:
        raise ValueError('Source checkpoint hash mismatch')
    hashes=code_hashes(args.baseline)
    hashes[str(audit_path)]=file_hash(audit_path)
    provenance=dict(original, audit_config=cfg.as_dict(),source_hashes=hashes,
                    audit_reward_weights=reward_weights(cfg),
                    trajectory_source=str(reference),comparison_reference=str(reference),
                    parent_provenance_sha256=file_hash(reference/'provenance.json'),
                    intervention='capacity counterfactual on existing capacity-20 teacher positions; no actor execution',
                    trajectory_unchanged=True)
    if out.exists():
        if not args.resume:
            raise FileExistsError(f'{out} exists; use --resume or --rebuild')
        saved=json.loads((out/'provenance.json').read_text())
        for key in ('source_hashes','audit_config','settings','trajectory_source','parent_provenance_sha256'):
            if saved[key]!=provenance[key]:
                raise ValueError(f'Re-audit resume mismatch: {key}')
    else:
        out.mkdir(parents=True)
        atomic_json(out/'provenance.json',provenance)
    execution=dict(status='running',trajectory_resampled=False,actor_executed=False,
                   source_validation=source_check,collection_seconds=0.,
                   trajectory_capacity=source_cfg.env.capacity,audit_capacity=cfg.env.capacity)
    before_files={str(p):file_hash(p) for folder in ('trajectories','audit')
                  for p in sorted((reference/folder).glob('scene_*.npz'))}
    started=time.perf_counter()
    audit_dir=out/'audit';audit_dir.mkdir(exist_ok=True)
    try:
        for i,scene in enumerate(settings['scenes']):
            path=reference/'trajectories'/f'scene_{scene}.npz'
            target=audit_dir/f'scene_{scene}.npz'
            with np.load(reference/'audit'/f'scene_{scene}.npz') as previous:
                if str(previous['trajectory_sha256'])!=before_files[str(path)]:
                    raise ValueError(f'Trajectory hash differs from source audit: {scene}')
            reuse=False
            if args.resume and target.exists():
                with np.load(target) as d:
                    reuse=(str(d['trajectory_sha256'])==before_files[str(path)]
                           and int(d['audit_capacity'])==cfg.env.capacity
                           and int(d['trajectory_capacity'])==source_cfg.env.capacity
                           and int(d['random_repeats'])==settings['random_repeats'])
                if not reuse:
                    raise ValueError(f'Stale capacity audit: {target}')
            if not reuse:
                audit_scene(path,target,cfg,settings['random_repeats'],
                            settings['random_score_seed'],trajectory_capacity=source_cfg.env.capacity,
                            reward_weights=reward_weights(cfg))
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
            if peak>12:
                raise MemoryError('Capacity re-audit exceeded 12 GiB RSS')
            execution.update(audited_scenes=i+1,peak_rss_gib=peak)
            atomic_json(out/'execution.json',execution)
            print(f'Capacity {cfg.env.capacity}: scene {scene} ({i+1}/{len(settings["scenes"])}), '
                  f'{time.perf_counter()-started:.1f}s, RSS peak={peak:.3f} GiB',flush=True)
        execution['audit_seconds']=time.perf_counter()-started
        stamp=time.perf_counter()
        build_report(out)
        execution['report_seconds']=time.perf_counter()-stamp
        if any(file_hash(p)!=h for p,h in before_files.items()):
            raise AssertionError('Source trajectories or audits changed')
        after_hashes=code_hashes(args.baseline);after_hashes[str(audit_path)]=file_hash(audit_path)
        if hashes!=after_hashes or file_hash(checkpoint)!=original['checkpoint_sha256']:
            raise AssertionError('Source code/config or checkpoint changed')
        execution.update(status='completed',source_trajectories_unchanged=True,
                         source_audits_unchanged=True,checkpoint_unchanged=True,
                         session_seconds=time.perf_counter()-started,
                         peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    except BaseException as error:
        execution.update(status='failed',error=repr(error))
        raise
    finally:
        atomic_json(out/'execution.json',execution)
    print(f'Completed capacity re-audit: {out}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--baseline', default=BASELINE, help='Physical baseline and shadow reward configuration')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=('all','collect','audit','report'), default='all')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild from saved audits only; no model loading')
    parser.add_argument('--resume', action='store_true', help='Reuse complete trajectories and verified scene audits')
    parser.add_argument('--scene-count', type=int, default=32)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--reuse-trajectories', help='Completed original diagnostic directory; no actor execution')
    parser.add_argument('--audit-config', default=BASELINE, help='Capacity counterfactual config, only with --reuse-trajectories')
    args = parser.parse_args()
    out = Path(args.output)
    if args.rebuild or args.phase == 'report':
        result = build_report(out)
        print(f'Rebuilt report: {out / "report.md"}; source coverage={result["source_coverage"]:.9f}', flush=True)
        return
    if not 1<=args.scene_count<=32 or not 1<=args.steps<=400 or args.random_repeats<1:
        parser.error('Require 1..32 scenes, 1..400 steps and at least one random repeat')
    if args.reuse_trajectories:
        reaudit_recorded(args)
        return
    settings = dict(scenes=list(range(10000,10000+args.scene_count)),steps=args.steps,
                    random_repeats=args.random_repeats, random_score_seed=970001,
                    action_seed=880001, batch_size=16, cpu_threads=8, numba_threads=4)
    began = time.perf_counter()
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = file_hash(checkpoint)
    sources = code_hashes(args.baseline)
    with isolated_rng():
        cfg,policy = policy_from_checkpoint(checkpoint,'cpu')
    baseline = load_config(args.baseline)
    differences = validate_environment(cfg,baseline)
    cfg.run.device='cpu'; cfg.run.cpu_threads=8; cfg.run.numba_threads=4
    cfg.run.num_envs=16; cfg.run.evaluation_action_seed=880001
    configure_runtime(cfg)
    policy.actor.eval().requires_grad_(False)
    policy.value.eval().requires_grad_(False)
    actor_before = parameter_hash(policy)
    cp_step = policy.provenance['checkpoint_env_steps']
    selection = json.loads((checkpoint.parent/'best_selection.json').read_text())
    if selection['env_steps']!=cp_step:
        raise ValueError('Checkpoint is not the selected best of the source run')
    with (checkpoint.parent/'validation.csv').open() as stream:
        validation=list(csv.DictReader(stream))
    matching=[r for r in validation if int(r['env_steps'])==cp_step]
    if len(matching)!=1:
        raise ValueError('Expected exactly one validation record for selected checkpoint')
    selected=matching[0]
    metric=selection.get('selection_metric','return')
    np.testing.assert_allclose(float(selected[metric]),max(float(r[metric]) for r in validation))
    np.testing.assert_allclose(float(selected['return']),selection['validation_return'])
    expected = float(selected['coverage'])
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['env_steps']!=1000000:
        raise ValueError('Expected a completed 1M source run')
    provenance = dict(schema_version=1, checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
                      checkpoint_env_steps=cp_step, source_run_env_steps=manifest['env_steps'],
                      source_config=cfg.as_dict(), baseline_config=baseline.as_dict(),
                      environment_differences=differences, settings=settings,
                      expected_source_coverage=expected, source_hashes=sources,
                      expected_source_return=float(selected['return']),
                      audit_reward_weights=reward_weights(baseline),
                      baseline_path=str(args.baseline),
                      source_training_reward_weights=reward_weights(cfg),
                      teacher_actor_sha256=actor_before,
                      random_generator='NumPy PCG64(SeedSequence([970001,scene,pre_action_t,repeat]))',
                      environment=dict(python=sys.version,platform=platform.platform(),
                                       numpy=np.__version__,torch=torch.__version__),
                      intervention='read-only shadow association on original greedy teacher trajectories')
    for name in ('config.yaml','validation.csv','best_selection.json','manifest.json'):
        sources[str(checkpoint.parent/name)]=file_hash(checkpoint.parent/name)
    if out.exists():
        if not args.resume:
            raise FileExistsError(f'{out} exists; use --resume or --rebuild')
        old=json.loads((out/'provenance.json').read_text())
        for key in ('checkpoint_sha256','source_hashes','settings','source_config','audit_reward_weights','baseline_config'):
            if old[key]!=provenance[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        provenance=old
    else:
        out.mkdir(parents=True)
        atomic_json(out/'provenance.json',provenance)
    execution=json.loads((out/'execution.json').read_text()) if (out/'execution.json').exists() else {}
    try:
        trajectory_paths=[out/'trajectories'/f'scene_{s}.npz' for s in settings['scenes']]
        if args.phase in ('all','collect') and not all(p.exists() for p in trajectory_paths):
            print(f'Collecting {len(settings["scenes"])} scenes × {args.steps} steps; frozen actor step={cp_step}',flush=True)
            stamp=time.perf_counter()
            collect(cfg,policy,out,settings['scenes'],args.steps)
            execution['collection_seconds']=time.perf_counter()-stamp
        coverage=check_source_coverage(out,provenance)
        execution['source_validation']=coverage
        print(f'Source coverage={coverage["coverage"]*100:.8f}%; full validation={coverage["full_protocol"]}',flush=True)
        if args.phase in ('all','audit'):
            audit_dir=out/'audit';audit_dir.mkdir(exist_ok=True)
            stamp=time.perf_counter()
            for i,(scene,path) in enumerate(zip(settings['scenes'],trajectory_paths)):
                target=audit_dir/f'scene_{scene}.npz'
                reuse=False
                if args.resume and target.exists():
                    with np.load(target) as d:
                        reuse=(str(d['trajectory_sha256'])==file_hash(path)
                               and int(d['random_repeats'])==args.random_repeats
                               and int(d['random_seed'])==970001
                               and np.array_equal(d['reward_weights'],provenance['audit_reward_weights']))
                    if not reuse:
                        raise ValueError(f'Stale saved audit: {target}')
                if not reuse:
                    audit_scene(path,target,cfg,args.random_repeats,
                                reward_weights=provenance['audit_reward_weights'])
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
                if peak>12:
                    raise MemoryError(f'RSS peak exceeded 12 GiB: {peak:.3f}')
                execution.update(audited_scenes=i+1,peak_rss_gib=peak)
                atomic_json(out/'execution.json',execution)
                print(f'Audited scene {scene} ({i+1}/{len(trajectory_paths)}), '
                      f'{time.perf_counter()-stamp:.1f}s, peak RSS={peak:.3f} GiB'+(' [reused]' if reuse else ''),flush=True)
            execution['audit_session_seconds']=time.perf_counter()-stamp
            stamp=time.perf_counter()
            build_report(out)
            execution['report_seconds']=time.perf_counter()-stamp
        if parameter_hash(policy)!=actor_before or file_hash(checkpoint)!=checkpoint_hash:
            raise AssertionError('Teacher actor or checkpoint changed')
        after_sources=code_hashes(args.baseline)
        for name in ('config.yaml','validation.csv','best_selection.json','manifest.json'):
            after_sources[str(checkpoint.parent/name)]=file_hash(checkpoint.parent/name)
        if after_sources!=sources:
            raise AssertionError('Source files changed during experiment')
        if any(p.grad is not None for p in policy.actor.parameters()):
            raise AssertionError('Frozen actor unexpectedly acquired gradients')
        execution.update(status='completed' if args.phase!='collect' else 'collected',
                         checkpoint_unchanged=True, actor_unchanged=True, source_unchanged=True,
                         peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                         session_seconds=time.perf_counter()-began,
                         output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    except BaseException as error:
        execution.update(status='failed',error=repr(error),session_seconds=time.perf_counter()-began)
        raise
    finally:
        atomic_json(out/'execution.json',execution)
    print(f'Completed: {out}',flush=True)


if __name__=='__main__':
    main()
