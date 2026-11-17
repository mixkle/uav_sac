"""User-started M2 controls: fresh 1M, automatic full continuation to 10M."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path

from uav_research.config import load_config
from uav_research.runtime import resolve_output
from uav_research.diffusion_dppo.config import contract
from uav_research.diffusion_dppo.storage import ensure_budget, load
from uav_research.diffusion_dppo.continuation import prepare_continuation
from resume_motion_only_10m import select_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'configs/m2_motion_neighbor_controls_10m/manifest.json'
GROUPS = ('M2-NoHop','M2-N5','M2-N7','M2-N9')


def resolve(group, device='cpu'):
    meta = json.loads(MANIFEST.read_text())
    entry = meta['runs'][group]
    source = ROOT/meta['reference_config']
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta['reference_sha256']:
        raise ValueError('Reference M2 config changed; review and rebuild this suite')
    configs = []
    for prefix in ('stage1_',''):
        path = ROOT/entry[prefix+'config']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry[prefix+'sha256']:
            raise ValueError('Control configuration hash changed')
        cfg = load_config(path)
        cfg.run.device = device
        cfg.run.cpu_threads = int(os.environ.get('UAV_TORCH_THREADS',cfg.run.cpu_threads))
        cfg.run.numba_threads = int(os.environ.get('UAV_NUMBA_THREADS',cfg.run.numba_threads))
        cfg.validate()
        configs.append(cfg)
    first, final = configs
    if first.run.total_steps != 1_000_000 or final.run.total_steps != 10_000_000 or contract(first) != contract(final):
        raise ValueError('Expected frozen 1M -> cumulative 10M schedule')
    return first, final, resolve_output(entry['stage1_output']), resolve_output(entry['output'])


def latest_full(cfg, output):
    """Refuse dirty log rollback instead of silently repeating updates."""
    candidates = sorted(p for p in (output/'checkpoints').glob('full_[0-9]*')
                        if (p/'index.json').is_file() and (p/'metadata.pt').is_file())
    if not candidates:
        raise FileNotFoundError(f'No complete checkpoint in existing run: {output}')
    checkpoint = candidates[-1]
    state = load(checkpoint, cfg, {'dppo_full'})
    for name in ('metrics.csv','validation.csv'):
        path = output/name
        if path.exists():
            with path.open() as stream:
                if any(int(r['env_steps']) > state['env_steps'] for r in csv.DictReader(stream)):
                    raise ValueError(f'{path} is newer than the checkpoint; refusing rollback')
    return checkpoint, state


def preflight(group, device='cpu', resume=False):
    first, final, source, target = resolve(group, device)
    ensure_budget(final, target)
    if not resume and (source.exists() or target.exists()):
        raise FileExistsError('Experiment output already exists; use --auto-resume explicitly')
    result = dict(group=group, output=str(target), stage1_output=str(source),
        stage1_steps=first.run.total_steps,total_steps=final.run.total_steps,
        neighbor_slots=final.env.neighbor_slots,fixed_observation_dim=final.env.fixed_obs_dim,
        critic_state_dim=final.env.state_dim,motion_rule=final.diffusion.ablation['motion_rule'],
        source_steps=0, resume_checkpoint=None, formal_training_started=False)
    if resume:
        if not source.exists():
            raise FileNotFoundError('No stage1 run; resume never starts a fresh experiment')
        checkpoint, state = latest_full(first, source)
        if state['env_steps'] > first.run.total_steps:
            raise ValueError('Stage1 exceeds its budget')
        if state['env_steps'] == first.run.total_steps:
            del state
            checkpoint, state = select_checkpoint(final, source, target,first.run.total_steps)
        elif target.exists():
            raise ValueError('Final stage exists but stage1 is incomplete')
        result.update(source_steps=state['env_steps'],resume_checkpoint=str(checkpoint))
    return result


def run(group, device='cpu', resume=False):
    first, final, source, target = resolve(group, device)
    lock_path = resolve_output(f'.locks/m2_controls_10m_{group}.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(preflight(group,device,resume),indent=2),flush=True)
        from uav_research.diffusion_dppo.runner import train_online
        checkpoint = None
        state = None
        if source.exists():
            checkpoint, state = latest_full(first,source)
        if state is None or state['env_steps'] < first.run.total_steps:
            del state
            print(f'{group}: stage1 start/resume, fresh origin -> 1M',flush=True)
            result = train_online(first,source,resume=checkpoint)
            if result['status'] != 'complete':
                print('Stage1 interrupted safely; NOT starting stage2.',flush=True)
                return result
        else:
            del state
        checkpoint, state = select_checkpoint(final,source,target,first.run.total_steps)
        del state
        if not (target/'continuation.json').exists():
            prepare_continuation(final,checkpoint,target)
        print(f'{group}: full-state continuation -> cumulative 10M',flush=True)
        return train_online(final,target,resume=checkpoint)


def report(args):
    from uav_research.system_ablation.report import collect_run,rebuild
    meta = json.loads(MANIFEST.read_text())
    out = resolve_output(args.output or 'analysis/m2_motion_neighbor_controls_10m')
    groups = args.groups or ['M2',*GROUPS]
    if not args.rebuild:
        for group in groups:
            if group == 'M2':
                path = resolve_output(meta['reference_output'])
            else:
                _,_,_,path = resolve(group,args.device)
            status = json.loads((path/'manifest.json').read_text())
            if status['env_steps'] != 10_000_000 or status['status'] != 'complete':
                raise ValueError(f'{group}: formal 10M not complete; no partial-budget comparison')
            collect_run(path,out/group,args.device,action_seeds=args.action_seeds)
    records = rebuild(out,comparisons_extra=[(g,'M2') for g in GROUPS],
        groupings={'motion_scaling':['M2','M2-NoHop'],
                   'uav_neighbors':['M2','M2-N5','M2-N7','M2-N9']})
    lines=['# M2：运动缩放与可见邻居数10M对照','',
        'M2为已完成的3邻居＋指数跳数缩放参考；NoHop仅关闭缩放，N5/N7/N9仅增加邻居槽位。',
        '均为仅运动Diffusion DPPO、规则SINR Top-K、无BC、种子0；新增四组从零开始1M，再完整恢复至10M，与参考阶段边界相同。',
        '主要比较同预算last，best保留原选模；先在场景内平均动作重复，再做2000次配对场景bootstrap，不是训练种子区间。','',
        '| 组 | 模型 | 实际步数 | 后100覆盖 | 后50覆盖 | 终态覆盖 | 后100吞吐Mbps | 后100σ pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['run_id']} | {row['checkpoint']} | {row['online_steps']} | "
            f"{row['last100_coverage']:.3%} | {row['last50_coverage']:.3%} | {row['final_coverage']:.3%} | "
            f"{row['last100_throughput_mbps']:.3f} | {row['last100_coverage_std_pp']:.3f} |")
    lines += ['', '固定场景10000/10001展示0/150/200/400步拓扑。CSV包含联网比例、骤降、最大跌幅、P05、回报、critic MC误差和动作行为。',
        '邻居增加不改变通信半径；不足槽位补零。输入宽度和actor参数随槽位数增加，不能把差异唯一归因于信息数量。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {out}/report.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['check','train','report'])
    p.add_argument('--group',choices=GROUPS)
    p.add_argument('--device',default='cpu')
    p.add_argument('--auto-resume',action='store_true')
    p.add_argument('--groups',nargs='+',choices=['M2',*GROUPS])
    p.add_argument('--output')
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--action-seeds',nargs='+',type=int,default=[880001,880002,880003])
    args=p.parse_args()
    if args.phase=='report':
        if args.auto_resume or args.group:
            p.error('Report uses --groups and never resumes training')
        report(args)
        return
    if not args.group or args.groups or args.rebuild or args.output:
        p.error('Training/check require --group and do not accept report-only arguments')
    if args.phase=='check':
        print(json.dumps(preflight(args.group,args.device,args.auto_resume),indent=2))
    else:
        print(json.dumps(run(args.group,args.device,args.auto_resume),indent=2))


if __name__=='__main__':
    main()
