"""User-started motion-only M1/M2 training and frozen evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from uav_research.config import load_config
from uav_research.runtime import resolve_output

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/motion_only_sinr_topk_1m/manifest.json'


def resolve(group, device=None, total_steps=1_000_000):
    if total_steps not in (1_000_000, 10_000_000):
        raise ValueError('Use the frozen 1M or cumulative 10M configuration')
    manifest = MANIFEST if total_steps == 1_000_000 else ROOT/'configs/motion_only_sinr_topk_10m/manifest.json'
    entry = json.loads(manifest.read_text())['runs'][group]
    path = ROOT / entry['config']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Configuration changed; regenerate and review the manifest')
    cfg = load_config(path)
    if device:
        cfg.run.device = device
    cfg.run.cpu_threads = int(os.environ.get('UAV_TORCH_THREADS', cfg.run.cpu_threads))
    cfg.run.numba_threads = int(os.environ.get('UAV_NUMBA_THREADS', cfg.run.numba_threads))
    cfg.validate()
    from uav_research.system_ablation.settings import options
    o = options(cfg)
    if (o['run_id'] != group or o['initialization'] != 'random' or
            o['initial_checkpoint'] or o['score_actions'] or o['association'] != 'sinr_topk' or
            cfg.diffusion.pretrained_checkpoint or cfg.diffusion.teacher_checkpoint):
        raise ValueError('M1/M2 require motion-only random initialization and fixed SINR Top-K')
    return cfg, resolve_output(entry['output'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['check', 'train', 'report'])
    p.add_argument('--group', choices=['M1', 'M2'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume')
    p.add_argument('--auto-resume', action='store_true')
    p.add_argument('--output')
    p.add_argument('--total-steps', type=int, choices=[1_000_000,10_000_000], default=1_000_000)
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--action-seeds', nargs='+', type=int, default=[880001, 880002, 880003])
    a = p.parse_args()
    if a.phase == 'report':
        if a.resume or a.auto_resume:
            p.error('Report never resumes training')
        from uav_research.system_ablation.report import collect_run, rebuild
        out = resolve_output(a.output or f'analysis/motion_only_sinr_topk_{a.total_steps//1_000_000}m')
        if not a.rebuild:
            for group in ([a.group] if a.group else ['M1', 'M2']):
                _, run = resolve(group, total_steps=a.total_steps)
                collect_run(run, out/group, a.device, action_seeds=a.action_seeds)
        records = rebuild(out, comparisons_extra=[('M2', 'M1')],
                          groupings={'motion_actor': ['M1', 'M2']})
        # Keep generic numerical rebuild, replace historical A--F prose here.
        lines = ['# M1/M2：仅运动策略＋固定 SINR Top-K', '',
            f'M1为Gaussian PPO，M2为8步Diffusion DPPO；均从零起步，目标总预算{a.total_steps//1_000_000}M。10M路径从原1M完整状态续训，无示范或BC。',
            '关联为动作前SINR评分、Top10提议、运动后过滤、UE选择，不补位；奖励为覆盖＋0.1数据率。',
            '主要比较last；best按后100步覆盖选取，数值相等时按后100步吞吐。',
            '每场景先平均三个动作种子，再做2000次配对场景bootstrap；不是训练种子区间。', '',
            '| 组 | 模型 | 后100覆盖 | 后50覆盖 | 终态覆盖 | 后100吞吐Mbps | 后100覆盖σ pp |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in records:
            lines.append(f"| {r['run_id']} | {r['checkpoint']} | {r['last100_coverage']:.3%} | "
                f"{r['last50_coverage']:.3%} | {r['final_coverage']:.3%} | "
                f"{r['last100_throughput_mbps']:.3f} | {r['last100_coverage_std_pp']:.3f} |")
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(records, indent=2)); return
    if not a.group:
        p.error('--group required')
    if a.rebuild:
        p.error('--rebuild is report-only')
    if a.resume and a.auto_resume:
        p.error('Choose --resume OR --auto-resume')
    cfg, out = resolve(a.group, a.device, a.total_steps)
    if a.output:
        out = resolve_output(a.output)
    resume = a.resume
    if a.auto_resume:
        full = sorted((out/'checkpoints').glob('full_[0-9]*'))
        if not full:
            raise FileNotFoundError('No full checkpoint; will not start fresh')
        resume = str(full[-1])
    from uav_research.diffusion_dppo.storage import ensure_budget, load
    from uav_research.diffusion_dppo.runner import train_online
    # Lock is outside run directory; preflight never creates a formal run.
    lock = out.parent / ('.'+out.name+'.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_budget(cfg, out)
        if resume:
            saved = load(resume, cfg, {'dppo_full'})
            if saved['env_steps'] >= cfg.run.total_steps:
                raise ValueError('Checkpoint already reaches the requested budget')
        elif any((out/name).exists() for name in ('manifest.json','step0.pt','last.pt','best.pt')):
            raise FileExistsError(f'Existing run; explicitly resume: {out}')
        print(json.dumps(dict(group=a.group, output=str(out), steps=cfg.run.total_steps,
            initialization='random', score_policy_parameters=False,
            resume=resume, phase=a.phase), indent=2), flush=True)
        if a.phase == 'train':
            print(json.dumps(train_online(cfg, out, resume=resume), indent=2))


if __name__ == '__main__':
    main()
