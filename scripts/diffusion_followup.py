"""Manual stages for NEW F2b/F2c/A1b. Old system_ablation CLI stays unchanged."""
import argparse
import json
from pathlib import Path
from uav_research.runtime import resolve_output
from uav_research.diffusion_followup.runtime import resolve,check_sources,require_initialization_gate,save_scale_milestone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['check','ready','audit-init','audit-coupling','frozen','pretrain','train'])
    parser.add_argument('--group',choices=['F2b','F2c','A1b'],default='F2b')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--auto-resume',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Audits only: small non-qualifying acceptance output')
    a=parser.parse_args()
    cfg,out,demos=resolve(a.group,a.device)
    data=resolve_output('runs/diffusion_followup_1m')
    audit=data/'audits'
    if a.smoke and a.phase not in ('audit-init','frozen'):parser.error('--smoke is for audits, never formal training')
    if a.resume and a.auto_resume:parser.error('Choose --resume or --auto-resume')
    check_sources(cfg)
    if a.phase=='check':
        from uav_research.diffusion_dppo.storage import ensure_budget
        ensure_budget(cfg,out)
        print(json.dumps(dict(group=a.group,config='configs/diffusion_followup_1m/'+a.group+'.yaml',
            output=str(out),steps=cfg.run.total_steps,source_check='passed'),indent=2));return
    if a.phase=='audit-init':
        from uav_research.diffusion_followup.audits import initialization_audit
        result=initialization_audit(a.output or audit/('A1b_smoke' if a.smoke else 'A1b_initialization'),
            a.device,repeats=2 if a.smoke else 64,independent=2 if a.smoke else 128)
        print(json.dumps(result,indent=2))
        if not a.smoke and not result['passed']:raise RuntimeError('Initialization gate FAILED; no training allowed')
        return
    if a.phase=='audit-coupling':
        from uav_research.diffusion_followup.audits import coupling_audit
        print(json.dumps(coupling_audit(a.output or audit/'F2c_coupling',a.device),indent=2));return
    if a.phase=='frozen':
        from uav_research.diffusion_followup.frozen import frozen_swap
        result=frozen_swap(a.output or audit/('F2b_frozen_smoke' if a.smoke else 'F2b_frozen'),a.device,
            seeds=[10000,10001] if a.smoke else None,
            action_seeds=[880001] if a.smoke else [880001,880002,880003])
        print(json.dumps(result,indent=2));return
    if a.phase=='pretrain':
        if a.group!='F2c':parser.error('Only F2c needs new BC; F2b copies A3 step0 and A1b has no BC')
        from uav_research.diffusion_dppo.data import pretrain
        from uav_research.diffusion_dppo.storage import load
        path=Path(a.output) if a.output else data/'pretraining/F2c'
        resume=a.resume
        if a.auto_resume and (path/'summary.json').exists():
            s=json.loads((path/'summary.json').read_text())
            if s.get('status')=='complete' and s.get('epoch')==50:
                load(path/'best.pt',cfg,{'dppo_pretrain'})
                print('F2c BC already complete and compatible');return
        if a.auto_resume and (path/'last.pt').exists():resume=str(path/'last.pt')
        print(json.dumps(pretrain(cfg,demos,path,resume),indent=2));return
    from uav_research.diffusion_dppo.runner import train_online
    from uav_research.system_ablation.initialization import initialize
    from uav_research.diffusion_dppo.storage import load
    from uav_research.diffusion_dppo.learner import DiffusionLearner
    from uav_research.runtime import seed_all
    if a.group=='A1b':require_initialization_gate(cfg)
    elif a.group=='F2b':
        frozen=json.loads((audit/'F2b_frozen/complete.json').read_text())
        if (not frozen.get('full') or frozen['scenes']!=cfg.run.validation_seeds or
                frozen['action_seeds']!=[880001,880002,880003]):
            raise ValueError('Complete full frozen A3 replay before F2b training')
    else:
        if not json.loads((audit/'F2c_coupling/summary.json').read_text()).get('passed'):
            raise ValueError('Complete F2c sensitivity audit first')
        summary=json.loads((data/'pretraining/F2c/summary.json').read_text())
        if summary.get('status')!='complete' or summary.get('epoch')!=50:raise ValueError('Finish F2c 50-epoch BC')
        from uav_research.topk_diagnostic import file_hash
        if summary['provenance']['dataset_sha256']!=file_hash(demos/'dataset.json'):
            raise ValueError('F2c BC did not use the frozen original A3 dataset')
    resume=a.resume
    if a.auto_resume:
        full=sorted((out/'checkpoints').glob('full_[0-9]*'))
        if not full:raise FileNotFoundError('No full checkpoint; never silently restart online training')
        resume=str(full[-1])
    if resume:load(resume,cfg,{'dppo_full'})
    else:
        # Validation only; train_online repeats seed0 and constructs fresh Adam/critic.
        seed_all(cfg.run.seed);check=DiffusionLearner(cfg,'cpu');initialize(check);del check
    if a.phase=='ready':
        print(json.dumps(dict(group=a.group,readiness='all preparation artifacts compatible',training_started=False)))
        return
    print(json.dumps(train_online(cfg,Path(a.output) if a.output else out,resume=resume,
        observer=save_scale_milestone if a.group=='A1b' else None),indent=2))


if __name__=='__main__':main()
