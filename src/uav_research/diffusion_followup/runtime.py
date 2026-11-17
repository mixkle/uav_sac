import hashlib
import json
from pathlib import Path
import numpy as np
from ..config import load_config
from ..runtime import resolve_output,atomic_save,isolated_rng,seed_all
from ..diffusion_dppo.config import contract
from ..diffusion_dppo.storage import artifact,load
from ..topk_diagnostic import file_hash

ROOT=Path(__file__).resolve().parents[3]
GROUPS=('F2b','F2c','A1b')


def resolve(group,device='cpu'):
    if group not in GROUPS:raise ValueError('Follow-up ID required: F2b F2c A1b')
    cfg=load_config(ROOT/f'configs/diffusion_followup_1m/{group}.yaml')
    data=resolve_output('runs/diffusion_followup_1m')
    old=resolve_output('runs/system_ablation_1m')
    cfg.diffusion.teacher_checkpoint=str(old/'G0_seed0/best.pt')
    if group=='F2b':cfg.diffusion.ablation['initial_checkpoint']=str(old/'A3_seed0/step0.pt')
    if group=='F2c':cfg.diffusion.ablation['initial_checkpoint']=str(data/'pretraining/F2c/best.pt')
    cfg.run.device=device
    import os
    cfg.run.cpu_threads=int(os.environ.get('UAV_TORCH_THREADS',cfg.run.cpu_threads))
    cfg.run.numba_threads=int(os.environ.get('UAV_NUMBA_THREADS',cfg.run.numba_threads))
    return cfg.validate(),data/f'{group}_seed0',old/'demonstrations'


def actor_digest(actor):
    digest=hashlib.sha256()
    for key,value in actor.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_sources(cfg):
    old=resolve_output('runs/system_ablation_1m')
    for g in ('G0','A1','A3','A5','F2a'):
        m=json.loads((old/f'{g}_seed0/manifest.json').read_text())
        if m['status']!='complete' or m['env_steps']!=1_000_000:raise ValueError('Finish reference '+g)
    if cfg.diffusion.ablation['run_id']=='F2b':
        model=load(cfg.diffusion.ablation['initial_checkpoint'],kinds={'dppo_model'})
        if model['env_steps']!=0 or model['config']['diffusion']['ablation']['run_id']!='A3':
            raise ValueError('F2b must copy A3 step0, never A3 best/last')


def require_initialization_gate(cfg):
    path=resolve_output('runs/diffusion_followup_1m/audits/A1b_initialization/gate.json')
    gate=json.loads(path.read_text())
    if gate.get('passed') is not True or gate['contract']!=contract(cfg):
        raise ValueError('A1b full initialization audit did not pass for this contract')
    from ..diffusion_dppo.learner import DiffusionLearner
    with isolated_rng():
        seed_all(cfg.run.seed)
        actor=DiffusionLearner(cfg,'cpu').actor
        if gate['actor_sha256']!=actor_digest(actor):raise ValueError('Audited initial actor changed')
    return gate


def save_scale_milestone(cfg,output,learner,steps):
    path=Path(output)/'audit_100k.pt'
    if steps>=100_000 and not path.exists():
        atomic_save(path,artifact(cfg,'model',env_steps=steps,learner=learner.state_dict(),
            provenance=dict(reason='first complete update at or above100k; scale audit only')))
