"""Actor-only stage changes. Never restore teacher critic/Adam/online counters."""
from pathlib import Path
from ..diffusion_dppo.storage import load
from ..diffusion_dppo.config import from_dict
from ..topk_diagnostic import file_hash
from .settings import options


def initialize(learner, override=None):
    cfg=learner.cfg; o=options(cfg)
    if o['initialization']=='random':
        if override: raise ValueError('Random group must not load weights')
        return dict(initialization='random',stage_online_steps=0)
    path=Path(override or o['initial_checkpoint'])
    source=load(path,kinds={'dppo_model','dppo_pretrain'})
    other=from_dict(source['config'])
    # A4 loads G0 actor; A3/A5/F load BC. F1 differs ONLY in dynamics.
    left,right=cfg.as_dict(),other.as_dict()
    for name in ('width','height','num_uavs','num_users','capacity','neighbor_slots',
                 'observation_history','observation_coordinates','history_motion_source'):
        if left['env'][name]!=right['env'][name]:raise ValueError('Actor input semantics changed: '+name)
    so=options(other)
    for name in ('score_actions','ue_context','observation_scope',
                 'disable_noisy_score_to_motion','diffusion_init_mode'):
        if o[name]!=so[name]:raise ValueError('Actor structure changed: '+name)
    if cfg.algo.name != other.algo.name:raise ValueError('Actor distribution mismatch')
    if o['initialization']=='actor' and source['kind']!='dppo_model':
        raise ValueError('Actor initialization requires an online model artifact')
    if o['initialization']=='bc' and source['kind']!='dppo_pretrain':
        raise ValueError('BC group requires pretrained actor')
    learner.actor.load_state_dict(source['actor'] if source['kind']=='dppo_pretrain' else source['learner']['actor'])
    return dict(initialization=o['initialization'],source_path=str(path),source_sha256=file_hash(path),
        source_online_steps=source.get('env_steps',0),source_pretrain_epoch=source.get('epoch'),
        source_training_budget=other.run.total_steps,
        source_provenance=source.get('provenance',{}),critic='fresh',online_adam='fresh',stage_online_steps=0)
