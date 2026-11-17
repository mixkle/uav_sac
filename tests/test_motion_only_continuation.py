"""Budget extension preserves optimizer/RNG state and the original 1M lineage."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch

from uav_research.config import load_config
from uav_research.diffusion_dppo.config import contract
from uav_research.diffusion_dppo.continuation import prepare_continuation
from uav_research.diffusion_dppo.runner import train_online
from uav_research.diffusion_dppo.storage import load


@pytest.mark.parametrize('group',['M1','M2'])
def test_only_total_budget_changes(group):
    a=load_config(f'configs/motion_only_sinr_topk_1m/{group}.yaml')
    b=load_config(f'configs/motion_only_sinr_topk_10m/{group}.yaml')
    assert b.run.total_steps==10_000_000
    assert contract(a)==contract(b)
    left,right=a.as_dict(),b.as_dict()
    left['run']['total_steps']=right['run']['total_steps']
    assert left==right


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('group',['M1','M2'])
def test_continuation_and_recovery_preserve_source_and_full_state(group,tmp_path):
    cfg=load_config(f'configs/motion_only_sinr_topk_1m/{group}.yaml')
    cfg.env.num_uavs=4;cfg.env.num_users=12;cfg.env.horizon=4
    cfg.run.device='cpu';cfg.run.num_envs=2;cfg.run.cpu_threads=2;cfg.run.numba_threads=2
    cfg.run.validation_seeds=[10000,10001];cfg.run.test_seeds=[20000,20001]
    cfg.run.total_steps=8;cfg.run.eval_interval=8;cfg.run.tensorboard=False
    cfg.algo.rollout_steps=2;cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=2
    cfg.diffusion.minimum_free_gib=0;cfg.diffusion.microbatch_agents=16
    cfg.validate()
    source=tmp_path/'source'
    train_online(cfg,source)
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}
    checkpoint=source/'checkpoints/full_000000008'
    new=copy.deepcopy(cfg);new.run.total_steps=16;new.validate()
    for name in ('continuous','resumed'):
        lineage=prepare_continuation(new,checkpoint,tmp_path/name)
        assert lineage['source_steps']==8 and lineage['additional_steps']==8
        assert (tmp_path/name/'step0.pt').read_bytes()==(source/'step0.pt').read_bytes()
    train_online(new,tmp_path/'continuous',resume=checkpoint)
    train_online(new,tmp_path/'resumed',resume=checkpoint,stop_after_updates=3)
    pause=tmp_path/'resumed/checkpoints/full_000000012'
    assert load(pause)['env_steps']==12
    train_online(new,tmp_path/'resumed',resume=pause)
    a,b=[load(tmp_path/name/'last.pt') for name in ('continuous','resumed')]
    assert a['env_steps']==b['env_steps']==16
    equal(a['learner'],b['learner'])
    after={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in source.rglob('*') if p.is_file()}
    assert hashes==after


def test_select_never_falls_back_or_restarts(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    helper=importlib.import_module('resume_motion_only_10m')
    cfg=load_config('configs/motion_only_sinr_topk_10m/M1.yaml')
    original=tmp_path/'source';target=tmp_path/'target'
    with pytest.raises(FileNotFoundError):helper.select_checkpoint(cfg,original,target)
    target.mkdir()
    with pytest.raises(FileExistsError):helper.select_checkpoint(cfg,original,target)
    full=target/'checkpoints/full_002000000';full.mkdir(parents=True)
    (full/'index.json').write_text('{}');(full/'metadata.pt').write_bytes(b'')
    with pytest.raises(ValueError,match='provenance'):helper.select_checkpoint(cfg,original,target)


def test_launcher_continuation_route_preserves_existing_launcher():
    text=Path('scripts/launch_motion_only_server.sh').read_text()
    assert 'UAV_MOTION_GPUS:-2,3' in text
    assert 'uav_session_prefix=motion-10m' in text
    assert 'resume_motion_only_10m_server.sh' in text
    assert 'train_motion_only_server.sh' in text
    assert text.index('if "$uav_check_only"')<text.index('new-session')
