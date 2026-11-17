"""Control-variable, probability and automatic continuation acceptance."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.runtime import seed_all
from uav_research.diffusion_dppo.config import contract
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.storage import load

GROUPS=('M2-NoHop','M2-N5','M2-N7','M2-N9')


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    return importlib.import_module('m2_control_experiments')


def small(group):
    c=load_config(f'configs/m2_motion_neighbor_controls_10m/{group}.yaml')
    c.env.num_uavs=4;c.env.num_users=12;c.env.horizon=4
    c.run.device='cpu';c.run.num_envs=2;c.run.cpu_threads=2;c.run.numba_threads=2
    c.run.validation_seeds=[10000,10001];c.run.test_seeds=[20000,20001]
    c.run.total_steps=8;c.run.eval_interval=8;c.run.tensorboard=False
    c.algo.rollout_steps=2;c.algo.ppo_epochs=1;c.algo.ppo_minibatches=2
    c.diffusion.minimum_free_gib=0;c.diffusion.microbatch_agents=16
    torch.set_num_threads(2)
    return c.validate()


@pytest.mark.parametrize('group',GROUPS)
def test_configuration_changes_are_exactly_the_intended_controls(group,cli):
    first,cfg,_,_=cli.resolve(group,'cuda:0')
    base=load_config('configs/motion_only_sinr_topk_10m/M2.yaml').as_dict()
    data=cfg.as_dict()
    data['diffusion']['ablation']['run_id']='M2'
    if group=='M2-NoHop':
        assert data['env']['motion_scaling']['enabled'] is False
        assert data['diffusion']['ablation']['motion_rule']=='none'
        data['env']['motion_scaling']['enabled']=True
        data['diffusion']['ablation']['motion_rule']='exponential'
    else:
        assert data['env']['neighbor_slots']==int(group[4:])
        data['env']['neighbor_slots']=3
    assert data==base
    assert first.run.total_steps==1_000_000 and cfg.run.total_steps==10_000_000
    assert contract(first)==contract(cfg)
    assert cfg.env.state_dim==6292
    meta=json.loads(Path('configs/m2_motion_neighbor_controls_10m/manifest.json').read_text())
    for prefix in ('','stage1_'):
        e=meta['runs'][group]
        assert hashlib.sha256(Path(e[prefix+'config']).read_bytes()).hexdigest()==e[prefix+'sha256']


@pytest.mark.parametrize('group',GROUPS)
def test_only_motion_probability_and_unchanged_critic(group):
    cfg=small(group);base=small('M2-NoHop')
    seed_all(0);reference=DiffusionLearner(base,'cpu')
    seed_all(0);learner=DiffusionLearner(cfg,'cpu')
    for key,value in reference.value.state_dict().items():
        torch.testing.assert_close(value,learner.value.state_dict()[key],atol=0,rtol=0)
    env=ServiceEnvironment(cfg.env);obs,_,_=env.reset(17)
    action,chain=learner.actor.sample(obs)
    assert not chain.scores.size and not chain.candidate_offsets.any()
    assert not list(learner.actor.score_head.parameters())
    new=learner.actor.evaluate_chain(obs,chain)
    torch.testing.assert_close(new,torch.tensor(chain.logp),atol=.002,rtol=.0002)
    assert np.isfinite(action.movement).all()
    assert (np.linalg.norm(action.movement,axis=-1)<=1).all()
    new.mean().backward()
    assert learner.actor.movement_head[-1].weight.grad.abs().sum()>0
    if group=='M2-NoHop':np.testing.assert_array_equal(env.motion_speed_factor([1,2,3,5]),1)


def test_neighbor_slots_do_not_expand_graph_or_change_physical_state():
    configs=[];envs=[]
    for n in (3,5,7,9):
        cfg=load_config('configs/motion_only_sinr_topk_10m/M2.yaml')
        cfg.env.neighbor_slots=n;cfg.run.cpu_threads=2;cfg.run.numba_threads=2;cfg.validate()
        env=ServiceEnvironment(cfg.env);env.reset(19)
        env.uav_xy[:]=np.column_stack([900+np.arange(18)*10,np.full(18,1200)])
        env.C.fill(0)
        env._refresh()
        configs.append(cfg);envs.append(env)
    base=envs[0]
    prefix=configs[0].env.current_obs_dim
    for cfg,env in zip(configs[1:],envs[1:]):
        np.testing.assert_array_equal(env.state,base.state)
        np.testing.assert_array_equal(env.A,base.A)
        np.testing.assert_array_equal(env.obs.fixed[:,:prefix],base.obs.fixed[:,:prefix])
        np.testing.assert_array_equal(env.obs.fixed[:,-4:],base.obs.fixed[:,-4:])
        start=cfg.env.self_dim+2*cfg.env.num_bs
        features=env.obs.fixed[:,start:cfg.env.current_obs_dim].reshape(18,cfg.env.neighbor_slots,-1)
        # Every additional slot carries a real neighbor in this dense graph.
        assert (np.abs(features[:,3:]).sum(axis=-1)>0).all()
        action,_=DiffusionLearner(cfg,'cpu').actor.sample(env.obs)
        action.movement[:]=.1
        if env is envs[1]:fixed=copy.deepcopy(action)
        env.step(copy.deepcopy(fixed))
    base.step(copy.deepcopy(fixed))
    for env in envs[1:]:
        for key in ('uav_xy','user_xy','A','B','C','state'):
            np.testing.assert_array_equal(getattr(env,key),getattr(base,key))


@pytest.mark.parametrize('group',GROUPS)
def test_automatic_pipeline_and_full_recovery(group,cli,monkeypatch,tmp_path):
    first=small(group);final=copy.deepcopy(first);final.run.total_steps=16
    def paths(name):
        return (tmp_path/name/'stage1',tmp_path/name/'final')
    active=['continuous']
    monkeypatch.setattr(cli,'resolve',lambda *a,**kw:(first,final,*paths(active[0])))
    monkeypatch.setattr(cli,'resolve_output',lambda p:tmp_path/'locks'/Path(p).name)
    cli.run(group)
    active[0]='resumed'
    from uav_research.diffusion_dppo.runner import train_online
    source,target=paths(active[0])
    # Stop within stage1, then let the suite finish both stages itself.
    train_online(first,source,stop_after_updates=1)
    assert cli.preflight(group,resume=True)['source_steps']==4
    cli.run(group,resume=True)
    actual=load(target/'last.pt');expected=load(paths('continuous')[1]/'last.pt')
    assert actual['env_steps']==expected['env_steps']==16
    for branch in ('actor','value'):
        for key,tensor in expected['learner'][branch].items():
            torch.testing.assert_close(tensor,actual['learner'][branch][key],atol=0,rtol=0)
    assert actual['learner']['optimizer_steps']==expected['learner']['optimizer_steps']
    assert json.loads((target/'continuation.json').read_text())['source_steps']==8
    assert (target/'step0.pt').read_bytes()==(source/'step0.pt').read_bytes()
    with pytest.raises(ValueError,match='completed'):cli.preflight(group,resume=True)
    with pytest.raises(FileExistsError):cli.preflight(group,resume=False)


def test_launcher_preflights_all_before_starting_any_job():
    text=Path('scripts/launch_m2_controls_server.sh').read_text()
    assert 'UAV_M2_CONTROL_GPUS:-0,1,2,3' in text
    assert 'M2-NoHop M2-N5 M2-N7 M2-N9' in text
    assert text.index('if "$uav_check_only"')<text.index('new-session')
    assert 'check --group' in text and 'train --group' in text


def test_stage2_resume_export_and_report(cli,monkeypatch,tmp_path):
    from uav_research.diffusion_dppo.runner import train_online,export_actor,DeployedServiceActor
    from uav_research.diffusion_dppo.continuation import prepare_continuation
    from uav_research.system_ablation.report import collect_run
    from types import SimpleNamespace
    group='M2-N9';first=small(group);final=copy.deepcopy(first);final.run.total_steps=16
    source=tmp_path/'source';target=tmp_path/'final'
    train_online(first,source)
    checkpoint=source/'checkpoints/full_000000008'
    prepare_continuation(final,checkpoint,target)
    train_online(final,target,resume=checkpoint,stop_after_updates=3)
    monkeypatch.setattr(cli,'resolve',lambda *a,**kw:(first,final,source,target))
    monkeypatch.setattr(cli,'resolve_output',lambda p:tmp_path/'locks'/Path(p).name)
    assert cli.preflight(group,resume=True)['source_steps']==12
    cli.run(group,resume=True)
    assert load(target/'last.pt')['env_steps']==16
    exported=tmp_path/'actor.pt';export_actor(target/'last.pt',exported)
    deployed=DeployedServiceActor(exported,'cpu')
    obs,_,_=ServiceEnvironment(final.env).reset(19)
    action,chain=deployed.actor.sample(obs)
    assert chain.scores.size==0 and np.isfinite(action.movement).all()
    out=tmp_path/'report'
    collect_run(target,out/group,'cpu',kinds=('best','last'))
    monkeypatch.setattr(cli,'resolve_output',lambda p:Path(p))
    cli.report(SimpleNamespace(output=str(out),groups=[group],rebuild=True))
    assert (out/'figures/uav_neighbors/last_episode_curves.png').exists()
    assert group in (out/'report.md').read_text()
    assert 'A4' not in (out/'report.md').read_text()
    with pytest.raises(ValueError):
        mismatch=copy.deepcopy(final);mismatch.env.neighbor_slots=3;mismatch.validate()
        load(target/'last.pt',mismatch,{'dppo_model'})
