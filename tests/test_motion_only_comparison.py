"""M1/M2 are fresh motion-only learners with one identical rule protocol."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.runtime import seed_all
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.observations import ServiceObservation, offsets
from uav_research.diffusion_dppo.policy import make_service_actor
from uav_research.diffusion_dppo.storage import load


def small(group):
    c = load_config(f'configs/motion_only_sinr_topk_1m/{group}.yaml')
    c.env.num_uavs=4; c.env.num_users=12; c.env.horizon=4
    c.run.device='cpu'; c.run.num_envs=2; c.run.cpu_threads=2; c.run.numba_threads=2
    c.run.validation_seeds=[10000,10001]; c.run.test_seeds=[20000,20001]
    c.run.total_steps=8; c.run.eval_interval=8; c.run.tensorboard=False
    c.algo.rollout_steps=2; c.algo.ppo_epochs=1; c.algo.ppo_minibatches=2
    c.diffusion.minimum_free_gib=0; c.diffusion.microbatch_agents=16
    torch.set_num_threads(2)
    return c.validate()


def test_paired_configs_and_no_initialization_sources():
    entries=json.loads(Path('configs/motion_only_sinr_topk_1m/manifest.json').read_text())['runs']
    configs=[load_config(entries[g]['config']) for g in ('M1','M2')]
    a,b=[c.as_dict() for c in configs]
    assert a['env']==b['env'] and a['run']==b['run']
    for c in configs:
        o=c.diffusion.ablation
        assert o['initialization']=='random' and not o['initial_checkpoint']
        assert not c.diffusion.teacher_checkpoint and not c.diffusion.pretrained_checkpoint
        assert not o['score_actions'] and o['association']=='sinr_topk'
        assert c.run.total_steps==1_000_000
        assert c.env.throughput_weight==.1 and not c.env.robustness_weight
        assert not c.env.disconnection_weight
    for x in (a,b):
        x['algo'].pop('name')
        x['diffusion']['ablation'].pop('run_id')
        x['diffusion']['ablation'].pop('bc_objective')
        x['diffusion']['ablation'].pop('diffusion_init_mode')
    assert a==b


def test_diffusion_random_initialization_has_usable_motion_scale():
    c=small('M2');seed_all(0)
    obs,_,_=ServiceEnvironment(c.env).reset(19)
    actor=make_service_actor(c)
    obs=__import__('uav_research.diffusion_dppo.observations',fromlist=['stack']).stack([obs]*32)
    action,chain=actor.sample(obs)
    norms=np.linalg.norm(action.movement.reshape(-1,2),axis=1)
    latent=np.linalg.norm(chain.movement[:,-1],axis=1)
    assert .1<norms.mean()<.95 and np.quantile(latent,.99)<15
    # Zero residual projections are initialization, not BC; they train normally.
    before=actor.movement_head[-1].weight.detach().clone()
    actor.evaluate_chain(obs,chain).mean().backward()
    assert actor.movement_head[-1].weight.grad.abs().sum()>0
    opt=torch.optim.Adam(actor.parameters(),lr=1e-4);opt.step()
    assert not torch.equal(before,actor.movement_head[-1].weight)


@pytest.mark.parametrize('group',['M1','M2'])
@pytest.mark.parametrize('empty',[False,True])
def test_only_two_random_coordinates_and_gradient(group,empty):
    c=small(group); seed_all(0)
    n=0 if empty else 13
    obs=ServiceObservation(np.zeros((4,c.env.fixed_obs_dim),np.float32),
        np.random.default_rng(1).normal(size=(n*4,5)).astype(np.float32),
        offsets([n]*4),np.tile(np.arange(n),4),np.ones(n*4,bool))
    actor=make_service_actor(c)
    action,chain=actor.sample(obs)
    # ServiceAction carries deterministic rule messages, never learned scores.
    assert len(action.scores)==n*4
    assert chain.scores.size==0 and not chain.candidate_offsets.any()
    assert not list(actor.score_head.parameters())
    assert chain.movement.shape[0]==4 and chain.movement.shape[-1]==2
    np.testing.assert_array_equal(chain.score_logp,0)
    new=actor.evaluate_chain(obs,chain)
    torch.testing.assert_close(new,torch.tensor(chain.logp),atol=2e-3,rtol=2e-4)
    torch.testing.assert_close((new-torch.tensor(chain.logp)).exp(),torch.ones_like(new),atol=2e-3,rtol=2e-4)
    new.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.movement_head.parameters())
    if not empty:
        if group=='M2':
            # Zero output projection blocks upstream gradients on the first
            # backward only. Once that projection updates, the encoder trains.
            torch.optim.Adam(actor.parameters(),lr=1e-4).step()
            actor.zero_grad(set_to_none=True)
            actor.evaluate_chain(obs,chain).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.ue.parameters())


def test_identical_critic_fresh_adam_and_fixed_action_physics():
    learners=[]; envs=[]
    for group in ('M1','M2'):
        c=small(group);seed_all(0)
        x=DiffusionLearner(c,'cpu');learners.append(x)
        assert not x.actor_opt.state and not x.value_opt.state
        env=ServiceEnvironment(c.env);env.reset(19);envs.append(env)
    for k,p in learners[0].value.state_dict().items():
        torch.testing.assert_close(p,learners[1].value.state_dict()[k],atol=0,rtol=0)
    a,_=learners[0].actor.sample(envs[0].obs)
    b=copy.deepcopy(a);b.scores=1-b.scores # ignored by fixed SINR rule
    x,y=envs[0].step(a),envs[1].step(b)
    for name in ('uav_xy','user_xy','C','B','history_motion'):
        if hasattr(envs[0],name):
            np.testing.assert_array_equal(getattr(envs[0],name),getattr(envs[1],name))
    assert x[2]==y[2]


@pytest.mark.parametrize('group',['M1','M2'])
def test_online_resume_and_motion_only_export(group,tmp_path):
    from uav_research.diffusion_dppo.runner import train_online,export_actor,DeployedServiceActor
    c=small(group)
    train_online(c,tmp_path/'full')
    train_online(c,tmp_path/'resume',stop_after_updates=1)
    checkpoint=sorted((tmp_path/'resume/checkpoints').glob('full_*'))[-1]
    train_online(c,tmp_path/'resume',resume=checkpoint)
    full=load(tmp_path/'full/last.pt');resumed=load(tmp_path/'resume/last.pt')
    assert full['env_steps']==resumed['env_steps']==8
    assert full['provenance']['initialization']=='random'
    for branch in ('actor','value'):
        for key,t in full['learner'][branch].items():
            torch.testing.assert_close(t,resumed['learner'][branch][key],atol=0,rtol=0)
    path=tmp_path/'export.pt';export_actor(tmp_path/'full/last.pt',path)
    deployed=DeployedServiceActor(path,'cpu')
    actor=deployed.actor
    env=ServiceEnvironment(c.env);obs,_,_=env.reset(19)
    action,chain=actor.sample(obs)
    assert chain.scores.size==0 and np.isfinite(action.movement).all()


def test_dedicated_report_rebuild(tmp_path):
    from uav_research.diffusion_dppo.runner import train_online
    from uav_research.system_ablation.report import collect_run
    spec=importlib.util.spec_from_file_location('motion_cli','scripts/motion_only_comparison.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    for group in ('M1','M2'):
        c=small(group);out=tmp_path/group
        train_online(c,out)
        collect_run(out,tmp_path/'report'/group,'cpu',kinds=('best','last'))
    import sys
    old=sys.argv
    try:
        sys.argv=['motion_only_comparison.py','report','--rebuild','--output',str(tmp_path/'report')]
        cli.main()
    finally:sys.argv=old
    assert (tmp_path/'report/figures/motion_actor/last_episode_curves.png').exists()
    assert 'A4' not in (tmp_path/'report/report.md').read_text()
    assert 'M2 - M1' in (tmp_path/'report/metrics/paired_differences.csv').read_text()
