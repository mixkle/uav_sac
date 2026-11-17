import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.runtime import seed_all,atomic_save,rng_state
from uav_research.diffusion_dppo.policy import make_service_actor,DiffusionActor
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.storage import artifact
from uav_research.diffusion_dppo.config import contract
from uav_research.diffusion_followup.frozen import swap_config,intervene


def small(group='A1b'):
 c=load_config(f'configs/diffusion_followup_1m/{group}.yaml')
 c.env.num_uavs=4;c.env.num_users=24;c.env.horizon=4;c.run.num_envs=2
 c.run.cpu_threads=2;c.run.numba_threads=2;c.run.device='cpu';c.run.total_steps=8
 c.run.validation_seeds=[10000,10001];c.run.test_seeds=[20000,20001];c.run.eval_interval=8
 c.run.tensorboard=False;c.algo.ppo_epochs=1;c.algo.ppo_minibatches=2;c.algo.rollout_steps=2
 c.diffusion.minimum_free_gib=0;c.diffusion.microbatch_agents=16
 c.diffusion.ablation.update(initialization='random',initial_checkpoint='')
 torch.set_num_threads(2)
 return c.validate()


def observation(c):
 env=ServiceEnvironment(c.env);o,_,_=env.reset(991)
 # A dense, connected layout guarantees nonempty candidates for gradient tests.
 env.set_positions(np.array([[600,600],[800,600],[600,800],[800,800]],float),
                   np.random.default_rng(12).uniform(600,800,(c.env.num_users,2)))
 return env,env.obs


@pytest.mark.parametrize('group',['F2b','F2c','A1b'])
def test_joint_probability_and_head_gradients(group):
 c=small(group);seed_all(0);env,o=observation(c);actor=make_service_actor(c)
 action,chain=actor.sample(o)
 assert len(chain.scores)==o.candidate_lengths.sum() and chain.scores.size>0
 p=actor.evaluate_chain(o,chain)
 torch.testing.assert_close(p,torch.tensor(chain.logp),atol=2e-3,rtol=2e-4)
 assert torch.max((p.detach()-torch.tensor(chain.logp)).abs())<2e-3
 p.mean().backward()
 for head in (actor.movement_head,actor.score_head):
  assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
 assert np.max(np.linalg.norm(action.movement,axis=-1))<=1.000001
 env.step(action)


def test_gate_only_removes_noise_path_same_parameters():
 cfg=small('F2c');base=copy.deepcopy(cfg);base.diffusion.ablation['disable_noisy_score_to_motion']=False;base.validate()
 seed_all(7);a=make_service_actor(base)
 seed_all(7);b=make_service_actor(cfg)
 assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
 for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
 _,o=observation(cfg);enc=b.encode(o.tensor('cpu'));m=torch.randn((o.groups,2));s=torch.randn(int(o.candidate_lengths.sum()))
 for step in range(8):
  first,score=b.predict_noise(enc,m,s,step);second,score2=b.predict_noise(enc,m,s+2,step)
  torch.testing.assert_close(first,second,atol=0,rtol=0)
  assert (score-score2).abs().sum()>0
  e=copy.copy(enc);e.context=enc.context+.2
  assert (b.predict_noise(e,m,s,step)[0]-first).abs().sum()>0


def test_reference_epsilon_and_exact_posterior_mean():
 c=small();_,o=observation(c);actor=make_service_actor(c);enc=actor.encode(o.tensor('cpu'))
 m=torch.randn((o.groups,2));s=torch.randn(int(o.candidate_lengths.sum()))
 for step in range(8):
  pm,ps=actor.predict_noise(enc,m,s,step)
  torch.testing.assert_close(pm,actor.sqrt_one_minus_abar[step]*m)
  torch.testing.assert_close(ps,actor.sqrt_one_minus_abar[step]*s)
  mm,ms=actor.transition(enc,m,s,step)
  previous=1. if step==0 else actor.sqrt_abar[step-1]
  keep=actor.sqrt_abar[step]/previous
  torch.testing.assert_close(mm,keep*m,atol=1e-6,rtol=1e-6)
  torch.testing.assert_close(ms,keep*s,atol=1e-6,rtol=1e-6)
  # Original epsilon posterior agrees to float32 cancellation tolerance.
  old=DiffusionActor.transition(actor,enc,m,s,step)
  torch.testing.assert_close(mm,old[0],atol=3e-4,rtol=3e-4)
  torch.testing.assert_close(ms,old[1],atol=3e-4,rtol=3e-4)
 # Vector-step path is used by PPO/BC, and must implement the same reference.
 steps=torch.arange(o.groups)%8
 p,q=actor.predict_noise(enc,m,s,steps)
 torch.testing.assert_close(p,actor.sqrt_one_minus_abar[steps,None]*m)
 torch.testing.assert_close(q,actor.sqrt_one_minus_abar[steps[enc.owners]]*s)


def test_reference_residual_is_trainable_not_frozen():
 c=small();_,o=observation(c);actor=make_service_actor(c)
 optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
 initial=copy.deepcopy(actor.state_dict())
 for _ in range(2):
  optimizer.zero_grad();mov,score=actor.denoising_loss(o,np.zeros((o.groups,2),np.float32),np.zeros(o.candidate_lengths.sum(),np.float32))
  (mov.mean()+score.mean()).backward();optimizer.step()
 assert not torch.equal(initial['movement_head.4.weight'],actor.state_dict()['movement_head.4.weight'])
 assert not torch.equal(initial['score_head.4.weight'],actor.state_dict()['score_head.4.weight'])
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.ue.parameters())


def test_frozen_intervention_keeps_physics_rng_history():
 c=small('F2b');c.diffusion.ablation['association']='learned_topk';c.validate()
 env,o=observation(c);action,chain=make_service_actor(c).sample(o)
 row,arrays=intervene(env.snapshot(),action,c,swap_config(c))
 assert row['physical_positions_equal'] and 0<=row['proposal_jaccard']<=1
 assert arrays['learned_association'].sum(0).max()<=1
 assert arrays['sinr_association'].sum(1).max()<=c.env.capacity


def test_per_step_shadow_proposals_equal_real_protocol():
 from uav_research.diffusion_followup.frozen import protocol_inputs,protocol_outcomes
 c=small('F2b');c.diffusion.ablation['association']='learned_topk';c.validate()
 env,o=observation(c);action,_=make_service_actor(c).sample(o)
 before=protocol_inputs(env,action);env.step(action);row=protocol_outcomes(env,before)
 assert row['learned_coverage']==pytest.approx(env.info['coverage'])
 assert row['learned_throughput_bps']==pytest.approx(env.info['throughput_bps'])
 assert row['learned_direct_switches']==env.info['direct_switches']


def test_legacy_alias_and_missing_flags_remain_unchanged():
 m=json.loads(Path('configs/system_ablation_1m/manifest.json').read_text())
 assert m['aliases']['F2b']=='A3'
 old=load_config('configs/system_ablation_1m/A3.yaml')
 assert 'disable_noisy_score_to_motion' not in contract(old)['diffusion']['ablation']
 seed_all(0);a=make_service_actor(old)
 b=copy.deepcopy(old);b.diffusion.ablation.update(disable_noisy_score_to_motion=False,diffusion_init_mode='epsilon');b.validate()
 seed_all(0);other=make_service_actor(b)
 for k,v in a.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],atol=0,rtol=0)


def test_actor_initialization_semantics_cannot_be_cross_loaded(tmp_path):
 from uav_research.system_ablation.initialization import initialize
 c=small('F2c');c.diffusion.ablation['disable_noisy_score_to_motion']=False;c.validate()
 seed_all(0);source=DiffusionLearner(c,'cpu')
 p=tmp_path/'bc.pt';atomic_save(p,artifact(c,'pretrain',actor=source.actor.state_dict(),epoch=50,provenance={}))
 target=small('F2c');target.diffusion.ablation.update(initialization='bc',initial_checkpoint=str(p));target.validate()
 with pytest.raises(ValueError,match='structure'):initialize(DiffusionLearner(target,'cpu'))


def test_F2b_exact_A3_actor_and_fresh_critic_adam(tmp_path):
 from uav_research.system_ablation.initialization import initialize
 source_cfg=small('F2b');source_cfg.diffusion.ablation.update(run_id='A3',association='learned_topk');source_cfg.validate()
 seed_all(0);source=DiffusionLearner(source_cfg,'cpu')
 with torch.no_grad():source.value[-1].bias.add_(100)
 path=tmp_path/'A3_step0.pt';atomic_save(path,artifact(source_cfg,'model',env_steps=0,learner=source.state_dict()))
 cfg=small('F2b');cfg.diffusion.ablation.update(initialization='actor',initial_checkpoint=str(path));cfg.validate()
 seed_all(0);target=DiffusionLearner(cfg,'cpu');before=copy.deepcopy(target.value.state_dict())
 initialize(target)
 for k,v in source.actor.state_dict().items():torch.testing.assert_close(v,target.actor.state_dict()[k],atol=0,rtol=0)
 for k,v in before.items():torch.testing.assert_close(v,target.value.state_dict()[k],atol=0,rtol=0)
 assert not target.actor_opt.state and not target.value_opt.state


@pytest.mark.parametrize('group',['F2b','F2c','A1b'])
def test_empty_candidates_remain_movement_only_density(group):
 from uav_research.diffusion_dppo.observations import ServiceObservation,offsets
 cfg=small(group)
 obs=ServiceObservation(np.zeros((4,cfg.env.fixed_obs_dim),np.float32),np.empty((0,5),np.float32),
      offsets([0]*4),np.empty(0,np.int64),np.empty(0,bool))
 actor=make_service_actor(cfg);action,chain=actor.sample(obs)
 assert chain.scores.shape==(0,9)
 torch.testing.assert_close(actor.evaluate_chain(obs,chain),torch.tensor(chain.logp),atol=2e-3,rtol=2e-4)


def test_F2c_shared_demonstrations_and_BC_resume(tmp_path):
 from uav_research.diffusion_dppo.data import collect_demonstrations,pretrain,DemoDataset
 from uav_research.diffusion_dppo.storage import load
 teacher=small('F2b');teacher.algo.name='gaussian_service_mappo'
 teacher.diffusion.ablation.update(run_id='G0',association='learned_topk',bc_objective='nll');teacher.validate()
 path=tmp_path/'teacher.pt';atomic_save(path,artifact(teacher,'model',env_steps=0,learner=DiffusionLearner(teacher,'cpu').state_dict()))
 a3=small('F2c');a3.diffusion.ablation.update(run_id='A3',disable_noisy_score_to_motion=False)
 a3.diffusion.teacher_checkpoint=str(path);a3.diffusion.fit_seeds=[300000];a3.diffusion.selection_seeds=[301000]
 a3.diffusion.pretrain_epochs=1;a3.diffusion.pretrain_batch=8;a3.validate()
 collect_demonstrations(a3,tmp_path/'demos')
 gated=copy.deepcopy(a3);gated.diffusion.ablation.update(run_id='F2c',disable_noisy_score_to_motion=True);gated.validate()
 x=DemoDataset(tmp_path/'demos','fit',a3).batch(np.arange(8))
 y=DemoDataset(tmp_path/'demos','fit',gated).batch(np.arange(8))
 np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_array_equal(x[2],y[2])
 pretrain(gated,tmp_path/'demos',tmp_path/'continuous')
 pretrain(gated,tmp_path/'demos',tmp_path/'resumed',max_batches=1)
 pretrain(gated,tmp_path/'demos',tmp_path/'resumed',resume=tmp_path/'resumed/last.pt')
 one=load(tmp_path/'continuous/last.pt');two=load(tmp_path/'resumed/last.pt')
 for k,v in one['actor'].items():torch.testing.assert_close(v,two['actor'][k],atol=0,rtol=0)


@pytest.mark.parametrize('group',['F2b','F2c','A1b'])
def test_exact_cpu_recovery(group,tmp_path):
 from uav_research.diffusion_dppo.runner import train_online
 from uav_research.diffusion_dppo.storage import load
 c=small(group)
 first=tmp_path/'continuous';second=tmp_path/'resumed'
 train_online(c,first)
 train_online(c,second,stop_after_updates=1)
 full=sorted((second/'checkpoints').glob('full_*'))[-1]
 train_online(c,second,resume=full)
 x=load(first/'last.pt');y=load(second/'last.pt')
 for key in ('actor','value','normalizer','average_normalizer'):
  for k,v in x['learner'][key].items():torch.testing.assert_close(v,y['learner'][key][k],atol=0,rtol=0)
