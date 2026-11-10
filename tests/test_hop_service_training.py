"""Scientific contracts for executed history and Gaussian ragged PPO."""
import copy
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.runtime import seed_all
from uav_research.diffusion_dppo.environment import ServiceEnvironment,ServiceVector
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.policy import DiffusionActor
from uav_research.diffusion_dppo.gaussian import GaussianServiceActor
from uav_research.diffusion_dppo.storage import load
from uav_research.diffusion_dppo.observations import offsets, ServiceObservation
from test_diffusion_dppo import synthetic,nested_equal,make_teacher


def config(kind='gaussian_service_mappo'):
 c=load_config('configs/diffusion_dppo_local_smoke.yaml')
 c.algo.name=kind;c.env.motion_scaling['enabled']=True
 c.env.history_motion_source='post_reflection_displacement'
 c.diffusion.minimum_free_gib=0;c.run.cpu_threads=2
 c.validate();return c


def test_executed_history_timing_restore_and_legacy_identity():
 c=config();env=ServiceEnvironment(c.env);o,_,_=env.reset(10)
 actor=GaussianServiceActor(c);a,_=actor.sample(o)
 before=env.uav_xy.copy();h=env.B.min(1).copy();counts=env.sensed.sum(1).copy()
 env.step(a)
 np.testing.assert_allclose(env.history[:,:2],(env.uav_xy-before)/30,atol=1e-7)
 np.testing.assert_allclose(env.history[:,2],h/5)
 np.testing.assert_allclose(env.history[:,3],counts/10)
 snap=env.snapshot();obs=env.obs.copy();env._refresh();np.testing.assert_array_equal(env.obs.fixed,obs.fixed)
 restored=ServiceEnvironment(c.env);restored.restore(snap)
 nested_equal(restored.snapshot(),snap)
 a,_=actor.sample(obs);left,right=env.step(a),restored.step(a)
 nested_equal(left[0].__dict__,right[0].__dict__);nested_equal(left[1:],right[1:])
 old=copy.deepcopy(c.env);old.history_motion_source='command'
 with pytest.raises(ValueError,match='contract'):ServiceEnvironment(old).restore(snap)
 # Scaling is execution only: UE random stream unchanged.
 old.motion_scaling['enabled']=False;legacy=ServiceEnvironment(old);legacy.reset(10)
 a,_=actor.sample(legacy.obs);legacy.step(a)
 first=ServiceEnvironment(c.env);first.reset(10);first.step(a)
 np.testing.assert_array_equal(first.user_xy,legacy.user_xy)


def test_gaussian_joint_density_empty_permutation_and_gradients():
 c=config();torch.set_num_threads(2);actor=GaussianServiceActor(c)
 obs=synthetic(c);action,chain=actor.sample(obs)
 p=actor.evaluate_chain(obs,chain)
 np.testing.assert_allclose(p.detach(),chain.logp,atol=2e-6)
 enc=actor.encode(obs.tensor('cpu'));mm,ml,sm,sl=actor.distribution(enc)
 m=torch.tensor(chain.movement[:,-1]);s=torch.tensor(chain.scores[:,-1])
 expected=torch.distributions.Normal(mm,ml.exp()).log_prob(m).sum(-1)
 lp=torch.distributions.Normal(sm,sl.exp()).log_prob(s)
 for i in range(obs.groups):expected[i]+=lp[chain.candidate_offsets[i]:chain.candidate_offsets[i+1]].sum()
 torch.testing.assert_close(p[:,0],expected)
 (-p.mean()).backward()
 for module in (actor.ue,actor.context,actor.movement_head,actor.score_head):
  assert sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None)>0
 # Reverse local rows and corresponding latent routing, never change UE identity.
 rows=np.concatenate([np.arange(obs.offsets[i],obs.offsets[i+1])[::-1] for i in range(obs.groups)])
 perm=ServiceObservation(obs.fixed.copy(),obs.users[rows],obs.offsets.copy(),obs.ue_ids[rows],obs.service_mask[rows])
 pm=actor.distribution(actor.encode(perm.tensor('cpu')))
 torch.testing.assert_close(mm,pm[0]);torch.testing.assert_close(ml,pm[1])
 for i in range(obs.groups):
  lo,hi=chain.candidate_offsets[i:i+2]
  torch.testing.assert_close(sm[lo:hi].flip(0),pm[2][lo:hi])
 empty=synthetic(c,lengths=(0,0,0,0),candidates=(0,0,0,0))
 a,ch=actor.sample(empty);assert a.scores.size==0
 torch.testing.assert_close(actor.evaluate_chain(empty,ch),torch.tensor(ch.logp))
 assert (np.linalg.norm(action.movement,axis=-1)<1.000001).all()


@pytest.mark.parametrize('kind',['gaussian_service_mappo','diffusion_dppo'])
def test_small_complete_resume_export_evaluate(kind,tmp_path):
 from uav_research.diffusion_dppo.runner import train_online,export_actor,DeployedServiceActor
 from uav_research.diffusion_dppo.report import evaluate_checkpoint
 from uav_research.diffusion_dppo.data import collect_demonstrations,pretrain
 c=config(kind);c.algo.ppo_epochs=1;c.algo.ppo_minibatches=1
 pretrained=None
 if kind=='diffusion_dppo':
  c.diffusion.teacher_checkpoint=str(tmp_path/'teacher.pt')
  tc=copy.deepcopy(c);tc.env.motion_scaling['enabled']=False;tc.env.history_motion_source='command'
  make_teacher(tc,tmp_path/'teacher.pt')
  collect_demonstrations(c,tmp_path/'demo')
  c.diffusion.pretrain_epochs=1
  # Collection/BC contracts include pretrain epochs; recollect with final config.
  import shutil;shutil.rmtree(tmp_path/'demo');collect_demonstrations(c,tmp_path/'demo')
  pretrain(c,tmp_path/'demo',tmp_path/'pretrain');pretrained=tmp_path/'pretrain/best.pt'
 train_online(c,tmp_path/'full',pretrained=pretrained)
 first=train_online(c,tmp_path/'resume',pretrained=pretrained,stop_after_updates=1)
 train_online(c,tmp_path/'resume',resume=first['full_checkpoint'])
 a=load(tmp_path/'full/last.pt');b=load(tmp_path/'resume/last.pt')
 nested_equal(a['learner'],b['learner'])
 export_actor(tmp_path/'full/last.pt',tmp_path/'actor.pt');deployed=DeployedServiceActor(tmp_path/'actor.pt')
 learner=DiffusionLearner(c,'cpu');learner.load_state_dict(a['learner']);o,_,_=ServiceEnvironment(c.env).reset(10000)
 seed_all(20);x=deployed.act(o);seed_all(20);y=learner.act(o);nested_equal(x.movement,y.movement);nested_equal(x.scores,y.scores)
 result=evaluate_checkpoint(tmp_path/'full/last.pt',tmp_path/'eval',moment_repeats=1)
 assert np.isfinite(result['summary']['return'])
 wrong=copy.deepcopy(c);wrong.algo.name='diffusion_dppo' if kind=='gaussian_service_mappo' else 'gaussian_service_mappo'
 with pytest.raises(ValueError):load(tmp_path/'full/last.pt',wrong)


def test_paired_critic_initialization_and_scalar_batch_process():
 from uav_research.diffusion_dppo.observations import stack
 torch.set_num_threads(2)
 a=config('diffusion_dppo');b=config()
 seed_all(0);left=DiffusionLearner(a,'cpu')
 seed_all(0);right=DiffusionLearner(b,'cpu')
 nested_equal(left.value.state_dict(),right.value.state_dict())
 actor=right.actor
 for backend in ('batch','process'):
  vector=ServiceVector(b.env,2,backend)
  envs=[ServiceEnvironment(b.env) for _ in range(2)]
  try:
   obs,_,_=vector.reset([100,101]);scalar=stack([e.reset(s)[0] for e,s in zip(envs,[100,101])])
   nested_equal(obs.__dict__,scalar.__dict__)
   for _ in range(2):
    action,_=actor.sample(obs);actual=vector.step(action)
    expected=[e.step(x) for e,x in zip(envs,action.split(b.env.num_uavs))]
    nested_equal(actual[0].__dict__,stack([e[0] for e in expected]).__dict__)
    np.testing.assert_allclose(actual[2],[e[2] for e in expected],rtol=1e-7,atol=1e-8)
    obs=actual[0]
   saved=vector.snapshot();vector.reset_at(0)
   np.testing.assert_array_equal(vector.snapshot()[1]['history'],saved[1]['history'])
  finally:vector.close()


@pytest.mark.parametrize('kind',['gaussian_service_mappo','diffusion_dppo'])
def test_large_microbatch_preserves_weighted_update(kind):
 from test_diffusion_dppo import collect_rollout
 c=config(kind);c.algo.ppo_epochs=1;c.algo.ppo_minibatches=1
 seed_all(81);base=DiffusionLearner(c,'cpu');rollout=collect_rollout(c,base,4)
 state=copy.deepcopy(base.state_dict());small,big=copy.deepcopy(c),copy.deepcopy(c)
 small.diffusion.microbatch_agents=3;big.diffusion.microbatch_agents=4096
 x,y=DiffusionLearner(small,'cpu'),DiffusionLearner(big,'cpu')
 x.load_state_dict(state);y.load_state_dict(state)
 seed_all(20);x.update(rollout);seed_all(20);y.update(rollout)
 nested_equal(x.actor.state_dict(),y.actor.state_dict(),atol=1e-5)
 nested_equal(x.value.state_dict(),y.value.state_dict(),atol=1e-6)
