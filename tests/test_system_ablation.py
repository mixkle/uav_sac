"""A--F causal controls, probabilities, masks, initialization and recovery."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.runtime import seed_all,atomic_save
from uav_research.diffusion_dppo.config import contract
from uav_research.diffusion_dppo.environment import ServiceEnvironment,ServiceVector
from uav_research.diffusion_dppo.policy import make_service_actor
from uav_research.diffusion_dppo.learner import DiffusionLearner,DppoRollout
from uav_research.diffusion_dppo.storage import artifact,load
from uav_research.system_ablation.settings import better,speed_factor,options

IDS=list(json.loads(Path('configs/system_ablation_1m/manifest.json').read_text())['runs'])


def small(key='G0'):
 c=load_config(f'configs/system_ablation_1m/{key}.yaml')
 c.env.num_uavs=4;c.env.num_users=12;c.env.neighbor_slots=3;c.env.horizon=4
 c.run.device='cpu';c.run.num_envs=2;c.run.cpu_threads=2;c.run.numba_threads=2
 c.run.validation_seeds=[10000,10001];c.run.test_seeds=[20000,20001]
 c.run.total_steps=8;c.run.tensorboard=False;c.run.eval_interval=8
 c.algo.rollout_steps=2;c.algo.ppo_epochs=1;c.algo.ppo_minibatches=2
 c.diffusion.minimum_free_gib=0;c.diffusion.microbatch_agents=16
 c.diffusion.ablation.update(initialization='random',initial_checkpoint='')
 c.validate();torch.set_num_threads(2)
 return c


@pytest.mark.parametrize('key',IDS)
def test_probability_update_all_variants(key):
 c=small(key);seed_all(0)
 env=ServiceEnvironment(c.env);obs,state,_=env.reset(19)
 actor=make_service_actor(c)
 action,chain=actor.sample(obs)
 assert np.isfinite(action.movement).all()
 assert np.all(np.linalg.norm(action.movement,axis=-1)<=1.000001)
 np.testing.assert_array_equal(action.ue_ids,obs.candidate_ids)
 torch.testing.assert_close(actor.evaluate_chain(obs,chain),torch.tensor(chain.logp),atol=2e-3,rtol=2e-4)
 loss=actor.evaluate_chain(obs,chain).mean();loss.backward()
 assert any(p.grad is not None for p in actor.movement_head.parameters())
 assert any(p.grad is not None for p in actor.ue.parameters())
 if not options(c)['score_actions']:
  assert chain.scores.size==0 and chain.candidate_offsets.sum()==0
  assert sum(p.numel() for p in actor.score_head.parameters())==0
 env.step(action)
 snap=env.snapshot();fixed=env.obs.fixed.copy();association=env.C.copy()
 env._refresh();np.testing.assert_array_equal(env.C,association)
 restored=ServiceEnvironment(c.env);restored.restore(snap)
 np.testing.assert_array_equal(restored.obs.fixed,fixed)
 np.testing.assert_array_equal(restored.C,association)
 if options(c)['observation_scope']=='global':
  np.testing.assert_array_equal(env.obs.global_context,restored.obs.global_context)


def test_motion_endpoints_and_constant():
 c=small('B2');np.testing.assert_allclose(speed_factor([1,2,5],c.env),[.25,.4375,1])
 c=small('B5');np.testing.assert_allclose(speed_factor([1,2,5],c.env),[.5,.5,.5])
 c=small('B4');np.testing.assert_allclose(speed_factor([1,2,5],c.env),1)


def test_lexicographic_selection():
 assert better(dict(last100_coverage=.7,last100_throughput_bps=1),(.69,1e9))[0]
 assert not better(dict(last100_coverage=.69,last100_throughput_bps=1e9),(.7,1))[0]
 assert better(dict(last100_coverage=.7,last100_throughput_bps=2),(.7,1))[0]


def test_initial_weights_and_common_critic():
 for key in ('B2','B3','B4','B5','D3'):
  a,b=small(),small(key)
  seed_all(0);x=DiffusionLearner(a,'cpu')
  seed_all(0);y=DiffusionLearner(b,'cpu')
  for p,q in zip(x.actor.parameters(),y.actor.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
  for p,q in zip(x.value.parameters(),y.value.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)


def test_sinr_and_scoring_noise_isolation():
 c,d=small('D3'),small('D4');e,f=ServiceEnvironment(c.env),ServiceEnvironment(d.env)
 o,_,_=e.reset(11);f.reset(11)
 a,_=make_service_actor(c).sample(o)
 b=copy.deepcopy(a);b.scores=1-b.scores
 x=e.step(a);y=f.step(b)
 np.testing.assert_array_equal(e.C,f.C)
 np.testing.assert_array_equal(e.uav_xy,f.uav_xy)
 np.testing.assert_array_equal(e.user_xy,f.user_xy)
 assert x[2]==y[2]


def test_context_truncation_keeps_all_candidates():
 from uav_research.diffusion_dppo.observations import ServiceObservation,offsets
 obs=ServiceObservation(np.zeros((4,small().env.fixed_obs_dim),np.float32),
  np.random.default_rng(1).normal(size=(80,5)).astype(np.float32),offsets([20]*4),np.tile(np.arange(20),4),np.ones(80,bool))
 for key in ('C2','C3','C4'):
  actor=make_service_actor(small(key));a,chain=actor.sample(obs)
  assert len(a.scores)==80 and len(chain.scores)==80
  assert actor.encode(obs.tensor('cpu')).candidates.shape[0]==80


def test_bc_losses_shared_labels():
 from uav_research.diffusion_dppo.observations import stack
 c=small();o,_,_=ServiceEnvironment(c.env).reset(17)
 a,chain=make_service_actor(c).sample(o)
 for key in ('A3','A5','F2a'):
  actor=make_service_actor(small(key))
  m,s=actor.denoising_loss(o,chain.movement[:,-1],chain.scores[:,-1])
  assert m.shape==s.shape==(4,)
  assert torch.isfinite(m+s).all()
  (m+s).mean().backward()


@pytest.mark.parametrize('key',['G0','A1','D4','F2a','E2'])
def test_short_training_resume(key,tmp_path):
 from uav_research.diffusion_dppo.runner import train_online
 c=small(key)
 train_online(c,tmp_path/'continuous')
 train_online(c,tmp_path/'resumed',stop_after_updates=1)
 full=sorted((tmp_path/'resumed/checkpoints').glob('full_*'))[-1]
 train_online(c,tmp_path/'resumed',resume=full)
 a=load(tmp_path/'continuous/last.pt')['learner']
 b=load(tmp_path/'resumed/last.pt')['learner']
 for k in ('actor','value','normalizer','average_normalizer'):
  for name,v in a[k].items():torch.testing.assert_close(v,b[k][name],atol=0,rtol=0)


def test_truncated_context_has_no_motion_information_leak():
 from uav_research.diffusion_dppo.observations import ServiceObservation,offsets
 rng=np.random.default_rng(9);features=rng.normal(size=(80,5)).astype(np.float32)
 features[:,2]=np.tile(np.arange(20),4)  # top10 are rows 10..19 in each group
 obs=ServiceObservation(np.zeros((4,small().env.fixed_obs_dim),np.float32),features,
     offsets([20]*4),np.tile(np.arange(20),4),np.ones(80,bool))
 changed=obs.copy();idx=np.concatenate([np.arange(i*20,i*20+10) for i in range(4)])
 changed.users[idx,0:2]+=100
 for key in ('C2','C3'):
  actor=make_service_actor(small(key))
  a=actor.distribution(actor.encode(obs.tensor('cpu')))
  b=actor.distribution(actor.encode(changed.tensor('cpu')))
  torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
  torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
  assert not torch.equal(a[2],b[2])  # FULL candidate score path still receives it


def test_demo_and_bc_stage_pairing(tmp_path):
 from uav_research.diffusion_dppo.runner import train_online
 from uav_research.diffusion_dppo.data import collect_demonstrations,pretrain,DemoDataset
 root=tmp_path/'stages'
 base=small();train_online(base,root/'G0')
 # All students use the SAME complete dataset and supervised latent actions.
 cfgs={k:small(k) for k in ('A3','A5','F2a')}
 for c in cfgs.values():
  c.diffusion.fit_seeds=[300000];c.diffusion.selection_seeds=[301000]
  c.diffusion.pretrain_epochs=1;c.diffusion.pretrain_batch=16
  c.diffusion.teacher_checkpoint=str(root/'G0/best.pt')
  c.diffusion.ablation['initialization']='bc'
  c.diffusion.ablation['initial_checkpoint']='unused_until_online'
  c.validate()
 collect_demonstrations(cfgs['A3'],root/'demos')
 data_hash=__import__('hashlib').sha256((root/'demos/dataset.json').read_bytes()).hexdigest()
 labels=[]
 for key,c in cfgs.items():
  ds=DemoDataset(root/'demos','fit',c);labels.append(ds.batch(np.arange(4))[1:])
  pretrain(c,root/'demos',root/key)
  initial=load(root/key/'best.pt')
  assert initial['provenance']['dataset_sha256']==data_hash
  c.diffusion.ablation['initial_checkpoint']=str(root/key/'best.pt')
  from uav_research.system_ablation.initialization import initialize
  learner=DiffusionLearner(c,'cpu');initialize(learner)
  assert not learner.actor_opt.state and not learner.value_opt.state
 for m,s in labels[1:]:
  np.testing.assert_array_equal(m,labels[0][0]);np.testing.assert_array_equal(s,labels[0][1])


def test_report_rebuild_and_metrics(tmp_path):
 from uav_research.diffusion_dppo.runner import train_online
 from uav_research.system_ablation.report import collect_run,rebuild,episode_metrics
 c=small();train_online(c,tmp_path/'run')
 root=tmp_path/'report'
 collect_run(tmp_path/'run',root/'G0','cpu',kinds=('best','last'),moments=0)
 records=rebuild(root);assert len(records)==2
 old=(root/'metrics/unified_best_last.csv').read_bytes()
 rebuild(root);assert (root/'metrics/unified_best_last.csv').read_bytes()==old
 assert (root/'figures/A_actor_type/last_episode_curves.png').exists()
 c=np.array([[.1,.2,.3,.4],[.8,.6,.4,.2]])
 data=np.stack([c,np.ones_like(c)*1e6,c],-1)
 rows=episode_metrics(data,['coverage','throughput_bps','reward'],[1,2])
 assert rows[0]['last100_coverage_std_pp']==pytest.approx(c[0].std()*100)
 assert rows[1]['last100_drop_events_10pp']==3


@pytest.mark.parametrize('key',['G0','C2','C3'])
def test_context_permutation_and_log_probability_parts(key):
 from uav_research.diffusion_dppo.observations import ServiceObservation,offsets
 rng=np.random.default_rng(49);lengths=[0,13,7,11];starts=offsets(lengths)
 ids=np.concatenate([np.arange(n) for n in lengths]);mask=np.ones(sum(lengths),bool)
 obs=ServiceObservation(np.zeros((4,small().env.fixed_obs_dim),np.float32),
     rng.normal(size=(sum(lengths),5)).astype(np.float32),starts,ids,mask)
 order=np.concatenate([np.arange(starts[i],starts[i+1])[rng.permutation(n)] for i,n in enumerate(lengths)])
 perm=ServiceObservation(obs.fixed.copy(),obs.users[order],starts,ids[order],mask[order])
 actor=make_service_actor(small(key))
 a=actor.distribution(actor.encode(obs.tensor('cpu')));b=actor.distribution(actor.encode(perm.tensor('cpu')))
 torch.testing.assert_close(a[0],b[0],atol=2e-6,rtol=1e-5)
 torch.testing.assert_close(a[1],b[1],atol=2e-6,rtol=1e-5)
 torch.testing.assert_close(a[2],b[2][np.argsort(order)],atol=2e-6,rtol=1e-5)
 _,chain=actor.sample(obs)
 np.testing.assert_allclose(chain.logp,chain.movement_logp+chain.score_logp,atol=2e-6)


def test_entirely_empty_diffusion_motion_only():
 from uav_research.diffusion_dppo.observations import ServiceObservation,offsets
 c=small('F2a');o=ServiceObservation(np.zeros((4,c.env.fixed_obs_dim),np.float32),
   np.empty((0,5),np.float32),offsets([0]*4),np.empty(0,np.int64),np.empty(0,bool))
 actor=make_service_actor(c);action,chain=actor.sample(o)
 assert action.scores.size==chain.scores.size==0
 np.testing.assert_allclose(chain.score_logp,0)
 torch.testing.assert_close(actor.evaluate_chain(o,chain),torch.tensor(chain.logp),atol=2e-3,rtol=2e-4)
