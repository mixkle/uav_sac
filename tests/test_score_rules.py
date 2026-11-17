import numpy as np
import pytest
from uav_research.score_rule_diagnostic import (quality,rule_scores,spearman,chosen,
    interruption_rows,ScoreRuleEnvironment,survival_rows)
from uav_research.association_timing import sinr_scores
from uav_research.diffusion_dppo.environment import resolve_topk
from uav_research.config import load_config
from uav_research.diffusion_dppo.observations import ServiceAction


def test_quality_monotonic_stable_and_global_order():
    y=np.array([-10000.,-3,0,10,10000.]);q=quality(y)
    assert np.isfinite(q).all() and np.all(np.diff(q)>0)
    assert q[3]==pytest.approx(1)
    rng=np.random.default_rng(3);y=rng.uniform(-3,40,(18,150));b=rng.random(y.shape)<.3
    a=rng.random(y.shape)<.7
    for x,z in zip(resolve_topk(b,a,quality(y),10),resolve_topk(b,a,sinr_scores(y,-3),10)):
        np.testing.assert_array_equal(x,z)


def test_local_load_constant_and_no_competition():
    y=np.array([[0.,1,3],[3.,2,1]]);c=np.array([[1,0,0],[0,1,0]])
    b=np.array([[1,1,0],[0,0,1]],bool)
    r0=resolve_topk(b,b,rule_scores(y,c,1),1)
    r2=resolve_topk(b,b,rule_scores(y,c,1,0,100),1)
    for x,z in zip(r0,r2):np.testing.assert_array_equal(x,z)
    assert chosen([2,1,0],[.5,.5,.5],2).tolist()==[0,1]


def test_keep_and_load_only_act_on_known_current_data():
    y=np.zeros((2,2));c=np.array([[0,1],[1,0]])
    b=np.ones_like(c,bool)
    q=rule_scores(y,c,2,.15,.3)
    assert q[0,1]-q[0,0]==pytest.approx(.15)
    assert np.all(q==rule_scores(y,c,2,.15,.3))
    with pytest.raises(ValueError):rule_scores(y,c,2,-1,0)


def test_ranks_ties_and_undefined():
    assert spearman([1,2,3],[3,2,1])==pytest.approx(-1)
    assert spearman([1,1,2],[2,2,4])==pytest.approx(1)
    assert spearman([1,1],[2,3]) is None
    assert spearman([],[]) is None


def test_exclusive_interruption_and_valid_proposal_impossible():
    prev=np.array([[1,1,1,1],[0,0,0,0]])
    cur=np.zeros_like(prev);b=np.ones_like(prev,bool);a=b.copy();p=b.copy()
    p[0,0]=False;p[:,0]=False;a[0,1]=False;p[:,1]=False;b[0,2]=False;p[:,2]=False;p[:,3]=False
    rows=interruption_rows(prev,cur,b,a,p,np.ones((2,3),int),np.zeros_like(prev),np.zeros_like(prev),np.ones(4,int),10)
    assert [r['cause'] for r in rows]==['not_in_topk','link_or_discovery','pre_candidate_missing','not_in_topk']
    p[0,0]=True
    with pytest.raises(AssertionError):interruption_rows(prev,cur,b,a,p,np.ones((2,3),int),prev,prev,np.ones(4,int),10)


def test_km_right_censor():
    rows=survival_rows([1,2,2],[1,1,0])
    assert rows[0]['survival']==pytest.approx(2/3)
    assert rows[1]['survival']==pytest.approx(1/3)
    assert rows[1]['censored']==1


def test_rule_adapter_zero_weights_same_as_t2_and_physics():
    from uav_research.association_timing import TimingEnvironment
    cfg=load_config('configs/system_ablation_1m/A3.yaml')
    cfg.env.num_uavs=4;cfg.env.num_users=24;cfg.env.horizon=3;cfg.validate()
    e=TimingEnvironment(cfg.env,'T2',False);e.reset(10000)
    f=ScoreRuleEnvironment(cfg.env,'R3',0,0);f.restore(e.snapshot())
    for _ in range(3):
        obs=e.obs;a=ServiceAction(np.full((4,2),.2,np.float32),np.full(len(obs.candidate_ids),.5),obs.candidate_ids,
            np.r_[0,np.cumsum(obs.candidate_lengths)])
        e.step(a);f.step(a)
        for name in ('uav_xy','user_xy','C','history','B','Y','service_age'):
            np.testing.assert_array_equal(getattr(e,name),getattr(f,name))
        for key in e.info:
            if key not in ('score_mean','score_std'):
                assert e.info[key]==f.info[key]


def test_fixed_snapshot_samples_reproducible_without_changing_execution_rng(tmp_path):
    import torch
    from uav_research.score_rule_experiment import sample_snapshot
    from uav_research.diffusion_dppo.policy import make_service_actor
    from uav_research.diffusion_dppo.observations import stack
    cfg=load_config('configs/system_ablation_1m/A3.yaml');cfg.run.device='cpu'
    torch.set_num_threads(2);cfg.validate()
    actor=make_service_actor(cfg).eval().requires_grad_(False)
    e=ScoreRuleEnvironment(cfg.env,'T1');e.reset(10000);obs=e.obs.copy()
    action,_=actor.sample(obs)
    e.recorded_scores=np.zeros_like(e.Y);e.recorded_logits=np.zeros_like(e.Y)
    for u in range(18):
        lo,hi=action.offsets[u:u+2];e.recorded_scores[u,action.ue_ids[lo:hi]]=action.scores[lo:hi]
    e.recorded_movement=action.movement.copy()
    prepared=e.prepare(action)
    from uav_research.environment import physical_links
    e.finish(prepared,physical_links(cfg.env,e.uav_xy,e.user_xy))
    before=e.snapshot();rng=torch.get_rng_state().clone()
    ctx=[dict(scene_seed=10000,pool='fixed',inclusion_probability=1.,design_weight=1.,extra_strata='')]
    identity=dict(checkpoint='best',action_seed=880001,begin=0,step=1)
    for dest in (tmp_path/'a',tmp_path/'b'):
        sample_snapshot(actor,stack([obs]),[e],[prepared],ctx,dest,identity,repeats=8,device='cpu')
    with np.load(tmp_path/'a/score_samples/b0_t001.npz') as a,np.load(tmp_path/'b/score_samples/b0_t001.npz') as b:
        for key in ('scores','score_logits','movement','before','after','previous_c'):
            np.testing.assert_array_equal(a[key],b[key])
    assert torch.equal(torch.get_rng_state(),rng)
    for key in ('uav_xy','user_xy','association','history'):
        np.testing.assert_array_equal(before[key],e.snapshot()[key])
    assert before['movement_rng']==e.snapshot()['movement_rng']
    assert all(p.grad is None for p in actor.parameters())
