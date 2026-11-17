import copy
import numpy as np
import pytest
from uav_research.association_timing import solutions, TimingEnvironment
from uav_research.config import load_config
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.observations import ServiceAction


def test_static_scores_and_candidates():
    b=np.array([[1,1,0],[0,1,1]],bool)
    q=np.array([[.8,.9,.1],[.2,.7,.6]])
    r=solutions(b,b,q,q,q,1)
    for name in r:
        np.testing.assert_array_equal(r[name][0],r['T2'][0])


def test_new_candidate_and_invalid_old_proposal():
    b=np.array([[1,1,0]],bool);a=np.array([[0,1,1]],bool)
    pre=np.array([[.9,.8,.1]]);post=np.array([[.1,.7,.95]])
    r=solutions(b,a,pre,pre,post,1)
    assert r['T2'][0].sum()==0
    assert r['P1'][0].sum()==0
    assert r['P2'][0][0,1]==1
    assert r['T3'][0][0,2]==1
    assert r['T3'][1].sum()==1  # single round, not iterative fill


def test_p1_only_changes_owner_not_support():
    b=np.ones((2,1),bool)
    pre=np.array([[.9],[.8]]);post=pre[::-1].copy()
    r=solutions(b,b,pre,pre,post,1)
    assert r['T2'][0][0,0]==1 and r['P1'][0][1,0]==1
    np.testing.assert_array_equal(r['P1'][1],r['T2'][1])
    np.testing.assert_array_equal(r['P1'][2],r['T2'][2])


def test_ties_empty_capacity_and_finite_validation():
    b=np.ones((2,3),bool);q=np.full((2,3),.5)
    r=solutions(b,b,q,q,q,2)
    for c,_,_ in r.values():
        np.testing.assert_array_equal(c,np.array([[1,1,0],[0,0,0]]))
    for c,_,_ in solutions(b,np.zeros_like(b),q,q,q,2).values():assert c.sum()==0
    q[0,0]=np.nan
    with pytest.raises(ValueError):solutions(b,b,q,q,q,2)


def small():
    cfg=load_config('configs/system_ablation_1m/A3.yaml')
    cfg.env.num_uavs=4;cfg.env.num_users=24;cfg.env.horizon=4
    cfg.run.num_envs=2;cfg.run.device='cpu'
    cfg.run.cpu_threads=2;cfg.run.numba_threads=2
    return cfg.validate()


def action(env):
    o=env.obs
    return ServiceAction(np.array([[.7,-.5]]*env.cfg.num_uavs,np.float32),
        np.linspace(.01,.99,len(o.candidate_ids)),o.candidate_ids.copy(),
        np.r_[0,np.cumsum(o.candidate_lengths)])


def test_t1_exact_production_and_shadow_isolation():
    cfg=small();base=ServiceEnvironment(cfg.env);base.reset(10000)
    a=TimingEnvironment(cfg.env,'T1',True);a.restore(base.snapshot())
    b=TimingEnvironment(cfg.env,'T1',False);b.restore(base.snapshot())
    for _ in range(4):
        act=action(base);r=base.step(act);ra=a.step(act);rb=b.step(act)
        for e in (a,b):
            for name in ('uav_xy','user_xy','C','Y','B','history','service_age','last_owner'):
                np.testing.assert_array_equal(getattr(base,name),getattr(e,name))
            assert base.rng.bit_generator.state==e.rng.bit_generator.state
            assert base.movement_rng.bit_generator.state==e.movement_rng.bit_generator.state
        assert r[2:]==ra[2:]==rb[2:]
    with pytest.raises(RuntimeError):a.step(action(a))


def test_protocol_physics_and_snapshot_history():
    cfg=small();base=ServiceEnvironment(cfg.env);base.reset(10001)
    envs=[TimingEnvironment(cfg.env,p,True) for p in ('T1','T2','T3')]
    for e in envs:e.restore(base.snapshot())
    act=action(base)
    for e in envs:e.step(act)
    for e in envs[1:]:
        for name in ('uav_xy','user_xy','Y','B','history'):
            np.testing.assert_array_equal(getattr(envs[0],name),getattr(e,name))
    e=envs[-1];restored=TimingEnvironment(cfg.env,'T3',True);restored.restore(e.snapshot())
    old=e.history.copy();e._refresh();np.testing.assert_array_equal(old,e.history)
    act=action(e);x=e.step(act);y=restored.step(act)
    np.testing.assert_array_equal(e.C,restored.C)
    np.testing.assert_array_equal(e.history,restored.history)
    assert x[2:]==y[2:]
    with pytest.raises(ValueError):envs[0].restore(e.snapshot())


def test_t2_matches_existing_frozen_sinr_protocol():
    cfg=small();base=ServiceEnvironment(cfg.env);base.reset(10002)
    timing=TimingEnvironment(cfg.env,'T2',True);timing.restore(base.snapshot())
    other=copy.deepcopy(cfg);other.diffusion.ablation['association']='sinr_topk';other.validate()
    production=ServiceEnvironment(other.env)
    snap=base.snapshot();snap['service_ablation']=copy.deepcopy(other.env._service_ablation)
    production.restore(snap)
    for _ in range(4):
        act=action(timing);timing.step(act);production.step(act)
        np.testing.assert_array_equal(timing.C,production.C)
        assert timing.info==production.info


def test_invalid_score_rejected_before_motion():
    e=TimingEnvironment(small().env);e.reset(10000);a=action(e)
    if len(a.scores):
        a.scores[0]=np.nan
        with pytest.raises(ValueError):e.step(a)
        assert e.t==0
