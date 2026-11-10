import numpy as np
import pytest
from uav_research.config import load_config
from uav_research.link_safety_diagnostic import maximum_link_scale,visible_parent


def test_largest_feasible_scale_and_empty_cases():
    s,ok=maximum_link_scale([630,0],[30,0],.5,640)
    assert ok and s==pytest.approx(1/3)
    assert maximum_link_scale([650,0],[5,0],.5,640)==(0.,False)
    assert maximum_link_scale([650,0],[-30,0],.5,640)==(.5,True)
    assert maximum_link_scale([650,0],[-10,0],.5,640)==(0.,False)
    assert maximum_link_scale([640,0],[0,0],.5,640)==(.5,True)
    assert maximum_link_scale([650,0],[0,0],.5,640)==(0.,False)
    assert maximum_link_scale([640,0],[1,0],.5,640)==(0.,True)
    s,ok=maximum_link_scale([0,640],[30,0],1,640)
    assert ok and abs(s)<1e-12
    # The segment enters and leaves the ball; the greatest root is required.
    s,ok=maximum_link_scale([-650,0],[2000,0],1,640)
    assert ok and s==pytest.approx(1290/2000)


def test_randomized_solver_against_dense_grid():
    rng=np.random.default_rng(70123)
    for _ in range(200):
        r=rng.uniform(-700,700,2);v=rng.uniform(-30,30,2);alpha=rng.uniform(0,1)
        s,ok=maximum_link_scale(r,v,alpha,640)
        grid=np.linspace(0,alpha,10001)
        feasible=np.linalg.norm(r+grid[:,None]*v,axis=1)<=640
        if ok:
            assert 0<=s<=alpha and np.linalg.norm(r+s*v)<=640+1e-7
            if feasible.any():assert s>=grid[feasible].max()-1e-10
            assert not np.any(feasible&(grid>s+1e-10))
        else:assert not feasible.any()


def test_parent_choice_only_reads_visible_slots_and_lower_hop():
    e=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml').env
    fixed=np.zeros((18,e.fixed_obs_dim),np.float32);fixed[0,3]=3/19
    start=e.self_dim+2*e.num_bs
    for k,(j,h,xy) in enumerate([(1,3,[.01,0]),(2,4,[.02,0]),(3,2,[.03,0])]):
        p=start+k*e.neighbor_dim;fixed[0,p]=(j+1)/18;fixed[0,p+1:p+3]=xy;fixed[0,p+4]=h/19
    j,xy,h=visible_parent(fixed,e,0,require_lower_hop=True)
    assert j==3 and h==3;np.testing.assert_allclose(xy,[105,0],atol=1e-5)
    fixed[0,start+2*e.neighbor_dim+4]=5/19
    assert visible_parent(fixed,e,0,require_lower_hop=True)[0]==-1
    assert visible_parent(fixed,e,0,require_lower_hop=False)[0]==1
    for own in (1,19):
        fixed[0,3]=own/19;assert visible_parent(fixed,e,0,require_lower_hop=False)[0]==-1


def test_environment_geometry_history_restoration_and_batch_equivalence():
    from uav_research.hop_speed_diagnostic import HopSpeedEnvironment
    from uav_research.link_safety_diagnostic import LinkSafetyEnvironment,LinkSafetyVector
    from uav_research.runtime import configure_runtime
    c=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml');c.run.device='cpu';configure_runtime(c)
    c.env.num_uavs=4;c.env.num_users=12;e=c.env
    constrained=LinkSafetyEnvironment(e,margin=5);hop=HopSpeedEnvironment(e)
    constrained.reset(0);hop.reset(0)
    xy=np.array([[1200,800],[1840,800],[2800,2600],[2800,3200]],np.float64)
    ue=np.full((12,2),[2000,1000])
    constrained.set_positions(xy,ue);hop.set_positions(xy,ue)
    assert constrained.B[0].min()==1 and constrained.B[1].min()==2
    action=np.array([[0,0],[1,0],[0,0],[0,0]],np.float32)
    origin=constrained.uav_xy.copy()
    a,b=constrained.step(action),hop.step(action)
    assert constrained.parents[1]==0 and constrained.constraint_feasible[1]
    assert constrained.speed_factor[1]==pytest.approx(5/30)
    assert np.linalg.norm(constrained.uav_xy[1]-origin[0])==pytest.approx(645)
    np.testing.assert_array_equal(constrained.history[:,:2],action)
    np.testing.assert_array_equal(constrained.user_xy,hop.user_xy)
    # Outside the buffered ball, an outward ray has no feasible scale.
    xy[1,0]=1848;constrained.set_positions(xy,ue)
    constrained.step(action)
    assert not constrained.constraint_feasible[1] and constrained.speed_factor[1]==0
    assert constrained.info['link_constraint_infeasible_count']>=1
    snapshot=constrained.snapshot();other=LinkSafetyEnvironment(e,margin=5);other.restore(snapshot)
    x,y=constrained.step(action),other.step(action)
    np.testing.assert_array_equal(x[1],y[1]);assert x[2]==y[2]
    with LinkSafetyVector(e,2,margin=5) as v:
        v.reset([30000,30001]);single=[LinkSafetyEnvironment(e,margin=5) for _ in range(2)]
        for env,seed in zip(single,[30000,30001]):env.reset(seed)
        for _ in range(3):
            actions=np.tile(action,(2,1,1));x=v.step(actions);ys=[env.step(action) for env in single]
            np.testing.assert_array_equal(x[1],np.stack([z[1] for z in ys]))


def test_static_parent_constraint_does_not_guarantee_joint_moving_link():
    from uav_research.link_safety_diagnostic import LinkSafetyEnvironment
    c=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml');c.env.num_uavs=4;c.env.num_users=12
    env=LinkSafetyEnvironment(c.env,margin=5);env.reset(0)
    xy=np.array([[1200,800],[1840,800],[2800,2600],[2800,3200]],np.float64)
    env.set_positions(xy,np.full((12,2),[2000,1000]))
    # U0 is directly connected and moves left 7.5 m. Child's old-parent
    # bound allows +5 m, so the actual joint distance becomes 652.5 m.
    env.step(np.array([[-1,0],[1,0],[0,0],[0,0]],np.float32))
    assert env.constraint_feasible[1]
    assert env.info['link_constraint_static_violation_count']==0
    assert np.linalg.norm(env.uav_xy[0]-env.uav_xy[1])==pytest.approx(652.5)
    assert env.info['link_constraint_feasible_but_actual_broken_count']>=1
