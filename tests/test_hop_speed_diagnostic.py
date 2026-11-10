import copy
import numpy as np
import pytest
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment, reflect
from uav_research.hop_speed_diagnostic import HopSpeedEnvironment, HopSpeedVector, hop_speed_factor
from uav_research.runtime import configure_runtime
from uav_research.vector import VectorEnvironment


def config():
    c=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    c.run.device='cpu';configure_runtime(c)
    c.env.num_uavs=4;c.env.num_users=12
    return c.env


def test_curve_and_unreachable_sentinel():
    h=np.arange(1,20);a=hop_speed_factor(h,18)
    assert a[0]==.25 and a[-1]==1 and a[-2]<1
    assert np.all(np.diff(a)>0)
    np.testing.assert_allclose(a[:18],.25+.75*(1-np.exp(-.4*(h[:18]-1)))/(1-np.exp(-.4*18)))
    np.testing.assert_array_equal(hop_speed_factor(h,18,1),1)
    for bad in ([0],[-1],[1.5],[np.nan]):
        with pytest.raises(ValueError):hop_speed_factor(bad,18)


def test_pre_action_scale_preserves_command_history_and_ue_stream():
    c=config();e=HopSpeedEnvironment(c);b=UAVEnvironment(c)
    e.reset(10000);b.reset(10000)
    # Override the pre-action graph to test h=1/2/U/disconnected explicitly.
    e.B[:]=5;e.B[:,0]=[1,2,4,5]
    origin=e.uav_xy.copy();action=np.tile([.4,.3],(4,1)).astype(np.float32)
    a=hop_speed_factor([1,2,4,5],4)
    old_discovery=e.sensed.sum(1).copy()
    e.step(action);b.step(action)
    np.testing.assert_allclose(e.uav_xy,reflect(origin+(action*c.max_speed*c.dt)*a[:,None],[c.width,c.height]))
    np.testing.assert_array_equal(e.history[:,:2],action)
    np.testing.assert_allclose(e.history[:,2],np.array([1,2,4,5])/5)
    np.testing.assert_allclose(e.history[:,3],old_discovery/c.capacity)
    np.testing.assert_array_equal(e.last_action,action)
    np.testing.assert_array_equal(e.user_xy,b.user_xy)
    assert e.movement_rng.bit_generator.state==b.movement_rng.bit_generator.state
    old=e.speed_factor.copy();e._refresh();np.testing.assert_array_equal(e.speed_factor,old)
    state=e.snapshot();other=HopSpeedEnvironment(c);other.restore(state)
    for _ in range(3):
        x,y=e.step(action),other.step(action)
        np.testing.assert_array_equal(x[1],y[1]);assert x[2]==y[2]


def test_disabled_and_alpha_one_reproduce_physical_environment():
    c=config()
    for options in (dict(enabled=False),dict(alpha_min=1.)):
        e=HopSpeedEnvironment(c,**options);b=UAVEnvironment(c)
        x,y=e.reset(10001),b.reset(10001)
        np.testing.assert_array_equal(x[0].fixed,y[0].fixed)
        action=np.tile([.6,-.2],(4,1)).astype(np.float32)
        for _ in range(8):
            x,y=e.step(action),b.step(action)
            np.testing.assert_array_equal(x[0].fixed,y[0].fixed)
            np.testing.assert_array_equal(x[1],y[1]);assert x[2]==y[2]
            for key in y[-1]:
                if np.isscalar(y[-1][key]):assert x[-1][key]==y[-1][key],key


def test_batch_and_scalar_agree_and_reflection_uses_scaled_command():
    c=config()
    with HopSpeedVector(c,2) as v:
        v.reset([10000,10001]);singles=[HopSpeedEnvironment(c) for _ in range(2)]
        for e,s in zip(singles,[10000,10001]):e.reset(s)
        action=np.full((2,4,2),.4,np.float32)
        for _ in range(3):
            x=v.step(action);ys=[e.step(a) for e,a in zip(singles,action)]
            np.testing.assert_array_equal(x[1],np.stack([y[1] for y in ys]))
    e=HopSpeedEnvironment(c);e.reset(1)
    e.uav_xy[:]=[3499,100];e.B[:]=5;e.B[:,0]=1
    e.step(np.tile([1,0],(4,1)).astype(np.float32))
    np.testing.assert_allclose(e.uav_xy[:,0],3493.5)
    assert e.info['boundary_reflection_fraction']==1
