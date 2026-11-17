import numpy as np
import pytest
from uav_research.tail_metrics import per_scene,read_series,summarize


def test_window_final_units_and_no_reset():
    c=np.zeros((2,400));c[:,300:]=np.array([[.4],[.8]]);c[:,-1]=[.5,.9]
    r=np.ones_like(c)*200e6;r[:,-1]=400e6
    s=per_scene(c,r,np.array([10,11]))
    np.testing.assert_allclose(s['tail100_coverage_pct'],[40.1,80.1])
    np.testing.assert_allclose(s['final_coverage_pct'],[50,90])
    np.testing.assert_allclose(s['tail100_throughput_mbps'],202)
    np.testing.assert_allclose(s['final_throughput_mbps'],400)


def test_between_scene_spread_is_not_temporal_variance():
    c=np.stack([np.zeros(400),np.ones(400)])
    s=per_scene(c,np.ones_like(c),np.array([1,2]))
    result,ci,_=summarize([s])
    assert result['tail100_coverage_std_pp']==0
    assert result['tail100_coverage_pct']==50
    assert ci['tail100_coverage_std_pp']==[0,0]


def test_opposing_repeats_do_not_cancel_temporal_variance():
    c=np.tile(np.arange(400)%2,(2,1)).astype(float);r=np.ones_like(c)
    a=per_scene(c,r,np.array([1,2]));b=per_scene(1-c,r,np.array([1,2]))
    result,_,_=summarize([a,b])
    assert result['tail100_coverage_std_pp']==50
    assert result['tail100_scene_std_pp']==50


def test_legacy_raw_tau_requires_explicit_physical_scale(tmp_path):
    p=tmp_path/'reward_curves.npz';c=np.full((2,400),.7)
    np.savez(p,diagnostics=c[:,:,None],diagnostic_names=['coverage'],raw=np.full((2,400,1),.3),raw_names=['tau'],scene_seeds=[1,2])
    with pytest.raises(ValueError,match='Throughput'):read_series(p)
    coverage,rate,scenes=read_series(p,throughput_scale_bps=750e6)
    np.testing.assert_array_equal(coverage,c);np.testing.assert_allclose(rate,225e6)


@pytest.mark.parametrize('bad',['short','nonfinite','duplicate','agent_matrix'])
def test_invalid_input_is_rejected(bad):
    c=np.zeros((2,400));r=np.ones_like(c);seeds=np.array([1,2])
    if bad=='short':c=c[:,:99];r=r[:,:99]
    elif bad=='nonfinite':c[0,0]=np.nan
    elif bad=='duplicate':seeds[:]=1
    elif bad=='agent_matrix':c=c[:,:,None];r=r[:,:,None]
    with pytest.raises(ValueError):per_scene(c,r,seeds)


def test_named_shadow_columns_do_not_use_actual_policy_metrics(tmp_path):
    p=tmp_path/'score_audit.npz';a=np.zeros((2,400,4));a[:,:,0]=.2;a[:,:,1]=.8;a[:,:,2]=30e6;a[:,:,3]=70e6
    np.savez(p,metrics=a,names=['coverage','sinr_score_coverage','throughput_bps','sinr_score_throughput_bps'],seeds=[1,2])
    c,r,_=read_series(p,coverage_name='sinr_score_coverage',throughput_name='sinr_score_throughput_bps')
    np.testing.assert_allclose(c,.8);np.testing.assert_allclose(r,70e6)
