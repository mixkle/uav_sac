import copy
import numpy as np
import pytest
import torch
from torch import nn

from uav_research.critic_calibration import discounted_episode_returns
from uav_research.critic_time_factor import (
    ARMS, OfflineCritic, horizon_factors, target_statistics, make_optimizer,
    fit_epoch, predict, save_offline, restore_offline, paired_bootstrap,
)
from uav_research.evaluate import policy_from_checkpoint


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_data():
    rng=np.random.default_rng(17)
    states=rng.normal(size=(3,4,5)).astype(np.float32)
    states[:,:,-1]=np.arange(4)/4
    returns=discounted_episode_returns(rng.uniform(.1,.8,(3,4)),.995)
    return dict(states=states,returns=returns)


def model(arm):
    source=nn.Sequential(nn.Linear(5,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),nn.Linear(8,1))
    return OfflineCritic(source,arm,target_statistics(small_data()['returns'],.995),4,.995)


@pytest.mark.parametrize('gamma',[0.,.5,.995,1.])
def test_factor_identity_terminal_and_coverage_bounds(gamma):
    r=np.array([[0.,.2,.8,1.],[1.,1.,1.,1.]])
    g=discounted_episode_returns(r,gamma)
    h=horizon_factors(np.arange(4,0,-1),gamma)
    assert h[-1]==1
    assert horizon_factors([0],gamma)[0]==0
    c=g/h
    assert ((c>=-1e-12)&(c<=1+1e-12)).all()
    np.testing.assert_allclose(c*h,g)
    if gamma==1:np.testing.assert_array_equal(h,[4,3,2,1])


@pytest.mark.parametrize('arm',ARMS)
def test_fit_only_stats_initialization_and_time_shuffle(arm):
    data=small_data(); stats=target_statistics(data['returns'],.995)
    g=data['returns']; h=horizon_factors(np.arange(4,0,-1),.995)
    assert stats['return_std']==g.std(ddof=0)
    assert stats['average_std']==(g/h).std(ddof=0)
    source=nn.Sequential(nn.Linear(5,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),nn.Linear(8,1))
    original=copy.deepcopy(source.state_dict())
    m=OfflineCritic(source,arm,stats,4,.995)
    for name,p in source.state_dict().items():torch.testing.assert_close(p,original[name],rtol=0,atol=0)
    for name,p in m.network.state_dict().items():
        if name not in ['4.weight','4.bias']:torch.testing.assert_close(p,original[name],rtol=0,atol=0)
    assert torch.count_nonzero(m.network[-1].weight)==0
    assert torch.count_nonzero(m.network[-1].bias)==0
    assert not make_optimizer(m).state
    pred=predict(m,data)
    expected=np.full((3,4),stats['return_mean']) if arm==ARMS[0] else np.broadcast_to(h*stats['average_mean'],(3,4))
    np.testing.assert_allclose(pred['value'],expected,rtol=1e-6)
    flat=torch.tensor(data['states'].reshape(-1,5))
    order=torch.tensor([9,1,7,3,11,0,10,5,6,2,4,8])
    torch.testing.assert_close(m(flat)[order],m(flat[order]),rtol=0,atol=0)
    torch.testing.assert_close(m.factors(flat[order]),torch.tensor(np.tile(h,3)[order],dtype=torch.float32))
    terminal=flat[:1].clone();terminal[:,-1]=1
    assert m(terminal).item()==0
    with pytest.raises(ValueError,match='Terminal'):
        m.loss(terminal,torch.zeros(1))
    bad=flat[:1].clone();bad[:,-1]=.123
    with pytest.raises(ValueError,match='State time'):
        m(bad)


def test_a_b_same_return_loss_and_factor_gradient():
    a,b=model(ARMS[0]),model(ARMS[1])
    x=torch.tensor(small_data()['states'][:1,2])
    h=b.factors(x).item(); c=.4; target=torch.tensor([.8])
    with torch.no_grad():
        a.network[-1].bias.fill_((h*c-a.statistics['return_mean'])/a.statistics['return_std'])
        b.network[-1].bias.fill_((c-b.statistics['average_mean'])/b.statistics['average_std'])
    torch.testing.assert_close(a(x),b(x))
    torch.testing.assert_close(a.loss(x,target),b.loss(x,target))
    loss=b.loss(x,target);loss.backward()
    expected=(b(x).item()-target.item())/b.statistics['return_std']**2*h*b.statistics['average_std']
    assert b.network[-1].bias.grad.item()==pytest.approx(expected,rel=1e-5)
    # B/C are identical before fitting despite different objectives.
    base=model(ARMS[1]); other=OfflineCritic(base.network,ARMS[2],base.statistics,4,.995)
    torch.testing.assert_close(base(x),other(x),rtol=0,atol=0)


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_nested(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_nested(x,y)
    else:assert a==b


@pytest.mark.parametrize('arm',ARMS)
def test_training_checkpoint_resume_and_online_rejection(tmp_path,arm):
    data=small_data(); m=model(arm);o=make_optimizer(m);rng=np.random.default_rng(940001)
    initial=copy.deepcopy(m.state_dict())
    update=fit_epoch(m,o,data,5,rng)
    assert update['optimizer_steps']==3
    save_offline(tmp_path/'offline.pt',m,o,rng,1,dict(test=True))
    expected=fit_epoch(m,o,data,5,rng)
    another=model(arm);opt=make_optimizer(another);gen=np.random.default_rng(8)
    restored=restore_offline(tmp_path/'offline.pt',another,opt,gen)
    assert restored['epoch']==1
    actual=fit_epoch(another,opt,data,5,gen)
    assert expected==actual
    equal_nested(m.state_dict(),another.state_dict())
    equal_nested(o.state_dict(),opt.state_dict())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items() if not k.startswith('network.'))
    # These offline artifacts deliberately lack the online config/learner schema.
    with pytest.raises(KeyError):policy_from_checkpoint(tmp_path/'offline.pt','cpu')
    wrong=model(ARMS[(ARMS.index(arm)+1)%3])
    with pytest.raises(ValueError,match='Incompatible'):
        restore_offline(tmp_path/'offline.pt',wrong,make_optimizer(wrong),gen)


def test_paired_order_and_bootstrap():
    data=small_data(); orders=[]
    for arm in ARMS:
        m=model(arm);o=make_optimizer(m);g=np.random.default_rng(940001)
        orders.append([fit_epoch(m,o,data,5,g)['permutation_sha256'] for _ in range(2)])
    assert orders[0]==orders[1]==orders[2]
    y=data['returns']
    pairs=paired_bootstrap(dict(bad=y+2,good=y),y,2000,950001)
    assert pairs==paired_bootstrap(dict(bad=y+2,good=y),y,2000,950001)
    assert all(r['delta']==pytest.approx(-2) and r['high']<0 for r in pairs)


def test_constant_targets_and_time_only_correlation():
    from uav_research.critic_time_factor import error_metrics
    stats=target_statistics(np.zeros((3,4)),.995)
    assert stats['return_std']==stats['average_std']==1e-6
    # Decimal reference curves can leave tiny roundoff after centering; no
    # between-scene signal should be reported for any time-only reference.
    curve=np.linspace(.01,70.73,400)
    p=np.broadcast_to(curve,(32,400))
    y=p+np.random.default_rng(5).normal(size=p.shape)
    assert error_metrics(p,y)['time_centered_correlation'] is None
