import copy
import numpy as np
import pytest
import torch

from uav_research.algorithms import make_learner, gae, mappo_value_loss
from uav_research.config import load_config
from uav_research.models import DiscountedHorizon, ValueNormalizer
from uav_research.runtime import configure_runtime, seed_all, load_checkpoint
from uav_research.train import train
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.deployment import export_actor, DeployedActor
from uav_research.environment import UAVEnvironment
from uav_research.plotting import curves, trace_plot


def factor_config():
    cfg=load_config('tests/fixtures/configs/discrete_mappo_smoke.yaml')
    cfg.env.throughput_weight=cfg.env.robustness_weight=cfg.env.disconnection_weight=0.
    cfg.algo.critic_parameterization='time_factor'
    cfg.run.cpu_threads=1
    return cfg.validate()


def assert_exact(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_exact(x,y)
    else:assert a==b


@pytest.mark.parametrize('gamma',[.995,1.])
def test_horizon_indices_and_terminal(gamma):
    m=DiscountedHorizon(400,gamma)
    x=torch.zeros(5,10);x[:,-1]=torch.tensor([399,0,400,123,380])/400
    h=m(x)
    assert h[0]==1 and h[2]==0
    np.testing.assert_allclose(h.numpy(),[sum(gamma**k for k in range(n)) for n in [1,400,0,277,20]],rtol=1e-6)
    with pytest.raises(ValueError):m(torch.tensor([[0.,.123456]]))


def test_same_actor_trunk_rng_and_scaled_initial_value():
    c=factor_config();d=copy.deepcopy(c);d.algo.critic_parameterization='direct'
    device=configure_runtime(c)
    seed_all(123);a=make_learner(d,device);state_after_direct=torch.get_rng_state().clone()
    seed_all(123);b=make_learner(c,device)
    assert_exact(a.actor.state_dict(),b.actor.state_dict())
    assert_exact(state_after_direct,torch.get_rng_state())
    for k,v in a.value.state_dict().items():
        if k not in ['4.weight','4.bias']:assert_exact(v,b.value.state_dict()[k])
    x=torch.randn(4,c.env.state_dim);x[:,-1]=torch.tensor([0.,.25,.75,1.])
    expected=a.values(x)*b.horizon_factor(x)/b.horizon_factor.table[-1].float()
    torch.testing.assert_close(b.values(x),expected,rtol=1e-5,atol=1e-6)
    assert b.values(x)[-1]==0
    legacy=a.state_dict();legacy.pop('critic_parameterization')
    a.load_state_dict(legacy)
    with pytest.raises(ValueError,match='parameterization'):b.load_state_dict(legacy)
    with pytest.raises(ValueError,match='parameterization'):a.load_state_dict(b.state_dict())


def test_popart_preserves_value_average_and_adam_scaling():
    cfg=factor_config();model=make_learner(cfg,configure_runtime(cfg))
    x=torch.randn(8,cfg.env.state_dim);x[:,-1]=torch.arange(8)/8
    value,average,_=model.value_components(x)
    model.value_opt.zero_grad();value.sum().backward();model.value_opt.step()
    before_v,before_c,_=model.value_components(x)
    before_v,before_c=before_v.detach().clone(),before_c.detach().clone()
    norm_old=model.average_normalizer.std.clone()
    opt_before=copy.deepcopy(model.value_opt.state_dict())
    output_parameters=list(model.value[-1].parameters())
    parameter_ids=model.value_opt.param_groups[0]['params']
    indices=[next(i for i,q in enumerate(parameter_ids) if q is p) for p in output_parameters]
    model.average_normalizer.update(torch.linspace(.1,.8,8),model.value[-1],model.value_opt)
    scale=norm_old/model.average_normalizer.std
    after=model.value_opt.state_dict()
    for i in indices:
        torch.testing.assert_close(after['state'][i]['exp_avg'],opt_before['state'][i]['exp_avg']*scale)
        torch.testing.assert_close(after['state'][i]['exp_avg_sq'],opt_before['state'][i]['exp_avg_sq']*scale**2)
    current=model.value_components(x)
    torch.testing.assert_close(before_v,current[0],rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(before_c,current[1],rtol=1e-5,atol=1e-6)
    weights=copy.deepcopy(model.value.state_dict())
    model.normalizer.update_statistics(torch.arange(8,dtype=torch.float32))
    assert_exact(weights,model.value.state_dict())
    torch.testing.assert_close(before_v,model.values(x),rtol=1e-5,atol=1e-5)


def test_direct_popart_matches_original_and_gae_value_units():
    n=ValueNormalizer();legacy=copy.deepcopy(n)
    layer=torch.nn.Linear(3,1);old_layer=copy.deepcopy(layer)
    opt=torch.optim.Adam(layer.parameters());old_opt=torch.optim.Adam(old_layer.parameters())
    for data in [torch.tensor([1.,2.,5.]),torch.tensor([-4.,7.])]:
        old_mean,old_std=legacy.mean.clone(),legacy.std.clone()
        values=data.detach().flatten();count=values.numel();mean,var=values.mean(),values.var(unbiased=False)
        total=legacy.count+count;delta=mean-legacy.mean
        new_mean=legacy.mean+delta*(count/total)
        m2=legacy.variance*legacy.count+var*count+delta.square()*legacy.count*count/total
        with torch.no_grad():
            legacy.mean.copy_(new_mean);legacy.variance.copy_((m2/total).float().clamp_min(1e-4));legacy.count.copy_(total)
            old_layer.weight.mul_(old_std/legacy.std)
            old_layer.bias.copy_((old_std*old_layer.bias+old_mean-legacy.mean)/legacy.std)
        n.update(data,layer,opt)
        assert_exact(n.state_dict(),legacy.state_dict());assert_exact(layer.state_dict(),old_layer.state_dict())
    adv,ret=gae(torch.tensor([[.4],[.3]]),torch.tensor([[2.],[1.]]),torch.tensor([[1.],[0.]]),torch.tensor([[False],[True]]),.995,.95)
    assert ret[-1]==pytest.approx(.3)
    _,fragment=gae(torch.tensor([[.4]]),torch.tensor([[2.]]),torch.tensor([[5.]]),torch.tensor([[False]]),.995,.95)
    assert fragment.item()==pytest.approx(.4+.995*5)


def test_factor_value_clipping_and_gradient():
    cfg=factor_config();m=make_learner(cfg,configure_runtime(cfg))
    x=torch.randn(2,cfg.env.state_dim);x[:,-1]=torch.tensor([0.,31/32])
    raw,c,h=m.value_components(x)
    m.normalizer.update_statistics(torch.tensor([2.,8.]))
    predicted=m.normalizer.normalize(raw)
    old=predicted.detach()-torch.tensor([.7,-.8]);target=torch.tensor([2.,-3.])
    clipped=old+(predicted-old).clamp(-.2,.2)
    manual=torch.maximum(torch.nn.functional.huber_loss(predicted,target,delta=10.,reduction='none'),torch.nn.functional.huber_loss(clipped,target,delta=10.,reduction='none')).mean()
    torch.testing.assert_close(mappo_value_loss(predicted,target,old,.2),manual)
    grad=torch.autograd.grad(raw.sum(),c,retain_graph=True)[0]
    torch.testing.assert_close(grad,h)
    assert mappo_value_loss(predicted,target,old,.2).requires_grad


@pytest.mark.parametrize('field,value',[('name','sac'),('critic_parameterization','bad')])
def test_invalid_configuration(field,value):
    cfg=factor_config();setattr(cfg.algo,field,value)
    with pytest.raises(ValueError):cfg.validate()


def test_new_config_is_only_parameterization_change():
    a=load_config('tests/fixtures/configs/discrete_mappo_ablation_c_mobile_coverage.yaml')
    b=load_config('tests/fixtures/configs/discrete_mappo_time_factor_250k.yaml')
    assert b.algo.value_clipping and b.run.total_steps==250000
    b.algo.critic_parameterization='direct'
    assert a.as_dict()==b.as_dict()


@pytest.mark.parametrize('threads',[1,8])
def test_complete_time_factor_pipeline_and_exact_resume(tmp_path,threads):
    cfg=factor_config();cfg.run.cpu_threads=threads
    cfg.run.total_steps=64;cfg.run.eval_interval=32;cfg.run.validation_seeds=[10000]
    complete=train(cfg,tmp_path/'full')
    part=copy.deepcopy(cfg);part.run.total_steps=32
    first=train(part,tmp_path/'split')
    resumed=train(cfg,tmp_path/'split',first['resume_checkpoint'])
    a,b=load_checkpoint(complete['resume_checkpoint']),load_checkpoint(resumed['resume_checkpoint'])
    assert_exact(a['learner'],b['learner'])
    assert_exact(a['environments'],b['environments'])
    ecfg,policy=policy_from_checkpoint(tmp_path/'full'/'best.pt','cpu')
    rows,summary=evaluate(ecfg,policy,[10000],tmp_path/'evaluation',trace_count=1)
    assert np.isfinite(rows[0]['return'])
    export_actor(tmp_path/'full'/'best.pt',tmp_path/'actor.pt')
    deployed=DeployedActor(tmp_path/'actor.pt');obs,_,_=UAVEnvironment(cfg.env).reset(10000)
    np.testing.assert_array_equal(deployed.act(obs),policy.act(obs,True))
    curves([tmp_path/'full'],tmp_path/'figures');trace_plot(tmp_path/'evaluation'/'trace_10000.npz',tmp_path/'figures')
    changed=copy.deepcopy(cfg);changed.algo.critic_parameterization='direct'
    with pytest.raises(ValueError,match='Environment/algorithm changed'):
        train(changed,tmp_path/'wrong',complete['resume_checkpoint'])
