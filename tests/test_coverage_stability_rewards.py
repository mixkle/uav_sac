import copy
import numpy as np
import pytest
import torch

from uav_research.config import load_config
from uav_research.coverage_stability import (
    REWARD_ABLATIONS, DISCONNECTION_SWEEP, BASELINE_CONFIG, ablation_config, config_path, coverage_drop_metrics,
)
from uav_research.algorithms import make_learner
from uav_research.runtime import configure_runtime, seed_all
from uav_research.vector import VectorEnvironment
from uav_research.train import train
from uav_research.evaluate import evaluate, policy_from_checkpoint


def test_thresholds_endpoints_drawdown_and_constant():
    c=np.array([.8,.7,.5,.5,.6,.4,.8])
    m=coverage_drop_metrics(c)
    assert m['max_drop_1']==pytest.approx(.2)
    assert m['max_drop_5']==pytest.approx(.4)
    assert m['max_drawdown']==pytest.approx(.4)
    assert m['drop_events_10pp']==3 and m['drop_events_20pp']==2
    assert m['peak_step']==1
    assert coverage_drop_metrics(np.full(400,.6))['max_drop_1']==0
    assert coverage_drop_metrics(np.full(400,.6))['maintain_coverage']==pytest.approx(0)
    assert coverage_drop_metrics([0])['max_drop_5']==0
    for bad in ([],[float('nan')],[-.1],[1.1],[[.2,.3]]):
        with pytest.raises(ValueError):coverage_drop_metrics(bad)


def test_configs_only_change_weights_and_device_and_initial_networks_match():
    base=load_config(BASELINE_CONFIG)
    device=configure_runtime(base)
    seed_all(0)
    reference=make_learner(base,device)
    for group in {**REWARD_ABLATIONS, **DISCONNECTION_SWEEP}:
        cfg=load_config(config_path(group))
        assert cfg.as_dict()==ablation_config(base,group).as_dict()
        assert cfg.run.total_steps==1_000_000 and cfg.run.cpu_threads==8 and cfg.run.numba_threads==4
        assert cfg.algo==base.algo and cfg.env.throughput_weight==.1
        cfg.run.device='cpu';seed_all(0)
        model=make_learner(cfg,device)
        for attr in ('actor','value'):
            left=getattr(reference,attr).state_dict();right=getattr(model,attr).state_dict()
            assert all(torch.equal(left[k],right[k]) for k in left)
        assert not model.actor_opt.state and not model.value_opt.state


def test_saved_report_keeps_common_return_separate_from_training_reward(tmp_path,monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    import uav_research.plotting
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    import report_mappo_stability_rewards as report
    monkeypatch.setattr(uav_research.plotting,'save_figure',lambda fig,*args:plt.close(fig))
    out=tmp_path/'report';out.mkdir()
    runs={}
    base=load_config(BASELINE_CONFIG)
    for group in report.GROUPS:
        run=tmp_path/'runs'/group;run.mkdir(parents=True);runs[group]=run
        cp=run/'last.pt';cp.write_bytes(b'test-only-frozen-checkpoint-identity')
        target=out/'last'/group;target.mkdir(parents=True)
        cfg=copy.deepcopy(base) if group=='baseline' else ablation_config(base,group,'cpu')
        (target/'evaluation_config.json').write_text(json.dumps(cfg.as_dict()))
        raw=np.zeros((32,400,4));raw[:,:,0]=.6;raw[:,100,0]=.4
        raw[:,:,1]=.8;raw[:,:,2]=.3;raw[:,:,3]=2/18
        weights=np.array([1,.1,cfg.env.robustness_weight,-cfg.env.disconnection_weight])
        contributions=raw*weights;reward=contributions.sum(-1)
        curve=target/'reward_curves.npz'
        np.savez_compressed(curve,scene_seeds=report.SCENES,raw=raw,contributions=contributions,reward=reward,
                            reward_weights=weights,diagnostic_names=['task_reward'],diagnostics=reward[:,:,None])
        (target/'stability_source.json').write_text(json.dumps(dict(evaluation=dict(
            sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),scenes=report.SCENES,action_seed=880001),
            reward_curves_sha256=hashlib.sha256(curve.read_bytes()).hexdigest())))
    args=SimpleNamespace(output=out,checkpoints=['last'])
    report.rebuild(args,runs)
    saved=json.loads((out/'comparison.json').read_text())['results']['last']['summary']
    assert len({round(s['common_return'],8) for s in saved.values()})==1
    assert saved['robustness']['training_return']>saved['baseline']['training_return']
    assert saved['disconnection']['training_return']<saved['baseline']['training_return']
    assert all(s['max_drop_1']==pytest.approx(.2) for s in saved.values())
    assert all(s['drop_events_10pp']==1 for s in saved.values())


@pytest.mark.parametrize('group',list(REWARD_ABLATIONS)+list(DISCONNECTION_SWEEP))
def test_reward_physics_training_evaluation_and_restore(tmp_path,group):
    cfg=load_config(config_path(group));cfg.run.device='cpu'
    cfg.env.num_uavs=4;cfg.env.num_users=12;cfg.env.horizon=8
    cfg.run.num_envs=2;cfg.run.validation_seeds=[30000,30001]
    cfg.run.total_steps=32;cfg.run.eval_interval=32;cfg.run.log_interval=16
    cfg.run.tensorboard=False;cfg.algo.rollout_steps=8
    cfg.algo.ppo_epochs=1;cfg.algo.ppo_minibatches=2
    cfg.validate();configure_runtime(cfg)
    base=copy.deepcopy(cfg);base.env.disconnection_weight=base.env.robustness_weight=0
    with VectorEnvironment(cfg.env,2) as v,VectorEnvironment(base.env,2) as b:
        v.reset([30000,30001]);b.reset([30000,30001])
        actions=np.full((2,4,2),.1,np.float32)
        for _ in range(3):
            x,y=v.step(actions),b.step(actions)
            np.testing.assert_array_equal(x[1],y[1])
            for reward,info in zip(x[2],x[-1]):
                expected=info['coverage']+.1*info['tau']+cfg.env.robustness_weight*info['robustness']-cfg.env.disconnection_weight*info['disconnection_ratio']
                assert reward==pytest.approx(expected)
                assert 0<=info['robustness']<=(1-1/4)*info['coverage']+1e-12
    full=train(cfg,tmp_path/'full')
    part=copy.deepcopy(cfg);part.run.total_steps=16
    paused=train(part,tmp_path/'split')
    resumed=train(cfg,tmp_path/'split',paused['resume_checkpoint'])
    from uav_research.runtime import load_checkpoint
    from test_online_time_factor import assert_exact
    assert_exact(load_checkpoint(full['resume_checkpoint'])['learner'],load_checkpoint(resumed['resume_checkpoint'])['learner'])
    evaluated,policy=policy_from_checkpoint(tmp_path/'full/last.pt','cpu')
    evaluate(evaluated,policy,[30000,30001],tmp_path/'evaluation',trace_count=1,reward_curves=True)
    with np.load(tmp_path/'evaluation/reward_curves.npz') as z:
        np.testing.assert_allclose(z['contributions'].sum(-1),z['reward'])
        assert all(np.isfinite(z[k]).all() for k in ['raw','contributions','reward'])
