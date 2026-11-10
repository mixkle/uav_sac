import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_research.boundary import directional_boundary_cost
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.evaluate import evaluate, policy_from_checkpoint
from uav_research.bridge_diagnostics import audit_policy
from uav_research.runtime import load_checkpoint, configure_runtime
from uav_research.train import train
from uav_research.vector import VectorEnvironment
from test_mappo_bridge import small
from test_online_time_factor import assert_exact

BASE = 'continuous_mappo_full_reward_time_factor_relative'


def config(central=False, penalty=False):
    cfg = small('M1')
    cfg.run.best_metric = 'task_return'
    cfg.env.boundary_penalty_weight = .05 if penalty else 0.
    cfg.env.boundary_penalty_margin_m = 100.
    if central:
        cfg.env.user_initialization = 'uniform_interior'
        cfg.env.user_initial_margin_fraction = 1/7
    return cfg.validate()


def test_cost_math():
    c = 1 / np.sqrt(2)
    positions = [[500,500],[0,500],[50,500],[100,500],[0,500],[0,500],[0,0],[0,0]]
    actions = [[-1,0],[-1,0],[-1,0],[-1,0],[1,0],[0,1],[-c,-c],[0,0]]
    result = directional_boundary_cost(positions,actions,1000,1000,100)
    np.testing.assert_allclose(result,[0,1,.25,0,0,0,1,0])
    rng=np.random.default_rng(1)
    xy=rng.uniform(0,1000,(10000,2)); a=rng.normal(size=(10000,2)); a/=np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
    v=directional_boundary_cost(xy,a,1000,1000,100)
    assert v.min() >= 0 and v.max() <= 1


def test_configs_and_initialization():
    base=load_config(f'tests/fixtures/configs/{BASE}_100k.yaml')
    for suffix,center,penalty in [('central',True,False),('boundary',False,True),('central_boundary',True,True)]:
        cfg=load_config(f'tests/fixtures/configs/{BASE}_{suffix}_100k.yaml')
        expected=copy.deepcopy(base); expected.run.best_metric='task_return'
        expected.env.boundary_penalty_weight=.05 if penalty else 0.
        if center:
            expected.env.user_initialization='uniform_interior';expected.env.user_initial_margin_fraction=1/7
        assert cfg.as_dict()==expected.as_dict()
        assert cfg.run.total_steps == 100000 and cfg.run.cpu_threads == 8
        a,b=UAVEnvironment(base.env),UAVEnvironment(cfg.env);a.reset(87);b.reset(87)
        np.testing.assert_array_equal(a.uav_xy,b.uav_xy)
        np.testing.assert_allclose(b.user_xy,500+a.user_xy*(5/7) if center else a.user_xy)
        assert a.movement_rng.bit_generator.state==b.movement_rng.bit_generator.state
    bad=copy.deepcopy(base);bad.env.boundary_penalty_weight=.05;bad.env.boundary_penalty_margin_m=1750
    with pytest.raises(ValueError,match='half'):bad.validate()
    # Disabled legacy modes are allowed on small fields.
    bad.env.boundary_penalty_weight=0;bad.validate()


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_reward_dynamics_and_snapshot(backend):
    base=config(); cfg=config(penalty=True);base.env.backend=cfg.env.backend=backend
    a,b=UAVEnvironment(base.env),UAVEnvironment(cfg.env);a.reset(45);b.reset(45)
    xy=np.tile([1.,cfg.env.height/2],(cfg.env.num_uavs,1))
    a.set_positions(xy,a.user_xy);b.set_positions(xy,b.user_xy)
    actions=np.tile([-1.,0.],(cfg.env.num_uavs,1))
    ra,rb=a.step(actions),b.step(actions)
    np.testing.assert_array_equal(a.state,b.state)
    assert b.info['boundary_penalty']>0
    assert b.info['reward']==b.info['task_reward']-b.info['boundary_penalty']
    np.testing.assert_allclose(ra[2]-rb[2],b.info['boundary_penalty'])
    np.testing.assert_allclose(b.trace_frame()['reward_contributions'].sum(),rb[2])
    np.testing.assert_allclose(b.info['boundary_cost'],(1-1/100)**2)
    assert b.actual_displacement[0,0]>0  # reflected inward, original command still penalized
    saved=b.snapshot();restored=UAVEnvironment(cfg.env);restored.restore(saved)
    assert_exact(saved,restored.snapshot());assert_exact(b.info,restored.info)
    b.step(actions);restored.step(actions);assert_exact(b.snapshot(),restored.snapshot())
    restored.reset(4);assert restored.info['boundary_penalty']==0
    assert restored.trace_frame()['reward_terms'].shape==(4,)


@pytest.mark.parametrize('backend',['batch','process'])
def test_vector_equivalence(backend):
    cfg=config(penalty=True);configure_runtime(cfg)
    refs=[UAVEnvironment(cfg.env) for _ in range(2)]
    for e,seed in zip(refs,[31,32]):e.reset(seed)
    with VectorEnvironment(cfg.env,2,backend) as vec:
        vec.reset([31,32])
        for _ in range(3):
            actions=np.tile([-1.,0.],(2,cfg.env.num_uavs,1))
            _,state,reward,_,_,infos=vec.step(actions)
            for i,e in enumerate(refs):
                x=e.step(actions[i]);np.testing.assert_allclose(state[i],x[1],atol=2e-7)
                np.testing.assert_allclose(reward[i],x[2]);assert_exact(infos[i],x[-1])


@pytest.mark.parametrize('center,penalty',[(True,False),(False,True),(True,True)])
def test_complete_resume_and_audit(tmp_path,center,penalty):
    cfg=config(center,penalty)
    full=train(cfg,tmp_path/'full')
    half=copy.deepcopy(cfg);half.run.total_steps=32
    first=train(half,tmp_path/'split');resumed=train(cfg,tmp_path/'split',first['resume_checkpoint'])
    a,b=[load_checkpoint(x['resume_checkpoint']) for x in [full,resumed]]
    for key in ['learner','environments','rng','best_score','best_selection']:
        assert_exact(a[key],b[key])
    pcfg,policy=policy_from_checkpoint(tmp_path/'full/last.pt','cpu')
    rows,summary=evaluate(pcfg,policy,pcfg.run.validation_seeds,tmp_path/'evaluation',trace_count=1,reward_curves=True)
    data,_=audit_policy(pcfg,policy,tmp_path/'audit',samples=4)
    with np.load(tmp_path/'evaluation/reward_curves.npz') as curves:
        np.testing.assert_array_equal(data['rewards'],curves['reward'])
        np.testing.assert_allclose(curves['contributions'].sum(-1),curves['reward'])
        assert len(curves['names'])==5 if penalty else len(curves['names'])==4
    np.testing.assert_allclose(data['component_returns'].sum(-1),data['value_labels'],atol=1e-10)
    for r in rows:np.testing.assert_allclose(r['task_return']-r['return'],r['boundary_penalty_return'],atol=1e-11)
    selection=json.loads((tmp_path/'full/best_selection.json').read_text())
    assert selection['selection_metric']=='task_return'
    assert selection['selection_score']==selection['validation_task_return']
    wrong=copy.deepcopy(cfg);wrong.run.best_metric='return'
    with pytest.raises(ValueError,match='best_metric'):train(wrong,tmp_path/'wrong',full['resume_checkpoint'])
    wrong=copy.deepcopy(cfg);wrong.env.boundary_penalty_weight+=.01
    with pytest.raises(ValueError,match='Environment/algorithm'):train(wrong,tmp_path/'other',full['resume_checkpoint'])


def test_best_uses_task_return(tmp_path,monkeypatch):
    import uav_research.train as module
    import importlib
    module=importlib.import_module('uav_research.train')
    real=module.evaluate
    scores=iter([(20.,21.),(10.,22.),(30.,19.)])
    def fake(*args,**kwargs):
        rows,summary=real(*args,**kwargs)
        train_score,task_score=next(scores)
        summary['mean']['return']=train_score;summary['mean']['task_return']=task_score
        return rows,summary
    monkeypatch.setattr(module,'evaluate',fake)
    cfg=config(penalty=True);train(cfg,tmp_path/'run')
    s=json.loads((tmp_path/'run/best_selection.json').read_text())
    assert s['env_steps']==32 and s['selection_score']==22 and s['validation_return']==10


def test_four_way_report_and_rebuild(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from report_mappo_boundary import LABELS,build,publish,validate_configs
    configs={label:config(c,p) for label,c,p in [('M1',False,False),('M1-C',True,False),('M1-B',False,True),('M1-CB',True,True)]}
    configs['M1'].run.best_metric='return';validate_configs(configs)
    runs=[]
    for label,cfg in configs.items():
        run=tmp_path/label;train(cfg,run);runs.append(run)
    out=tmp_path/'report';result=build(runs,out,samples=4)
    assert len(result['paired'])==102
    assert all(g['config']['algo']['critic_parameterization']=='time_factor' for g in result['groups'].values())
    assert (out/'figures/boundary_last.png').exists()
    assert (out/'M1-CB/figures/topology_last_30020.png').exists()
    before=(out/'policy_metrics.csv').read_bytes()
    # Rebuilding may only use report data, never access source runs.
    for run in runs:run.rename(run.with_name(run.name+'_moved'))
    publish(out,json.loads((out/'comparison.json').read_text()))
    assert before==(out/'policy_metrics.csv').read_bytes()


def test_legacy_log_migration(tmp_path):
    from uav_research.runtime import CSVLogger
    path=tmp_path/'legacy.csv';path.write_text('return,coverage\n12.5,0.4\n')
    fields=['return','task_return','coverage','boundary_penalty_return']
    log=CSVLogger(path,fields,additive_defaults={'task_return':lambda r:r['return'],'boundary_penalty_return':0.})
    log.write(dict(zip(fields,[13.,13.,.5,0.])));log.close()
    import csv
    rows=list(csv.DictReader(path.open()))
    assert rows[0]=={'return':'12.5','task_return':'12.5','coverage':'0.4','boundary_penalty_return':'0.0'}
    assert len(rows)==2
    with pytest.raises(ValueError,match='schema'):CSVLogger(path,['return'])


def test_empty_region_is_not_zero_service():
    from uav_research.evaluate import summarize
    rows=[dict(scene_seed=1,train_seed=0,method='mappo',coverage=.4,edge_coverage=None,interior_coverage=None),
          dict(scene_seed=2,train_seed=0,method='mappo',coverage=.6,edge_coverage=.8,interior_coverage=None)]
    for ordered in [rows,rows[::-1]]:
        summary=summarize(ordered)
        assert summary['mean']['coverage']==.5 and summary['mean']['edge_coverage']==.8
        assert summary['valid_episode_counts']['edge_coverage']==1
        assert summary['valid_episode_counts']['interior_coverage']==0
        assert 'interior_coverage' not in summary['mean']
